"""F1 查询期计划。只读原问题，不改变 V3 查询、时间排序或 D1 解析。"""
from __future__ import annotations

from dataclasses import dataclass
import re

from app.governance.planning import contains, plan as governance_plan
from app.governance.registry import PROPERTIES


@dataclass(frozen=True)
class SummaryRequirement:
    key: str
    label: str
    trigger: str


@dataclass(frozen=True)
class SummaryPlan:
    applicable: bool = False
    task: str = ''
    scopes: tuple[str, ...] = ()
    slots: frozenset = frozenset()
    global_scope: bool = False
    requirements: tuple[SummaryRequirement, ...] = ()
    reference: int | None = None
    window: object = None
    knowledge_cutoff: int | None = None
    warnings: tuple[str, ...] = ()
    order: str = 'source'
    constraints: tuple[str, ...] = ()
    predicates: frozenset[str] = frozenset()
    focus_terms: tuple[str, ...] = ()


FACETS = {
    'goals': ('目标', r'\b(?:goals?|objectives?|aims?)\b|目标|目的'),
    'decisions': ('决定', r'\b(?:decisions?|decided|approved|cancelled|canceled)\b|决定|决策|取消|批准'),
    'reasons': ('明确原因', r'\b(?:why|reasons?|because|due to)\b|原因|因为|由于'),
    'rules': ('规则与例外', r'\b(?:rules?|exceptions?|except|unless|only if|only after)\b|规则|例外|除非|仅当|只有'),
    'plans': ('计划', r'\b(?:plans?|planned|next steps?|intend\w*)\b|计划|打算|下一步'),
    'issues': ('未解决事项', r'\b(?:unresolved|pending|blocked|risks?|remaining issues?)\b|未解决|待确认|阻塞|风险|遗留'),
    'current': ('当前状态', r'\b(?:current|currently|status|now)\b|当前|目前|状态|现状'),
    'changes': ('变化', r'\b(?:changes?|changed|shift|evol\w*|trajectory)\b|变化|转变|演变|轨迹'),
    'structure': ('已约定的结构与格式', r'\b(?:structure|format|sections?|outline)\b|结构|格式|章节|提纲'),
}


def mentioned_projects(text):
    names = {m[1] for m in re.finditer(r'\bproject\s+([A-Za-z][\w-]*)', text, re.I)
             if m[1].casefold() not in ('status', 'history', 'progress', 'report', 'rules', 'is')}
    names.update(m[1] for m in re.finditer(r'项目\s*([\u3400-\u9fffA-Za-z0-9_-]+?)(?=的|[，。；：\s]|$)', text))
    return names


def task_text(query):
    """只用请求区规划；材料、引用、选项仍保留在 V3 的原始查询中。"""
    text = re.sub(r'```[\s\S]*?```|<(?P<tag>context|source|document|materials)>[\s\S]*?</(?P=tag)>', '', query, flags=re.I)
    def quoted_name(match):
        body = match[0][1:-1].strip()
        # 简短名称可以作为范围；整段引文及其中的命令、日期不能触发规划。
        if (re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,40}|[\u3400-\u9fff]{1,20}', body)
                and not re.search(r'summar|recap|review|总结|概括|要求|指令|回顾', body, re.I)):
            return body
        return ''
    text = re.sub(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」', quoted_name, text)
    material = re.compile(r'^\s*(?:#{1,6}\s*|\[)?(?:参考资料|材料|背景|上下文|引用|选项|'
                          r'sources?|context|background|reference materials?|excerpts?|documents?|options)'
                          r'\s*(?:\]|[:：])', re.I)
    request = re.compile(r'^\s*(?:#{1,6}\s*|\[)?(?:任务|问题|请求|要求|(?:final\s+)?task|question|request|instructions)'
                         r'\s*(?:\]|[:：])', re.I)
    kept = []
    in_material = False
    for line in text.splitlines():
        if request.match(line):
            in_material = False
            kept.append(request.sub('', line, count=1))
        elif material.match(line):
            in_material = True
        elif not in_material and not re.match(r'^\s*(?:>|[A-H][.)、]\s)', line):
            # 内联 “任务……。参考资料：……” 的后半段也不参与时间和意图解释。
            kept.append(re.split(r'(?:参考资料|参考材料|引用材料|以下材料|source material|reference material)\s*[:：]',
                                 line, maxsplit=1, flags=re.I)[0])
    return '\n'.join(kept).strip()


def build_summary_plan(query, snapshot, *, reference=None, window=None):
    task = task_text(query)
    if re.search(r"\b(?:do not|don't|never)\s+(?:summari[sz]e|recap)|(?:不要|无需|不必)(?:进行)?(?:总结|概括|回顾)", task, re.I):
        return SummaryPlan()
    signal = re.search(r'\b(?:summari[sz]e|recap|synthesi[sz]e)\b|'
                       r'\b(?:give|provide|create|make)\b[^.?!\n]{0,35}\b(?:summary|overview|recap)\b|'
                       r'总结|概括|摘要|概览|梳理|回顾', task, re.I)
    across = re.search(r'\b(?:taken together|across|over the course|throughout)\b[^.?!\n]*'
                       r'\b(?:discussions?|sessions?|conversations?|history)\b|'
                       r'(?:结合|综合|纵观|贯穿).*(?:历次|此前|会话|记录|讨论)', task, re.I)
    report = re.search(r'\b(?:write|draft|prepare|produce|generate)\b[^.?!\n]*\b(?:report|review|brief)\b|'
                       r'(?:撰写|写出|生成|整理).*(?:报告|综述|总结)', task, re.I)
    signal = signal or across or report
    if not signal:
        return SummaryPlan()
    # 材料日期不能收缩为检索窗口。只在请求文本未被剥离时复用 V3 的窗口。
    p = governance_plan(task, snapshot, reference=reference, window=window if task == query.strip() else None)
    if set(p.unresolved) - {'ambiguous_identity'}:
        return SummaryPlan(warnings=p.unresolved)
    scopes = set(p.scope_keys)
    scopes.update(f.subject_name for f in snapshot.facts if f.subject_id in p.named_subjects)
    # 目录可能只有 no_fact 原文，不能要求一个项目先有结构化事实才能检索。
    scopes.update(mentioned_projects(task))
    requirements = [SummaryRequirement(key, label, m[0]) for key, (label, pattern) in FACETS.items()
                    if (m := re.search(pattern, task, re.I))]
    if not requirements:
        requirements = [SummaryRequirement('overview', '相关经历', signal[0])]
    predicates = p.predicates - {'event', 'rules'}
    slots = frozenset(f.slot for f in snapshot.facts if
        (not scopes or any(contains(f.scope, s) or contains(f.subject_name, s) for s in scopes))
        and (scopes or not predicates or f.predicate in predicates))
    terms = tuple(dict.fromkeys(a for prop in PROPERTIES if prop.key in predicates for a in prop.aliases))
    if 'employer' in predicates:
        terms += ('employment', 'work', 'working', '任职', '工作', '公司')
    if not scopes and not predicates:
        # 未注册主题也应保持范围；只有明确全局请求或没有主题的概览才扫描全局。
        global_request = re.search(r'\b(?:all|entire|whole)\s+(?:my\s+|our\s+)?(?:history|discussions|sessions|conversations)\b|'
                                   r'(?:全部|所有|整个|全局).*(?:历史|经历|讨论|会话)', task, re.I)
        match = re.search(r'\b(?:summari[sz]e|recap)\s+([^.!?;\n]+)', task, re.I)
        if match and not global_request:
            stop = set('my our the a an all entire whole previous past history experiences experience discussions sessions '
                       'overview summary and or of in with including please goals objectives decisions reasons rules '
                       'exceptions current status plans changes events format structure'.split())
            terms = tuple(dict.fromkeys(t for t in re.findall(r'[A-Za-z][\w-]*', match[1]) if t.casefold() not in stop))
        chinese = re.search(r'(?:总结|概括|梳理|回顾)(?:一下)?(?:我的|我们的)?([^，。；\n]+)', task)
        if chinese and not global_request and not terms:
            topic = re.sub(r'(?:的)?(?:经历|历程|历史|记录|讨论|情况)$', '', chinese[1]).strip()
            if topic and topic not in ('全部', '所有', '整体', '全局'):
                terms = (topic,)
        if terms and not any(contains(s.content, t) for s in snapshot.sources.values() for t in terms):
            return SummaryPlan(warnings=('unrecognized_scope',))
    order = 'chronological' if across or any(r.key == 'changes' for r in requirements) else 'source'
    constraints = tuple(m[0] for m in re.finditer(r'(?:保留|包括|包含|不得|不要)[^。；\n]+|'
                         r'\b(?:include|retain|preserve|do not)\b[^.;\n]+', task, re.I))
    warnings = ('身份待确认：同名主体分开展示，不推断跨会话为同一身份',) if p.ambiguous_identity else ()
    return SummaryPlan(True, task, tuple(sorted(scopes)), slots, not scopes and not predicates and not terms,
                       tuple(requirements), p.reference if p.reference is not None else reference,
                       p.window, p.knowledge_cutoff, warnings, order, constraints, frozenset(predicates), terms)
