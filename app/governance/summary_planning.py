"""F1 查询期计划。只读原问题，不改变 V3 查询、时间排序或 D1 解析。"""
from __future__ import annotations

from dataclasses import dataclass
import re

from app.governance.planning import contains, non_recall_request, plan as governance_plan, task_text
from app.governance.registry import PROPERTIES
from app.governance.task_requirements import memory_task_text, procedure_request, task_requirements


@dataclass(frozen=True)
class SummaryRequirement:
    key: str
    label: str
    trigger: str
    terms: tuple[str, ...] = ()
    mandatory: bool = False
    kind: str = 'facet'


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
    order_basis: str = 'observation'  # observation | event | message
    procedure_requirements: tuple[SummaryRequirement, ...] = ()


FACETS = {
    'goals': ('目标', r'\b(?:goals?|objectives?|aims?)\b|目标|目的'),
    'decisions': ('决定', r'\b(?:decisions?|decided|approved|cancelled|canceled)\b|决定|决策|取消|批准'),
    'reasons': ('明确原因', r'\b(?:why|reasons?|because|due to)\b|原因|因为|由于'),
    'rules': ('规则与例外', r'\b(?:rules?|exceptions?|except|unless|only if|only after)\b|规则|例外|除非|仅当|只有'),
    'plans': ('计划', r'\b(?:plans?|planned|next steps?|intend\w*)\b|计划|打算|下一步'),
    'advice': ('既有建议', r'\b(?:advice|recommendations?|suggestions?|suggested|recommended)\b|建议|推荐'),
    'issues': ('未解决事项', r'\b(?:unresolved|pending|blocked|risks?|remaining issues?)\b|未解决|待确认|阻塞|风险|遗留'),
    'current': ('当前状态', r'\b(?:current|currently|status|now)\b|当前|目前|状态|现状'),
    'changes': ('变化', r'\b(?:changes?|changed|shift|evol\w*|trajectory)\b|变化|转变|演变|轨迹'),
    'structure': ('已约定的结构与格式', r'\b(?:structure|format|sections?|outline)\b|结构|格式|章节|提纲'),
    'timeline': ('事件与阶段顺序', r'\b(?:events?|stages?|sequence|chronological|timeline|first|then|finally)\b|事件|阶段|时间顺序|先后顺序|首先|随后|最终'),
}


def mentioned_projects(text):
    names = {m[1] for m in re.finditer(r'\bproject\s+([A-Za-z][\w-]*)', text, re.I)
             if m[1].casefold() not in ('status', 'history', 'progress', 'report', 'rules', 'is')}
    names.update(m[1] for m in re.finditer(r'项目\s*([\u3400-\u9fffA-Za-z0-9_-]+?)(?=的|[，。；：\s]|$)', text))
    return names


def build_summary_plan(query, snapshot, *, reference=None, window=None):
    task = memory_task_text(task_text(query))
    if non_recall_request(task) or re.search(r"\b(?:do not|don't|never)\s+(?:summari[sz]e|recap)|(?:不要|无需|不必)(?:进行)?(?:总结|概括|回顾)", task, re.I):
        return SummaryPlan()
    dependencies = tuple(SummaryRequirement(r.key, r.label, r.trigger, r.terms, r.mandatory, r.kind)
                         for r in task_requirements(task))
    procedure_signal = next((r for r in dependencies if r.kind == 'procedure'), None) if procedure_request(task) else None
    signal = re.search(r'\b(?:summari[sz]e|recap|synthesi[sz]e)\b|'
                       r'\b(?:give|provide|create|make)\b[^.?!\n]{0,35}\b(?:summary|overview|recap)\b|'
                       r'总结|概括|摘要|概览|梳理|回顾', task, re.I)
    across = re.search(r'\b(?:taken together|across|over the course|throughout)\b[^.?!\n]*'
                       r'\b(?:discussions?|sessions?|conversations?|history)\b|'
                       r'(?:结合|综合|纵观|贯穿).*(?:历次|此前|会话|记录|讨论)', task, re.I)
    report = re.search(r'\b(?:write|draft|prepare|produce|generate)\b[^.?!\n]*\b(?:report|review|brief)\b|'
                       r'(?:撰写|写出|生成|整理).*(?:报告|综述|总结)', task, re.I)
    listing = re.search(r'\b(?:list|enumerate|outline)\b[^.?!\n]*'
                        r'\b(?:decisions?|risks?|issues?|goals?|rules?|exceptions?|plans?|events?|stages?)\b|'
                        r'(?:列出|列举|罗列).*(?:决定|决策|风险|问题|目标|规则|例外|计划|事件|阶段)', task, re.I)
    comparison = re.search(r'\b(?:compare|contrast)\b[^.?!\n]*'
                           r'\b(?:initial|earlier|original)\b[^.?!\n]*\b(?:final|later|current)\b|'
                           r'(?:比较|对比).*(?:最初|早期|原先|原始).*(?:最终|后期|当前|后来)', task, re.I)
    chronological = re.search(r'\b(?:chronological\s+order|in\s+(?:time|temporal)\s+order|timeline)\b|'
                              r'\bin\s+what\s+order\b|(?:时间|先后)顺序|时间线', task, re.I)
    mention_order = re.search(r'\b(?:order|sequence)\b[^.?!\n]{0,100}'
                              r'\b(?:mention(?:ed|ing|s)?|discuss(?:ed|ing|es)?|brought\b[^.?!\n]{0,25}\bup)\b|'
                              r'\b(?:mention(?:ed|ing|s)?|discuss(?:ed|ing|es)?)\b[^.?!\n]{0,100}\b(?:order|sequence)\b|'
                              r'\b(?:order|sequence)\s+of\s+(?:(?:the|my|our|these|those)\s+)?(?:discussions?|mentions?)\b|'
                              r'(?:提及|提到|讨论).*(?:顺序|先后)|(?:顺序|先后).*(?:提及|提到|讨论)', task, re.I)
    signal = signal or across or report or listing or comparison or chronological or mention_order
    if not signal and not procedure_signal:
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
    if comparison and not any(r.key == 'changes' for r in requirements):
        requirements.append(SummaryRequirement('changes', FACETS['changes'][0], comparison[0]))
    order_request = mention_order or chronological
    if order_request and not any(r.key == 'timeline' for r in requirements):
        requirements.append(SummaryRequirement('timeline', FACETS['timeline'][0], order_request[0]))
    if not requirements and signal:
        requirements = [SummaryRequirement('overview', '相关经历', signal[0])]
    requirements.extend(dependencies)
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
        match = re.search(r'\b(?:summari[sz]e|recap|list|enumerate|compare|contrast|outline)\s+([^.!?;\n]+)', task, re.I)
        if match and not global_request:
            stop = set('my our the a an all entire whole previous past history experiences experience discussions sessions '
                       'overview summary and or of in with including please goals objectives decisions reasons rules '
                       'exceptions current status plans changes events format structure risks issues stages initial final '
                       'advice recommendations recommendation suggestions suggestion '
                       'earlier later original chronology chronological temporal order timeline for about over across '
                       'conversations from to between on after before during as at since until january february march '
                       'april may june july august september october november december'.split())
            topic = re.split(r'\b(?:include|including|retain|preserve)\b', match[1], maxsplit=1, flags=re.I)[0]
            if procedure_signal:
                topic = re.split(r'\b(?:using|according to|following|with)\b', topic, maxsplit=1, flags=re.I)[0]
            terms = tuple(dict.fromkeys(t for t in re.findall(r'[A-Za-z][\w-]*', topic) if t.casefold() not in stop))
        chinese = re.search(r'(?:总结|概括|梳理|回顾)(?:一下)?(?:我的|我们的)?([^，。；\n]+)', task)
        if chinese and not global_request and not terms:
            topic = re.split(r'(?:使用|按照|根据|保留|包括|包含)', chinese[1], maxsplit=1)[0]
            topic = re.sub(r'(?:的)?(?:经历|历程|历史|记录|讨论|情况)$', '', topic).strip()
            if topic and topic not in ('全部', '所有', '整体', '全局'):
                terms = (topic,)
        if terms and not any(contains(s.content, t) for s in snapshot.sources.values() for t in terms):
            return SummaryPlan(warnings=('unrecognized_scope',))
        if not terms and not global_request and procedure_signal:
            # Procedure-only requests do not authorize scanning unrelated history.
            terms = tuple(dict.fromkeys(t for r in dependencies for t in r.terms))
            slots = frozenset()
    order = 'chronological' if across or order_request or comparison or any(r.key == 'changes' for r in requirements) else 'source'
    order_basis = 'message' if mention_order else ('event' if chronological else 'observation')
    constraints = tuple(m[0] for m in re.finditer(r'(?:保留|包括|包含|不得|不要)[^。；\n]+|'
                         r'\b(?:include|retain|preserve|do not)\b[^.;\n]+', task, re.I))
    warnings = ('身份待确认：同名主体分开展示，不推断跨会话为同一身份',) if p.ambiguous_identity else ()
    return SummaryPlan(True, task, tuple(sorted(scopes)), slots, not scopes and not predicates and not terms,
                       tuple(requirements), p.reference if p.reference is not None else reference,
                       p.window, p.knowledge_cutoff, warnings, order, constraints, frozenset(predicates), terms, order_basis,
                       dependencies)
