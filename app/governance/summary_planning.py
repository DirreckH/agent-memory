"""F1 查询期计划。只读原问题，不改变 V3 查询、时间排序或 D1 解析。"""
from __future__ import annotations

from dataclasses import dataclass
import re

from app.governance.planning import contains, plan as governance_plan
from app.governance.registry import PROPERTIES, RELATION_ALIASES


@dataclass(frozen=True)
class SummaryRequirement:
    key: str
    label: str
    trigger: str
    terms: tuple[str, ...] = ()
    origin: str = 'explicit'


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
    task_kind: str = 'overview'
    direct_requirements: tuple[SummaryRequirement, ...] = ()
    attribution_requirements: tuple[SummaryRequirement, ...] = ()


# F1-only evidence requests. These aliases do not register mutable D1 properties.
DIRECT_FIELDS = {
    **{p.key: p.aliases for p in PROPERTIES if p.key not in ('event', 'rules')},
    **RELATION_ALIASES,
    'identifier': ('identifier', 'serial number', 'reference number', '编号', '标识符'),
    'amount': ('amount', 'budget', '金额', '预算金额'),
}
ATTRIBUTION_ACTIONS = {
    'propose': ('propose', 'proposed', 'proposes', 'suggested', 'suggest', '提出', '提议', '建议者'),
    'report': ('said', 'says', 'told', 'reported', 'reporter', 'relayed', '转述', '说过', '告诉', '发言者'),
    'execute': ('implemented', 'executed', 'performed', '执行', '实施者'),
    'approve': ('approved', 'authorized', '批准', '审批者'),
}


def protected_demands(task):
    """Only explicit tasks request protection; supplied answers/materials never do.

    Matching a phrase requests preservation, not certification of its truth. Unknown
    action/field syntax remains on the existing F1 path rather than guessing actors.
    """
    # The older scope planner does not recognize bare "Materials:". Limit this
    # extra sanitization to new protection requests; keep its routing unchanged.
    task = task_text(re.sub(r'^\s*materials?\s*[:：]', 'Reference materials:', task, flags=re.I | re.M))
    cue = re.search(r'\b(?:who|whose|what|which|where|state|specify|identify|include|retain|preserve)\b|'
                    r'谁|哪|什么|多少|说明|指出|包括|包含|保留|负责人', task, re.I)
    if not cue:
        return (), ()
    direct = []
    for key, aliases in DIRECT_FIELDS.items():
        trigger = next((t for t in aliases if contains(task, t)), None)
        if trigger:
            # Keep an exact query substring for the audit, including original case.
            match = re.search(re.escape(trigger), task, re.I)
            direct.append(SummaryRequirement('direct:' + key, trigger, match[0], aliases, 'direct'))
    attribution = []
    who = re.search(r'\b(?:who|whose|by whom|speaker|attribution)\b|谁|归因|提出者|转述者|说话人|执行者', task, re.I)
    if who:
        for key, aliases in ATTRIBUTION_ACTIONS.items():
            trigger = next((t for t in aliases if contains(task, t)), None)
            if trigger:
                match = re.search(re.escape(trigger), task, re.I)
                attribution.append(SummaryRequirement('attribution:' + key, '归因:' + trigger,
                                                       match[0], aliases, 'attribution'))
    return tuple(direct), tuple(attribution)


FACETS = {
    'goals': ('目标', r'\b(?:goals?|objectives?|aims?)\b|目标|目的'),
    'decisions': ('决定', r'\b(?:decisions?|decided|approved|cancelled|canceled)\b|决定|决策|取消|批准'),
    'reasons': ('明确原因', r'\b(?:why|reasons?|because|due to)\b|原因|因为|由于'),
    'rules': ('规则', r'\b(?:rules?|exceptions?|except|unless|only if|only after)\b|规则|例外|除非|仅当|只有'),
    'exceptions': ('明确例外', r'\b(?:exceptions?|except|unless)\b|例外|除非'),
    'plans': ('计划', r'\b(?:plans?|planned|next steps?|intend\w*)\b|计划|打算|下一步'),
    'issues': ('未解决事项', r'\b(?:unresolved|pending|blocked|risks?|remaining issues?)\b|未解决|待确认|阻塞|风险|遗留'),
    'current': ('当前状态', r'\b(?:current|currently|status|now)\b|当前|目前|状态|现状'),
    'changes': ('变化', r'\b(?:changes?|changed|shift|evol\w*|trajectory)\b|变化|转变|演变|轨迹'),
    'structure': ('已约定的结构与格式', r'\b(?:structure|format|sections?|outline)\b|结构|格式|章节|提纲'),
    'results': ('结果与数据', r'\b(?:results?|measurements?|findings?|outcomes?)\b|结果|测量|数据'),
    'limitations': ('限制与限定', r'\b(?:limitations?|caveats?|uncertainty|not tested|only tested)\b|局限|限制|不确定|未测试'),
}


def evidence_demands(task):
    """只从任务明确列出的证据类别派生需求，不把材料或预测答案转为事实。

    首版支持 combining/using both X and Y data/results 等限定语法。未知组合
    仍由现有概览处理，不猜测任意问句的隐含关系。
    """
    demands = []
    patterns = [r'\b(?:combining|combine|combined|using both|based on both)\s+(?:the\s+)?'
                r'(?P<a>[\w-]+(?:\s+[\w-]+){0,2}?)\s+and\s+(?P<b>[\w-]+(?:\s+[\w-]+){0,2}?)\s+'
                r'(?:data|results?|measurements?|evidence|records?)\b',
                r'(?:结合|综合)(?P<a>[\w-]{1,16})和(?P<b>[\w-]{1,16})(?:数据|结果|记录)']
    for pattern in patterns:
        for match in re.finditer(pattern, task, re.I):
            for term in (match['a'].strip(), match['b'].strip()):
                key = 'aspect:' + term.casefold()
                if not any(r.key == key for r in demands):
                    demands.append(SummaryRequirement(key, term + '证据', match[0], (term,), 'task_evidence'))
    return demands


def mentioned_projects(text):
    names = {m[1] for m in re.finditer(r'\bproject\s+([A-Za-z][\w-]*)', text, re.I)
             if m[1].casefold() not in ('status', 'history', 'progress', 'report', 'rules', 'is')}
    names.update(m[1] for m in re.finditer(r'项目\s*([\u3400-\u9fffA-Za-z0-9_-]+?)(?=的|[，。；：\s]|$)', text))
    return names


def task_text(query):
    """只用请求区规划；材料、引用、选项仍保留在 V3 的原始查询中。"""
    text = re.sub(r'```[\s\S]*?```|~~~[\s\S]*?~~~|<(?P<tag>context|source|document|materials)>[\s\S]*?</(?P=tag)>', '', query, flags=re.I)
    def quoted_name(match):
        body = match[0][1:-1].strip()
        # 简短名称可以作为范围；整段引文及其中的命令、日期不能触发规划。
        if (re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,40}|[\u3400-\u9fff]{1,20}', body)
                and not re.search(r'summar|recap|review|总结|概括|要求|指令|回顾', body, re.I)):
            return body
        return ''
    text = re.sub(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」', quoted_name, text)
    material = re.compile(r'^\s*(?:#{1,6}\s*|\[)?(?:参考资料|材料|背景|上下文|引用|选项|'
                          r'sources?|context|background|reference materials?|excerpts?|documents?|options|'
                          r'(?:conversation\s+)?transcripts?|artifacts?|prior outputs?)'
                          r'\s*(?:\]\s*[:：]?|[:：]|$)', re.I)
    request = re.compile(r'^\s*(?:#{1,6}\s*|\[)?(?:任务|问题|请求|要求|(?:final\s+)?task|question|request|instructions)'
                         r'\s*(?:\]\s*[:：]?|[:：]|$)', re.I)
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
                       r'\b(?:discussions?|sessions?|conversations?|exchanges?|history)\b|'
                       r'(?:结合|综合|纵观|贯穿).*(?:历次|此前|会话|记录|讨论)', task, re.I)
    report = re.search(r'\b(?:write|draft|prepare|produce|generate)\b[^.?!\n]*\b(?:report|review|brief)\b|'
                       r'(?:撰写|写出|生成|整理).*(?:报告|综述|总结)', task, re.I)
    demands = evidence_demands(task)
    synthesis = re.search(r'\b(?:overall|conclusion|conclude|assess|assessment|evaluate)\b|总体|结论|综合评价', task, re.I)
    signal = signal or across or report or (synthesis if len(demands) > 1 else None)
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
    stance = re.search(r'\b(?:attitude|stance|position|opinion)\s+(?:towards?|on|about)\s+'
                       r'(?P<topic>[\w-]+(?:\s+[\w-]+){0,5}?)(?=\s+(?:shift|chang\w*|evol\w*)\b|[?.!,;]|$)', task, re.I)
    if stance and across and not scopes:
        topic = stance['topic'].strip()
        if not any(contains(s.content, topic) for s in snapshot.sources.values()):
            return SummaryPlan(warnings=('unrecognized_scope',))
        scopes.add(topic)
    if (across and re.search(r'\bexchanges?\b', across[0], re.I) and not report
            and not scopes and re.search(r'\b(?:stance|position|attitude)\b', task, re.I)):
        # 新支持的对话轨迹只有在主体/主题范围明确时接管。不能把歧义指代扩展成全部历史。
        return SummaryPlan(warnings=('unrecognized_scope',))
    requirements = [SummaryRequirement(key, label, m[0]) for key, (label, pattern) in FACETS.items()
                    if (m := re.search(pattern, task, re.I))]
    requirements.extend(demands)
    if not requirements:
        requirements = [SummaryRequirement('overview', '相关经历', signal[0])]
    predicates = p.predicates - {'event', 'rules'}
    if stance and across:
        predicates = predicates - {'job_role', 'preference'}
    # like(s) 也用于互动指标与举例。只有显式个人偏好请求才选择偏好槽位。
    # 此消歧仅属于 F1；D1 的属性注册与版本解释保持原样。
    if 'preference' in predicates and not re.search(
            r'\b(?:preferences?|prefer\w*|favorite\w*)\b|\b(?:I|we|you)\s+(?:like|love)\b|'
            r'\b(?:my|our|your|his|her|their)\s+(?:likes|dislikes)\b|偏好|喜好|喜欢', task, re.I):
        predicates = predicates - {'preference'}
    slots = frozenset(f.slot for f in snapshot.facts if
        (not scopes or any(contains(f.scope, s) or contains(f.subject_name, s) for s in scopes))
        and (scopes or not predicates or f.predicate in predicates))
    terms = tuple(dict.fromkeys(a for prop in PROPERTIES if prop.key in predicates for a in prop.aliases))
    if 'employer' in predicates:
        terms += ('employment', 'work', 'working', '任职', '工作', '公司')
    if demands and not scopes and not predicates:
        terms = tuple(t for r in demands for t in r.terms)
    if report and not scopes and not predicates:
        # 报告中的材料指标不是用户属性。以报告主题和明确要求回忆的约定作检索范围。
        topic = re.search(r'\b(?:write|draft|prepare|produce|generate)\s+(?:the\s+|an?\s+)?'
                          r'([\w -]{1,60}?)\s+(?:report|review|brief)\b', task, re.I)
        terms = tuple(t for t in re.findall(r'[A-Za-z][\w-]*', topic[1] if topic else '')
                      if t.casefold() not in {'my', 'our', 'previously', 'agreed', 'new', 'final'})
        if any(r.key == 'structure' for r in requirements):
            terms += ('report', 'format', 'structure', 'outline', '报告', '格式', '结构', '提纲')
    if not scopes and not predicates and not terms:
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
    direct, attribution = protected_demands(task)
    # Reuse F1's established distinction between engagement likes and preferences.
    direct = tuple(r for r in direct if r.key != 'direct:preference' or 'preference' in predicates)
    return SummaryPlan(True, task, tuple(sorted(scopes)), slots, not scopes and not predicates and not terms,
                       tuple(requirements), p.reference if p.reference is not None else reference,
                       p.window, p.knowledge_cutoff, warnings, order, constraints, frozenset(predicates), terms,
                       'report' if report else 'trajectory' if order == 'chronological' else
                       'synthesis' if demands else 'overview', direct, attribution)
