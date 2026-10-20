"""受控治理计划。只读 V3 上下文；不猜答案、不回写 V3 时间窗口。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re

from app.governance.registry import PROPERTIES, RELATION_ALIASES
from app.temporal import TemporalWindow, resolve_relative_window
from app.temporal_extraction import extract_temporal_rules


RELATIONS = RELATION_ALIASES


def contains(text, word):
    if word.isascii():
        plural = r's?' if re.fullmatch(r'[A-Za-z]+',word) and not word.endswith('s') else ''
        return bool(re.search(r'(?<!\w)' + re.escape(word) + plural + r'(?!\w)', text, re.I))
    return word.casefold() in text.casefold()


def relation_key(predicate):
    return next((key for key, aliases in RELATIONS.items() if predicate.casefold() in aliases), predicate.casefold())


def task_text(query):
    """Extract request sections without treating supplied material as query instructions."""
    text = re.sub(r'```[\s\S]*?```|<(?P<tag>context|source|document|materials)>[\s\S]*?</(?P=tag)>',
                  '', query, flags=re.I)

    def quoted_name(match):
        body = match[0][1:-1].strip()
        # A standalone calendar-day literal is a task argument. Preserve it for
        # normal validation, including invalid days, without promoting sentences.
        if any(re.fullmatch(pattern, body, re.I) for pattern, _ in _DATE_PATTERNS):
            return body
        if re.fullmatch(rf'\d{{4}}(?:-\d{{1,2}})?|(?:{_MONTH})\.?\s+\d{{4}}|'
                       r'\d{4}\s*年(?:\s*\d{1,2}\s*月)?', body, re.I):
            return body  # Keep unsupported precision visible to the temporal guard.
        if (re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,40}|[\u3400-\u9fff]{1,20}', body)
                and not re.search(r'summar|recap|review|总结|概括|要求|指令|回顾', body, re.I)):
            return body
        return ''

    text = re.sub(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」', quoted_name, text)
    prefix = r'^\s*(?:#{1,6}\s*|[-=]{2,}\s*)?(?:\[)?'
    suffix = r'\s*(?:\]\s*[:：]?|[:：]|\s*$)'
    material = re.compile(prefix + r'(?:参考资料|材料|背景|上下文|引用|选项|'
                          r'sources?|context|background|reference materials?|excerpts?|documents?|options|'
                          r'(?:conversation\s+)?transcripts?|artifacts?|prior\s+outputs?)' + suffix, re.I)
    request = re.compile(prefix + r'(?:最终任务|任务|问题|请求|要求|(?:final\s+)?task|question|request|instructions)'
                         + suffix, re.I)
    kept = []
    in_material = False
    for line in text.splitlines():
        if request.match(line):
            in_material = False
            kept.append(request.sub('', line, count=1))
        elif material.match(line):
            in_material = True
        elif not in_material and not re.match(r'^\s*(?:>|[A-H][.)、]\s)', line):
            kept.append(re.split(r'(?:参考资料|参考材料|引用材料|以下材料|source material|reference material)\s*[:：]',
                                 line, maxsplit=1, flags=re.I)[0])
    return '\n'.join(kept).strip()


def non_recall_request(query):
    """Advice, counterfactuals and configuration instructions do not assert a memory lookup."""
    return bool(re.search(
        r'\b(?:what if|suppose|hypothetically|should I|should we|would I|would we)\b|'
        r'\b(?:how (?:do|can|should) (?:I|we)|how to)\b[^.?!\n]*'
        r'\b(?:change|set|configure|enable|disable|implement|adjust)\b|'
        r'(?:^|[.!?]\s*)(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?'
        r'(?:recommend|suggest|advise)\b|'
        r'\b(?:give|provide|offer)\s+(?:me|us)\s+(?:some\s+)?(?:advice|recommendations?|suggestions?)\b|'
        r'假如|假设|如果我|如果我们|(?:我|我们)(?:应该|该)|'
        r'^(?:请)?(?:推荐|建议)|(?:给我|给我们)(?:一些|提供)?建议|如何(?:设置|修改|配置)',
        query, re.I))


_BROAD_ALIASES = frozenset({'position', 'like', 'live', 'living', 'company', 'progress'})


def _property_keys(query, snapshot):
    # Broad nouns/verbs are not sufficient to identify a fact slot.
    keys = {spec.key for spec in PROPERTIES if any(
        alias not in _BROAD_ALIASES and contains(query, alias) for alias in spec.aliases)}
    if re.search(r'\b(?:employment|employer)\b|\bwork\s+(?:experience|history)\b|'
                 r'\bwhere\s+(?:do|does|did)\b[^.?!\n]*\bwork\b|'
                 r'\b(?:what|which)\s+company\b[^.?!\n]*\b(?:work|employed)\b|'
                 r'工作经历|工作历史|任职|所在公司|在哪(?:里)?工作', query, re.I):
        keys.add('employer')
    if re.search(r'\bwhere\s+(?:do|does|did)\b[^.?!\n]*\blive\b', query, re.I):
        keys.add('residence')
    if re.search(r'\bwhat\s+(?:do|does|did)\b[^.?!\n]*\blike\b', query, re.I):
        keys.add('preference')
    if re.search(r'\bjob\s+position\b', query, re.I):
        keys.add('job_role')
    if re.search(r'\b(?:status|progress)\b[^.?!\n]*\bproject\b|'
                 r'\bproject\b[^.?!\n]*\b(?:status|progress)\b', query, re.I):
        keys.add('project_status')
    keys.update(f.predicate for f in snapshot.facts
                if f.predicate not in _BROAD_ALIASES | {'work'} and contains(query, f.predicate))
    return keys


def _fact_question(query):
    return bool(re.search(r'\b(?:what|which|who|where)\s+(?:is|are|was|were|do|does|did)\b|'
                          r'\b(?:tell me|show me|recall)\b|什么|哪(?:个|里|家)|是谁|多少', query, re.I))


@dataclass(frozen=True)
class QueryPlan:
    intent: str = 'ordinary'
    slots: frozenset = frozenset()
    ambiguous_identity: bool = False
    operations: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    relation: str | None = None
    reference: int | None = None
    window: TemporalWindow | None = None
    knowledge_cutoff: int | None = None
    evidence: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    predicates: frozenset = frozenset()
    named_subjects: frozenset = frozenset()
    global_summary: bool = False
    scope_keys: frozenset = frozenset()


_SUMMARY = r'\b(?:summari[sz]e|summary|overview|key decisions)\b|总结|摘要|概览|梳理|整体回顾|全局|长历史'
_CHANGE = r'\b(?:changes|trajectory|evolution)\b|\bhow\b.*\bchang\w*|变化|变迁|轨迹'
_CURRENT = r'\b(?:currently|current|now|as of)\b|现在|目前|当前|最新|截至|截止到'
_HISTORY = r'\b(?:history|previous|formerly|used to|did)\b|历史|以前|曾经|过去|当时'
_KNOWN = r'\b(?:known at|knew at|knew then|records available|known by)\b|当时知道|当时已知|当时的记录|截至.*已知'
_MONTHS = {name: i for i,names in enumerate((('jan','january'),('feb','february'),('mar','march'),
    ('apr','april'),('may',),('jun','june'),('jul','july'),('aug','august'),('sep','sept','september'),
    ('oct','october'),('nov','november'),('dec','december')),1) for name in names}
_MONTH = '|'.join(sorted(_MONTHS, key=len, reverse=True))
_DATE_PATTERNS = (
    (r'(?<!\d)(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?!\d)', False),
    (r'(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日?', False),
    (rf'\b(?P<m>{_MONTH})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(?P<y>\d{{4}})\b', True),
    (rf'\b(?P<d>\d{{1,2}})\s+(?P<m>{_MONTH})\.?[,]?\s+(?P<y>\d{{4}})\b', True),
)


def _dates(query):
    found=[]
    for pattern,english in _DATE_PATTERNS:
        for match in re.finditer(pattern,query,re.I):
            month=_MONTHS[match['m'].casefold()] if english else int(match['m'])
            found.append((match.start(), datetime(int(match['y']),month,int(match['d']),tzinfo=timezone.utc),match[0]))
    return sorted(found)


def plan(query, snapshot, *, reference=None, window=None, expansion=''):
    # expansion 只为接口复用而接收；没有 query span 支持的扩展不能触发治理。
    task = task_text(query)
    if task != query.strip():
        window = None  # A window inferred from material cannot constrain the actual task.
    query = task
    if non_recall_request(query):
        return QueryPlan()
    summary=re.search(_SUMMARY,query,re.I)
    change=re.search(_CHANGE,query,re.I)
    current=re.search(_CURRENT,query,re.I)
    history=re.search(_HISTORY,query,re.I)
    known=re.search(_KNOWN,query,re.I)
    try:
        dates=_dates(query)
    except ValueError:
        return QueryPlan(unresolved=('invalid_date',))
    rules=extract_temporal_rules(query)
    if rules.blocked or len(dates)>2:
        return QueryPlan(unresolved=('unsupported_time',))
    relative = rules.constraints.relative_window if rules.constraints else None
    if relative and (dates or reference is None and window is None):
        return QueryPlan(unresolved=('unsupported_time',))
    if dates and re.search(r'\d{1,2}:\d{2}|\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b', query, re.I):
        return QueryPlan(unresolved=('unsupported_time_precision',))
    if len(dates)==2 and not re.search(r'\b(?:between|from)\b|至|到',query,re.I):
        return QueryPlan(unresolved=('multiple_dates',))
    if dates and re.search(r'可能|大约|\b(?:maybe|approximately)\b',query,re.I):
        return QueryPlan(unresolved=('uncertain_date',))
    directional = None
    if dates and re.search(r'\b(?:before|after|since|until|except|not on)\b|之前|之后|以外|除外|不是.*日期',query,re.I):
        # Only a single adjacent, unqualified before/after date has an unambiguous day boundary.
        if len(dates) == 1:
            start, _, text = dates[0]
            before = query[:start]
            after = query[start + len(text):]
            prefix = re.search(r'\b(before|after)\s*$', before, re.I)
            suffix = re.match(r'\s*(之前|之后)', after)
            directional = prefix[1].casefold() if prefix else ({'之前': 'before', '之后': 'after'}[suffix[1]] if suffix else None)
        operators = re.findall(r'\b(?:before|after|since|until)\b|之前|之后', query, re.I)
        if (directional is None or len(operators) != 1
                or re.search(r'\b(?:except|not|since|until|approximately|around|about)\b|除外|以外|大约', query, re.I)):
            return QueryPlan(unresolved=('unsupported_date_operator',))
    if not dates and window is None:
        temporal_text = re.sub(r'(?:之前|以前)(?:的)?(?:讨论|会话|记录|建议)|(?:之前|以前)关于', '', query)
        if (re.search(r'\b(?:before|after|since|until)\b|之前|之后|以前|以后', temporal_text, re.I)
                or relative and not rules.complete
                or re.search(rf'\b(?:in|during|on|as of|known at|known by)\s+(?:\d{{4}}|{_MONTH})\b|'
                             r'\d{4}\s*年|\d{4}-\d{1,2}', query, re.I)):
            return QueryPlan(unresolved=('unsupported_time',))
    keys = _property_keys(query, snapshot)
    relations=[key for key,aliases in RELATIONS.items() if any(contains(query,a) for a in aliases)]
    if len(relations)>1:
        return QueryPlan(unresolved=('unsupported_relation_chain',))
    relation=relations[0] if relations else None
    if relation:
        keys.update(f.predicate for f in snapshot.facts if relation_key(f.predicate)==relation)
    slots={f.slot for f in snapshot.facts if f.predicate in keys}
    named={f.subject_id for f in snapshot.facts if f.subject_name!='self'
           and any(contains(query,a) for a in (f.subject_name,*f.aliases))}
    subject_text = re.sub(r'(?:告诉|告知|给|向|为)我(?!的|们)', '', query)
    self_target = bool(re.search(r'\b(?:I|my|we|our)\b|我',subject_text,re.I))
    scope_keys={f.scope for f in snapshot.facts if f.scope not in ('general','work','home','temporary')
                and contains(query,f.scope)}
    if named:
        slots={s for s in slots if s[0] in named}
    elif self_target:
        slots={s for s in slots if s[0]=='self'}
    if scope_keys:
        slots = {s for s in slots if s[2] in scope_keys}
    if re.search(r'\b(?:temporary|hotel|business trip)\b|临时|酒店|出差',query,re.I):
        slots={s for s in slots if s[2]=='temporary'}
    elif re.search(r'\b(?:home|residence|address)\b|住址|住所|地址|住在',query,re.I):
        slots={s for s in slots if s[1]!='residence' or s[2]=='home'}
    for identity in {f.identity_context for f in snapshot.facts if f.identity_context}:
        if contains(query,identity):
            slots={s for s in slots if any(f.slot==s and f.identity_context==identity for f in snapshot.facts)}
    date_window=None
    if dates:
        if dates[-1][1]<dates[0][1]:
            return QueryPlan(unresolved=('reversed_dates',))
        start = int(dates[0][1].timestamp()*1000)
        end = int((dates[-1][1]+timedelta(days=1)).timestamp()*1000)-1
        if directional == 'before':
            date_window = TemporalWindow(None, start, hard_boundary=True)
            reference = start - 1
        elif directional == 'after':
            date_window = TemporalWindow(end + 1, None, hard_boundary=True)
        else:
            reference=end
            date_window=TemporalWindow(start,reference,end_inclusive=True)
    elif window is not None:
        date_window=window
        if window.end_ms is not None:
            reference=window.end_ms if window.end_inclusive else window.end_ms-1
    elif rules.complete and rules.constraints and rules.constraints.relative_window and reference is not None:
        date_window=resolve_relative_window(rules.constraints.relative_window,reference)
        reference=date_window.end_ms if date_window.end_inclusive else date_window.end_ms-1
    signal=summary or change or current or history or dates or date_window
    # 明确的复合属性关系也可触发，不依赖 V3 多跳语法启发式。
    chain=bool(relation and keys & {'employer','residence'} and not named and self_target)
    explicit_lookup = bool(slots and (named or self_target or scope_keys) and _fact_question(query)
                           and keys - {'event', 'rules'})
    if not signal and not chain and not explicit_lookup:
        return QueryPlan(slots=frozenset(slots), predicates=frozenset(keys),
                         named_subjects=frozenset(named), scope_keys=frozenset(scope_keys))
    # No subject inference from whichever user records happen to match the property.
    if not summary and keys - {'event', 'rules'} and not (named or self_target or scope_keys):
        return QueryPlan()
    global_summary=bool(summary and not scope_keys and not named and (
        re.search(r'\b(?:entire|whole|all|global)\b|全部|所有|全局|整体|整个',query,re.I) or not keys))
    if summary:
        # “截至某日的综述”覆盖此前阶段，不等同于“当天发生的事件”。
        if len(dates)==1 and re.search(r'\b(?:as of)\s*$|(?:截至|截止到)\s*$', query[:dates[0][0]], re.I):
            date_window=None
        if global_summary:
            slots={f.slot for f in snapshot.facts}
        elif scope_keys:
            slots={f.slot for f in snapshot.facts if f.scope in scope_keys}
        elif named and not keys:
            slots={f.slot for f in snapshot.facts if f.subject_id in named}
        intent,ops='summary',('summarize',)
    elif directional:
        intent,ops='history',('events_between',)
    elif change or history and not dates and not date_window and not current:
        intent,ops='changes' if change else 'history',('history_until',)
    elif keys and not keys <= {'event','rules'} or chain:
        intent,ops='current',('state_at',)
    else:
        intent,ops='history',('events_between',) if date_window else ('history_until',)
    if chain and intent!='summary':
        intent,ops='current',('state_at','relation')
    spans=tuple(m[0] for m in (summary,change,current,history,known) if m)+tuple(d[2] for d in dates)
    evidence=spans+tuple(a for s in PROPERTIES for a in s.aliases if contains(query,a))
    ambiguous=len(named & {s[0] for s in slots})>1
    return QueryPlan(intent,frozenset(slots),ambiguous,ops,ops,relation if chain else None,reference,
                     date_window,reference if known else None,evidence,
                     ('ambiguous_identity',) if ambiguous else (),frozenset(keys),frozenset(named),global_summary,
                     frozenset(scope_keys))


def needs_governance(query):
    return bool(re.search('|'.join((_SUMMARY,_CHANGE,_CURRENT,_HISTORY,_KNOWN)),query,re.I)
                or re.search(r'\d{4}[-年]|\b(?:as of|since|until)\b',query,re.I))
