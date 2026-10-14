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


def _dates(query):
    found=[]
    patterns=[(r'(?<!\d)(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?!\d)',False),
              (r'(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日?',False),
              (rf'\b(?P<m>{_MONTH})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(?P<y>\d{{4}})\b',True),
              (rf'\b(?P<d>\d{{1,2}})\s+(?P<m>{_MONTH})\.?[,]?\s+(?P<y>\d{{4}})\b',True)]
    for pattern,english in patterns:
        for match in re.finditer(pattern,query,re.I):
            month=_MONTHS[match['m'].casefold()] if english else int(match['m'])
            found.append((match.start(), datetime(int(match['y']),month,int(match['d']),tzinfo=timezone.utc),match[0]))
    return sorted(found)


def plan(query, snapshot, *, reference=None, window=None, expansion=''):
    # expansion 只为接口复用而接收；没有 query span 支持的扩展不能触发治理。
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
    if len(dates)==2 and not re.search(r'\b(?:between|from)\b|至|到',query,re.I):
        return QueryPlan(unresolved=('multiple_dates',))
    if dates and re.search(r'可能|大约|\b(?:maybe|approximately)\b',query,re.I):
        return QueryPlan(unresolved=('uncertain_date',))
    if dates and (re.search(r'\b(?:before|after|since|until|except|not on)\b|之前|之后|以外|除外|不是.*日期',query,re.I)):
        # 首版不把方向/排除窗口错误收缩成当天。由原有时间流程继续处理。
        return QueryPlan(unresolved=('unsupported_date_operator',))
    keys={s.key for s in PROPERTIES if any(contains(query,a) for a in s.aliases)}
    if re.search(r'\b(?:employment|employer|work|working)\b|工作|任职|所在公司',query,re.I):
        keys.add('employer')
    keys.update(f.predicate for f in snapshot.facts if contains(query,f.predicate))
    relations=[key for key,aliases in RELATIONS.items() if any(contains(query,a) for a in aliases)]
    if len(relations)>1:
        return QueryPlan(unresolved=('unsupported_relation_chain',))
    relation=relations[0] if relations else None
    if relation:
        keys.update(f.predicate for f in snapshot.facts if relation_key(f.predicate)==relation)
    slots={f.slot for f in snapshot.facts if f.predicate in keys}
    named={f.subject_id for f in snapshot.facts if f.subject_name!='self'
           and any(contains(query,a) for a in (f.subject_name,*f.aliases))}
    if named:
        slots={s for s in slots if s[0] in named}
    elif re.search(r'\b(?:I|my|me|we|our)\b|我',query,re.I):
        slots={s for s in slots if s[0]=='self'}
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
        reference=int((dates[-1][1]+timedelta(days=1)).timestamp()*1000)-1
        date_window=TemporalWindow(int(dates[0][1].timestamp()*1000),reference,end_inclusive=True)
    elif window is not None:
        date_window=window
        if window.end_ms is not None:
            reference=window.end_ms if window.end_inclusive else window.end_ms-1
    elif rules.complete and rules.constraints and rules.constraints.relative_window and reference is not None:
        date_window=resolve_relative_window(rules.constraints.relative_window,reference)
        reference=date_window.end_ms if date_window.end_inclusive else date_window.end_ms-1
    signal=summary or change or current or history or dates or date_window
    # 明确的复合属性关系也可触发，不依赖 V3 多跳语法启发式。
    chain=bool(relation and keys & {'employer','residence'} and not named)
    if not signal and not chain:
        return QueryPlan()
    scope_keys={f.scope for f in snapshot.facts if f.scope not in ('general','work','home','temporary')
                and contains(query,f.scope)}
    global_summary=bool(summary and not scope_keys and not named and (
        re.search(r'\b(?:entire|whole|all|global)\b|全部|所有|全局|整体|整个',query,re.I) or not keys))
    if summary:
        # “截至某日的综述”覆盖此前阶段，不等同于“当天发生的事件”。
        if current and len(dates)==1:
            date_window=None
        if global_summary:
            slots={f.slot for f in snapshot.facts}
        elif scope_keys:
            slots={f.slot for f in snapshot.facts if f.scope in scope_keys}
        elif named and not keys:
            slots={f.slot for f in snapshot.facts if f.subject_id in named}
        intent,ops='summary',('summarize',)
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
