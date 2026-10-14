"""治理的来源视图；永不修改存储记录、V3 文本或向量。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.governance.extraction import digest


@dataclass(frozen=True)
class EvidenceItem:
    section: str
    text: str
    source_ids: tuple[str, ...]
    fact_ids: tuple[str, ...] = ()
    requirement: str = ''
    topic: str = ''
    moment: int = -1


@dataclass(frozen=True)
class GovernanceResult:
    applicable: bool
    evidence: tuple[EvidenceItem, ...] = ()
    covered_requirements: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()
    conflicts: tuple[tuple[str, ...], ...] = ()
    superseded_spans: tuple[tuple[str, int, int, str], ...] = ()
    warnings: tuple[str, ...] = ()


def date_text(timestamp):
    if timestamp is None:
        return '时间未知'
    try:
        return datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return '时间未知'


def valid_evidence(ev, snapshot):
    source = snapshot.sources.get(ev.source_id)
    return bool(source and 0 <= ev.start < ev.end <= len(source.content)
                and source.content[ev.start:ev.end] == ev.quote and digest(ev.quote) == ev.content_hash)


def fact_item(fact, snapshot, section, requirement=''):
    if not fact.evidence or not all(valid_evidence(e, snapshot) for e in fact.evidence):
        return None
    source = snapshot.sources[fact.source_id]
    # 完整断言 + 时间依据。更新说明由执行器按查询语义单独加入。
    quotes = dict.fromkeys(e.quote for e in fact.evidence if e.purpose in ('assertion', 'time'))
    label = f'提及日:{date_text(source.timestamp)};消息时间:{date_text(source.timestamp)}'
    if fact.valid_from is not None:
        label += f';明确有效日:{date_text(fact.valid_from)}'
    if fact.valid_to is not None:
        label += f';有效期结束:{date_text(fact.valid_to)}'
    ev = fact.evidence[0]
    text = (f'[{label};来源:{ev.source_id}:{ev.start}-{ev.end};角色:{source.role};'
            f'归因:{fact.subject_name};场景:{fact.scope};性质:{fact.modality}] ' + '\n'.join(quotes))
    return EvidenceItem(section, text, (fact.source_id,), (fact.id,), requirement,
                        str(fact.slot), fact.observation_time if fact.observation_time is not None else -1)


def raw_item(source, label='待确认原文', requirement='', *, text=None):
    content = source.content if text is None else text
    header = (f'[{label};来源:{source.id};会话:{source.session_id};角色:{source.role};'
              f'消息时间:{date_text(source.timestamp)}] ')
    return EvidenceItem('pending', header + content, (source.id,), requirement=requirement,
                        topic=source.session_id, moment=source.timestamp if source.timestamp is not None else -1)
