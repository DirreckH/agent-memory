"""按完整证据单元打包，并在打包后报告实际覆盖。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import json
import re

from app.governance.evidence import date_text, valid_evidence
from app.governance.extraction import digest, token_count


@dataclass(frozen=True)
class SummaryCoverage:
    key: str
    label: str
    found: int
    emitted: int
    status: str  # emitted / partial / omitted / missing；只评价候选证据，不代表答案正确。


@dataclass(frozen=True)
class SummaryOutcome:
    hits: tuple = ()
    emitted: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    coverage: tuple[SummaryCoverage, ...] = ()
    evidence: tuple = ()
    packet_tokens: int = 0
    support_errors: tuple[str, ...] = ()


def source_metadata(source):
    return (f'[角色:{source.role};消息时间:{date_text(source.timestamp)};'
            f'文档位置:{source.session_id}/{source.ordinal}]')


def span_reference(ev):
    return f'[来源:{ev.source_id}:{ev.start}-{ev.end}]'


def required_support_present(item, snapshot):
    """Check encoded range dependencies, independently of query facet counts."""
    if not item.spans or not all(valid_evidence(ev, snapshot) for ev in item.spans):
        return False
    return all(valid_evidence(required, snapshot) and any(
        ev.source_id == required.source_id and ev.start <= required.start and ev.end >= required.end
        for ev in item.spans) for required in item.required_spans)


def ordered_spans(item):
    source_order = {sid: i for i, sid in enumerate(dict.fromkeys(ev.source_id for ev in item.spans))}
    return sorted(dict.fromkeys(item.spans), key=lambda ev: (source_order[ev.source_id], ev.start, ev.end))


def render_evidence_item(item, snapshot=None, seen_sources=None):
    parts = []
    seen_quotes = set()
    seen_sources = set() if seen_sources is None else seen_sources
    # Context headings precede their body; update sources remain after the source
    # they qualify. The order is documentary, not an inferred event ordering.
    for ev in ordered_spans(item):
        citation = span_reference(ev)
        if snapshot is not None and ev.source_id not in seen_sources:
            citation += source_metadata(snapshot.sources[ev.source_id])
            seen_sources.add(ev.source_id)
        if item.support_key and ev.quote.strip() in seen_quotes:
            parts.append(citation)
        else:
            parts.append(citation + ' ' + ev.quote)
            seen_quotes.add(ev.quote.strip())
    return ('[' + '；'.join(item.labels) + ']\n' if item.labels else '') + '\n'.join(parts)


def emitted_item_pattern(item, snapshot):
    """Match exact group text while allowing metadata only at its first source use."""
    parts = []
    seen_quotes = set()
    for ev in ordered_spans(item):
        part = re.escape(span_reference(ev))
        part += '(?:' + re.escape(source_metadata(snapshot.sources[ev.source_id])) + ')?'
        if not (item.support_key and ev.quote.strip() in seen_quotes):
            part += re.escape(' ' + ev.quote)
            seen_quotes.add(ev.quote.strip())
        parts.append(part)
    labels = ('[' + '；'.join(item.labels) + ']\n') if item.labels else ''
    return re.escape(labels) + '\n'.join(parts)


def audit_serialized_support(serialized, items, snapshot):
    """Validate encoded support against the actual caller-visible JSON contents.

    This proves preservation of known source ranges and dependencies only. It is
    not an entailment, identity-resolution, or final-answer correctness judge.
    """
    try:
        packets = json.loads(serialized)
        if not isinstance(packets, list):
            return ('invalid_packet',)
        contents = [packet['content'] for packet in packets
                    if isinstance(packet, dict) and isinstance(packet.get('content'), str)]
    except (ValueError, TypeError, KeyError):
        return ('invalid_packet',)
    errors = []
    for index, item in enumerate(items):
        if not required_support_present(item, snapshot):
            errors.append(f'incomplete_dependency:{index}')
            continue
        expected = emitted_item_pattern(item, snapshot)
        # A source's shared attributes must remain attached to one of its actual
        # emitted range references in the same hit, not to an unrelated source.
        metadata_anchors = [tuple(span_reference(ev) + source_metadata(snapshot.sources[sid])
                                  for candidate in items for ev in candidate.spans if ev.source_id == sid)
                            for sid in dict.fromkeys(ev.source_id for ev in item.spans)]
        if not any(re.search(expected, content) and all(any(anchor in content for anchor in anchors)
                   for anchors in metadata_anchors) for content in contents):
            errors.append(f'missing_emitted_support:{index}')
    return tuple(errors)


def assemble_summary(collection, summary_plan, baseline, snapshot, *, limit, token_budget):
    from app.service import SearchHit
    items = [e for e in collection.evidence if required_support_present(e, snapshot)]
    if not items or limit <= 0:
        rows = tuple(SummaryCoverage(r.key, r.label, 0, 0, 'missing') for r in summary_plan.requirements)
        return SummaryOutcome(missing=tuple(r.key for r in rows), coverage=rows)
    header = '[F1 抽取式证据；提及时间不等于生效时间；未结构化原文不自动认定为当前状态]'

    rendered = {id(e): render_evidence_item(e, snapshot) for e in items}
    costs = {id(e): token_count(rendered[id(e)]) for e in items}
    requirements = {r.key: r.label for r in collection.requirements or summary_plan.requirements}
    found_counts = Counter(k for e in items for k in e.facets if k in requirements)

    def coverage(chosen):
        counts = Counter(k for e in chosen for k in e.facets if k in requirements)
        return tuple(SummaryCoverage(k, label, found_counts[k], counts[k],
            'missing' if not found_counts[k] else 'omitted' if not counts[k] else
            'partial' if counts[k] < found_counts[k] else 'emitted') for k, label in requirements.items())

    def render(chosen):
        rows = coverage(chosen)
        lines = [header, *('[' + w + ']' for w in collection.warnings)]
        labels = {'emitted': '已输出', 'partial': '部分输出；预算未覆盖其余',
                  'omitted': '已找到但预算未覆盖', 'missing': '证据不足'}
        lines.append('[候选证据覆盖：' + '；'.join(
            f'{c.label} {c.emitted}/{c.found} {labels[c.status]}' for c in rows) + ']')
        if len(chosen) < len(items):
            lines.append(f'[内容预算未覆盖 {len(items) - len(chosen)} 个完整证据单元；未宣称语义完整]')
        displayed = chosen
        if summary_plan.order == 'chronological':
            displayed = sorted(chosen, key=lambda e: (not bool(e.moments), min(e.moments, default=0),
                snapshot.sources[e.spans[0].source_id].session_id,
                snapshot.sources[e.spans[0].source_id].ordinal, e.spans[0].start, e.spans[0].source_id))
        # Every span retains a full source ID and character range. Repeated role,
        # message-time and document-position fields are shared within this hit.
        seen_sources = set()
        lines.extend(render_evidence_item(e, snapshot, seen_sources) for e in displayed)
        return '\n'.join(lines)

    def make_hit(chosen):
        text = render(chosen)
        ids = {ev.source_id for e in chosen for ev in e.spans}
        anchors = [h for h in baseline if h.id in ids]
        score = max((h.score for h in anchors), default=baseline[0].score if baseline else 0.0)
        stamp = anchors[0].created_at if anchors else max(snapshot.sources[s].recorded_at for s in ids)
        uid = 'f1_' + digest(str((snapshot.user_id, snapshot.generation, snapshot.revision, text)))[:24]
        return SearchHit(uid, text, score, stamp)

    def serialize(hit):
        return json.dumps([asdict(hit)], ensure_ascii=False, sort_keys=True, separators=(',', ':'))

    def packet_cost(hit):
        # 统一评测可见 memories 的序列化形式；包括换行转义和派生 ID 等开销。
        return token_count(serialize(hit))

    chosen = []
    pending = list(items)
    frequency = Counter(k for item in items for k in item.facets if k != 'overview')
    while pending:
        covered = set().union(*(e.facets for e in chosen)) if chosen else set()
        # 先补尚未覆盖的需求，再选低频信息；不以重复次数作为重要性。
        item = max(pending, key=lambda e: (
            len((e.facets & requirements.keys()) - covered),
            sum(1 / frequency[k] for k in e.facets if k != 'overview' and k not in covered),
            bool(set(e.moments) - {t for c in chosen for t in c.moments}),
            -costs[id(e)]))
        pending.remove(item)
        if packet_cost(make_hit(chosen + [item])) <= token_budget:
            chosen.append(item)
    if not chosen:
        rows = coverage(())
        return SummaryOutcome(omitted=tuple(c.key for c in rows if c.found),
                              missing=tuple(c.key for c in rows if not c.found), coverage=rows)
    hit = make_hit(chosen)
    support_errors = audit_serialized_support(serialize(hit), chosen, snapshot)
    if support_errors:
        # Never return a packet which failed its encoded dependency checks.
        rows = coverage(())
        return SummaryOutcome(omitted=tuple(c.key for c in rows if c.found),
                              missing=tuple(c.key for c in rows if not c.found),
                              coverage=rows, support_errors=support_errors)
    rows = coverage(chosen)
    return SummaryOutcome((hit,), tuple(c.key for c in rows if c.emitted),
                          tuple(c.key for c in rows if c.status in ('partial', 'omitted')),
                          tuple(c.key for c in rows if not c.found), rows, tuple(chosen), packet_cost(hit))
