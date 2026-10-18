"""按完整证据单元打包，并在打包后报告实际覆盖。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import json

from app.governance.evidence import date_text, valid_evidence
from app.governance.extraction import digest, token_count


@dataclass(frozen=True)
class SummaryCoverage:
    key: str
    label: str
    found: int
    emitted: int
    status: str  # emitted / partial / omitted / missing；只评价候选证据，不代表答案正确。
    support_status: str = 'missing'  # source_supported / candidate / budget_omitted / missing / ambiguous
    supported: int = 0
    emitted_supported: int = 0


@dataclass(frozen=True)
class SummaryOutcome:
    hits: tuple = ()
    emitted: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    coverage: tuple[SummaryCoverage, ...] = ()
    evidence: tuple = ()
    packet_tokens: int = 0
    protection: object = None


def assemble_summary(collection, summary_plan, baseline, snapshot, *, limit, token_budget, protected=()):
    from app.service import SearchHit
    items = [e for e in collection.evidence if all(valid_evidence(s, snapshot) for s in e.spans)]
    if not items or limit <= 0:
        rows = tuple(SummaryCoverage(r.key, r.label, 0, 0, 'missing') for r in summary_plan.requirements)
        return SummaryOutcome(missing=tuple(r.key for r in rows), coverage=rows)
    header = '[F1 抽取式证据；提及时间不等于生效时间；未结构化原文不自动认定为当前状态]'

    source_order = {sid: i for i, sid in enumerate(snapshot.sources)}
    refs = {sid: f'S{i + 1}' for sid, i in source_order.items()}

    def render_item(item, seen_spans=None):
        parts = []
        seen_quotes = set()
        seen_spans = seen_spans if seen_spans is not None else set()
        for ev in item.spans:
            citation = f'[{refs[ev.source_id]}:{ev.start}-{ev.end}]'
            if item.local_attribution:
                citation += f'[消息角色:{snapshot.sources[ev.source_id].role};归因以原文为准]'
            span_key = (ev.source_id, ev.start, ev.end)
            if span_key in seen_spans or (item.support_key and ev.quote.strip() in seen_quotes):
                parts.append(citation)
            else:
                parts.append(citation + '\n' + ev.quote)
                seen_quotes.add(ev.quote.strip())
            seen_spans.add(span_key)
        # 未结构化性质由包头统一说明；D1 的逐片段版本标签仍与对应单元一同输出。
        labels = tuple(label for label in item.labels if label != '未结构化原文；适用性待核验')
        return ('[' + '；'.join(labels) + ']\n' if labels else '') + '\n'.join(parts)

    rendered = {id(e): render_item(e) for e in items}
    costs = {id(e): token_count(rendered[id(e)]) for e in items}
    requirements = {r.key: r.label for r in collection.requirements or summary_plan.requirements}
    found_counts = Counter(k for e in items for k in e.facets if k in requirements)
    supported_counts = Counter(k for e in items for k in e.supported if k in requirements)

    def coverage(chosen):
        counts = Counter(k for e in chosen for k in e.facets if k in requirements)
        supported = Counter(k for e in chosen for k in e.supported if k in requirements)
        return tuple(SummaryCoverage(k, label, found_counts[k], counts[k],
            'missing' if not found_counts[k] else 'omitted' if not counts[k] else
            'partial' if counts[k] < found_counts[k] else 'emitted',
            'ambiguous' if summary_plan.warnings and any('身份待确认' in w for w in summary_plan.warnings) else
            'source_supported' if supported[k] else 'budget_omitted' if supported_counts[k] else
            'candidate' if counts[k] else 'budget_omitted' if found_counts[k] else 'missing',
            supported_counts[k], supported[k]) for k, label in requirements.items())

    def render(chosen):
        rows = coverage(chosen)
        lines = [header, *('[' + w + ']' for w in collection.warnings)]
        labels = {'emitted': '已输出', 'partial': '部分输出；预算未覆盖其余',
                  'omitted': '已找到但预算未覆盖', 'missing': '证据不足'}
        support_labels = {'source_supported': '原文支持', 'candidate': '关系待核验',
                          'budget_omitted': '预算缺口', 'missing': '未找到', 'ambiguous': '范围待确认'}
        lines.append('[需求证据覆盖（不代表答案正确）：' + '；'.join(
            f'{c.label} {c.emitted}/{c.found} {labels[c.status]}，{support_labels[c.support_status]}' for c in rows) + ']')
        if len(chosen) < len(items):
            lines.append(f'[内容预算未覆盖 {len(items) - len(chosen)} 个完整证据单元；未宣称语义完整]')
        displayed = sorted(chosen, key=lambda e: (source_order[e.spans[0].source_id], e.spans[0].start))
        if summary_plan.order == 'chronological':
            displayed = sorted(chosen, key=lambda e: (not bool(e.moments), min(e.moments, default=0),
                snapshot.sources[e.spans[0].source_id].session_id,
                snapshot.sources[e.spans[0].source_id].ordinal, e.spans[0].start, e.spans[0].source_id))
        ids = dict.fromkeys(s.source_id for e in displayed for s in e.spans)
        lines.append('[来源目录；短编号仅在本条结果内有效]')
        for sid in ids:
            source = snapshot.sources[sid]
            lines.append(f'[{refs[sid]}={sid};角色:{source.role};消息时间:{date_text(source.timestamp)};'
                         f'文档位置:{source.session_id}/{source.ordinal}]')
        seen_spans = set()
        lines.extend(render_item(e, seen_spans) for e in displayed)
        return '\n'.join(lines)

    def make_hit(chosen):
        text = render(chosen)
        ids = {ev.source_id for e in chosen for ev in e.spans}
        anchors = [h for h in baseline if h.id in ids]
        score = max((h.score for h in anchors), default=baseline[0].score if baseline else 0.0)
        stamp = anchors[0].created_at if anchors else max(snapshot.sources[s].recorded_at for s in ids)
        uid = 'f1_' + digest(str((snapshot.user_id, snapshot.generation, snapshot.revision, text)))[:24]
        return SearchHit(uid, text, score, stamp)

    def packet_cost(hit):
        # 统一评测可见 memories 的序列化形式；包括换行转义和派生 ID 等开销。
        return token_count(json.dumps([asdict(hit)], ensure_ascii=False, sort_keys=True, separators=(',', ':')))

    # The caller supplies whole, version-labelled units, including dependencies.
    # Never split a unit to make the protected set fit. The guard audits shortfalls.
    chosen = []
    protected = set(protected)
    def unit_key(item):
        return tuple((s.source_id, s.start, s.end, s.content_hash, s.purpose) for s in item.spans)
    for item in items:
        if unit_key(item) in protected and packet_cost(make_hit(chosen + [item])) <= token_budget:
            chosen.append(item)
    pending = [e for e in items if e not in chosen]
    frequency = Counter(k for item in items for k in item.facets if k != 'overview')
    while pending:
        covered = set().union(*(e.facets for e in chosen)) if chosen else set()
        supported = set().union(*(e.supported for e in chosen)) if chosen else set()
        # 先补尚未覆盖的需求，再选低频信息；不以重复次数作为重要性。
        item = max(pending, key=lambda e: (
            len((e.supported & requirements.keys()) - supported),
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
    rows = coverage(chosen)
    return SummaryOutcome((hit,), tuple(c.key for c in rows if c.emitted),
                          tuple(c.key for c in rows if c.status in ('partial', 'omitted')),
                          tuple(c.key for c in rows if not c.found), rows, tuple(chosen), packet_cost(hit))
