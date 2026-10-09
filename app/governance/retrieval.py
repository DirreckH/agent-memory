from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import numpy as np

from app.governance.extraction import digest, token_count
from app.governance.summaries import LABELS, SECTIONS, SummaryBuilder, fact_line
from app.multihop import supplement_multihop
from app.temporal import TemporalWindow, resolve_relative_window
from app.temporal_extraction import extract_temporal_rules

from app.governance.registry import PROPERTIES, property_spec
from app.governance.resolution import StateResolver


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    slots: frozenset
    ambiguous_identity: bool = False
    keywords: tuple[str, ...] = ()
    historical_reference: int | None = None
    window: object = None
    relative_window: object = None


@dataclass(frozen=True)
class GovernanceSearchContext:
    query: str
    baseline: list
    reference: int | None
    settings: object
    limit: int
    steps: tuple = ()
    query_texts: tuple[str, ...] = ()
    query_vectors: object = None
    score_record: object = None
    window: object = None


_GOVERNANCE_SIGNAL = re.compile(
    r"summari[sz]e|summary|overview|key decisions|总结|摘要|概览|梳理|整体回顾|全局|长历史|"
    r"how.*chang|changes|trajectory|evolv|变化|变迁|轨迹|"
    r"history|previous|formerly|used to|\bdid\b|历史|以前|曾经|过去|当时|去年|上周|上个月|last (?:week|month|year)|"
    r"currently|current|now|现在|目前|当前|最新|\b\d{4}-\d{2}-\d{2}\b", re.I)


def needs_governance(query: str) -> bool:
    """普通查询不需要实体目录或派生快照，其治理对照与基础路径恒等。"""
    return bool(_GOVERNANCE_SIGNAL.search(query))


def plan(query, snapshot):
    summary = re.search(r"summari[sz]e|summary|overview|key decisions|总结|摘要|概览|梳理|整体回顾|全局|长历史", query, re.I)
    trajectory = re.search(r"how.*chang|changes|trajectory|evolv|变化|变迁|轨迹", query, re.I)
    history = re.search(r"history|previous|formerly|used to|\bdid\b|历史|以前|曾经|过去|当时|去年|上周|上个月|last (?:week|month|year)", query, re.I)
    current = re.search(r"currently|current|now|现在|目前|当前|最新", query, re.I)
    date_ref, date_window = None, None
    explicit_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
    try:
        if explicit_dates and not current:
            days = [datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc) for s in explicit_dates]
            if len(days) == 1 or (len(days) == 2 and re.search(r"between|from|至|到", query, re.I) and days[0] <= days[1]):
                start, end = days[0], days[-1] + timedelta(days=1)
                date_ref = int(end.timestamp() * 1000) - 1
                date_window = TemporalWindow(int(start.timestamp() * 1000), date_ref, end_inclusive=True)
                history = True
    except ValueError:
        pass
    if not summary and not trajectory and not history and not current:
        return QueryPlan("ordinary", frozenset())
    keys = {s.key for s in PROPERTIES if any(a in query.casefold() for a in s.aliases)}
    if re.search(r"employment|\bwork\b|工作|任职", query, re.I):
        keys.add("employer")
    keys.update(f.predicate for f in snapshot.facts if f.predicate in query.casefold())
    slots = {f.slot for f in snapshot.facts if f.predicate in keys}
    named = {f.subject_id for f in snapshot.facts if f.subject_name != "self"
             and any(a.casefold() in query.casefold() for a in (f.subject_name, *f.aliases))}
    if named:
        slots = {s for s in slots if s[0] in named}
    elif re.search(r"\b(I|my|me|we|our)\b|我", query, re.I):
        slots = {s for s in slots if s[0] == "self"}
    if re.search(r"temporary|hotel|business trip|临时|酒店|出差", query, re.I):
        slots = {s for s in slots if s[2] == "temporary"}
    elif re.search(r"home|residence|address|住址|住所|地址|住在", query, re.I):
        slots = {s for s in slots if s[1] != "residence" or s[2] == "home"}
    for context in {f.identity_context for f in snapshot.facts if f.identity_context}:
        if context.casefold() in query.casefold():
            slots = {s for s in slots if any(f.slot == s and f.identity_context == context for f in snapshot.facts)}
    identity_ids = {s[0] for s in slots}
    keywords = tuple(scope for scope in {f.scope for f in snapshot.facts}
                     if scope not in ("general", "work", "home", "temporary") and scope.casefold() in query.casefold())
    if summary:
        if re.search(r"entire|whole|all|global|全部|所有|全局|整体|整个", query, re.I):
            slots = {f.slot for f in snapshot.facts}
        elif keywords:
            slots = {f.slot for f in snapshot.facts if f.scope in keywords}
        elif not keys and not named:
            slots = {f.slot for f in snapshot.facts}
        return QueryPlan("summary", frozenset(slots), len(named & identity_ids) > 1, keywords, date_ref, date_window)
    rules = extract_temporal_rules(query)
    relative = rules.constraints.relative_window if rules.complete and rules.constraints else None
    return QueryPlan("changes" if trajectory else "history" if history else "current", frozenset(slots),
                     len(named & identity_ids) > 1, historical_reference=date_ref, window=date_window, relative_window=relative)


class GovernanceRetriever:
    def __init__(self, *, versions=True, summaries=True):
        self.versions = versions
        self.summaries = summaries

    def retrieve(self, snapshot, search_context):
        from app.service import SearchHit
        ctx = search_context
        query, baseline, reference = ctx.query, ctx.baseline, ctx.reference
        query_plan = plan(query, snapshot)
        intent, slots = query_plan.intent, query_plan.slots
        if intent == "ordinary":
            return baseline
        if not self.versions and intent != "summary":
            return baseline
        deadline = time.perf_counter() + ctx.settings.governance_query_budget_ms / 1000
        if query_plan.historical_reference is not None:
            reference = query_plan.historical_reference
        history_window = query_plan.window
        if query_plan.relative_window is not None and reference is not None:
            history_window = resolve_relative_window(query_plan.relative_window, reference)
            if history_window.end_ms is not None:
                reference = history_window.end_ms
        ctx = replace(ctx, reference=reference)
        header = "[抽取式证据；提及时间不等于生效时间，不推断未知历史区间]"
        if query_plan.ambiguous_identity:
            header += "\n[身份待确认：同名主体缺少跨会话身份依据]"
        if snapshot.incomplete:
            header += f"\n[覆盖不完整：{len(snapshot.incomplete)} 条来源的适用范围待确认]"
        items = []
        if intent == "summary":
            if not self.summaries:
                # 版本消融：保留基础检索排序，仅投影适用片段，不做主题/阶段聚合。
                state = StateResolver().resolve(snapshot, intent="current", reference=reference)
                allowed = {f.id for f in state.selected}
                allowed.update(f.id for f in snapshot.facts if f.kind in ('event','decision','rule','other')
                               and f.modality == 'asserted' and state.excluded.get(f.id) == 'not_confirmed_state')
                candidates = []
                for record in snapshot.records:
                    facts = [f for f in snapshot.facts if f.source_id == record.id and f.id in allowed]
                    if not facts:
                        continue
                    content = '\n'.join(fact_line(f) for f in facts)
                    projected = replace(record, content=content, search_text=content)
                    scores = [ctx.score_record(t, v, projected) for t, v in zip(ctx.query_texts, ctx.query_vectors)]
                    valid = [s for s in scores if s is not None]
                    if valid and max(valid) >= ctx.settings.min_relevance_score:
                        candidates.append((max(valid), record.id, facts))
                candidates.sort(key=lambda i: (-i[0], i[1]))
                items = [('current', fact_line(f), f.slot, f.observation_time or -1)
                         for _, _, facts in candidates[:ctx.limit] for f in facts]
            else:
                use_cache = self.versions and history_window is None and slots == frozenset(f.slot for f in snapshot.facts)
                units = snapshot.summaries if use_cache and all(u.reference_ms == reference for u in snapshot.summaries) else ()
                if not units and time.perf_counter() < deadline:
                    built = SummaryBuilder(versions=self.versions).build(
                        snapshot, slots, budget_ms=max(0, (deadline - time.perf_counter()) * 1000), reference=reference,
                        time_window=history_window)
                    units = built.units
                if not units:
                    header += "\n[摘要覆盖待确认：构建预算不足，以下为必要原文]"
                fact_by_id = {f.id: f for f in snapshot.facts}
                seen = set()
                for unit in units:
                    if unit.level != "session" or unit.section == "directory":
                        continue
                    if unit.section == "conflicts":
                        items.append((unit.section, unit.text, unit.topic, -1))
                        continue
                    for fid in unit.display_fact_ids or unit.fact_ids:
                        fact = fact_by_id[fid]
                        key = (fact.subject_id, fact.predicate, fact.scope, fact.value, fact.kind,
                               fact.modality, fact.valid_from, fact.quote)
                        if key in seen:
                            continue
                        seen.add(key)
                        items.append((unit.section, fact_line(fact, label=fact.modality if unit.section == "plans" else ""),
                                      unit.topic, fact.observation_time or -1))
        else:
            resolution = StateResolver().resolve(snapshot, slots, "current" if intent == "current" else "history", reference)
            facts = resolution.selected if self.versions else tuple(f for f in snapshot.facts if f.slot in slots)
            if resolution.conflicts:
                header += "\n[存在未解决冲突；不能确定唯一当前值]"
            if not facts or resolution.unknown_slots:
                header += "\n[当前状态待确认：部分槽位没有适用证据]"
            for fact in facts:
                items.append(("current" if intent == "current" else "history", fact_line(fact), fact.slot, fact.observation_time or -1))
            if intent == "current" and resolution.conflicts:
                conflict_ids = {fid for group in resolution.conflicts for fid in group}
                for fact in snapshot.facts:
                    if fact.id in conflict_ids and fact not in facts:
                        items.append(("conflicts", fact_line(fact, label="相反证据，未消解"), fact.slot, fact.observation_time or -1))
            if intent != "current":
                for ev in resolution.update_evidence:
                    if ev.purpose == "update":
                        items.append(("conflicts", "[纠正/撤回记录；错误旧说法不作为历史事实] " + ev.quote,
                                      ev.source_id, -1))
            if ctx.steps and ctx.score_record and not resolution.conflicts and not query_plan.ambiguous_identity and time.perf_counter() < deadline:
                extra = self._multihop(snapshot, ctx, query_plan, resolution, deadline)
                for fact in extra:
                    if fact not in facts:
                        items.append(("events", fact_line(fact), fact.slot, fact.observation_time or -1))
        # 未覆盖来源只作为待确认原文，禁止把它们当作当前值或桥接实体。
        if snapshot.incomplete or not items:
            ids = snapshot.incomplete or tuple(snapshot.sources)
            for sid in ids:
                if time.perf_counter() >= deadline:
                    break
                source = snapshot.sources[sid]
                if sid in snapshot.incomplete or intent == "summary":
                    items.append(("pending", f"[待确认原文;来源:{sid};归因:{source.role}] {source.content}",
                                  source.session_id, source.timestamp or -1))
        budget_expired = time.perf_counter() >= deadline
        if budget_expired:
            header += "\n[处理预算不足；覆盖待确认]"
        text, omitted = self._pack(header, items, ctx.settings.governance_content_tokens, deadline)
        if omitted:
            # 预留预算内的明确覆盖标记，不将未装入的片段说成已概括。
            text, _ = self._pack(header + f"\n[内容预算不足；{omitted} 个完整证据片段未覆盖]", items,
                                 ctx.settings.governance_content_tokens, deadline)
        uid = "gov_" + digest(str((snapshot.user_id, snapshot.generation, snapshot.revision, intent, text)))[:24]
        return [SearchHit(uid, text, 1.0, max((s.recorded_at for s in snapshot.sources.values()), default=""))]

    @staticmethod
    def _pack(header, items, budget, deadline=None):
        # 先轮询主题/阶段，防止重复提及或单个长主题耗尽全局预算。
        buckets = defaultdict(list)
        for item in sorted(items, key=lambda i: (i[3], str(i[2]), i[1])):
            buckets[(item[0], str(item[2]))].append(item)
        ordered = []
        section_order = {s: i for i, s in enumerate((*SECTIONS, "pending"))}
        section_queues = {s: [k for k in sorted(buckets) if k[0] == s] for s in section_order}
        while any(section_queues.values()):
            for section in section_order:
                queue = section_queues[section]
                if not queue:
                    continue
                key = queue.pop(0)
                ordered.append(buckets[key].pop(0))
                if buckets[key]:
                    queue.append(key)
        chosen = defaultdict(list)
        used, omitted = token_count(header), 0
        for section, line, _, _ in ordered:
            label = LABELS.get(section, "待确认原文")
            addition = (f"\n【{label}】\n" if not chosen[section] else "\n") + line
            cost = token_count(addition)
            if used + cost > budget or (deadline is not None and time.perf_counter() >= deadline):
                omitted += 1
                continue
            chosen[section].append(line)
            used += cost
        # 最后按固定分区输出；不会切断原文限定条件或否定句。
        output = header + "".join(f"\n【{LABELS.get(s, '待确认原文')}】\n" + "\n".join(chosen[s])
                                   for s in (*SECTIONS, "pending") if chosen[s])
        return output, omitted

    def _multihop(self, snapshot, ctx, query_plan, primary, deadline):
        all_state = StateResolver().resolve(snapshot, intent="current" if query_plan.intent == "current" else "history",
                                            reference=ctx.reference)
        by_id = {f.id: f for f in snapshot.facts}
        blocked = {by_id[fid].value.casefold() for fid, reason in primary.excluded.items()
                   if reason in ("superseded_observation", "corrected_claim", "retracted_claim")
                   and by_id[fid].predicate in ("employer", "residence")}
        blocked -= {f.value.casefold() for f in primary.selected}
        grouped = defaultdict(list)
        anchors = set()
        for fact in primary.selected:
            anchors.update(a for a in fact.object_entities if a.casefold() not in blocked)
            if fact.predicate in ("employer", "residence"):
                anchors.add(fact.value)
        candidates = []
        for fact in all_state.selected:
            if fact.subject_name.casefold() in blocked:
                continue
            grouped[fact.source_id].append(fact)
            candidates.append(fact)
        eligible = {f.id: f for f in primary.selected}
        for _ in range(ctx.settings.multihop_max_rounds + 1):
            reached = [f for f in candidates if f.id not in eligible
                       and any((re.search(r"(?<![a-zA-Z0-9_])" + re.escape(a) + r"(?![a-zA-Z0-9_])", f.quote, re.I)
                                if re.search(r"[a-zA-Z0-9]", a) else a in f.quote) for a in anchors)]
            if not reached:
                break
            for fact in reached:
                eligible[fact.id] = fact
                anchors.update(a for a in fact.object_entities if a.casefold() not in blocked)
        records, bridge_map = [], {}
        for record in snapshot.records:
            if time.perf_counter() >= deadline:
                return ()
            facts = grouped.get(record.id, ())
            if not facts:
                continue
            content = "\n".join(fact_line(f) for f in facts)
            records.append(replace(record, content=content, search_text=content))
            values = {a for f in facts for a in f.object_entities}
            values.update(f.subject_name for f in facts if f.subject_name != "self")
            values.update(f.value for f in facts if f.predicate in ("employer", "residence"))
            bridge_map[record.id] = tuple(sorted(v for v in values if v.casefold() not in blocked))
        scored = []
        for record in records:
            if time.perf_counter() >= deadline:
                return ()
            scores = [ctx.score_record(t, v, record) for t, v in zip(ctx.query_texts, ctx.query_vectors)]
            valid = [s for s in scores if s is not None]
            if valid and max(valid) >= ctx.settings.min_relevance_score:
                scored.append((max(valid), record, record.source_timestamp))
        scored.sort(key=lambda i: (-i[0], i[1].created_at, i[1].id))
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return ()
        fused = supplement_multihop(
            steps=ctx.steps, records=records, baseline=scored,
            step_vectors=np.repeat(ctx.query_vectors[:1], len(ctx.steps), axis=0),
            score_record=ctx.score_record, min_score=ctx.settings.min_relevance_score,
            max_rounds=ctx.settings.multihop_max_rounds, seed_limit=ctx.settings.multihop_seed_limit,
            bridge_limit=ctx.settings.multihop_bridge_limit, supplement_limit=ctx.settings.multihop_supplement_limit,
            budget_seconds=min(remaining, ctx.settings.multihop_budget_seconds), protected_prefix=ctx.limit,
            context_radius=ctx.settings.multihop_context_radius, bridge_anchors=bridge_map)
        ids = {r.id for _, r, _ in fused}
        return tuple(f for f in eligible.values() if f.source_id in ids)
