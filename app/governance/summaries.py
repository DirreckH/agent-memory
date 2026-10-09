from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone

from app.governance.extraction import digest
from app.governance.models import SummarySnapshot, SummaryUnit, UserSnapshot, StateResolution
from app.governance.registry import property_spec
from app.governance.resolution import StateResolver


SECTIONS = ("current", "history", "events", "plans", "conflicts")
LABELS = {"current": "当前状态", "history": "历史变化", "events": "事件与决定",
          "plans": "计划", "conflicts": "冲突与纠正", "directory": "来源目录"}


def stage(fact):
    moment = fact.valid_from if fact.valid_from is not None else fact.source_time
    if moment is None:
        return "时间未知"
    try:
        day = datetime.fromtimestamp(moment / 1000, timezone.utc).date().isoformat()
    except (ValueError, OSError, OverflowError):
        return "时间未知"
    return ("明确有效日:" if fact.valid_from is not None else "提及日:") + day


def fact_line(fact, *, label=""):
    ev = fact.evidence[0]
    return (f"[{label + ';' if label else ''}{stage(fact)};来源:{ev.source_id}:{ev.start}-{ev.end};"
            f"归因:{fact.subject_name};场景:{fact.scope}] {ev.quote}")


class SummaryBuilder:
    """直接读取事实和原文构建派生视图，从不以旧摘要作为唯一输入。"""
    def __init__(self, *, versions=True):
        self.versions = versions

    def build(self, snapshot: UserSnapshot, scope=None, *, budget_ms=50, reference=None, time_window=None) -> SummarySnapshot:
        deadline = time.perf_counter() + budget_ms / 1000
        if budget_ms <= 0:
            return SummarySnapshot(snapshot.generation, snapshot.revision, (), False)
        if self.versions:
            current = StateResolver().resolve(snapshot, intent="current", reference=reference)
            history = StateResolver().resolve(snapshot, intent="history")
        else:
            current = history = StateResolution(snapshot.facts, {}, (), (), snapshot.incomplete)
        selected = {f.id for f in current.selected} if self.versions else {f.id for f in snapshot.facts}
        historical = {f.id for f in history.selected} if self.versions else selected
        conflict_ids = {fid for group in current.conflicts for fid in group}
        groups = defaultdict(list)
        for fact in snapshot.facts:
            if time.perf_counter() >= deadline:
                return SummarySnapshot(snapshot.generation, snapshot.revision, (), False)
            if scope is not None and fact.slot not in scope:
                continue
            if time_window is not None and fact.observation_time is not None and not time_window.contains(fact.observation_time):
                continue
            if (self.versions and current.excluded.get(fact.id) == "duplicate_support"
                    and any(f.slot == fact.slot and f.value == fact.value and f.quote == fact.quote for f in current.selected)):
                continue
            if fact.id in conflict_ids:
                section = "conflicts"
            elif fact.modality != "asserted" or current.excluded.get(fact.id) == "not_yet_effective":
                section = "plans"
            elif fact.operation == "retract":
                section = "conflicts"
            elif fact.kind in ("event", "decision", "rule", "other") and not (
                    history.excluded.get(fact.id) in ("corrected_claim", "retracted_claim")
                    or history.excluded.get(fact.id, "").endswith(("_corrects", "_retracts"))):
                section = "events"
            elif fact.id not in historical:
                continue
            elif fact.id in selected:
                section = "current"
            else:
                section = "history"
            source = snapshot.sources[fact.source_id]
            spec = property_spec(fact.predicate)
            topic = f"{fact.subject_name}/{spec.category}/{fact.scope}"
            groups[(source.session_id, topic, stage(fact), section)].append(fact)
        units = []

        def make(level, key, section, text, facts=(), sources=(), children=(), coverage=(), topic="", phase="", display=()):
            fid = tuple(sorted(set(facts)))
            sid = tuple(sorted(set(sources)))
            child_ids = tuple(u.id for u in children)
            uid = "sum_" + digest(str((snapshot.user_id, snapshot.generation, snapshot.revision,
                                       level, key, section, fid, sid, child_ids, text)))[:24]
            return SummaryUnit(uid, level, key, section, text, fid, sid, child_ids,
                               tuple(sorted(set(coverage))), topic, phase, reference_ms=reference, display_fact_ids=tuple(display))

        for (session, topic, phase, section), facts in sorted(groups.items()):
            # 同阶段相同完整表达只压缩文字，所有来源仍进入依赖与覆盖清单。
            lines = dict.fromkeys(fact_line(f, label=f.modality if section == "plans" else "") for f in facts)
            dependencies = list(facts)
            if section == "current" and self.versions:
                dependencies.extend(f for f in snapshot.facts if current.excluded.get(f.id) == "duplicate_support"
                                    and any(f.slot == basis.slot and f.value == basis.value and f.quote == basis.quote for basis in facts))
            units.append(make("session", f"{session}/{topic}/{phase}", section, "\n".join(lines),
                              [f.id for f in dependencies], [f.source_id for f in dependencies],
                              coverage=[(e.source_id, e.start, e.end) for f in dependencies for e in f.evidence],
                              topic=topic, phase=phase, display=[f.id for f in facts]))
        # 纠正记录单独保留；错误旧值不会进入当前或真实历史分区。
        correction_groups = defaultdict(list)
        for rel in snapshot.relations:
            if rel.relation in ("corrects", "retracts"):
                correction_groups[rel.newer].append(rel)
        by_id = {f.id: f for f in snapshot.facts}
        for fid, relations in sorted(correction_groups.items()):
            fact = by_id[fid]
            evs = { (e.source_id, e.start, e.end): e for r in relations for e in r.evidence }
            text = "[纠正/撤回记录；错误旧说法不作为历史事实]\n" + "\n".join(
                f"[来源:{e.source_id}:{e.start}-{e.end}] {e.quote}" for e in evs.values())
            units.append(make("session", "correction/" + fid, "conflicts", text, [fid],
                              [e.source_id for e in evs.values()], coverage=tuple(evs),
                              topic=f"{fact.subject_name}/{property_spec(fact.predicate).category}/{fact.scope}", phase=stage(fact)))
        categorized = {f.source_id for f in snapshot.facts}
        directory_sources = set(snapshot.sources) - categorized
        if directory_sources:
            # 无法归类的内容仍可从会话目录取回；目录本身不宣称拥有事实覆盖。
            text = "\n".join(f"[来源:{sid};会话:{snapshot.sources[sid].session_id};"
                             f"{'时间未知' if snapshot.sources[sid].timestamp is None else '仅有提及时间'}]"
                             for sid in sorted(directory_sources))
            units.append(make("session", "source-directory", "directory", text, sources=directory_sources))
        topic_groups = defaultdict(list)
        for unit in units:
            topic_groups[(unit.topic, unit.section)].append(unit)
        for (topic, section), children in sorted(topic_groups.items()):
            units.append(make("topic", topic, section, "\n".join(u.text for u in children),
                              [f for u in children for f in u.fact_ids], [s for u in children for s in u.source_ids],
                              children, [c for u in children for c in u.coverage], topic))
        topics = [u for u in units if u.level == "topic"]
        sections = []
        for section in (*SECTIONS, "directory"):
            children = [u for u in topics if u.section == section]
            if children:
                sections.append(f"【{LABELS[section]}】\n" + "\n".join(u.text for u in children))
        overview = make("overview", "global", "overview", "\n".join(sections),
                        [f for u in topics for f in u.fact_ids], [s for u in topics for s in u.source_ids],
                        topics, [c for u in topics for c in u.coverage])
        units.append(overview)
        return SummarySnapshot(snapshot.generation, snapshot.revision, tuple(units), time.perf_counter() < deadline)
