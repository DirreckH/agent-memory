from __future__ import annotations

from collections import defaultdict
import re

from app.governance.models import Fact, StateResolution, UserSnapshot, VersionRelation
from app.governance.registry import property_spec


def _precedes(old: Fact, new: Fact, *, observation=False) -> bool:
    a = old.observation_time if observation else old.source_time
    b = new.observation_time if observation else new.source_time
    if observation and (a is None or b is None):
        return False
    if a is not None and b is not None and a != b:
        return a < b
    if old.commit_sequence is not None and new.commit_sequence is not None:
        if old.commit_sequence == new.commit_sequence:
            return (old.ordinal, old.evidence[0].start) < (new.ordinal, new.evidence[0].start)
        if not observation or (a is not None and b is not None):
            return old.commit_sequence < new.commit_sequence
    return False


def derive_relations(facts: tuple[Fact, ...]) -> tuple[VersionRelation, ...]:
    """在提交事务的最新快照内计算，禁止模型指定数据库事实 ID。"""
    grouped = defaultdict(list)
    for fact in facts:
        grouped[fact.slot].append(fact)
    relations = {}
    for slot, group in grouped.items():
        spec = property_spec(slot[1])
        for new in group:
            if new.modality != "asserted":
                continue
            support = None
            target_candidates = [f for f in group if f.id != new.id and f.modality == "asserted"
                                 and new.target_value is not None and f.value.casefold() == new.target_value.casefold()
                                 and _precedes(f, new)]
            target_times = {f.observation_time for f in target_candidates}
            broad_target = any(re.search(r"\bnever\b|all (?:earlier|previous)|从未|所有.*(?:说法|记录)", e.quote, re.I)
                               for e in new.evidence if e.purpose == "update")
            for old in group:
                implicit = new.operation == "assert"
                if old.id == new.id or old.modality != "asserted" or not _precedes(old, new, observation=implicit):
                    continue
                evidence = tuple(e for e in new.evidence if e.purpose == "update")
                relation = None
                if new.operation in ("correct", "retract", "change") and evidence:
                    if new.target_value is not None and new.target_value.casefold() == old.value.casefold():
                        if new.target_date is not None:
                            moment = old.observation_time
                            if moment is not None and moment // 86400000 == new.target_date // 86400000:
                                relation = {"correct": "corrects", "retract": "retracts", "change": "changes"}[new.operation]
                        elif new.operation in ("correct", "retract") and len(target_candidates) > 1 and len(target_times) > 1 and not broad_target:
                            relation = "conflicts"
                        else:
                            relation = {"correct": "corrects", "retract": "retracts", "change": "changes"}[new.operation]
                    elif new.operation == "change" and new.target_value is None and spec.cardinality == "single" and spec.mutable:
                        relation = "changes"
                elif new.polarity == old.polarity == "positive" and old.operation != "retract":
                    if old.value.casefold() == new.value.casefold():
                        relation = "supports"
                    elif (implicit and spec.cardinality == "single" and spec.mutable and new.kind in ("state", "preference")
                          and old.kind in ("state", "preference") and _precedes(old, new, observation=True)):
                        relation = "changes"
                    elif spec.cardinality != "multi":
                        relation = "conflicts"
                elif new.polarity != old.polarity and new.value.casefold() == old.value.casefold():
                    relation = "conflicts"
                if relation:
                    rel = VersionRelation(new.id, old.id, relation, evidence or new.evidence)
                    if relation == "supports":
                        # 支持关系连接最近一次同值观察即可；原文和摘要依赖仍完整保留。
                        if support is None or _precedes(support[0], old, observation=True):
                            support = (old, rel)
                    else:
                        relations[(new.id, old.id, relation)] = rel
            if support is not None:
                rel = support[1]
                relations[(rel.newer, rel.older, rel.relation)] = rel
    return tuple(relations.values())


class StateResolver:
    def resolve(self, snapshot: UserSnapshot, slots=None, intent="current", reference=None) -> StateResolution:
        considered = {f.id: f for f in snapshot.facts if slots is None or f.slot in slots}
        excluded, updates, unresolved = {}, [], []
        for fid, reason in snapshot.blocked_facts:
            if fid in considered and (intent == "current" or not reason.endswith("_changes")):
                excluded[fid] = reason
        relations = snapshot.relations
        by_id = {f.id: f for f in snapshot.facts}
        for rel in relations:
            if rel.older not in considered:
                continue
            new = by_id.get(rel.newer)
            if (rel.relation == "conflicts" and new is not None and new.operation in ("correct", "retract")
                    and rel.newer in considered):
                unresolved.append((rel.older, rel.newer))
                updates.extend(rel.evidence)
            if (intent == "current" and rel.relation == "changes" and reference is not None and new is not None
                    and new.observation_time is not None and new.observation_time > reference):
                continue
            if rel.relation in ("corrects", "retracts"):
                excluded[rel.older] = "corrected_claim" if rel.relation == "corrects" else "retracted_claim"
                updates.extend(rel.evidence)
            elif rel.relation == "changes" and intent == "current":
                excluded.setdefault(rel.older, "superseded_observation")
                updates.extend(rel.evidence)
        grouped = defaultdict(list)
        all_slots = set()
        for fact in considered.values():
            all_slots.add(fact.slot)
            if fact.id in excluded:
                continue
            if fact.modality != "asserted" or fact.polarity != "positive" or fact.operation == "retract":
                excluded[fact.id] = "not_confirmed_state"
                continue
            if intent == "history" and reference is not None and fact.observation_time is not None and fact.observation_time > reference:
                excluded[fact.id] = "after_reference"
                continue
            if intent == "current" and reference is not None and fact.valid_from is not None and fact.valid_from > reference:
                excluded[fact.id] = "not_yet_effective"
                continue
            if (intent == "current" and reference is not None and fact.valid_from is None
                    and fact.source_time is not None and fact.source_time > reference):
                excluded[fact.id] = "after_reference"
                continue
            if intent == "current" and reference is not None and fact.valid_to is not None and fact.valid_to <= reference:
                excluded[fact.id] = "expired_interval"
                continue
            grouped[fact.slot].append(fact)
        selected, conflicts = [], list(unresolved)
        for slot, group in grouped.items():
            unique = {}
            for fact in sorted(group, key=lambda f: (f.observation_time or -1, f.commit_sequence or -1, f.ordinal, f.id)):
                key = (fact.value.casefold(), fact.polarity)
                if intent == "current" and key in unique:
                    excluded[unique[key].id] = "duplicate_support"
                unique[key if intent == "current" else fact.id] = fact
            selected.extend(unique.values())
            if intent == "current" and property_spec(slot[1]).cardinality != "multi" and len(unique) > 1:
                conflicts.append(tuple(f.id for f in unique.values()))
        selected.sort(key=lambda f: (f.observation_time is None, f.observation_time or 0, f.source_id, f.id))
        if intent == "current":
            for negative in considered.values():
                if (negative.polarity != "negative" or negative.modality != "asserted" or negative.operation == "retract"
                        or excluded.get(negative.id) not in (None, "not_confirmed_state")):
                    continue
                for positive in selected:
                    if positive.slot != negative.slot or positive.value.casefold() != negative.value.casefold():
                        continue
                    if (negative.observation_time is not None and positive.observation_time is not None
                            and negative.observation_time < positive.observation_time):
                        continue
                    conflicts.append((positive.id, negative.id))
                    updates.extend(negative.evidence)
        seen, evidence = set(), []
        for ev in updates:
            key = (ev.source_id, ev.start, ev.end, ev.purpose)
            if key not in seen:
                evidence.append(ev)
                seen.add(key)
        unknown = tuple(sorted(all_slots - {f.slot for f in selected}))
        return StateResolution(tuple(selected), excluded, tuple(evidence), tuple(conflicts), snapshot.incomplete, unknown)
