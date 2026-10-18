"""Query-local F1 protection and constrained assembly; no new facts or D1 decisions.

The frozen-style packet is the fallback, not unfiltered V3 hits. Thus a failed
packing attempt cannot resurrect stale values, future records or unrelated data.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re

from app.governance.evidence import valid_evidence
from app.governance.extraction import token_count
from app.governance.planning import contains
from app.governance.summary_assembly import assemble_summary


@dataclass(frozen=True)
class ProtectionAudit:
    passed: bool
    decision: str
    required_units: int = 0
    retained_units: int = 0
    missing_requirements: tuple[str, ...] = ()
    omitted_units: tuple = ()
    coverage_regressions: tuple[str, ...] = ()
    invalid_sources: int = 0


def unit_key(item):
    return tuple((s.source_id, s.start, s.end, s.content_hash, s.purpose) for s in item.spans)


def matches(item, requirement, collection, snapshot, plan):
    text = '\n'.join(s.quote for s in item.spans if s.purpose != 'update')
    if requirement.origin == 'direct':
        predicate = requirement.key.split(':', 1)[1]
        facts = [f for f in snapshot.facts if f.id in item.fact_ids and f.predicate == predicate]
        if facts:
            # The resolver already decided versions. An excluded/planned assertion
            # cannot become a direct current-state anchor just through word overlap.
            allowed = collection.current_fact_ids | collection.conflict_fact_ids
            if plan.order == 'chronological' or re.search(
                    r'\b(?:history|previous|past|formerly)\b|历史|以前|曾经|此前', plan.task, re.I):
                allowed |= collection.history_fact_ids
            return any(f.id in allowed for f in facts)
    return any(contains(text, term) for term in requirement.terms)


def guarded_summary(collection, plan, baseline, snapshot, *, limit, token_budget, governance_result=None,
                    preserve_coverage=True):
    original = assemble_summary(collection, plan, baseline, snapshot, limit=limit, token_budget=token_budget)
    demands = plan.direct_requirements + plan.attribution_requirements
    if not demands:
        return original
    valid_items = tuple(e for e in collection.evidence if all(valid_evidence(s, snapshot) for s in e.spans))
    rejected = len(collection.evidence) - len(valid_items)
    collection = replace(collection, evidence=valid_items)
    matching = {r.key: tuple(e for e in collection.evidence if matches(e, r, collection, snapshot, plan)) for r in demands}
    missing = tuple(r.key for r in demands if not matching[r.key])
    keys = {unit_key(e) for units in matching.values() for e in units}
    # The existing D1 evidence and base ranking only order safe, in-scope units;
    # their rendered text is never reparsed or appended without source validation.
    governed = {f for e in getattr(governance_result, 'evidence', ()) for f in e.fact_ids}
    ranks = {h.id: i for i, h in enumerate(baseline)}
    items = []
    for item in collection.evidence:
        if unit_key(item) in keys:
            item = replace(item, local_attribution=True)
        items.append(item)
    # Stable order among equal anchors; source chronology for final display remains
    # in assemble_summary. No score, vector, threshold or temporal ranking changes.
    items.sort(key=lambda e: (unit_key(e) not in keys, not bool(set(e.fact_ids) & governed),
                              min((ranks[s.source_id] for s in e.spans if s.source_id in ranks), default=len(ranks))))
    protected_collection = replace(collection, evidence=tuple(items))

    # Retain a source-backed witness for every existing explicit coverage need.
    # This is a structural floor, not a certificate of semantic completeness.
    floor = set(keys)
    selected = [e for e in items if unit_key(e) in floor]
    for row in original.coverage if preserve_coverage else ():
        if row.key == 'overview':
            continue
        attr = 'supported' if row.supported else 'facets'
        if not row.emitted or any(row.key in getattr(e, attr) for e in selected):
            continue
        candidates = [e for e in original.evidence if row.key in getattr(e, attr)]
        if candidates:
            witness = min(candidates, key=lambda e: token_count('\n'.join(s.quote for s in e.spans)))
            floor.add(unit_key(witness))
            selected.append(witness)
    candidate = assemble_summary(protected_collection, plan, baseline, snapshot, limit=limit,
                                 token_budget=token_budget, protected=floor)
    emitted = {unit_key(e) for e in candidate.evidence}
    omitted = tuple(sorted(keys - emitted))
    new_rows = {r.key: r for r in candidate.coverage}
    regressions = tuple(r.key for r in original.coverage if
        (r.emitted and not getattr(new_rows.get(r.key), 'emitted', 0)) or
        (r.supported and not getattr(new_rows.get(r.key), 'supported', 0)))
    invalid = rejected + sum(not valid_evidence(s, snapshot) for e in candidate.evidence for s in e.spans)
    passed = not (missing or omitted or (regressions if preserve_coverage else ()) or invalid)
    returned = candidate if passed else original
    retained = {unit_key(e) for e in returned.evidence}
    final_missing = tuple(r.key for r in demands if not matching[r.key] or
                          any(unit_key(e) not in retained for e in matching[r.key]))
    audit = ProtectionAudit(passed, 'protected_merge' if passed else 'preserved_f1_baseline', len(keys),
                            len(keys & retained), final_missing,
                            tuple(sorted(keys - retained)), regressions, invalid)
    # Failed protection remains visible to callers/audits. Never call fallback a
    # successful repair, and never silently widen scope to make an empty packet nonempty.
    return replace(returned, protection=audit)
