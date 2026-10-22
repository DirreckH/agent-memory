"""Choose a final, budgeted packet from already scope/version-checked evidence.

Never fall back to an unfiltered raw search result. These guards preserve encoded
source dependencies and explicit needs; they do not certify semantic entailment.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json

from app.governance.extraction import token_count
from app.governance.summary_assembly import (
    SummaryCoverage, SummaryOutcome, assemble_summary, audit_serialized_support,
    required_support_present,
)


def serialized_hits(hits):
    return json.dumps([asdict(hit) for hit in hits], ensure_ascii=False,
                      sort_keys=True, separators=(',', ':'))


def _baseline_supports(item, baseline, snapshot):
    """All group dependencies must be present in their own raw source hit."""
    return all(any(hit.id == ev.source_id and hit.id in snapshot.sources
                   and ev.quote in hit.content for hit in baseline) for ev in item.spans)


def _coverage(items, chosen, requirements):
    rows = []
    for req in requirements:
        found = sum(req.key in item.facets for item in items)
        emitted = sum(req.key in item.facets for item in chosen)
        status = 'missing' if not found else 'omitted' if not emitted else (
            'partial' if emitted < found else 'emitted')
        rows.append(SummaryCoverage(req.key, req.label, found, emitted, status))
    return tuple(rows)


def _raw_hit_matches_source(hit, source, snapshot):
    if hit.content == source.content:
        return True
    # The snapshot records are loaded for this same user and transaction. Accept
    # only the persisted display text plus a prefix supported by source metadata;
    # a matching suffix or arbitrary search-result prefix is not sufficient.
    if not any(record.id == hit.id and record.content == hit.content
               for record in snapshot.records):
        return False
    prefix = f'[{source.role}]\n'
    if source.timestamp is not None:
        try:
            when = datetime.fromtimestamp(source.timestamp / 1000, timezone.utc)
            stamp = when.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
            prefix = f'[{source.role} | {stamp}]\n'
        except (OverflowError, OSError, ValueError):
            pass
    return hit.content == prefix + source.content


def _raw_candidate(items, requirements, baseline, snapshot, *, limit, token_budget):
    """A narrow fallback for whole, unstructured, fully qualified source hits.

    Mixed-scope text, structured status labels, omitted dependencies or unknown
    source IDs disqualify the raw path. All collected items must survive. For
    other cases the compact renderer preserves labels, warnings and references.
    """
    selected = []
    source_ids = {ev.source_id for item in items for ev in item.spans}
    fact_sources = {ev.source_id for fact in snapshot.facts for ev in fact.evidence}
    for hit in baseline:
        if hit.id not in source_ids or hit.id in fact_sources or hit.id not in snapshot.sources:
            continue
        source = snapshot.sources[hit.id]
        if not source.raw_available or not _raw_hit_matches_source(hit, source, snapshot):
            continue
        ranges = sorted((ev.start, ev.end) for item in items for ev in item.spans if ev.source_id == hit.id)
        cursor = 0
        complete = True
        for start, end in ranges:
            if start > cursor and source.content[cursor:start].strip():
                complete = False
                break
            cursor = max(cursor, end)
        if source.content[cursor:].strip():
            complete = False
        if complete and not any(old.id == hit.id for old in selected):
            selected.append(hit)
    if (not selected or len(selected) > limit
            or not all(_baseline_supports(item, selected, snapshot) for item in items)):
        return None
    rows = _coverage(items, items, requirements)
    if any(row.status != 'emitted' for row in rows):
        return None
    tokens = token_count(serialized_hits(selected))
    if tokens > token_budget:
        return None
    return SummaryOutcome(tuple(selected), tuple(row.key for row in rows),
                          coverage=rows, evidence=tuple(items), packet_tokens=tokens)


def select_summary_outcome(collection, plan, baseline, snapshot, outcome, *, limit, token_budget):
    """Keep a valid primary packet unless a qualified alternative preserves more.

    Both candidates use the same serialized budget. Explicit requirement groups
    precede eligible baseline support. Unknown, out-of-scope and obsolete raw
    fragments never enter through this gate: the collection is the eligibility
    boundary. No state/version rules are reimplemented here.
    """
    requirements = collection.requirements or plan.requirements
    items = tuple(item for item in collection.evidence if required_support_present(item, snapshot))
    if limit <= 0 or not items:
        rows = _coverage(items, (), requirements)
        return SummaryOutcome(omitted=tuple(row.key for row in rows if row.found),
                              missing=tuple(row.key for row in rows if not row.found),
                              coverage=rows, selection='no_eligible_packet')
    mandatory = {req.key for req in requirements if getattr(req, 'mandatory', False)}
    protected = tuple(item for item in items if _baseline_supports(item, baseline, snapshot)
                      and item.facets & {req.key for req in requirements if req.key != 'overview'})

    def valid(candidate):
        if not candidate.hits or len(candidate.hits) > limit:
            return False
        serialized = serialized_hits(candidate.hits)
        return (token_count(serialized) <= token_budget and not candidate.support_errors
                and all(item in items for item in candidate.evidence)
                and bool(candidate.evidence)
                and not audit_serialized_support(serialized, candidate.evidence, snapshot))

    def rank(candidate):
        chosen = candidate.evidence
        # Complete explicit requirements precede partial coverage. The source
        # group itself, not a token match or a pre-serialization count, is scored.
        needs = [tuple(item for item in items if key in item.facets) for key in mandatory]
        return (sum(bool(group) and all(item in chosen for item in group) for group in needs),
                sum(bool(item.facets & mandatory) for item in chosen),
                sum(item in chosen for item in protected),
                len({req.key for req in requirements for item in chosen if req.key in item.facets}))

    primary_valid = valid(outcome)
    needed = tuple(item for item in items if item.facets & mandatory)
    if primary_valid and all(item in outcome.evidence for item in (*needed, *protected)):
        return outcome
    alternatives = []
    compact = assemble_summary(collection, plan, baseline, snapshot, limit=limit,
                               token_budget=token_budget, compact=True, preferred=protected)
    if valid(compact):
        alternatives.append(replace(compact, selection='compact_qualified'))
    # Raw hits cannot carry collector warnings. Use this path only when none are
    # present, and no structured facts/known missing coverage require a label.
    if not collection.warnings:
        raw = _raw_candidate(items, requirements, baseline, snapshot,
                             limit=limit, token_budget=token_budget)
        if raw is not None:
            alternatives.append(replace(raw, selection='raw_qualified'))
    empty_rows = _coverage(items, (), requirements)
    best = outcome if primary_valid else SummaryOutcome(
        omitted=tuple(row.key for row in empty_rows if row.found),
        missing=tuple(row.key for row in empty_rows if not row.found),
        coverage=empty_rows, support_errors=outcome.support_errors,
        selection='no_eligible_packet')
    for candidate in alternatives:
        if not best.hits or rank(candidate) > rank(best):
            best = candidate
    return best
