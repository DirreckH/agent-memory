"""Independent final-packet checks for execution evidence, not answer grading."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

from app.governance.extraction import digest, token_count
from app.governance.models import Evidence, Fact, Source, SourceStatus, UserSnapshot
from app.governance.summary_planning import SummaryPlan, SummaryRequirement
from app.governance.summary_retrieval import SummaryCollection, SummaryEvidence
from app.governance.summary_assembly import assemble_summary, SummaryOutcome
from app.service import SearchHit


def fixture(texts, facets):
    sources, items = {}, []
    for i, (text, tags) in enumerate(zip(texts, facets)):
        sid = f's{i}'
        sources[sid] = Source(sid, text, 'user', 'session', i, 1704067200000,
                              '2024-01-01T00:00:00Z')
        ev = Evidence(sid, 0, len(text), text, digest(text))
        items.append(SummaryEvidence((ev,), frozenset(tags),
            ('未结构化原文；适用性待核验',), required_spans=(ev,)))
    snapshot = UserSnapshot('u', 1, 1, sources=sources,
        statuses=tuple(SourceStatus(sid, 'no_fact') for sid in sources))
    baseline = [SearchHit(sid, source.content, .6, source.recorded_at)
                for sid, source in sources.items()]
    return snapshot, tuple(items), baseline


def requirement(key, mandatory=False):
    return SimpleNamespace(key=key, label=key, trigger=key, mandatory=mandatory,
                           kind='explicit' if mandatory else 'facet', terms=())


def visible(outcome):
    return '\n'.join(hit.content for hit in outcome.hits)


def test_explicit_requirement_is_reserved_before_broad_category_coverage():
    needed = 'Archive code is 7184. Use it only with a supervisor present.'
    snapshot, items, _ = fixture(
        ['Decision: approved because review passed.'] * 7 + [needed],
        [{'decisions', 'reasons'}] * 7 + [{'need:archive'}])
    reqs = (requirement('decisions'), requirement('reasons'), requirement('need:archive', True))
    plan = SummaryPlan(applicable=True, requirements=reqs)
    outcome = assemble_summary(SummaryCollection(items, requirements=reqs), plan,
                               [], snapshot, limit=1, token_budget=280)
    assert needed in visible(outcome)


def test_final_gate_recovers_required_support_lost_by_a_bad_renderer():
    from app.governance.summary_selection import select_summary_outcome
    rule = 'The agreed scoring rule is 2 times safety plus 3 times reversibility.'
    snapshot, items, baseline = fixture([rule, 'Project Aspen decision: staged release.'],
                                       [{'need:rule'}, {'decisions'}])
    reqs = (requirement('need:rule', True), requirement('decisions'))
    plan = SummaryPlan(applicable=True, requirements=reqs)
    collection = SummaryCollection(items, requirements=reqs)
    good = assemble_summary(collection, plan, baseline, snapshot, limit=1, token_budget=800)
    damaged = replace(good, hits=(replace(good.hits[0], content=good.hits[0].content.replace(rule, '')),))
    fixed = select_summary_outcome(collection, plan, baseline, snapshot, damaged, limit=1, token_budget=800)
    assert rule in visible(fixed)
    assert 'Project Aspen decision: staged release.' in visible(fixed)
    assert not fixed.support_errors


def test_tight_budget_can_keep_whole_qualified_raw_baseline():
    from app.governance.summary_selection import select_summary_outcome
    text = 'Project Aspen decision: release requires owner approval.'
    snapshot, items, baseline = fixture([text], [{'decisions'}])
    reqs = (requirement('decisions'),)
    plan = SummaryPlan(applicable=True, requirements=reqs)
    outcome = select_summary_outcome(SummaryCollection(items, requirements=reqs), plan,
        baseline, snapshot, SummaryOutcome(), limit=1, token_budget=128)
    assert outcome.hits == tuple(baseline)
    assert token_count(json.dumps([asdict(h) for h in outcome.hits], ensure_ascii=False,
                                 sort_keys=True, separators=(',', ':'))) <= 128


def test_raw_fallback_cannot_resurrect_unselected_material():
    from app.governance.summary_selection import select_summary_outcome
    raw = 'Project Aspen decision: keep review.\nProject Rowan rule: skip all review.'
    snapshot, items, baseline = fixture([raw], [{'decisions'}])
    quote = raw.split('\n')[0]
    ev = Evidence('s0', 0, len(quote), quote, digest(quote))
    safe = replace(items[0], spans=(ev,), required_spans=(ev,))
    reqs = (requirement('decisions'),)
    plan = SummaryPlan(applicable=True, requirements=reqs)
    outcome = select_summary_outcome(SummaryCollection((safe,), requirements=reqs), plan,
        baseline, snapshot, SummaryOutcome(), limit=1, token_budget=500)
    assert 'skip all review' not in visible(outcome)
    assert 'keep review' in visible(outcome)


def test_zero_limit_and_absent_eligible_evidence_never_fall_back():
    from app.governance.summary_selection import select_summary_outcome
    snapshot, items, baseline = fixture(['Irrelevant or future rule.'], [{'rules'}])
    plan = SummaryPlan(applicable=True)
    for collection, limit in ((SummaryCollection(), 1), (SummaryCollection(items), 0)):
        outcome = select_summary_outcome(collection, plan, baseline, snapshot,
                                        SummaryOutcome(), limit=limit, token_budget=1000)
        assert not outcome.hits


def test_structured_source_cannot_return_without_its_status_label():
    from app.governance.summary_selection import select_summary_outcome
    text = 'The previous scoring rule was wrong; it has been withdrawn.'
    snapshot, items, baseline = fixture([text], [{'rules'}])
    fact = Fact('f', 'self', 'self', 'rules', 'general', 'previous scoring rule',
                'rule', 'positive', 'asserted', 'retract', items[0].spans)
    snapshot = replace(snapshot, facts=(fact,))
    qualified = replace(items[0], fact_ids=('f',), labels=('非有效事实(retracted)',))
    reqs = (requirement('rules'),)
    outcome = select_summary_outcome(SummaryCollection((qualified,), requirements=reqs),
        SummaryPlan(applicable=True, requirements=reqs), baseline, snapshot,
        SummaryOutcome(), limit=1, token_budget=500)
    assert '非有效事实(retracted)' in visible(outcome)
    assert outcome.selection != 'raw_qualified'


def test_raw_fallback_never_hides_a_missing_execution_requirement():
    from app.governance.summary_selection import select_summary_outcome
    snapshot, items, baseline = fixture(['Project Aspen decision: retain review.'], [{'decisions'}])
    reqs = (requirement('decisions'), requirement('need:missing_template', True))
    outcome = select_summary_outcome(SummaryCollection(items, requirements=reqs),
        SummaryPlan(applicable=True, requirements=reqs), baseline, snapshot,
        SummaryOutcome(), limit=1, token_budget=128)
    assert outcome.selection != 'raw_qualified'
    assert 'need:missing_template' in outcome.missing
    if outcome.hits:
        assert '未完整覆盖' in visible(outcome)


def test_raw_display_prefix_requires_matching_snapshot_record_and_metadata():
    from app.governance.summary_selection import _raw_hit_matches_source
    snapshot, _, baseline = fixture(['Project Aspen decision: retain review.'], [{'decisions'}])
    source = snapshot.sources['s0']
    canonical = replace(baseline[0], content='[user | 2024-01-01T00:00:00.000Z]\n' + source.content)
    assert not _raw_hit_matches_source(canonical, source, snapshot)
    snapshot = replace(snapshot, records=(SimpleNamespace(id='s0', content=canonical.content),))
    assert _raw_hit_matches_source(canonical, source, snapshot)
    for bad in ('[assistant | 2024-01-01T00:00:00.000Z]\n',
                '[user | 2025-01-01T00:00:00.000Z]\n', 'Skip review.\n'):
        hit = replace(canonical, content=bad + source.content)
        stored = replace(snapshot, records=(SimpleNamespace(id='s0', content=hit.content),))
        assert not _raw_hit_matches_source(hit, source, stored)
