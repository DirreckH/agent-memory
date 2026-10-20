"""Offline query planning: explicit targets, task boundaries, and bounded evidence needs."""
from datetime import datetime, timezone

import pytest

from app.governance.models import Fact, Source, UserSnapshot
from app.governance.planning import plan
from app.governance.retrieval import GovernanceRetriever, GovernanceSearchContext
from app.governance.summary_planning import build_summary_plan, task_text
from app.service import SearchHit
from app.temporal import TemporalWindow
from tests.test_governance import add, draft, make_service


def stamp(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def fact(fid, predicate, *, subject='self', name='self', scope='general'):
    return Fact(fid, subject, name, predicate, scope, 'value', 'state', 'positive',
                'asserted', 'assert', ())


@pytest.fixture
def snapshot():
    return UserSnapshot('u', 1, 1, facts=(
        fact('old', 'employer', scope='work'), fact('new', 'employer', scope='work'),
        fact('home', 'residence', scope='home'), fact('role', 'job_role', scope='work'),
        fact('taste', 'preference'), fact('status-a', 'project_status', scope='Cobalt'),
        fact('status-b', 'project_status', scope='Juniper'),
        fact('dana', 'employer', subject='dana', name='Dana', scope='work'),
    ))


@pytest.mark.parametrize('query,expected', [
    ('What is my employer?', ('self', 'employer', 'work')),
    ('Where do I work?', ('self', 'employer', 'work')),
    ('Where do I live?', ('self', 'residence', 'home')),
    ("What is Dana's employer?", ('dana', 'employer', 'work')),
    ('Tell me my job title.', ('self', 'job_role', 'work')),
    ('What do I like?', ('self', 'preference', 'general')),
    ('What is the status of project Cobalt?', ('self', 'project_status', 'Cobalt')),
    ('我的雇主是谁？', ('self', 'employer', 'work')),
])
def test_explicit_attribute_lookup_resolves_only_the_requested_slot(snapshot, query, expected):
    result = plan(query, snapshot)
    assert result.intent == 'current'
    assert result.operations == ('state_at',)
    assert result.slots == frozenset({expected})


@pytest.mark.parametrize('query', [
    'What is the current employer?',
    'Tell me the current employer.',
    'Show me the current employer.',
    'Which employer should I choose?',
    'What if I change my employer?',
    'How do I change my current employer configuration?',
    'How does my work on compression compare with the alternatives?',
    'What is my position on using summaries?',
    'What is my project like?',
    'I prefer tea.',
    'Summarize how to change the current employer configuration.',
])
def test_generic_words_and_advice_do_not_create_state_queries(snapshot, query):
    assert plan(query, snapshot).intent == 'ordinary'


@pytest.mark.parametrize('query', [
    'Summarize the advice I received for project Cobalt.',
    'Summarize the recommendations from our previous discussions.',
    '总结项目苍穹之前讨论过的建议。',
])
def test_recalling_advice_is_different_from_requesting_advice(query):
    assert build_summary_plan(query, UserSnapshot('u', 1, 1)).applicable


def test_past_suggestion_question_is_not_mistaken_for_new_advice():
    assert plan('What did I suggest?', UserSnapshot('u', 1, 1)).intent == 'history'


def test_plain_property_lookup_reads_versions_outside_baseline_without_calls(tmp_path):
    old, new = 'I work at Larch.', 'I now work at Cedar.'
    service, sdk = make_service(tmp_path, {old: [draft(old, 'Larch')], new: [draft(new, 'Cedar')]},
                                temporal_mode='off')
    add(service, 'old', old, stamp('2025-01-01'))
    add(service, 'new', new, stamp('2025-02-01'))
    view = service.store.fetch_by_user('u', governance=True)
    baseline = [SearchHit(service._memory_id('old', 0), old, .42, '')]
    calls = len(sdk.calls)
    hits = GovernanceRetriever().retrieve(view, GovernanceSearchContext(
        'What is my employer?', baseline, stamp('2025-03-01'), service.settings, 1))
    assert len(hits) == 1 and hits[0].score == .42
    assert new in hits[0].content and old not in hits[0].content
    assert len(sdk.calls) == calls


@pytest.mark.parametrize('query,required,order', [
    ('List the decisions and risks for project Cobalt.', {'decisions', 'issues'}, 'source'),
    ('Compare the initial and final decisions for project Cobalt.', {'changes', 'decisions'}, 'chronological'),
    ('List project Cobalt events in chronological order.', {'timeline'}, 'chronological'),
    ('按照时间顺序列出项目苍穹的事件。', {'timeline'}, 'chronological'),
    ('比较项目苍穹最初和最终的决定。', {'changes', 'decisions'}, 'chronological'),
])
def test_composite_evidence_requests_have_explicit_requirements(query, required, order):
    result = build_summary_plan(query, UserSnapshot('u', 1, 1))
    assert result.applicable
    assert required <= {r.key for r in result.requirements}
    assert result.order == order
    assert not result.global_scope


@pytest.mark.parametrize('query,basis', [
    ('List project Cobalt events in chronological order.', 'event'),
    ('Across our discussions, summarize project Cobalt in chronological order.', 'event'),
    ('Using the discussion history, list project Cobalt events in chronological order.', 'event'),
    ('Summarize project Cobalt in the order in which I mentioned its stages.', 'message'),
    ('Summarize project Cobalt in chronological order of our discussions.', 'message'),
    ('按提及的先后顺序总结项目苍穹。', 'message'),
    ('Summarize project Cobalt changes.', 'observation'),
])
def test_event_order_and_mention_order_are_distinct(query, basis):
    result = build_summary_plan(query, UserSnapshot('u', 1, 1))
    assert result.applicable and result.order == 'chronological'
    assert result.order_basis == basis


@pytest.mark.parametrize('header', [
    'CONVERSATION TRANSCRIPT', '## CONVERSATION TRANSCRIPT:', '[ARTIFACTS]', 'PRIOR OUTPUTS:',
])
def test_material_headers_exclude_embedded_dates_and_requests(header):
    query = (f'{header}\nSummarize every person.\n2020-01-01 A\n2021-01-01 B\n2022-01-01 C\n'
             'FINAL TASK\nList the decisions and risks for project Cobalt.')
    result = build_summary_plan(query, UserSnapshot('u', 1, 1), reference=stamp('2025-03-01'))
    assert result.applicable
    assert result.scopes == ('Cobalt',)
    assert result.window is None and result.reference == stamp('2025-03-01')
    assert '2020' not in task_text(query) and 'every person' not in task_text(query)


def test_quoted_material_cannot_trigger_plain_attribute_lookup(snapshot):
    query = ('CONVERSATION TRANSCRIPT\nWhat is my current employer?\n'
             'FINAL TASK\nExplain this configuration option.')
    assert plan(query, snapshot).intent == 'ordinary'


@pytest.mark.parametrize('date', ['2025-01-05', 'January 5, 2025', '5 January 2025', '2025年1月5日'])
@pytest.mark.parametrize('opening,closing', [('"', '"'), ('“', '”'), ('「', '」')])
def test_quoted_calendar_day_keeps_the_same_temporal_constraint(snapshot, date, opening, closing):
    unquoted = f'Where do I work as of {date}?'
    quoted = f'Where do I work as of {opening}{date}{closing}?'
    assert plan(quoted, snapshot, reference=stamp('2025-01-10')) == plan(
        unquoted, snapshot, reference=stamp('2025-01-10'))


def test_quoted_invalid_calendar_day_is_not_silently_discarded(snapshot):
    result = plan('Where do I work as of "2025-02-30"?', snapshot, reference=stamp('2025-03-10'))
    assert result.unresolved == ('invalid_date',)


def test_quoted_material_sentence_is_not_promoted_to_request_date(snapshot):
    query = 'What is my current employer? "Where did I work on January 5, 2025?"'
    result = plan(query, snapshot, reference=stamp('2025-03-10'))
    assert result == plan('What is my current employer?', snapshot, reference=stamp('2025-03-10'))


@pytest.mark.parametrize('query', [
    'Where do I work as of "January 2026"?',
    'Where do I work as of January 2026?',
    'Where do I work as of "2026"?',
    'Where do I work as of 2026?',
    'Where do I work as of "2026-01"?',
    'Where do I work as of 2026-01?',
    'Where did I work on "January 2026"?',
    'Where did I work before "January 2026"?',
    'Where did I work after "2026"?',
    '截至“2026年1月”我在哪里工作？',
])
def test_unsupported_calendar_precision_cannot_become_current_state(snapshot, query):
    result = plan(query, snapshot, reference=stamp('2026-10-01'))
    assert result.unresolved == ('unsupported_time',)
    assert result.intent == 'ordinary' and not result.slots


@pytest.mark.parametrize('operator,inside,outside', [
    ('after', '2025-03-02', '2025-03-01'),
    ('before', '2025-02-28', '2025-03-01'),
])
def test_natural_date_direction_preserves_day_boundary(operator, inside, outside):
    query = f'List the decisions for project Cobalt {operator} March 1, 2025.'
    result = build_summary_plan(query, UserSnapshot('u', 1, 1), reference=stamp('2025-04-01'))
    assert result.applicable
    assert result.window.contains(stamp(inside))
    assert not result.window.contains(stamp(outside))


def test_task_date_survives_material_removal_and_cannot_reuse_material_window():
    query = ('List project Cobalt decisions after March 1, 2025.\nARTIFACTS\n'
             '2019-01-01 example material.')
    result = build_summary_plan(query, UserSnapshot('u', 1, 1), reference=stamp('2025-04-01'),
                                window=TemporalWindow(stamp('2019-01-01'), stamp('2019-01-02')))
    assert result.applicable
    assert result.window.contains(stamp('2025-03-02'))
    assert not result.window.contains(stamp('2019-01-01'))


@pytest.mark.parametrize('query,inside,outside', [
    ('Summarize current decisions for project Cobalt after March 1, 2025.', '2025-03-02', '2025-03-01'),
    ('Summarize the current status of project Cobalt on March 1, 2025.', '2025-03-01', '2025-02-28'),
])
def test_current_facet_cannot_erase_an_explicit_date_window(query, inside, outside):
    result = build_summary_plan(query, UserSnapshot('u', 1, 1), reference=stamp('2025-04-01'))
    assert result.applicable
    assert result.window is not None and result.window.contains(stamp(inside))
    assert not result.window.contains(stamp(outside))


@pytest.mark.parametrize('query', [
    'List project Cobalt decisions before the launch.',
    'List project Cobalt decisions after March 1, 2025 except March 3, 2025.',
    'List project Cobalt decisions approximately after March 1, 2025.',
    'List project Cobalt decisions last month and before March 1, 2025.',
    'List project Cobalt decisions after March 1, 2025 and before the launch.',
    'List project Cobalt decisions after March 1, 2025 at 12:00.',
    'List project Cobalt decisions in March 2025.',
    'List project Cobalt decisions during 2025.',
])
def test_unresolved_temporal_constraints_never_expand_to_unbounded_synthesis(query):
    result = build_summary_plan(query, UserSnapshot('u', 1, 1), reference=stamp('2025-04-01'))
    assert not result.applicable
    assert result.warnings


def test_relative_window_without_reference_is_not_an_unbounded_summary():
    result = build_summary_plan('List project Cobalt decisions last month.', UserSnapshot('u', 1, 1))
    assert not result.applicable
    assert result.warnings


def test_relative_task_window_survives_material_removal():
    query = ('List project Cobalt decisions last month.\nARTIFACTS\n2019-01-01 example.')
    result = build_summary_plan(query, UserSnapshot('u', 1, 1), reference=stamp('2025-04-05'))
    assert result.applicable
    assert result.window.contains(stamp('2025-03-02'))
    assert not result.window.contains(stamp('2025-02-28'))


def test_summary_generic_like_and_work_words_do_not_shrink_to_profile(snapshot):
    raw = 'Project Cobalt work was approved. The report looks like the agreed outline.'
    snapshot.sources['a'] = Source('a', raw, 'user', 's', 0, stamp('2025-01-01'))
    result = build_summary_plan('Summarize what our work looked like across all conversations.', snapshot)
    assert result.applicable
    assert not result.predicates
    assert result.global_scope
