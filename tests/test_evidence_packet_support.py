"""Known evidence dependencies must survive the caller-visible packet budget.

These are synthetic, model-free regression cases. Coverage of these dependencies
does not certify arbitrary semantic entailment or a final answer.
"""
from dataclasses import asdict, replace
import json

import pytest

from app.governance.extraction import digest, token_count
from app.governance.models import Evidence, Fact, Source, SourceStatus, UserSnapshot
from app.governance.summary_assembly import assemble_summary
from app.governance.summary_planning import build_summary_plan
from app.governance.summary_retrieval import collect_summary


def raw_snapshot(raw, *, source_id='source-a', role='user'):
    source = Source(source_id, raw, role, 'meeting-a', 0, 1704067200000, '2024-01-01')
    return UserSnapshot('test-user', 1, 1, sources={source_id: source},
                        statuses=(SourceStatus(source_id, 'no_fact'),))


def packet(raw, query='Summarize project Cobalt decisions.', budget=320):
    snapshot = raw_snapshot(raw)
    plan = build_summary_plan(query, snapshot)
    assert plan.applicable
    collection = collect_summary(snapshot, plan)
    result = assemble_summary(collection, plan, [], snapshot, limit=1, token_budget=budget)
    text = '\n'.join(hit.content for hit in result.hits)
    serialized = json.dumps([asdict(hit) for hit in result.hits], ensure_ascii=False,
                            sort_keys=True, separators=(',', ':'))
    assert token_count(serialized) <= budget
    return text, result, collection


@pytest.mark.parametrize('budget', [128, 200, 320, 800])
def test_pronoun_action_never_outlives_its_local_actor(budget):
    raw = 'Project Cobalt review\nAlice presented the proposal.\nShe approved rollout.'
    text, result, collection = packet(raw, budget=budget)
    if 'She approved rollout.' in text:
        assert 'Alice presented the proposal.' in text
    if budget >= 320:
        assert 'She approved rollout.' in text
    assert any('Alice presented the proposal.' in ' '.join(e.quote for e in item.spans)
               and 'She approved rollout.' in ' '.join(e.quote for e in item.spans)
               for item in collection.evidence)


def test_local_actor_group_does_not_absorb_unrelated_preamble():
    unrelated = 'The office printer was replaced last week. ' * 200
    raw = unrelated + '\n\nProject Cobalt review\nAlice presented the proposal.\nShe approved rollout.'
    text, _, _ = packet(raw, budget=320)
    assert 'Alice presented the proposal.' in text and 'She approved rollout.' in text
    assert unrelated not in text


@pytest.mark.parametrize('budget', [180, 280, 600])
def test_heading_and_body_remain_bound(budget):
    raw = '## Project Cobalt\n### Release decision\nWe approved launch.'
    text, _, _ = packet(raw, budget=budget)
    if 'We approved launch.' in text:
        assert '### Release decision' in text
        assert '## Project Cobalt' in text
    if budget >= 280:
        assert 'We approved launch.' in text


@pytest.mark.parametrize('budget', [180, 260, 500])
def test_fenced_code_is_an_indivisible_unit(budget):
    code = '```python\nif not reviewed:\n    raise RuntimeError("release blocked")\nlaunch()\n```'
    raw = 'Project Cobalt rule implementation:\n' + code
    text, _, collection = packet(raw, 'Summarize project Cobalt rules.', budget)
    if any(part in text for part in ('```python', 'raise RuntimeError', 'launch()')):
        assert code in text
    assert any(code in ev.quote for item in collection.evidence for ev in item.spans)
    if budget >= 500:
        assert code in text


@pytest.mark.parametrize('budget', [180, 280, 600])
def test_table_rows_keep_their_header_and_other_rows(budget):
    table = '| Stage | Decision |\n| --- | --- |\n| Trial | approved only after audit |\n| Release | cancelled |'
    text, _, collection = packet('Project Cobalt rollout decisions\n' + table, budget=budget)
    if '| Trial |' in text or '| Release |' in text:
        assert table in text
    assert any(table in ev.quote for item in collection.evidence for ev in item.spans)
    if budget >= 600:
        assert table in text


@pytest.mark.parametrize('budget', [180, 280, 600])
def test_rule_and_explicit_exception_compete_together(budget):
    rule = 'Project Cobalt rule: every release requires review.'
    exception = 'Exception: emergency repairs need only one reviewer.'
    text, _, collection = packet(rule + '\n' + exception, 'Summarize project Cobalt rules.', budget)
    assert (rule in text) == (exception in text)
    assert any(rule in ' '.join(ev.quote for ev in item.spans)
               and exception in ' '.join(ev.quote for ev in item.spans) for item in collection.evidence)


def test_structure_does_not_cross_project_boundary():
    wanted = '## Project Cobalt\nWe approved retention.'
    other = '## Project Juniper\nWe approved deletion.'
    text, _, _ = packet(wanted + '\n\n' + other, budget=600)
    assert 'approved retention' in text
    assert 'Juniper' not in text and 'approved deletion' not in text


def test_nested_heading_does_not_merge_distinct_project_scopes():
    raw = '# Project Cobalt\nWe approved retention.\n## Project Juniper\nWe approved deletion.'
    text, _, _ = packet(raw, budget=600)
    assert 'approved retention' in text
    assert 'Juniper' not in text and 'approved deletion' not in text


def test_explicit_other_project_in_body_overrides_inherited_heading():
    raw = '# Project Cobalt\nWe approved retention.\nProject Juniper approved deletion.'
    text, _, _ = packet(raw, budget=600)
    assert 'approved retention' in text
    assert 'Juniper' not in text and 'approved deletion' not in text


def test_unclosed_code_fence_is_never_emitted_as_short_fragments():
    code = '```python\n' + '\n'.join(f'review_stage_{i} = "pending"' for i in range(100))
    text, result, _ = packet('Project Cobalt rule implementation:\n' + code,
                            'Summarize project Cobalt rules.', 220)
    assert 'review_stage_' not in text and '```' not in text
    assert not result.hits


def test_shared_source_metadata_is_rendered_once_with_locatable_spans():
    raw = 'Project Cobalt: we approved the archive.\nProject Cobalt: we cancelled the launch.'
    text, result, _ = packet(raw, budget=1000)
    assert 'approved the archive' in text and 'cancelled the launch' in text
    assert text.count('角色:user') == 1
    assert text.count('文档位置:meeting-a/0') == 1
    for item in result.evidence:
        for ev in item.spans:
            assert f'[来源:{ev.source_id}:{ev.start}-{ev.end}]' in text


def test_known_dependency_missing_from_candidate_spans_is_rejected():
    raw = 'Project Cobalt review\nAlice presented the proposal.\nShe approved rollout.'
    snapshot = raw_snapshot(raw)
    plan = build_summary_plan('Summarize project Cobalt decisions.', snapshot)
    collection = collect_summary(snapshot, plan)
    actor_item = next(item for item in collection.evidence
                      if any('She approved' in ev.quote for ev in item.spans))
    assert getattr(actor_item, 'required_spans', ())
    source = snapshot.sources['source-a']
    start = raw.index('She approved')
    short = Evidence(source.id, start, len(raw), raw[start:], digest(raw[start:]))
    broken = replace(actor_item, spans=(short,))
    result = assemble_summary(replace(collection, evidence=(broken,)), plan, [], snapshot,
                              limit=1, token_budget=1000)
    assert not result.hits


def test_serialized_support_audit_detects_omitted_dependency_text():
    from app.governance.summary_assembly import audit_serialized_support
    raw = 'Project Cobalt review\nAlice presented the proposal.\nShe approved rollout.'
    text, result, _ = packet(raw, budget=600)
    serialized = json.dumps([asdict(hit) for hit in result.hits], ensure_ascii=False)
    assert audit_serialized_support(serialized, result.evidence, raw_snapshot(raw)) == ()
    tampered = serialized.replace('Alice presented the proposal.', 'An actor is omitted.')
    assert audit_serialized_support(tampered, result.evidence, raw_snapshot(raw))


def test_serialized_support_audit_rejects_rebound_source_reference():
    from app.governance.summary_assembly import audit_serialized_support
    raw = 'Project Cobalt: we approved the archive.'
    _, result, _ = packet(raw, budget=600)
    serialized = json.dumps([asdict(hit) for hit in result.hits], ensure_ascii=False)
    tampered = serialized.replace('[来源:source-a:', '[来源:other-user-source:')
    assert audit_serialized_support(tampered, result.evidence, raw_snapshot(raw))


def test_actual_renderer_omission_fails_closed(monkeypatch):
    import app.governance.summary_assembly as assembly
    original = assembly.render_evidence_item
    def incomplete_renderer(*args, **kwargs):
        return original(*args, **kwargs).replace('Alice presented the proposal.', '')
    monkeypatch.setattr(assembly, 'render_evidence_item', incomplete_renderer)
    raw = 'Project Cobalt review\nAlice presented the proposal.\nShe approved rollout.'
    _, result, _ = packet(raw, budget=600)
    assert not result.hits
    assert result.support_errors


def test_serialized_support_audit_rejects_wrong_source_role():
    from app.governance.summary_assembly import audit_serialized_support
    raw = 'Project Cobalt: we approved the archive.'
    _, result, _ = packet(raw, budget=600)
    serialized = json.dumps([asdict(hit) for hit in result.hits], ensure_ascii=False)
    tampered = serialized.replace('角色:user', '角色:assistant')
    assert audit_serialized_support(tampered, result.evidence, raw_snapshot(raw))


@pytest.mark.parametrize('budget', [128, 240, 500, 1500])
def test_corrected_source_never_loses_its_update_dependency(tmp_path, budget):
    from tests.test_governance import add, draft, make_service
    old = 'I work at Larch. My notebook code is MN-73.'
    correction = 'Correction: I never worked at Larch. I worked at Cedar.'
    service, _ = make_service(tmp_path, {
        old: [draft('I work at Larch.', 'Larch')],
        correction: [draft('I worked at Cedar.', 'Cedar', operation='correct',
                           target_value='Larch', update_quote=correction)]},
        f1_mode='active', temporal_mode='off')
    add(service, 'old', old, 1704067200000)
    add(service, 'correction', correction, 1706745600000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    plan = build_summary_plan('Summarize my entire history.', snapshot)
    collection = collect_summary(snapshot, plan)
    result = assemble_summary(collection, plan, [], snapshot, limit=1, token_budget=budget)
    text = '\n'.join(hit.content for hit in result.hits)
    if old in text:
        assert correction in text and '非有效事实(corrected_claim)' in text
    if budget >= 1500:
        assert old in text and correction in text


def dated_events_snapshot(*, include_time_evidence=True, modality='asserted'):
    rows = [
        ('launch', 'Project Cobalt: I approved launch on January 4, 2024.', 'launch',
         'January 4, 2024', 1704326400000, 1704412800000),
        ('design', 'Project Cobalt: I approved design review on January 2, 2024.', 'design review',
         'January 2, 2024', 1704153600000, 1704844800000),
    ]
    sources, facts = {}, []
    for sid, raw, value, date_quote, event_time, message_time in rows:
        source = Source(sid, raw, 'user', sid, 0, message_time, '2024-01-10')
        sources[sid] = source
        assertion = Evidence(sid, 0, len(raw), raw, digest(raw))
        a = raw.index(date_quote)
        date_evidence = Evidence(sid, a, a + len(date_quote), date_quote, digest(date_quote), 'time')
        facts.append(Fact('fact-' + sid, 'self', 'self', 'event', 'Cobalt', value,
            'event', 'positive', modality, 'assert',
            (assertion, date_evidence) if include_time_evidence else (assertion,),
            source_time=message_time, valid_from=event_time, time_precision='day'))
    return UserSnapshot('u', 1, 1, sources=sources, facts=tuple(facts),
                        statuses=tuple(SourceStatus(sid, 'ready') for sid in sources))


def test_event_order_prefers_supported_event_dates_to_message_dates():
    snapshot = dated_events_snapshot()
    plan = build_summary_plan('Summarize project Cobalt in chronological order.', snapshot)
    collection = collect_summary(snapshot, plan)
    result = assemble_summary(collection, plan, [], snapshot, limit=1, token_budget=1000)
    text = result.hits[0].content
    assert text.index(snapshot.sources['design'].content) < text.index(snapshot.sources['launch'].content)
    assert '明确有效日:2024-01-02' in text and '消息时间:2024-01-10' in text


def test_event_dates_survive_extraction_and_drive_synthesis_order(tmp_path):
    from tests.test_governance import add, draft, make_service
    launch = 'Project Cobalt: I approved launch on January 4, 2024.'
    design = 'Project Cobalt: I approved design review on January 2, 2024.'
    service, _ = make_service(tmp_path, {
        launch: [draft(launch, 'launch', predicate='event', scope='Cobalt', kind='event',
                       valid_from='2024-01-04', time_quote='January 4, 2024')],
        design: [draft(design, 'design review', predicate='event', scope='Cobalt', kind='event',
                       valid_from='2024-01-02', time_quote='January 2, 2024')]},
        f1_mode='active', temporal_mode='off')
    add(service, 'launch', launch, 1704412800000)
    add(service, 'design', design, 1704844800000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    assert len(snapshot.facts) == 2 and not snapshot.incomplete
    text = service.search('Across our discussions, summarize project Cobalt in chronological order.', 'u', 1)[0].content
    assert text.index(design) < text.index(launch)
    assert '明确有效日:2024-01-02' in text and '消息时间:2024-01-10' in text


def test_explicit_mention_order_uses_message_dates_even_with_event_facts():
    snapshot = dated_events_snapshot()
    plan = build_summary_plan('List the order in which I mentioned project Cobalt events across our conversations.', snapshot)
    collection = collect_summary(snapshot, plan)
    result = assemble_summary(collection, plan, [], snapshot, limit=1, token_budget=1000)
    text = result.hits[0].content
    assert text.index(snapshot.sources['launch'].content) < text.index(snapshot.sources['design'].content)


@pytest.mark.parametrize('kind', ['raw', 'missing_time_support', 'planned'])
def test_unknown_event_dates_do_not_become_message_or_commit_dates(kind):
    snapshot = dated_events_snapshot(include_time_evidence=kind != 'missing_time_support',
                                     modality='planned' if kind == 'planned' else 'asserted')
    if kind == 'raw':
        snapshot = replace(snapshot, facts=(), statuses=tuple(SourceStatus(sid, 'no_fact') for sid in snapshot.sources))
    plan = build_summary_plan('Summarize project Cobalt in chronological order.', snapshot)
    collection = collect_summary(snapshot, plan)
    assert collection.evidence
    assert all(not item.moments for item in collection.evidence)
    result = assemble_summary(collection, plan, [], snapshot, limit=1, token_budget=1500)
    text = result.hits[0].content
    assert ('事件日期未知' in text if kind != 'planned' else '非确认记录' in text)
    assert '不推断事件先后' in text


@pytest.mark.parametrize('query,included,excluded', [
    ('List project Cobalt events in chronological order before January 3, 2024.', 'design', 'launch'),
    ('List project Cobalt events in chronological order after January 3, 2024.', 'launch', 'design'),
])
def test_known_event_window_is_independent_of_display_eligibility(query, included, excluded):
    snapshot = dated_events_snapshot()
    plan = build_summary_plan(query, snapshot, reference=1704844800000)
    collection = collect_summary(snapshot, plan)
    result = assemble_summary(collection, plan, [], snapshot, limit=1, token_budget=1000)
    text = result.hits[0].content
    assert snapshot.sources[included].content in text
    assert snapshot.sources[excluded].content not in text


def test_out_of_window_future_event_is_not_reintroduced_as_unknown(tmp_path):
    from tests.test_governance import add, draft, make_service
    launch = 'Project Cobalt: I approved launch on January 4, 2024.'
    design = 'Project Cobalt: I approved design review on January 2, 2024.'
    service, _ = make_service(tmp_path, {
        launch: [draft(launch, 'launch', predicate='event', scope='Cobalt', kind='event',
                       valid_from='2024-01-04', time_quote='January 4, 2024')],
        design: [draft(design, 'design review', predicate='event', scope='Cobalt', kind='event',
                       valid_from='2024-01-02', time_quote='January 2, 2024')]},
        f1_mode='active', temporal_mode='off')
    add(service, 'launch', launch, 1704412800000)
    add(service, 'design', design, 1704844800000)
    text = service.search('List project Cobalt events in chronological order before January 3, 2024.', 'u', 1)[0].content
    assert design in text and launch not in text
    assert '事件日期未知' not in text
