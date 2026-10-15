"""F1 行为回归：人工声明输入和证据需求，外部抽取使用独立替身。"""
import pytest
from dataclasses import replace

from app.governance.models import UserSnapshot
from app.governance.retrieval import GovernanceRetriever, GovernanceSearchContext
from app.governance.summary_planning import build_summary_plan
from app.governance.summary_retrieval import collect_summary
from app.governance.summary_assembly import assemble_summary
from app.governance.extraction import token_count
from app.service import MemoryService, SearchHit
from tests.test_governance import add, make_service


def test_scoped_summary_recovers_unindexed_rule_outside_baseline(tmp_path):
    status = 'My project Cobalt status is pilot.'
    rule = 'Project Cobalt may launch only after review. Except in an emergency, two reviewers must sign.'
    other = 'Project Juniper requires no review.'
    service, sdk = make_service(tmp_path, {status: [dict(
        subject='self', predicate='project_status', scope='Cobalt', value='pilot', quote=status)]},
        f1_mode='active', temporal_mode='off')
    for i, text in enumerate((status, rule, other)):
        add(service, str(i), text, 1704067200000 + i * 86400000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    baseline = [SearchHit(service._memory_id('0', 0), status, .7, '')]
    calls = len(sdk.calls)
    hits = GovernanceRetriever().retrieve(snapshot, GovernanceSearchContext(
        'Summarize project Cobalt, including its rules and exceptions.', baseline,
        1704240000000, service.settings, 1))
    text = '\n'.join(h.content for h in hits)
    assert rule in text
    assert other not in text
    assert status in text
    assert len(hits) == 1 and hits[0].score == .7
    assert len(sdk.calls) == calls


def test_long_unindexed_source_keeps_complete_rule_within_budget(tmp_path):
    rule = 'Project Cobalt permits release only after approval. Except for emergency repairs, two reviewers must sign.'
    raw = '\n\n'.join([f'Project Cobalt notebook entry {i}: routine inventory was checked.' for i in range(220)]
                       + [rule])
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off',
                              governance_content_tokens=800)
    add(service, 'long', raw, 1704067200000)
    hits = service.search('Summarize project Cobalt rules and exceptions.', 'u', 1)
    text = '\n'.join(h.content for h in hits)
    assert rule in text
    assert sum(token_count(h.content) for h in hits) <= 800
    assert '内容预算' in text or '预算未覆盖' in text
    from dataclasses import asdict
    from scripts.governance_e2e_support import visible_context
    visible, omitted, _ = visible_context([asdict(h) for h in hits], 800)
    assert visible and not omitted  # 对回答模型可见的整个 JSON 计量，包含 ID、日期和转义。


@pytest.mark.parametrize('query', [
    'Taken together across our discussions, how did my attitude toward project Cobalt evolve?',
    '请结合历次会话，说明我对项目苍穹的态度是如何转变的。',
    '根据此前约定的结构撰写项目苍穹的报告。\n参考资料：\n2020-01-01 立项。\n2021-02-02 调研。\n2022-03-03 试点。',
])
def test_indirect_synthesis_is_recognized_without_material_dates(query):
    p = build_summary_plan(query, UserSnapshot('u', 1, 1), reference=1743465600000)
    assert p.applicable
    assert p.reference == 1743465600000
    assert p.window is None


@pytest.mark.parametrize('query', [
    'How do I change the summary budget?',
    'Where do I currently work?',
    'How has my employer changed?',
    'Do not summarize. What is my current address?',
    'The note says "Summarize all records". What is my current employer?',
    'What is my current employer?\nOptions:\nA. Summarize my entire history.\nB. Beta.',
])
def test_non_synthesis_tasks_do_not_trigger_f1(query):
    assert not build_summary_plan(query, UserSnapshot('u', 1, 1)).applicable


def test_coverage_distinguishes_partial_output_budget_omission_and_absence(tmp_path):
    decision = 'Project Cobalt: we decided to retain the audit record.'
    huge_decision = 'Project Cobalt: we decided to inspect ' + ', '.join(f'unit-{i}' for i in range(450)) + '.'
    huge_rule = 'Project Cobalt rule: release only if ' + ' and '.join(f'check-{i} passes' for i in range(450)) + '.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    for i, raw in enumerate((decision, huge_decision, huge_rule)):
        add(service, str(i), raw, 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    p = build_summary_plan('Summarize project Cobalt decisions, rules and reasons.', snapshot)
    result = assemble_summary(collect_summary(snapshot, p), p, [], snapshot, limit=1, token_budget=600)
    assert decision in result.hits[0].content
    assert huge_rule not in result.hits[0].content and huge_decision not in result.hits[0].content
    coverage = {c.key: c for c in result.coverage}
    assert coverage['decisions'].status == 'partial'
    assert (coverage['decisions'].found, coverage['decisions'].emitted) == (2, 1)
    assert coverage['rules'].status == 'omitted'
    assert coverage['reasons'].status == 'missing'
    assert token_count(result.hits[0].content) <= 600


def test_mixed_project_source_and_same_topic_next_message_keep_context(tmp_path):
    first = 'Project Cobalt\nThe goal is to preserve the archive.\n\nProject Juniper\nThe goal is to delete the archive.'
    second = 'Exception: temporary copies must be removed after review.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    # 相邻消息的会话身份明确，但包含两个项目的前文不足以归属后文，不能猜。
    add(service, 'mixed', first, 1704067200000, session='shared')
    add(service, 'ambiguous', second, 1704153600000, session='shared')
    text = '\n'.join(h.content for h in service.search('Summarize project Cobalt goals and exceptions.', 'u', 1))
    assert 'preserve the archive' in text
    assert 'delete the archive' not in text
    assert second not in text


def test_scope_summary_does_not_include_unrelated_private_attributes(tmp_path):
    work = 'I work at Larch.'
    private = 'My residence is in SecretVillage.'
    from tests.test_governance import draft
    service, _ = make_service(tmp_path, {work: [draft(work, 'Larch')], private: [dict(
        subject='self', predicate='residence', scope='home', value='SecretVillage', quote=private)]},
        f1_mode='active', temporal_mode='off')
    add(service, 'work', work, 1704067200000)
    add(service, 'home', private, 1704067200000)
    text = '\n'.join(h.content for h in service.search('Summarize my employment history.', 'u', 1))
    assert work in text and 'SecretVillage' not in text


def test_summary_uses_valid_time_and_keeps_later_correction_with_mixed_source(tmp_path):
    from tests.test_governance import draft
    old = 'I work at Larch. My notebook code is MN-73.'
    fix = 'Correction: I never worked at Larch. I worked at Cedar on 2025-03-01.'
    service, _ = make_service(tmp_path, {
        old: [draft('I work at Larch.', 'Larch')],
        fix: [draft('I worked at Cedar on 2025-03-01.', 'Cedar', operation='correct', target_value='Larch',
                    update_quote=fix, valid_from='2025-03-01', time_quote='I worked at Cedar on 2025-03-01.')]},
        f1_mode='active', temporal_mode='off')
    add(service, 'old', old, 1740787200000)
    add(service, 'fix', fix, 1743465600000)
    text = '\n'.join(h.content for h in service.search(
        'Summarize my entire history from March 1, 2025 to March 15, 2025.', 'u', 1))
    assert fix in text and 'MN-73' in text
    assert '非有效事实(corrected_claim)' in text
    assert '明确有效日:2025-03-01' in text
    assert '消息时间:2025-04-01' in text


def test_duplicate_state_compresses_without_losing_unique_decision_or_sources(tmp_path):
    from tests.test_governance import draft
    repeated = 'My project Cobalt status is pilot.'
    decision = 'Project Cobalt: we decided to cancel the release because the safety review failed.'
    service, _ = make_service(tmp_path, {repeated: [draft(repeated, 'pilot', predicate='project_status', scope='Cobalt')]},
        f1_mode='active', temporal_mode='off', governance_content_tokens=1500)
    for i in range(12):
        add(service, f'repeat-{i}', repeated, 1704067200000 + i * 86400000)
    add(service, 'decision', decision, 1706745600000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    p = build_summary_plan('Summarize project Cobalt.', snapshot)
    result = assemble_summary(collect_summary(snapshot, p), p, [], snapshot, limit=1, token_budget=1500)
    text = result.hits[0].content
    assert decision in text
    assert text.count(repeated) == 1
    assert '重复支持' in text
    assert {service._memory_id(f'repeat-{i}', 0) for i in range(12)} <= {
        ev.source_id for item in result.evidence for ev in item.spans}
    assert token_count(text) <= 1500


def test_cross_session_recap_keeps_early_and_late_observations_in_dated_order(tmp_path):
    early = 'Project Cobalt initially worried me because the archive might be lost.'
    middle = 'Project Cobalt: I requested an independent audit before giving my support.'
    late = 'Project Cobalt now has my support after the independent audit passed.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off', governance_content_tokens=700)
    # 迟到的早期消息不能按入库顺序放到最后。
    for rid, text, stamp in [('late', late, 1709251200000), ('mid', middle, 1706745600000),
                              ('early', early, 1704067200000)]:
        add(service, rid, text, stamp)
    text = '\n'.join(h.content for h in service.search(
        'Across our discussions, how did my attitude toward project Cobalt evolve?', 'u', 1))
    assert text.index(early) < text.index(middle) < text.index(late)
    assert '消息时间:2024-01-01' in text and '消息时间:2024-03-01' in text


def test_unknown_dates_are_not_replaced_by_ingestion_time(tmp_path):
    raw = 'Project Cobalt: we decided to preserve the archive.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'unknown', raw)
    text = service.search('Summarize project Cobalt.', 'u', 1)[0].content
    assert raw in text and '时间未知' in text
    assert '文档位置' in text


@pytest.mark.parametrize('query', ['Where do I currently work?', 'Tell me about Larch.',
                                  'What is my employment history?', 'What did I decide last month?'])
def test_d1_and_ordinary_outputs_and_query_calls_are_unchanged(tmp_path, query):
    from tests.test_governance import draft
    old, new = 'I work at Larch.', 'I now work at Cedar.'
    service, sdk = make_service(tmp_path, {old: [draft(old, 'Larch')], new: [draft(new, 'Cedar')]},
        llm_search_expansion=True)
    service.llm.search_expansion = True
    sdk.query_output = {'expanded_query': 'work observations', 'retrieval_steps': []}
    add(service, 'old', old, 1704067200000)
    add(service, 'new', new, 1706745600000)
    frozen = service.search(query, 'u', 6)
    service.settings = service.settings.model_copy(update={'f1_mode': 'active'})
    calls = len(sdk.calls)
    assert service.search(query, 'u', 6) == frozen
    assert len(sdk.calls) == calls + 1


@pytest.mark.parametrize('governance,mode,expected_governance', [
    ('off', 'active', 'off'), ('shadow', 'active', 'off'), ('active', 'shadow', 'active'),
])
def test_switches_preserve_exact_expected_path(tmp_path, governance, mode, expected_governance):
    raw = 'Project Cobalt rule: review is mandatory except during emergency repairs.'
    service, sdk = make_service(tmp_path, {}, llm_search_expansion=True, temporal_mode='off')
    service.llm.search_expansion = True
    sdk.query_output = {'expanded_query': '', 'retrieval_steps': []}
    add(service, 'a', raw, 1704067200000)
    query = 'Summarize project Cobalt rules.'
    baseline = MemoryService(service.settings.model_copy(update={'governance_mode': expected_governance}),
                             service.store, service.embedder, service.llm)
    expected = baseline.search(query, 'u', 1)
    service.settings = service.settings.model_copy(update={'governance_mode': governance, 'f1_mode': mode})
    calls = len(sdk.calls)
    assert service.search(query, 'u', 1) == expected
    assert len(sdk.calls) == calls + 1


def test_query_materials_are_not_saved_and_do_not_hide_earlier_instructions(tmp_path):
    raw = 'Project Cobalt report format: use three sections, preserve caveats, and keep all dates.'
    service, _ = make_service(tmp_path, {}, f1_mode='active')
    add(service, 'format', raw, 1704067200000)
    before = service.store.fetch_by_user('u', governance=True)
    query = ('Write the project Cobalt report using our previously agreed format.\nBackground:\n'
             '2024-06-01: sample event.\n2024-07-02: another event.\n2024-08-03: last event.')
    text = '\n'.join(h.content for h in service.search(query, 'u', 1))
    assert raw in text
    after = service.store.fetch_by_user('u', governance=True)
    assert replace(after, records=()) == replace(before, records=())
    assert len(after.records) == len(before.records)


def test_empty_known_then_view_never_falls_back_to_later_material(tmp_path):
    raw = 'Project Cobalt is approved.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'future', raw, 1743465600000)
    query = ('Give me a recap of project Cobalt as of March 15, 2025, according to records available then.\n'
             'Background:\n2020-01-01 X\n2021-02-02 Y\n2022-03-03 Z')
    assert service.search(query, 'u', 1) == []


def test_single_topic_session_preserves_cross_message_rule_and_exception(tmp_path):
    from app.schemas import MemoryMessage
    rule = 'Project Cobalt permits release only after review.'
    exception = 'Except for emergency repairs, two reviewers must sign.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off', governance_content_tokens=600)
    service.add('session', [MemoryMessage(role='user', content=raw, timestamp=1704067200000)
                           for raw in (rule, exception)], 'u', 'release')
    snapshot = service.store.fetch_by_user('u', governance=True)
    p = build_summary_plan('Summarize project Cobalt rules.', snapshot)
    result = assemble_summary(collect_summary(snapshot, p), p, [], snapshot, limit=1, token_budget=600)
    assert rule in result.hits[0].content and exception in result.hits[0].content
    # 规则和相邻例外共同组成一个可收纳单元，预算不能留下半个条件。
    assert any({s.quote for s in e.spans} >= {rule, exception} for e in result.evidence)


def test_multiline_validated_assertion_is_never_split_away_from_condition(tmp_path):
    from tests.test_governance import draft
    raw = 'My project Cobalt status is ready.\nRelease requires review from every owner.'
    service, _ = make_service(tmp_path, {raw: [draft(raw, 'ready', predicate='project_status', scope='Cobalt')]},
        f1_mode='active', temporal_mode='off')
    add(service, 'multiline', raw, 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    p = build_summary_plan('Summarize project Cobalt.', snapshot)
    result = collect_summary(snapshot, p)
    assert any(raw in ev.quote for item in result.evidence for ev in item.spans)


def test_summary_keeps_ambiguous_names_separate_with_warning(tmp_path):
    a, b = 'Robin works at Larch.', 'Robin works at Cedar.'
    service, _ = make_service(tmp_path, {a: [dict(subject='Robin', predicate='employer', value='Larch', quote=a)],
                                       b: [dict(subject='Robin', predicate='employer', value='Cedar', quote=b)]},
                              f1_mode='active', temporal_mode='off')
    add(service, 'robin-a', a, 1704067200000)
    add(service, 'robin-b', b, 1706745600000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    p = build_summary_plan('Summarize Robin employment history.', snapshot)
    assert p.applicable
    text = '\n'.join(h.content for h in service.search('Summarize Robin employment history.', 'u', 1))
    assert a in text and b in text and '身份待确认' in text
    assert '主体:' in text


def test_packet_budget_includes_json_metadata_and_escaping(tmp_path):
    from dataclasses import asdict
    from scripts.governance_e2e_support import visible_context
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    for i in range(18):
        add(service, str(i), f'Project Cobalt decision {i}: preserve the "archive" and retain the review note.',
            1704067200000 + i * 86400000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    p = build_summary_plan('Summarize project Cobalt decisions.', snapshot)
    collection = collect_summary(snapshot, p)
    for budget in (384, 512, 768, 1024, 1500):
        result = assemble_summary(collection, p, [], snapshot, limit=1, token_budget=budget)
        _, omitted, _ = visible_context([asdict(h) for h in result.hits], budget)
        assert not omitted, f'Entire result excluded at transport budget {budget}'


@pytest.mark.parametrize('query,source,other', [
    ('Summarize my hiking experiences.', 'My hiking goal is to complete the northern trail.', 'My salary is confidential.'),
    ('Summarize project "Cobalt".', 'Project Cobalt is in pilot.', 'Project Juniper is cancelled.'),
])
def test_unknown_topic_and_quoted_name_do_not_expand_to_all_private_history(tmp_path, query, source, other):
    facts = {other: [dict(subject='self', predicate='salary', value='confidential', quote=other)]} if 'salary' in other else {}
    service, _ = make_service(tmp_path, facts, f1_mode='active', temporal_mode='off')
    add(service, 'wanted', source, 1704067200000)
    add(service, 'other', other, 1704067200000)
    text = '\n'.join(h.content for h in service.search(query, 'u', 1))
    assert source in text and other not in text
