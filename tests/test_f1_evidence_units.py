"""人工定义的 F1 结构与任务需求回归；金标不由待测解析器生成。"""
from app.governance.summary_planning import build_summary_plan
from app.governance.summary_retrieval import collect_summary
from tests.test_governance import add, make_service
from app.governance.summary_assembly import assemble_summary
from app.governance.summary_planning import task_text


def test_code_table_and_wrapped_paragraph_are_complete_evidence_units(tmp_path):
    paragraph = 'Project Aster validation uses two checks.\nBoth checks must pass before publication.'
    code = '```python\ndef accepted(row):\n    if row["verified"]:\n        return row["approved"]\n    return False\n```'
    table = '| Check | Limit (ms) |\n| --- | --- |\n| normal | 20 |\n| emergency | 40 |\nNote: emergency limits apply only after approval.'
    raw = paragraph + '\n\nValidation implementation:\n' + code + '\n\n' + table
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'structure', raw, 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    plan = build_summary_plan('Summarize project Aster validation.', snapshot)
    collection = collect_summary(snapshot, plan)
    quotes = [span.quote for item in collection.evidence for span in item.spans]
    for required in (paragraph, code, table):
        assert any(required in quote for quote in quotes)
    assert not any(q.strip() in ('```', 'return False', '| --- | --- |') for q in quotes)


def test_compact_packet_preserves_all_distinct_paragraphs_and_one_source_directory(tmp_path):
    paragraphs = [f'Project Aster validation step {i} checks sensor-{i} before publication.' for i in range(18)]
    raw = '\n\n'.join(paragraphs)
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'packet', raw, 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    plan = build_summary_plan('Summarize project Aster validation.', snapshot)
    result = assemble_summary(collect_summary(snapshot, plan), plan, [], snapshot, limit=1, token_budget=1400)
    assert len(result.hits) == 1
    assert all(p in result.hits[0].content for p in paragraphs)
    assert result.hits[0].content.count(service._memory_id('packet', 0)) == 1
    assert result.packet_tokens <= 1400


def test_explicit_task_section_does_not_treat_material_dates_or_engagement_likes_as_preferences(tmp_path):
    raw = ('Report format: begin with results, then limitations.\n'
           'Retain uncertainty; do not infer causation from engagement counts.')
    irrelevant = 'I like jasmine tea.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'format', raw, 1704067200000)
    add(service, 'tea', irrelevant, 1704067200000)
    query = ('CONVERSATION TRANSCRIPT\n2020-01-01: sample\n2021-02-02: sample\n2022-03-03: sample\n'
             'PRIOR OUTPUTS\nSummarize all personal preferences.\n'
             'FINAL TASK\nPrepare the engagement report using our previously agreed format. '
             'Include likes, views and shares from the supplied material.')
    snapshot = service.store.fetch_by_user('u', governance=True)
    plan = build_summary_plan(query, snapshot, reference=1743465600000)
    assert plan.applicable
    assert plan.reference == 1743465600000 and plan.window is None
    assert 'preference' not in plan.predicates
    assert '2020-01-01' not in task_text(query)
    text = '\n'.join(h.content for h in service.search(query, 'u', 1))
    assert raw in text and irrelevant not in text
    personal = build_summary_plan('Summarize my likes.', snapshot)
    assert personal.applicable and 'preference' in personal.predicates


def test_requested_comparison_reserves_each_named_evidence_family_under_budget(tmp_path):
    # 重复的常规结果不能挤掉任务明确要求的另一类测量及其限定。
    tensile = 'Project Aster tensile measurements: strength was 42 MPa on dry specimens only.'
    fatigue = 'Project Aster fatigue measurements: life was 9000 cycles; wet specimens were not tested.'
    routine = ['Project Aster tensile measurements: routine sample ' + str(i) + ' passed inspection.' for i in range(25)]
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'measurements', '\n\n'.join([tensile, *routine, fatigue]), 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    query = 'Combining the tensile and fatigue measurements, summarize project Aster results and limitations.'
    plan = build_summary_plan(query, snapshot)
    assert {r.terms for r in plan.requirements} >= {('tensile',), ('fatigue',)}
    assert all(r.trigger in plan.task for r in plan.requirements)
    outcome = assemble_summary(collect_summary(snapshot, plan), plan, [], snapshot, limit=1, token_budget=850)
    text = outcome.hits[0].content
    assert tensile in text and fatigue in text
    assert outcome.packet_tokens <= 850


def test_reason_heading_is_only_a_candidate_not_verified_causal_support(tmp_path):
    decision = 'Project Aster: we decided to postpone publication.'
    vague = 'Project Aster reasons:\nNo explanation was recorded.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'decision', decision, 1704067200000)
    add(service, 'reason', vague, 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    plan = build_summary_plan('Summarize project Aster decisions and reasons.', snapshot)
    result = assemble_summary(collect_summary(snapshot, plan), plan, [], snapshot, limit=1, token_budget=1000)
    reason = next(c for c in result.coverage if c.key == 'reasons')
    assert reason.support_status == 'candidate'
    assert reason.supported == 0
    assert '待核验' in result.hits[0].content


def test_stance_across_exchanges_keeps_named_topic_without_job_title_slot(tmp_path):
    early = 'I initially opposed Arbor because the data export was missing.'
    middle = 'For Arbor I requested an export audit before changing my position.'
    late = 'I now support Arbor after the export audit passed, only for internal use.'
    irrelevant = 'I am a staff engineer.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    for i, text in enumerate((early, middle, late, irrelevant)):
        add(service, str(i), text, 1704067200000 + i * 86400000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    query = 'Across these exchanges, how did my position on Arbor shift?'
    plan = build_summary_plan(query, snapshot)
    assert plan.applicable and plan.order == 'chronological'
    assert not plan.predicates and 'Arbor' in plan.scopes
    text = '\n'.join(h.content for h in service.search(query, 'u', 1))
    assert all(x in text for x in (early, middle, late))
    assert irrelevant not in text


def test_new_implicit_trajectory_with_ambiguous_scope_defers_to_existing_path(tmp_path):
    raw = 'We continued talking with the other team.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'dialogue', raw, 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    query = "Taken together, Robin's remarks suggest what shift in his stance toward the other team over the course of the exchange?"
    assert not build_summary_plan(query, snapshot).applicable


def test_oversize_code_block_is_omitted_whole_while_small_supported_rule_survives(tmp_path):
    from app.governance.extraction import token_count
    rule = 'Project Aster rule: deployment requires review; only after approval may it proceed.'
    block = '```python\n' + '\n'.join(f'check_{i} = result[{i}]' for i in range(200)) + '\n```'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'large-code', rule + '\n\nImplementation:\n' + block, 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    plan = build_summary_plan('Summarize project Aster rules and implementation.', snapshot)
    result = assemble_summary(collect_summary(snapshot, plan), plan, [], snapshot, limit=1, token_budget=700)
    assert rule in result.hits[0].content and 'check_' not in result.hits[0].content
    assert '预算' in result.hits[0].content and result.packet_tokens <= 700
    assert token_count(block) > 700
