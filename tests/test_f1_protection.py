"""Independent evidence contracts; no official score or parser-generated answer gold."""
from dataclasses import asdict, replace

import pytest

from app.governance.retrieval import GovernanceRetriever, GovernanceSearchContext
from app.governance.summary_planning import build_summary_plan
from app.service import SearchHit
from scripts.governance_e2e_support import visible_context
from tests.test_governance import add, draft, make_service


def notes():
    return [f'Project Alder notebook {i}: routine inventory was checked.' for i in range(36)]


def run_summary(service, query, source_ids=(), budget=None):
    snapshot = service.store.fetch_by_user('u', governance=True)
    baseline = [SearchHit(sid, snapshot.sources[sid].content, .71, snapshot.sources[sid].recorded_at)
                for sid in source_ids]
    settings = service.settings if budget is None else service.settings.model_copy(
        update={'governance_content_tokens': budget})
    ctx = GovernanceSearchContext(query, baseline, 1743465600000, settings, 1)
    retriever = GovernanceRetriever()
    return retriever.retrieve(snapshot, ctx), retriever.summarize(snapshot, ctx)


def test_direct_identifier_survives_competing_short_background(tmp_path):
    required = ('Project Alder archive identifier is AX-734; it belongs to the audit archive, '
                'not the deployment archive. Use it for restricted internal requests. '
                'The archived note retains the review history and the original access restrictions.')
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off',
                              governance_content_tokens=850)
    add(service, 'notes', '\n\n'.join([*notes(), required]), 1704067200000)
    hits, _ = run_summary(service, 'Summarize project Alder and specify its archive identifier.',
                          [service._memory_id('notes', 0)])
    assert required in hits[0].content
    assert len(hits) == 1 and hits[0].score == .71
    assert not visible_context([asdict(h) for h in hits], 850)[1]


def test_actor_reporter_and_negative_attribution_survive_together(tmp_path):
    required = ('Project Alder: Mira reported that Jon proposed the trial; Mira did not propose it. '
                'This was a suggestion, without authorization to proceed. Expansion requires review '
                'and must not proceed until the reviewers have signed the complete audit record.')
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off',
                              governance_content_tokens=850)
    add(service, 'attribution', '\n\n'.join([*notes(), required]), 1704067200000, role='assistant')
    hits, _ = run_summary(service, 'Summarize project Alder and identify who proposed the trial.',
                          [service._memory_id('attribution', 0)])
    assert required in hits[0].content
    assert '消息角色:assistant' in hits[0].content
    assert '归因以原文为准' in hits[0].content


def test_protection_does_not_restore_corrected_value_or_remove_its_dependency(tmp_path):
    old = 'My project Alder status is approved. The archive identifier is AX-734.'
    fix = 'Correction: My project Alder status is pending. The earlier approved entry was wrong.'
    service, _ = make_service(tmp_path, {
        old: [draft('My project Alder status is approved.', 'approved', predicate='project_status', scope='Alder')],
        fix: [draft('My project Alder status is pending.', 'pending', predicate='project_status', scope='Alder', operation='correct',
                    target_value='approved', update_quote=fix)]}, f1_mode='active', temporal_mode='off')
    add(service, 'old', old, 1704067200000)
    add(service, 'fix', fix, 1706745600000)
    hits, _ = run_summary(service, 'Summarize project Alder and specify its archive identifier and current project status.',
                          [service._memory_id('old', 0)])
    text = hits[0].content
    assert old in text and fix in text
    assert '非有效事实(corrected_claim)' in text


def test_guard_never_falls_back_to_wrong_scope_or_future_baseline(tmp_path):
    wanted = 'Project Alder archive identifier is AX-734.'
    other = 'Project Birch archive identifier is SECRET-12.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'wanted', wanted, 1704067200000)
    add(service, 'other', other, 1704067200000)
    hits, _ = run_summary(service, 'Summarize project Alder and specify its archive identifier.',
                          [service._memory_id('other', 0)])
    assert wanted in hits[0].content and 'SECRET-12' not in hits[0].content
    query = 'Summarize project Alder as of January 1, 2020, according to records available then, including its identifier.'
    hits, _ = run_summary(service, query, [service._memory_id('wanted', 0)])
    assert hits == []


def test_protection_missing_or_oversize_is_audited_not_silently_passed(tmp_path):
    raw = 'Project Alder archive identifier is AX-734; ' + 'retain every qualifier and condition; ' * 400
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'large', raw, 1704067200000)
    add(service, 'small', 'Project Alder: we decided to retain the archive.', 1704067200000)
    hits, (_, outcome) = run_summary(service, 'Summarize project Alder decisions and specify its identifier.', budget=600)
    assert raw not in '\n'.join(h.content for h in hits)
    assert outcome.protection is not None and not outcome.protection.passed
    assert outcome.protection.missing_requirements
    assert not visible_context([asdict(h) for h in hits], 600)[1]


@pytest.mark.parametrize('query', [
    'Where do I currently work?', 'Who proposed the trial?',
    'What is my archive identifier?', 'Do not summarize. What is my current employer?',
])
def test_non_f1_path_and_d1_output_are_untouched(tmp_path, query):
    raw = 'I work at Cedar.'
    service, _ = make_service(tmp_path, {raw: [draft(raw, 'Cedar')]}, temporal_mode='off')
    add(service, 'work', raw, 1704067200000)
    before = service.search(query, 'u', 1)
    service.settings = service.settings.model_copy(update={'f1_mode': 'active'})
    assert not build_summary_plan(query, service.store.fetch_by_user('u', governance=True)).applicable
    assert service.search(query, 'u', 1) == before


def test_materials_and_options_do_not_create_protected_requirements(tmp_path):
    service, _ = make_service(tmp_path, {}, f1_mode='active')
    add(service, 'a', 'Project Alder decided to retain the archive.', 1704067200000)
    plan = build_summary_plan('Summarize project Alder decisions.\nMaterials:\n'
                             'Who proposed the trial? What is the secret identifier?',
                             service.store.fetch_by_user('u', governance=True))
    assert not plan.direct_requirements and not plan.attribution_requirements


def test_historical_demands_keep_real_old_state_and_new_state(tmp_path):
    old, new = 'I work at Larch.', 'I now work at Cedar.'
    service, _ = make_service(tmp_path, {old: [draft(old, 'Larch')], new: [draft(new, 'Cedar')]},
                              f1_mode='active', temporal_mode='off')
    add(service, 'old', old, 1704067200000)
    add(service, 'new', new, 1706745600000)
    hits, (_, outcome) = run_summary(service, 'Summarize my employment history and include each employer.')
    assert old in hits[0].content and new in hits[0].content
    assert outcome.protection.passed and outcome.protection.required_units == 2


def test_local_roles_keep_shared_source_directory_compact(tmp_path):
    raw = 'Project Alder: Mira reported that Jon proposed the trial; Mira did not propose it.'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'a', raw, 1704067200000, role='assistant')
    hits, _ = run_summary(service, 'Summarize project Alder and identify who proposed the trial.')
    assert hits[0].content.count(service._memory_id('a', 0)) == 1
    assert '消息角色:assistant' in hits[0].content and raw in hits[0].content


def test_same_name_different_sessions_is_not_collapsed_by_protection(tmp_path):
    a, b = 'Robin works at Larch.', 'Robin works at Cedar.'
    service, _ = make_service(tmp_path, {
        a: [dict(subject='Robin', predicate='employer', value='Larch', quote=a)],
        b: [dict(subject='Robin', predicate='employer', value='Cedar', quote=b)]},
        f1_mode='active', temporal_mode='off')
    add(service, 'a', a, 1704067200000)
    add(service, 'b', b, 1706745600000)
    hits, (_, outcome) = run_summary(service, 'Summarize Robin history and specify the employer.')
    assert a in hits[0].content and b in hits[0].content and '身份待确认' in hits[0].content
    assert outcome.protection.required_units == 2


def test_corrupt_or_foreign_source_is_rejected_before_protection(tmp_path):
    from app.governance.summary_retrieval import collect_summary
    from app.governance.summary_guard import guarded_summary
    raw = 'Project Alder identifier is AX-734.'
    service, _ = make_service(tmp_path, {}, f1_mode='active')
    add(service, 'a', raw, 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    plan = build_summary_plan('Summarize project Alder and specify its identifier.', snapshot)
    collection = collect_summary(snapshot, plan)
    item = collection.evidence[0]
    for bad in (replace(item.spans[0], source_id='other-user-source'),
                replace(item.spans[0], content_hash='invalid')):
        corrupt = replace(collection, evidence=(replace(item, spans=(bad,)),))
        outcome = guarded_summary(corrupt, plan, [], snapshot, limit=1, token_budget=850)
        assert not outcome.hits and not outcome.protection.passed
        assert outcome.protection.invalid_sources == 1


def test_conflicting_demands_cannot_remove_existing_f1_coverage(tmp_path):
    from app.governance.summary_retrieval import collect_summary
    from app.governance.summary_assembly import assemble_summary
    from app.governance.summary_guard import guarded_summary
    field = 'Project Alder identifier is AX-734; ' + 'retain every qualifier; ' * 80
    rule = ('Project Alder: we decided to retain the archive because the review failed; except for public copies, disclosure requires review. '
            + 'The reviewers retain the complete audit trail. ' * 30).rstrip()
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'a', field, 1704067200000)
    add(service, 'b', rule, 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    plan = build_summary_plan('Summarize project Alder decisions, reasons and exceptions and specify its identifier.', snapshot)
    collection = collect_summary(snapshot, plan)
    original = assemble_summary(collection, plan, [], snapshot, limit=1, token_budget=850)
    unguarded = guarded_summary(collection, plan, [], snapshot, limit=1, token_budget=850, preserve_coverage=False)
    outcome = guarded_summary(collection, plan, [], snapshot, limit=1, token_budget=850)
    assert rule in original.hits[0].content
    assert rule not in unguarded.hits[0].content and field.rstrip() in unguarded.hits[0].content
    assert unguarded.protection.coverage_regressions
    assert outcome.hits == original.hits
    assert not outcome.protection.passed and outcome.protection.missing_requirements


def test_engagement_likes_do_not_create_personal_preference_protection(tmp_path):
    service, _ = make_service(tmp_path, {}, f1_mode='active')
    add(service, 'format', 'Report format: include the measured engagement counts.', 1704067200000)
    snapshot = service.store.fetch_by_user('u', governance=True)
    plan = build_summary_plan('Prepare the engagement report using the agreed format. Include likes and shares.', snapshot)
    assert plan.applicable and not plan.direct_requirements


def test_chinese_direct_field_and_reporter_are_preserved(tmp_path):
    raw = '项目榆木：档案编号为AH-42。陈宁转述林越提出的试点建议；陈宁并非建议提出者。仅在审查完成后才能扩大。'
    service, _ = make_service(tmp_path, {}, f1_mode='active', temporal_mode='off')
    add(service, 'cn', raw, 1704067200000)
    query = '总结项目榆木，说明档案编号，并指出谁提出试点。'
    hits, (plan, outcome) = run_summary(service, query)
    assert raw in hits[0].content and '归因以原文为准' in hits[0].content
    assert {r.key for r in plan.direct_requirements} == {'direct:identifier'}
    assert {r.key for r in plan.attribution_requirements} == {'attribution:propose'}
    assert outcome.protection.passed
