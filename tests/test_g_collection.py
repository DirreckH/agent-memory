"""Offline procedural-evidence collection, independent of query parser rules."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.governance.extraction import digest
from app.governance.models import Evidence, Fact, Source, SourceStatus, UserSnapshot
from app.governance.resolution import derive_relations
from app.governance.summary_planning import SummaryPlan
from app.governance.summary_retrieval import collect_summary


def requirement(terms=('response template',), kind='procedure'):
    return SimpleNamespace(key='need:' + kind + ':target', label='required target',
                           trigger='use our earlier target', terms=terms,
                           mandatory=True, kind=kind)


def plan(req=None, **changes):
    values = dict(vars(SummaryPlan(applicable=True, scopes=('Cobalt',), reference=4000)))
    values.update(requirements=(req,) if req else (), procedure_requirements=(req,) if req else ())
    values.update(changes)
    return SimpleNamespace(**values)


def snapshot(*texts, facts=(), relations=None):
    sources = {f's{i}': Source(f's{i}', text, 'user', f'session{i}', 0,
                               1000 * (i + 1), '2026-01-01') for i, text in enumerate(texts)}
    return UserSnapshot('u', 1, 1, sources=sources, facts=tuple(facts),
                        relations=tuple(relations if relations is not None else derive_relations(tuple(facts))),
                        statuses=tuple(SourceStatus(s, 'ready' if any(f.source_id == s for f in facts)
                                                   else 'no_fact') for s in sources))


def rule(sid, text, value, **updates):
    ev = Evidence(sid, 0, len(text), text, digest(text))
    values = dict(id='f' + sid, subject_id='self', subject_name='self', predicate='rules',
                  scope='reporting', value=value, kind='rule', polarity='positive', modality='asserted',
                  operation='assert', evidence=(ev,), source_time=1000 * (int(sid[1:]) + 1))
    values.update(updates)
    return Fact(**values)


def required_items(collection, req):
    return [item for item in collection.evidence if req.key in item.facets]


def quotes(items):
    return '\n'.join(ev.quote for item in items for ev in item.spans)


def test_user_generic_template_is_added_to_project_facts_without_claiming_current_validity():
    req = requirement()
    template = 'Our reusable response template:\nFields: result, evidence, limitations.\nException: omit empty fields.'
    data = snapshot('Project Cobalt: the pilot is ready.', template,
                    'Our expense template:\nFields: amount, currency.')
    found = required_items(collect_summary(data, plan(req)), req)
    assert len(found) == 1
    assert 'Fields: result, evidence, limitations.' in quotes(found)
    assert 'Exception: omit empty fields.' in quotes(found)
    assert 'expense template' not in quotes(found)
    assert '未结构化原文；适用性待核验' in found[0].labels
    assert not any('适用观察' in label for label in found[0].labels)


@pytest.mark.parametrize('foreign', [
    'Project Juniper response template:\nFields: secret owner, approval.',
    "Alice's response template:\nFields: secret owner, approval.",
    'Our response template applies only to project Juniper.\nFields: secret owner, approval.',
])
def test_same_name_foreign_project_or_owner_does_not_satisfy_requirement(foreign):
    req = requirement()
    data = snapshot('Project Cobalt: the pilot is ready.', foreign)
    collection = collect_summary(data, plan(req))
    assert not required_items(collection, req)
    assert 'secret owner' not in quotes(collection.evidence)


def test_explicit_detail_marks_the_matching_project_evidence_only():
    req = requirement(('archive access code',), kind='explicit')
    data = snapshot('Project Cobalt: the archive access code is 8472.',
                    'Project Juniper: the archive access code is 9921.',
                    'My archive access code is 1111.')
    found = required_items(collect_summary(data, plan(req)), req)
    assert len(found) == 1 and '8472' in quotes(found)
    assert '9921' not in quotes(found) and '1111' not in quotes(found)


def test_formula_variables_units_and_example_form_one_required_group():
    req = requirement(('scoring formula',))
    raw = ('Our scoring formula:\nscore = benefit / cost\nVariables:\n'
           'benefit = expected saving in USD\ncost = setup expense in USD\n'
           'Example: benefit=20 and cost=10 gives score=2.\n'
           'Exception: if cost=0, report undefined.')
    collection = collect_summary(snapshot(raw), plan(req))
    found = required_items(collection, req)
    assert len(found) == 1
    for text in ('score = benefit / cost', 'expected saving in USD', 'setup expense in USD',
                 'gives score=2', 'if cost=0'):
        assert text in quotes(found)
    assert found[0].required_spans


def test_unrelated_adjacent_paragraph_is_not_a_formula_dependency():
    req = requirement(('scoring formula',))
    raw = ('Our scoring formula is score = benefit / cost.\n'
           'Variables: benefit and cost are measured in USD.\n'
           'My bicycle needs a new chain.')
    found = required_items(collect_summary(snapshot(raw), plan(req)), req)
    assert len(found) == 1 and 'measured in USD' in quotes(found)
    assert 'bicycle' not in quotes(found)


def test_matching_title_without_body_does_not_satisfy_requirement():
    req = requirement()
    found = required_items(collect_summary(snapshot('# Our response template'), plan(req)), req)
    assert not found


def test_full_rule_slot_resolves_current_correction_even_without_repeated_template_name():
    req = requirement()
    old = 'Our response template uses fields A and B.'
    new = 'Correction: replace fields A and B with fields result and evidence.'
    f0 = rule('s0', old, 'A and B')
    ev = Evidence('s1', 0, len(new), new, digest(new), 'update')
    f1 = rule('s1', new, 'result and evidence', operation='correct', target_value='A and B',
              evidence=(replace(ev, purpose='assertion'), ev))
    found = required_items(collect_summary(snapshot(old, new, facts=(f0, f1)), plan(req)), req)
    assert found and 'result and evidence' in quotes(found)
    assert not any('fs0' in item.fact_ids for item in found)
    assert any('适用观察' in label for item in found for label in item.labels)


@pytest.mark.parametrize('updates', [
    {'valid_to': 2000}, {'valid_from': 9000}, {'modality': 'planned'}, {'polarity': 'negative'},
])
def test_inapplicable_rule_does_not_count_as_required_execution_evidence(updates):
    req = requirement()
    raw = 'Our response template uses fields result and evidence.'
    fact = rule('s0', raw, 'result and evidence', **updates)
    assert not required_items(collect_summary(snapshot(raw, facts=(fact,)), plan(req)), req)


def test_retracted_rule_is_not_reintroduced_as_unstructured_fallback():
    req = requirement()
    old = 'Our response template uses fields A and B.'
    new = 'I retract the response template using fields A and B; no replacement is available.'
    f0 = rule('s0', old, 'A and B')
    ev = Evidence('s1', 0, len(new), new, digest(new), 'update')
    f1 = rule('s1', new, 'A and B', operation='retract', target_value='A and B',
              evidence=(replace(ev, purpose='assertion'), ev))
    collection = collect_summary(snapshot(old, new, facts=(f0, f1)), plan(req))
    assert not required_items(collection, req)
    old_items = [item for item in collection.evidence if 'fs0' in item.fact_ids]
    for item in old_items:
        assert 'no replacement is available' in quotes([item])
        assert any('retracted_claim' in label for label in item.labels)


def test_rules_use_multi_value_resolver_without_implicit_latest_wins():
    req = requirement()
    a = 'Our response template requires a result field.'
    b = 'Our response template requires an evidence field.'
    found = required_items(collect_summary(snapshot(a, b, facts=(rule('s0', a, 'result'),
                                                               rule('s1', b, 'evidence'))), plan(req)), req)
    assert 'result field' in quotes(found) and 'evidence field' in quotes(found)


def test_foreign_structured_rule_does_not_gain_requirement_by_target_name():
    req = requirement()
    raw = 'Our response template uses private fields.'
    foreign = rule('s0', raw, 'private fields', subject_id='Alice', subject_name='Alice')
    collection = collect_summary(snapshot(raw, facts=(foreign,)), plan(req))
    assert not required_items(collection, req)
    assert 'private fields' not in quotes(collection.evidence)


def test_procedure_respects_source_knowledge_cutoff_and_time_window():
    from app.temporal import TemporalWindow
    req = requirement()
    raw = 'Our response template uses fields result and evidence.'
    assert not required_items(collect_summary(snapshot(raw), plan(req, knowledge_cutoff=500)), req)
    assert not required_items(collect_summary(snapshot(raw), plan(req, window=TemporalWindow(2000, 3000))), req)


def test_no_procedure_request_preserves_project_only_collection():
    data = snapshot('Project Cobalt: approved pilot.', 'Our response template uses fields result and evidence.')
    collection = collect_summary(data, plan())
    assert 'approved pilot' in quotes(collection.evidence)
    assert 'response template' not in quotes(collection.evidence)
    assert not any(key.startswith('need:') for item in collection.evidence for key in item.facets)


def test_unrelated_rule_in_same_multi_value_slot_is_not_a_target_version():
    req = requirement()
    a = 'Our response template requires a result field.'
    b = 'Our expense template requires an amount field.'
    collection = collect_summary(snapshot(a, b, facts=(rule('s0', a, 'result'),
                                                       rule('s1', b, 'amount'))), plan(req))
    found = required_items(collection, req)
    assert 'result field' in quotes(found)
    assert 'expense template' not in quotes(collection.evidence)


def test_procedure_only_request_does_not_collect_unrelated_personal_history():
    req = requirement(('scoring rule',))
    data = snapshot('Our scoring rule is score = safety + reversibility.',
                    'I bought a bicycle yesterday.')
    p = plan(req, scopes=(), global_scope=False, focus_terms=('scoring rule',))
    found = collect_summary(data, p)
    assert 'safety + reversibility' in quotes(required_items(found, req))
    assert 'bicycle' not in quotes(found.evidence)


def test_assistant_definition_remains_unconfirmed_source_when_explicitly_requested():
    req = requirement(('scoring method',))
    req.trigger = 'scoring method you taught'
    data = snapshot('The scoring method is score = safety + reversibility.')
    data = replace(data, sources={'s0': replace(data.sources['s0'], role='assistant')})
    found = required_items(collect_summary(data, plan(req)), req)
    assert found
    assert any('适用性待核验' in label for item in found for label in item.labels)
    assert not any('适用观察' in label for item in found for label in item.labels)


def test_user_taught_requirement_does_not_change_assistant_speaker_into_user():
    req = requirement(('scoring method',))
    req.trigger = 'scoring method I taught'
    data = snapshot('The scoring method is score = safety + reversibility.')
    data = replace(data, sources={'s0': replace(data.sources['s0'], role='assistant')})
    assert not required_items(collect_summary(data, plan(req)), req)


def test_unqualified_template_from_other_project_session_is_not_global():
    req = requirement()
    data = snapshot('Project Juniper: prepare the review.',
                    'Our response template uses fields secret and owner.')
    data = replace(data, sources={key: replace(source, session_id='juniper-session')
                                  for key, source in data.sources.items()})
    assert not required_items(collect_summary(data, plan(req)), req)


def test_project_reference_to_template_without_definition_is_not_need_coverage():
    req = requirement()
    data = snapshot('Project Cobalt uses our response template.')
    assert not required_items(collect_summary(data, plan(req)), req)


def test_current_correction_binds_separate_update_and_definition_evidence():
    req = requirement()
    old = 'Our response template requires two fields.'
    assertion = 'Our response template requires three fields.'
    update = 'Correction: the earlier two-field requirement was wrong.'
    new = assertion + '\n' + update
    f0 = rule('s0', old, 'two fields')
    ev = Evidence('s1', len(assertion) + 1, len(new), update, digest(update), 'update')
    f1 = rule('s1', assertion, 'three fields', operation='correct', target_value='two fields',
              evidence=(Evidence('s1', 0, len(assertion), assertion, digest(assertion)), ev))
    found = required_items(collect_summary(snapshot(old, new, facts=(f0, f1)), plan(req)), req)
    assert found
    for item in found:
        assert assertion in quotes([item]) and update in quotes([item])
        assert ev in item.required_spans


def test_real_planner_standalone_procedure_keeps_only_matching_definition():
    from app.governance.summary_planning import build_summary_plan
    data = snapshot('The scoring rule I taught you is score = safety + reversibility.',
                    'I bought a bicycle yesterday.')
    p = build_summary_plan('Apply the scoring rule I taught you.', data)
    assert p.applicable and p.procedure_requirements
    collection = collect_summary(data, p)
    assert 'safety + reversibility' in quotes(collection.evidence)
    assert 'bicycle' not in quotes(collection.evidence)
