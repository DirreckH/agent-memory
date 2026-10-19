"""Guard evaluation integrity; no tests here call real providers."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.governance_e2e_support import (CacheMiss, CompletionCache, VectorCache, canonical,
    evidence_metrics, freeze_persistence_clock, reviewed_metrics, runtime_corpus, validate_dataset, visible_context)

DATASET = json.loads((Path(__file__).parent / 'fixtures/governance_e2e_cases.json').read_text(encoding='utf-8'))


def request(text='hello'):
    return dict(model='test', messages=[{'role': 'system', 'content': 'test'},
        {'role': 'user', 'content': canonical({'query': text, 'options': []})}],
        temperature=0, max_tokens=512, response_format={'type': 'json_object'}, stream=False)


def test_gold_is_source_backed_and_does_not_enter_runtime():
    validate_dataset(DATASET)
    runtime = runtime_corpus(DATASET)
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    assert not {'gold', 'answer', 'reason', 'oracle_facts', 'forbidden_claims'} & keys(runtime)
    assert {s['id'] for s in runtime_corpus(DATASET, 'holdout')['scenarios']}.isdisjoint(
        {s['id'] for s in runtime_corpus(DATASET, 'development')['scenarios']})


def test_invalid_gold_span_is_rejected():
    data = deepcopy(DATASET)
    data['scenarios'][0]['questions'][0]['gold']['atoms'][0]['support'][0]['quote'] = 'fabricated quote'
    with pytest.raises(AssertionError, match='Unbacked gold'):
        validate_dataset(data)


def test_frozen_completion_replay_never_calls_network_and_detects_mismatch(tmp_path):
    calls = []
    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='length',
                message=SimpleNamespace(content='{"expanded_query":"literal"}'))], usage=None)
    recorded = CompletionCache(tmp_path, {'model': 'test'}, client=Client(), recording=True)
    first = recorded.create(**request())
    replay = CompletionCache(tmp_path, {'model': 'test'})
    assert replay.create(**request()).choices[0].finish_reason == first.choices[0].finish_reason == 'length'
    recorded.create(**request())
    assert len(calls) == 1
    with pytest.raises(CacheMiss):
        replay.create(**request('different query'))
    with pytest.raises(CacheMiss):
        CompletionCache(tmp_path, {'model': 'another'}).create(**request())
    with pytest.raises(ValueError, match='Unapproved'):
        replay.create(**request(), api_key='must never be saved')
    assert 'must never be saved' not in ''.join(p.read_text() for p in tmp_path.glob('*.json'))


def test_failed_call_is_not_fabricated_or_cached(tmp_path):
    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
        def create(self, **kwargs):
            raise PermissionError('secret upstream detail')
    cache = CompletionCache(tmp_path, {}, client=Client(), recording=True)
    with pytest.raises(PermissionError):
        cache.create(**request())
    assert not list(tmp_path.glob('*.json'))
    assert cache.stats['errors'][0]['type'] == 'PermissionError'
    assert 'secret upstream detail' not in canonical(cache.stats)


def test_vector_cache_reuses_exact_values_and_respects_input_order(tmp_path):
    import numpy as np
    class Provider:
        def embed(self, texts):
            return np.array([[len(t), 1] for t in texts], dtype=np.float32)
    live = VectorCache(tmp_path, {'model': 'test'}, provider=Provider(), recording=True)
    values = live.embed(['alpha', 'b', 'alpha'])
    frozen = VectorCache(tmp_path, {'model': 'test'}).embed(['b', 'alpha'])
    np.testing.assert_array_equal(values[[1, 0]], frozen)
    with pytest.raises(CacheMiss):
        VectorCache(tmp_path, {'model': 'test'}).embed(['missing'])


def test_visible_budget_counts_metadata_and_never_slices_qualifiers():
    hit = {'id': 'a', 'content': '允许购买，但必须补批。', 'score': 0.5, 'created_at': '2025-01-01'}
    visible, _, size = visible_context([hit], 1000)
    assert visible == [hit]
    small, omitted, _ = visible_context([hit], size - 1)
    assert small == [] and omitted == ['a']
    two, omitted, count = visible_context([hit, dict(hit, id='b')], size)
    assert two == [hit] and omitted == ['b'] and count <= size


def test_quote_coverage_does_not_grade_answer_or_semantics():
    gold = {'atoms': [{'claim': 'Current employer is B', 'support': [{'source_id': 'b', 'quote': 'B'}]}]}
    metric = evidence_metrics(gold, [{'content': 'Plan to join B; currently at A.'}])
    assert metric['verbatim_atom_coverage'] == 1
    assert metric['answer_correctness'] is None and metric['wrong_state_pollution'] is None


def test_alternative_evidence_counts_without_requiring_repeated_sources():
    gold = {'atoms': [{'claim': 'acceptance stage', 'support': [{'source_id': 'a', 'quote': 'entered acceptance'}],
                      'support_alternatives': [[{'source_id': 'b', 'quote': 'still in acceptance'}]]}]}
    assert evidence_metrics(gold, [{'content': 'still in acceptance'}])['verbatim_atom_coverage'] == 1


def review_item(**kwargs):
    return dict(mode='full', scenario='s', question='q', answer_correct=None, necessary_evidence_valid=None,
        wrong_current_state=None, qualifiers_preserved=None, first_failure_stage=None,
        reviewer='reviewer', reason='Read all sources and response', evidence_locations=['runs/full.json'], **kwargs)


def test_cannot_grade_missing_answers_or_hide_unreviewed_cases():
    rows = [{'scenario': 's', 'question': 'q', 'split': 'holdout', 'modes': {'full': {}, 'v3': {}}}]
    item = review_item()
    item['answer_correct'] = True
    with pytest.raises(ValueError, match='never produced'):
        reviewed_metrics({'items': [item]}, rows, set())
    empty = reviewed_metrics({'items': []}, rows, set())
    assert empty['by_mode_split']['full']['holdout']['complete_accuracy'] is None
    assert empty['v3_full_pairs'][0]['change'] == 'pending'
    graded = reviewed_metrics({'items': [item]}, rows, {('full', 's', 'q')})
    assert graded['by_mode_split']['full']['holdout']['complete_accuracy'] == 1
    assert graded['v3_full_pairs'][0]['change'] == 'pending'


def test_review_requires_traceable_reasons():
    rows = [{'scenario': 's', 'question': 'q', 'split': 'holdout', 'modes': {'full': {}}}]
    item = review_item()
    item.update(first_failure_stage='version', reviewer='')
    with pytest.raises(ValueError, match='review requires|Review requires'):
        reviewed_metrics({'items': [item]}, rows, set())


def test_f1_review_compares_frozen_current_with_full_without_inventing_answers():
    rows = [{'scenario': 's', 'question': 'q', 'split': 'development',
             'modes': {'legacy': {}, 'full': {}, 'f1_plan': {}, 'f1_pack': {}}}]
    items = []
    for mode, correct in [('legacy', False), ('full', True)]:
        item = review_item()
        item.update(mode=mode, answer_correct=correct)
        items.append(item)
    result = reviewed_metrics({'items': items}, rows, {('legacy', 's', 'q'), ('full', 's', 'q')})
    assert result['legacy_full_pairs'][0]['change'] == 'improved'
    assert result['by_mode_split']['f1_plan']['development']['complete_accuracy'] is None


def test_freezing_persistence_includes_governed_hit_without_baseline(tmp_path, monkeypatch):
    import app.service as service_module
    import app.storage as storage_module
    from tests.test_governance import make_service, add, draft
    stamp = '2025-07-01T00:00:00+00:00'
    freeze_persistence_clock(service_module, storage_module, stamp, monkeypatch.setattr)
    raw = 'I currently live in Suzhou.'
    service, _ = make_service(tmp_path, {raw: [draft(raw, 'Suzhou', predicate='residence', scope='home')]},
                              min_relevance_score=1.0, temporal_mode='off')
    add(service, 'clock', raw, 1740787200000)
    hits = service.search('What is my current residence?', 'u', 1)
    assert hits and hits[0].id.startswith('gov_')
    assert hits[0].created_at == '2025-07-01T00:00:00.000Z'


@pytest.mark.parametrize('finish,content,expected', [
    ('stop', '{"answer":"unknown","sources":[]}', 'awaiting_semantic_review'),
    ('length', '{"answer":"unknown","sources":[]}', 'blocked'),
    ('stop', '{"answer":"unknown","sources":"invalid"}', 'blocked'),
])
def test_answer_controller_uses_fixed_inputs_and_handles_incomplete_answers(tmp_path, monkeypatch, finish, content, expected):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import run_governance_e2e as runner
    import app.config
    import app.llm
    calls = []
    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
                message=SimpleNamespace(content=content))], usage=None)
    fingerprint = {'model': 'mock', 'provider': 'test'}
    client = Client()
    monkeypatch.setattr(app.config, 'Settings', lambda: SimpleNamespace(model_copy=lambda **kwargs: None))
    monkeypatch.setattr(app.llm, 'build_memory_llm', lambda _: SimpleNamespace(_get_client=lambda: client))
    monkeypatch.setattr(runner, 'fingerprints', lambda _: {'llm': fingerprint})
    manifest = {'fingerprints': {'llm': fingerprint}, 'answer_max_tokens': 768}
    one = {'scenarios': [{'id': 's', 'queries': [{'id': 'q', 'query': 'What is known?',
        'visible_hits': [], 'gold': 'MUST_NOT_ENTER_REQUEST'}]}]}
    runs = {mode: one for mode in ('v3', 'v4', 'preserve', 'full', 'oracle')}
    result = runner.answer(SimpleNamespace(output=tmp_path), manifest, runs)
    assert result['status'] == expected
    assert len(calls) == 1  # Identical visible input across modes reuses the exact same answer.
    assert 'MUST_NOT_ENTER_REQUEST' not in canonical(calls)
    assert calls[0]['temperature'] == 0 and calls[0]['max_tokens'] == 768
    if expected == 'awaiting_semantic_review':
        assert len(result['cases']) == 5
        assert all(c['correct'] is None for c in result['cases'])
    else:
        assert result['cases'] == []
