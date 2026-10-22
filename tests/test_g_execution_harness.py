"""Synthetic G evidence obligations; no online scores or model-answer claims."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_g_execution_regression as g
from scripts import run_g_answer_probe as answers
from scripts import run_memory_joint_regression as replay
from scripts.governance_e2e_support import runtime_corpus, validate_dataset


DATA = replay.read(g.DATASET)
QUESTIONS = [(s['id'], q['id']) for s in DATA['scenarios'] for q in s['questions']
             if not q.get('expect_base_response')]


def fixture_question(scene_id, question_id=None):
    scene = next(s for s in DATA['scenarios'] if s['id'] == scene_id)
    question = next(q for q in scene['questions'] if question_id is None or q['id'] == question_id)
    return scene, question


def raw_rows(scene, ids=None):
    mapped = g.source_ids(scene)
    return [{'id': mapped[s['id']], 'content': s['content'], 'score': 0.5, 'created_at': s['date']}
            for s in scene['sources'] if ids is None or s['id'] in ids]


def test_gold_is_source_backed_and_not_sent_to_runtime():
    validate_dataset(DATA)
    runtime = runtime_corpus(DATA)
    assert len(runtime['scenarios']) == 10
    assert sum(q.get('answer_probe', False) for s in DATA['scenarios'] for q in s['questions']) == 4
    assert all(set(q) == {'id', 'query', 'as_of', 'top_k'}
               for s in runtime['scenarios'] for q in s['questions'])
    assert all('oracle_facts' not in s for scene in runtime['scenarios'] for s in scene['sources'])


def test_whole_condition_and_negative_example_are_required():
    scene, question = fixture_question('conditions_and_example')
    good = {'visible_hits': raw_rows(scene, ['procedure'])}
    assert g.evaluate(scene, question, good, 1500)['passed']
    bad = copy.deepcopy(good)
    bad['visible_hits'][0]['content'] = bad['visible_hits'][0]['content'].replace(
        ' only if the incident ticket is linked', '').replace(
        'Example: a security reviewer without a linked incident ticket does not authorize an emergency repair.', '')
    result = g.evaluate(scene, question, bad, 1500)
    assert not result['checks']['complete_support:conditional exception']
    assert not result['checks']['complete_support:negative example']
    assert not result['checks']['dependency:exception_needs_ticket_and_example']


def test_quote_under_wrong_source_cannot_pass_provenance_check():
    scene, question = fixture_question('global_template')
    rows = raw_rows(scene, ['template', 'decision'])
    rows[0]['id'] = 'unrelated'
    rows[0]['content'] = '[来源:wrong:0-999] ' + rows[0]['content']
    result = g.evaluate(scene, question, {'visible_hits': rows}, 1500)
    assert not result['checks']['complete_support:complete learned template']


def test_different_project_and_missing_retraction_are_detected():
    scene, question = fixture_question('rule_retraction')
    result = g.evaluate(scene, question, {'visible_hits': raw_rows(scene, ['old', 'wrong'])}, 1500)
    assert not result['checks']['exclude_source:wrong']
    assert not result['checks']['dependency:old_exception_needs_withdrawal']
    assert not result['checks']['complete_support:withdrawal and absent replacement']


def test_serialized_metadata_counts_against_visible_budget():
    scene, question = fixture_question('tight_budget')
    result = g.evaluate(scene, question, {'visible_hits': raw_rows(scene)}, 8)
    assert not result['checks']['caller_visible_budget']


@pytest.fixture(scope='module')
def offline_rows(tmp_path_factory):
    folder = tmp_path_factory.mktemp('g-execution')
    result = replay.run_worker(replay.ROOT, g.DATASET, folder / 'worker.json', 1500)
    assert result['isolation'] == {'dotenv_reads': 0, 'network_attempts': 0, 'outside_database_attempts': 0}
    return {(r['scenario'], r['question'], r['mode']): r for r in result['rows']}


@pytest.mark.parametrize('scene_id,question_id', QUESTIONS)
def test_add_search_preserves_complete_g_support(offline_rows, scene_id, question_id):
    scene, question = fixture_question(scene_id, question_id)
    row = offline_rows[(scene_id, question_id, 'full')]
    result = g.evaluate(scene, question, row, 1500)
    assert result['passed'], {key: value for key, value in result['checks'].items() if not value}
    assert row['answer_correct'] is None


def test_pure_advice_preserves_ordinary_output_and_calls(offline_rows):
    full, off, shadow = [offline_rows[('pure_advice', 'ordinary', mode)] for mode in ('full', 'off', 'shadow')]
    for key in ('hits', 'query_call_hashes', 'query_vector_calls'):
        assert full[key] == off[key] == shadow[key]


def test_short_complete_decision_survives_128_token_visible_budget(tmp_path):
    report = replay.run_worker(replay.ROOT, g.DATASET, tmp_path / 'tight.json', 128, ['tight_budget'])
    row = next(r for r in report['rows'] if r['mode'] == 'full')
    scene, question = fixture_question('tight_budget')
    result = g.evaluate(scene, question, row, 128)
    assert result['passed'], result['checks']


def frozen_answer_pair(folder):
    scene, question = fixture_question('global_template')
    report = {'status': 'complete', 'real_model_evaluation': False, 'rows': [
        {'scenario': scene['id'], 'question': question['id'], 'mode': 'full',
         'visible_hits': raw_rows(scene, ['template', 'decision'])}]}
    replay.write(folder / 'baseline.json', report)
    replay.write(folder / 'current.json', report)
    replay.write(folder / 'comparison.json', {'dataset_sha256': replay.digest(DATA),
        'visible_budget': 1500, 'baseline_commit': g.PROTECTION_BASELINE,
        'current_app_sha256': 'synthetic-code-fingerprint'})
    return answers.prepare_jobs(DATA, folder, ['global_template/apply'])


class AnswerDouble:
    def __init__(self, *, fail=False, finish='stop'):
        self.calls, self.fail, self.finish = [], fail, finish
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError('SENSITIVE_UPSTREAM_DETAIL_MUST_NOT_BE_SAVED')
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=self.finish,
            message=SimpleNamespace(content=json.dumps({'answer': 'synthetic answer', 'sources': ['source-a']})))])


def test_answer_inputs_exclude_gold_and_share_identical_prompt_and_budget(tmp_path):
    jobs, context = frozen_answer_pair(tmp_path)
    assert len(jobs) == 2
    assert all(set(job['payload']) == {'question', 'memories'} for job in jobs)
    assert context['visible_budget'] == 1500
    assert 'gold' not in replay.canonical(jobs) and 'oracle_facts' not in replay.canonical(jobs)
    assert 'apply that supported and applicable rule' in answers.ANSWER_PROMPT
    assert 'not an independent instruction' in answers.ANSWER_PROMPT


def test_answer_success_cache_avoids_duplicate_provider_charges(tmp_path):
    jobs, context = frozen_answer_pair(tmp_path / 'pairs')
    identity = {'provider': 'openai', 'model': 'synthetic', 'configuration_sha256': 'fixture'}
    client = AnswerDouble()
    first = answers.execute_jobs(jobs, context, identity, client, tmp_path / 'answer')
    assert first['status'] == 'awaiting_semantic_review'
    # Identical old/new evidence deliberately shares a request cache entry.
    assert len(client.calls) == 1 and first['provider_attempts'] == 1 and first['cache_hits'] == 1
    second = answers.execute_jobs(jobs, context, identity, client, tmp_path / 'answer')
    assert second['provider_attempts'] == 0 and second['cache_hits'] == 2
    assert all(row['answer_correct'] is None for row in second['cases'])
    assert client.calls[0]['temperature'] == 0 and client.calls[0]['max_tokens'] == 768


@pytest.mark.parametrize('failure,finish', [(True, 'stop'), (False, 'length')])
def test_answer_failure_stops_and_keeps_only_exception_type(tmp_path, failure, finish):
    jobs, context = frozen_answer_pair(tmp_path / 'pairs')
    client = AnswerDouble(fail=failure, finish=finish)
    folder = tmp_path / 'answer'
    result = answers.execute_jobs(jobs, context, {'provider': 'openai', 'model': 'synthetic'}, client, folder)
    assert result['status'] == 'blocked' and result['provider_attempts'] == 1
    assert len(client.calls) == 1 and result['cases'] == []
    assert not (folder / 'cache').exists()
    assert 'SENSITIVE_UPSTREAM_DETAIL' not in (folder / 'answers.json').read_text(encoding='utf-8')


def test_answer_hard_limit_and_identity_do_not_expose_secrets(tmp_path):
    jobs, context = frozen_answer_pair(tmp_path / 'pairs')
    with pytest.raises(ValueError, match='AnswerProbeLimit'):
        answers.execute_jobs(jobs * 5, context, {}, AnswerDouble(), tmp_path / 'answer')
    identity = answers.model_identity(SimpleNamespace(llm_provider='openai', openai_model='synthetic',
        openai_base_url='https://private.example/endpoint', openai_api_key='NEVER_OUTPUT_ME'))
    encoded = replay.canonical(identity)
    assert set(identity) == {'provider', 'model', 'configuration_sha256'}
    assert 'private.example' not in encoded and 'NEVER_OUTPUT_ME' not in encoded


def test_answer_dry_run_does_not_load_provider_settings(tmp_path):
    frozen_answer_pair(tmp_path / 'pairs')
    (tmp_path / '.env').write_text('LLM_PROVIDER=invalid_not_to_be_read\n', encoding='utf-8')
    result = subprocess.run([sys.executable, str(Path(answers.__file__).resolve()),
        '--paired-output', str(tmp_path / 'pairs'), '--case', 'global_template/apply'],
        cwd=tmp_path, env=dict(os.environ, LLM_PROVIDER='invalid_not_to_be_read',
                             PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8'),
        capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)['status'] == 'dry_run_valid'
    assert not (tmp_path / 'pairs' / 'real_answers').exists()
