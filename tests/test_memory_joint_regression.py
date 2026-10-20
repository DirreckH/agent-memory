"""Evaluation integrity checks, independent of retrieval's own expected answers."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

from scripts import run_memory_joint_regression as runner


def test_worker_isolates_environment_database_and_application_entrypoint(tmp_path):
    # Reading this .env would be blocked by the worker's audit hook. Process
    # variables must also not override the explicitly isolated test settings.
    (tmp_path / '.env').write_text('EMBEDDING_PROVIDER=invalid_fixture_value\n', encoding='utf-8')
    protected = tmp_path / 'not-a-test-database.db'
    protected.write_bytes(b'unchanged sentinel')
    output = tmp_path / 'worker.json'
    result = subprocess.run([
        sys.executable, str(Path(runner.__file__).resolve()), '--worker', '--root', str(runner.ROOT),
        '--dataset', str(runner.ROOT / 'tests/fixtures/memory_joint_cases.json'),
        '--scenario', 'pronoun_packet', '--budget', '256', '--output', str(output)],
        cwd=tmp_path, env=dict(os.environ, DATABASE_PATH=str(protected),
            EMBEDDING_PROVIDER='invalid_fixture_value', GOVERNANCE_MODE='invalid_fixture_value',
            PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8'),
        capture_output=True, text=True, encoding='utf-8', timeout=45)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['isolation'] == {'dotenv_reads': 0, 'network_attempts': 0, 'outside_database_attempts': 0}
    assert len(report['rows']) == 5 and report['real_model_evaluation'] is False
    assert all(r['answer_correct'] is None for r in report['rows'])
    full = next(r for r in report['rows'] if r['mode'] == 'full')
    assert all(full['declared_checks'].values())
    assert protected.read_bytes() == b'unchanged sentinel'


def test_declared_ordinary_response_change_fails_even_with_identical_calls(tmp_path, monkeypatch):
    rows = [{'scenario': 'control', 'question': 'ordinary', 'category': 'ordinary', 'mode': mode,
             'hits': [{'id': 'same', 'content': 'original', 'score': 0.5, 'created_at': 'fixed'}],
             'declared_checks': {}, 'query_call_hashes': [], 'query_vector_calls': [],
             'expect_baseline_unchanged': True, 'expect_base_response': True}
            for mode in ('full', 'off', 'shadow', 'legacy', 'f1_shadow')]

    def fake_worker(root, *args):
        result = copy.deepcopy(rows)
        if root == runner.ROOT:
            result[0]['hits'][0]['content'] = 'unrelated active output'
        return {'rows': result}

    monkeypatch.setattr(runner, 'run_worker', fake_worker)
    dataset = tmp_path / 'fixture.json'
    dataset.write_text('{}', encoding='utf-8')
    args = SimpleNamespace(baseline=runner.DEFAULT_BASELINE, dataset=dataset,
                           output=tmp_path / 'paired', budget=256, scenario=[])
    assert runner.compare(args) == 1
    result = json.loads((args.output / 'comparison.json').read_text(encoding='utf-8'))
    failures = {item['check'] for item in result['compatibility'] if not item['passed']}
    assert failures == {'declared_full_response_unchanged', 'declared_ordinary_uses_base'}
