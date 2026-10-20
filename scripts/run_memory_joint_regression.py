"""Offline paired evidence regression against a frozen Git revision.

Uses synthetic fixtures and deterministic SDK/vector doubles in isolated temporary
databases. It never reads .env, production databases, logs, or calls a provider.
Reported coverage and declared checks are diagnostics, not answer accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = 'f930ac62b730b4cd6cd558cd412b0136d3c4437b'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def worker(args):
    # Import only after selecting the baseline/current source tree.
    sys.path.insert(0, str(args.root))
    isolation = {'dotenv_reads': 0, 'network_attempts': 0, 'outside_database_attempts': 0}
    database_root = None

    def audit(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
            if Path(os.fsdecode(values[0])).name.casefold() == '.env':
                isolation['dotenv_reads'] += 1
                raise RuntimeError('Offline evaluation cannot read .env')
        if event in ('socket.connect', 'socket.getaddrinfo'):
            isolation['network_attempts'] += 1
            raise RuntimeError('Offline evaluation cannot access the network')
        if event == 'sqlite3.connect':
            if database_root is None or not Path(os.fsdecode(values[0])).resolve().is_relative_to(database_root):
                isolation['outside_database_attempts'] += 1
                raise RuntimeError('Offline evaluation cannot open an external database')

    sys.addaudithook(audit)
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import numpy as np
    from app.config import Settings
    from app.embeddings import normalize_rows
    from app.llm import LLMConnection, OpenAICompatibleMemoryLLM
    from app.schemas import MemoryMessage
    import app.service as service_module
    import app.storage as storage_module
    from scripts.governance_e2e_support import freeze_persistence_clock, visible_context, evidence_metrics
    freeze_persistence_clock(service_module, storage_module, '2026-10-20T00:00:00+00:00')

    class OfflineSettings(Settings):
        @classmethod
        def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                       dotenv_settings, file_secret_settings):
            return (init_settings,)

    class FrozenSDK:
        def __init__(self, facts):
            self.facts, self.calls = facts, []
            self.query_output = {'expanded_query': '', 'retrieval_steps': []}
            self.chat = SimpleNamespace(completions=self)

        def with_options(self, **kwargs):
            return self

        def create(self, **kwargs):
            self.calls.append(kwargs)
            payload = json.loads(kwargs['messages'][1]['content'])
            response = self.query_output
            if 'chunks' in payload:
                response = {'items': [{'chunk_id': c['chunk_id'], 'complete': True,
                                      'facts': self.facts.get(c['text'], [])} for c in payload['chunks']]}
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content=json.dumps(response)))])

    class FrozenVectors:
        # Same deterministic vocabulary algorithm as the project's unit fixtures,
        # kept local so importing API tests cannot instantiate app.main's app.
        dimension = 512

        def __init__(self):
            self.vocabulary = {}

        def embed(self, texts):
            matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for row, text in enumerate(texts):
                normalized = ''.join(c.casefold() if c.isalnum() else ' ' for c in text)
                raw_tokens = normalized.split()
                tokens = list(raw_tokens)
                for token in raw_tokens:
                    if len(token) > 5 and token.endswith('ing'):
                        tokens.append(token[:-3])
                    if len(token) > 4 and token.endswith('ed'):
                        tokens.append(token[:-2])
                    if len(token) > 4 and token.endswith('es'):
                        tokens.append(token[:-2])
                    elif len(token) > 4 and token.endswith('s'):
                        tokens.append(token[:-1])
                for token in tokens:
                    index = self.vocabulary.setdefault(token, len(self.vocabulary))
                    if index >= self.dimension:
                        raise ValueError('Offline vocabulary capacity exceeded')
                    matrix[row, index] += 1
            return normalize_rows(matrix)

    def make_service(folder, facts, mode, **updates):
        config = OfflineSettings(_env_file=None, database_path=folder / 'memory.db',
            governance_mode=mode, llm_provider='none', llm_add_enrichment=False,
            temporal_extraction_mode='rules', min_relevance_score=0.0, **updates)
        llm = OpenAICompatibleMemoryLLM(LLMConnection('test', 'https://example.invalid', 'test'),
                                       1, 0, False, False)
        sdk = FrozenSDK(facts)
        llm._client = sdk
        service = service_module.MemoryService(config, storage_module.SQLiteMemoryStore(config.database_path),
                                               FrozenVectors(), llm)
        service.initialize()
        return service, sdk

    def timestamp(value):
        if not value:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)

    dataset = read(args.dataset)
    scenes = dataset['scenarios']
    if args.scenario:
        unknown = set(args.scenario) - {s['id'] for s in scenes}
        if unknown:
            raise ValueError(f'Unknown scenario IDs: {sorted(unknown)}')
        scenes = [s for s in scenes if s['id'] in args.scenario]
    results = []
    modes = {'full': ('active', 'active'), 'off': ('off', 'active'),
             'shadow': ('shadow', 'active'), 'f1_shadow': ('active', 'shadow'),
             'legacy': ('active', 'legacy')}
    with tempfile.TemporaryDirectory(prefix='memory-joint-db-') as temp:
        database_root = Path(temp).resolve()
        for scene in scenes:
            facts = {s['content']: s.get('oracle_facts', []) for s in scene['sources']}
            for mode, (governance, synthesis) in modes.items():
                folder = Path(temp) / scene['id'] / mode
                folder.mkdir(parents=True)
                service, sdk = make_service(folder, facts, mode=governance, f1_mode=synthesis,
                    temporal_mode='off', input_log_enabled=False, governance_content_tokens=args.budget,
                    governance_query_budget_ms=1000, governance_summary_budget_ms=1000,
                    llm_search_expansion=True)
                service.llm.search_expansion = True
                sdk.query_output = {'expanded_query': '', 'retrieval_steps': []}
                vector_calls = []
                base_embed = service.embedder.embed

                def traced_embed(texts):
                    values = base_embed(texts)
                    vector_calls.append({'texts_sha256': digest(texts), 'vectors_sha256': digest(values.tolist())})
                    return values

                service.embedder.embed = traced_embed
                # Preserve explicitly shared write requests/adjacency from fixtures.
                groups = {}
                for source in scene['sources']:
                    key = (source.get('request_id', source['id']), source.get('session', source['id']))
                    groups.setdefault(key, []).append(source)
                for (request, session), sources in groups.items():
                    service.add(request, [MemoryMessage(role=s.get('role', 'user'), content=s['content'],
                        timestamp=timestamp(s.get('date'))) for s in sources], scene['id'], session)
                snapshot = service.store.fetch_by_user(scene['id'], governance=True)
                fact_view = [{'subject': f.subject_name, 'predicate': f.predicate, 'value': f.value,
                              'valid_from': f.valid_from, 'valid_to': f.valid_to,
                              'time_precision': f.time_precision, 'quote': f.quote} for f in snapshot.facts]
                for question in scene['questions']:
                    before, before_vectors = len(sdk.calls), len(vector_calls)
                    hits = service.search(question['query'], scene['id'], question.get('top_k', 1),
                        query_time_ms=timestamp(question.get('as_of', '2026-10-20')))
                    serialized = [asdict(h) for h in hits]
                    visible, omitted, count = visible_context(serialized, args.budget)
                    text = '\n'.join(h['content'] for h in visible)
                    checks = {}
                    for quote in question.get('required', []):
                        checks['required:' + quote] = quote in text
                    for quote in question.get('forbidden', []):
                        checks['forbidden:' + quote] = quote not in text
                    for a, b in question.get('ordered', []):
                        checks['ordered:' + a + ' < ' + b] = a in text and b in text and text.index(a) < text.index(b)
                    for expected in question.get('required_facts', []):
                        checks['fact:' + canonical(expected)] = any(all(f.get(k) == v for k, v in expected.items()) for f in fact_view)
                    for expected in question.get('forbidden_facts', []):
                        checks['not_fact:' + canonical(expected)] = not any(all(f.get(k) == v for k, v in expected.items()) for f in fact_view)
                    row = {'scenario': scene['id'], 'question': question['id'], 'mode': mode,
                           'category': scene.get('category', ''), 'hits': serialized, 'visible_hits': visible,
                           'expect_baseline_unchanged': question.get('expect_baseline_unchanged', False),
                           'expect_base_response': question.get('expect_base_response', False),
                           'omitted_ids': omitted, 'visible_tokens': count, 'declared_checks': checks,
                           'query_call_hashes': [digest(c) for c in sdk.calls[before:]],
                           'query_vector_calls': vector_calls[before_vectors:],
                           'facts': fact_view, 'incomplete_sources': len(snapshot.incomplete),
                           'answer_correct': None}
                    if 'gold' in question:
                        row['evidence_metrics'] = evidence_metrics(question['gold'], visible)
                    results.append(row)
    if any(isolation.values()) or 'app.main' in sys.modules:
        raise RuntimeError('Offline isolation checks failed')
    write(args.output, {'status': 'complete', 'real_model_evaluation': False,
                        'isolation': isolation, 'rows': results})
    return 0


def run_worker(root, dataset, output, budget, scenarios=()):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--root', str(root),
        '--dataset', str(dataset), '--output', str(output), '--budget', str(budget),
        *(arg for scene in scenarios for arg in ('--scenario', scene))],
        cwd=root, env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0'),
        capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        # This process uses only synthetic inputs and cannot contact providers.
        raise RuntimeError(f'Offline worker failed ({result.returncode}): {result.stderr[-3000:]}')
    return read(output)


def compare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(['git', 'rev-parse', '--verify', args.baseline + '^{commit}'], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    archive = subprocess.run(['git', 'archive', '--format=zip', revision], cwd=ROOT,
                             capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory(prefix='memory-joint-baseline-') as temp:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            bundle.extractall(temp)
        baseline = run_worker(Path(temp), args.dataset, args.output / 'baseline.json', args.budget, args.scenario)
    current = run_worker(ROOT, args.dataset, args.output / 'current.json', args.budget, args.scenario)
    key = lambda r: (r['scenario'], r['question'], r['mode'])
    left = {key(r): r for r in baseline['rows']}
    right = {key(r): r for r in current['rows']}
    if left.keys() != right.keys():
        raise ValueError('Incomplete paired results')
    paired = []
    compatibility = []

    def same_response(a, b):
        return all(a[k] == b[k] for k in ('hits', 'query_call_hashes', 'query_vector_calls'))

    for ident, b in right.items():
        a = left[ident]
        if ident[2] == 'full':
            if b['expect_baseline_unchanged']:
                compatibility.append({'scenario': ident[0], 'question': ident[1],
                    'check': 'declared_full_response_unchanged', 'passed': same_response(a, b)})
            if b['expect_base_response']:
                compatibility.append({'scenario': ident[0], 'question': ident[1],
                    'check': 'declared_ordinary_uses_base',
                    'passed': same_response(b, right[(ident[0], ident[1], 'off')])})
            old = a.get('evidence_metrics', {}).get('verbatim_atom_coverage')
            new = b.get('evidence_metrics', {}).get('verbatim_atom_coverage')
            changes = {k: ('improved' if v and not a['declared_checks'][k] else
                          'regressed' if not v and a['declared_checks'][k] else 'unchanged')
                       for k, v in b['declared_checks'].items()}
            paired.append({'scenario': ident[0], 'question': ident[1], 'category': b['category'],
                'baseline_quote_coverage': old, 'current_quote_coverage': new,
                'coverage_change': None if old is None else 'improved' if new > old else 'regressed' if new < old else 'unchanged',
                'check_changes': changes, 'baseline_checks': a['declared_checks'], 'current_checks': b['declared_checks'],
                'exact_response_unchanged': a['hits'] == b['hits'],
                'query_calls_unchanged': (a['query_call_hashes'], a['query_vector_calls']) == (b['query_call_hashes'], b['query_vector_calls'])})
        if ident[2] == 'off':
            compatibility.append({'scenario': ident[0], 'question': ident[1], 'check': 'baseline_off_vs_current_off',
                'passed': a['hits'] == b['hits'] and a['query_call_hashes'] == b['query_call_hashes']
                          and a['query_vector_calls'] == b['query_vector_calls']})
    for ident, row in right.items():
        counterpart = {'off': 'shadow', 'legacy': 'f1_shadow'}.get(ident[2])
        if counterpart:
            other = right[(ident[0], ident[1], counterpart)]
            compatibility.append({'scenario': ident[0], 'question': ident[1], 'check': ident[2] + '_vs_' + counterpart,
                'passed': row['hits'] == other['hits'] and row['query_call_hashes'] == other['query_call_hashes']
                          and row['query_vector_calls'] == other['query_vector_calls']})
    report = {'baseline_commit': revision, 'dataset_sha256': digest(read(args.dataset)), 'visible_budget': args.budget,
              'scenarios': args.scenario or 'all',
              'current_app_sha256': digest({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted((ROOT / 'app').rglob('*.py'))}),
              'real_model_evaluation': False, 'answer_status': 'not_run', 'answer_accuracy': None,
              'evaluation': 'Synthetic frozen-draft/vector replay; quote presence and declared structural checks only.',
              'pairs': paired, 'compatibility': compatibility,
              'coverage_changes': dict(Counter(r['coverage_change'] for r in paired if r['coverage_change'])),
              'declared_check_changes': dict(Counter(v for r in paired for v in r['check_changes'].values())),
              'current_failed_checks': [{'scenario': r['scenario'], 'question': r['question'],
                                         'check': k} for r in paired for k, v in r['current_checks'].items() if not v],
              'limitations': ['Development regression fixtures, not unseen generalization or official scores.',
                             'Passing declared checks does not establish arbitrary semantic correctness.']}
    write(args.output / 'comparison.json', report)
    failed = [c for c in compatibility if not c['passed']]
    regressions = [r for r in paired if r['coverage_change'] == 'regressed' or 'regressed' in r['check_changes'].values()]
    call_changes = [r for r in paired if not r['query_calls_unchanged']]
    print(canonical({'pairs': len(paired), 'compatibility_failures': len(failed), 'regressions': len(regressions),
                     'query_call_changes': len(call_changes), 'current_failed_checks': len(report['current_failed_checks']),
                     'coverage_changes': report['coverage_changes'], 'declared_check_changes': report['declared_check_changes']}))
    return 1 if failed or regressions or call_changes or report['current_failed_checks'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default=DEFAULT_BASELINE)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'tests/fixtures/memory_joint_cases.json')
    parser.add_argument('--budget', type=int, default=1500)
    parser.add_argument('--scenario', action='append', default=[], help='Only run this scenario ID; may be repeated.')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/joint_optimization/paired')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.dataset, args.output, args.root = args.dataset.resolve(), args.output.resolve(), args.root.resolve()
    if args.budget < 128:
        parser.error('--budget must be at least 128')
    return worker(args) if args.worker else compare(args)


if __name__ == '__main__':
    raise SystemExit(main())
