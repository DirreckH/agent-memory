"""Paired real-chain governance evaluation. No production database/config writes.

run: record real preprocessing once, replay identical outputs through frozen code,
     then answer with one fixed model and prompt. selftest uses explicit fixtures.
review: export per-case manual semantic audit; never grade by answer keywords.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import io
import inspect
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zipfile

from governance_e2e_support import (ANSWER_PROMPT, CompletionCache, InvalidCompletion, VectorCache,
    canonical, evidence_metrics, freeze_persistence_clock, namespace, read_json, reviewed_metrics, runtime_corpus, sha, validate_dataset,
    visible_context, write_json)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'tests/fixtures/governance_e2e_cases.json'
BASELINES = {'v3': '9f096cf', 'v4': 'e90b153', 'legacy': 'fe846f7'}
BASELINES.update({mode: '0ec14657f7aaf80b5ab873bb156d3b49b507d768' for mode in
                 ('active_baseline', 'baseline_off', 'baseline_shadow', 'baseline_f1_shadow')})
MODES = ('v3', 'v4', 'preserve', 'full', 'off', 'shadow', 'oracle')
F1_MODES = ('legacy', 'f1_plan', 'f1_pack', 'f1_shadow')
F1_UNITS_MODES = ('active_baseline', 'full', 'off', 'shadow', 'baseline_off',
                  'baseline_shadow', 'f1_shadow', 'baseline_f1_shadow')
FIXED_PERSISTENCE = '2025-07-01T00:00:00+00:00'


def date_ms(date, *, end=False):
    if date is None:
        return None
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp() * 1000) + (86399999 if end else 0)


def encode(value):
    if is_dataclass(value):
        return encode(asdict(value))
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        values = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [encode(v) for v in values]
    return value


def settings_manifest(settings):
    # Never serialize Settings wholesale: it contains secrets.
    names = ('embedding_provider', 'embedding_model', 'embedding_dimensions', 'embedding_threads',
             'semantic_weight', 'lexical_weight', 'min_relevance_score', 'max_top_k', 'candidate_pool_size',
             'temporal_mode', 'temporal_reference_mode', 'temporal_extraction_mode', 'temporal_weight',
             'temporal_decay_half_life_days', 'temporal_event_anchor_min_score', 'query_dual_channel',
             'multihop_enabled', 'multihop_structured_planner', 'multihop_embed_goals', 'multihop_context_radius',
             'multihop_max_rounds', 'multihop_seed_limit', 'multihop_bridge_limit', 'multihop_supplement_limit',
             'multihop_budget_seconds', 'multihop_min_top_k', 'governance_input_tokens', 'governance_output_tokens',
             'governance_add_budget_seconds', 'governance_summary_budget_ms', 'governance_query_budget_ms',
             'governance_content_tokens', 'llm_add_enrichment', 'llm_search_expansion', 'llm_timeout_seconds')
    return {k: getattr(settings, k) for k in names}


def fingerprints(settings):
    prefix = 'deepseek' if settings.llm_provider == 'deepseek' else 'openai'
    return {
        'llm': {'provider': settings.llm_provider, 'model': getattr(settings, prefix + '_model'),
                'endpoint_sha256': sha(getattr(settings, prefix + '_base_url'))},
        'embedding': {'provider': settings.embedding_provider, 'model': settings.embedding_model,
                      'dimensions': settings.embedding_dimensions,
                      'endpoint_sha256': sha(settings.dashscope_base_url) if settings.embedding_provider == 'dashscope' else None}}


def code_digest():
    return sha({p.relative_to(ROOT).as_posix(): sha(p.read_text(encoding='utf-8-sig'))
                for p in sorted((ROOT / 'app').rglob('*.py'))})


def prepare(args, settings):
    dataset = read_json(args.dataset)
    validate_dataset(dataset)
    runtime = runtime_corpus(dataset, args.split)
    if args.visible_tokens < 128:
        raise ValueError('visible-tokens must be at least 128')
    if args.command != 'selftest':
        if args.expect_embedding_model and settings.embedding_model != args.expect_embedding_model:
            raise ValueError('Embedding model differs from requested evaluation model')
        if settings.multihop_structured_planner or settings.multihop_embed_goals:
            raise ValueError('Paired V3 prompts require MULTIHOP_STRUCTURED_PLANNER=false and MULTIHOP_EMBED_GOALS=false')
    oracle = {x['content']: x['oracle_facts'] for s in dataset['scenarios'] for x in s['sources']}
    assert len(oracle) == sum(len(s['sources']) for s in dataset['scenarios']), 'Duplicate source text needs scoped oracle keys'
    modes = F1_UNITS_MODES if args.f1_units_comparison else MODES + F1_MODES if args.f1_comparison else MODES
    baselines = {m: r for m, r in BASELINES.items() if m in modes}
    commits = {mode: subprocess.run(['git', 'rev-parse', '--verify', revision + '^{commit}'], cwd=ROOT,
                capture_output=True, text=True, check=True).stdout.strip() for mode, revision in baselines.items()}
    manifest = {'schema': 1, 'dataset_sha256': sha(dataset), 'code_sha256': code_digest(), 'baselines': baselines,
                'modes': list(modes), 'f1_comparison': args.f1_comparison,
                'f1_units_comparison': args.f1_units_comparison,
                'baseline_commits': commits,
                'split': args.split, 'selftest': args.command == 'selftest', 'settings': settings_manifest(settings),
                'fingerprints': fingerprints(settings), 'visible_tokens': args.visible_tokens,
                'answer_prompt': ANSWER_PROMPT, 'answer_max_tokens': 768, 'temperature': 0,
                'persistence_clock': {'value': FIXED_PERSISTENCE, 'layers': ['service', 'storage']},
                'evaluation_overrides': {'llm_failure_mode': 'strict', 'llm_max_retries': 0,
                    'dashscope_max_retries': 0, 'governance_strict_write': False, 'input_log_enabled': False,
                    **({'governance_query_budget_ms': 1000, 'governance_summary_budget_ms': 1000,
                        'governance_content_tokens': args.visible_tokens}
                       if args.f1_comparison or args.f1_units_comparison else {})},
                'scenarios': len(runtime['scenarios']),
                'questions': sum(len(s['questions']) for s in runtime['scenarios']),
                'gold_review': dataset['authorship']}
    path = args.output / 'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Manifest differs. Use a new output directory; never mix models/code/data.')
    write_json(path, manifest)
    write_json(args.output / 'runtime.json', runtime)
    write_json(args.output / 'gold.json', dataset)
    write_json(args.output / 'oracle.json', oracle)
    return manifest


def validation_audit(folder, calls, source_lookup, oracle=None):
    """Read-only replay of individual validator decisions; no change to indexing."""
    from app.governance.extraction import FactIndexer
    from app.governance.models import Chunk, Source
    audit = []
    for call in calls:
        if call['kind'] != 'extraction':
            continue
        path = folder / 'cache/llm' / (call['key'] + '.json')
        if not path.exists():
            continue
        entry = read_json(path)
        payload = json.loads(entry['request']['messages'][1]['content'])
        try:
            response = json.loads(entry['response']['choices'][0]['message']['content'])
            items = response['items']
        except (KeyError, ValueError, TypeError):
            audit.append({'request': call['key'], 'error': 'InvalidExtractionJSON'})
            continue
        if not isinstance(items, list):
            audit.append({'request': call['key'], 'error': 'InvalidExtractionItems'})
            continue
        for c in payload['chunks']:
            original = source_lookup[c['source_id']]
            source = Source(c['source_id'], original['content'], c['role'], c['session_id'], c['ordinal'], c['source_timestamp'])
            chunk = Chunk(c['chunk_id'], source, c['start'], c['end'])
            matching = [i for i in items if isinstance(i, dict) and i.get('chunk_id') == c['chunk_id']]
            if any(not isinstance(i.get('facts'), list) for i in matching):
                audit.append({'request': call['key'], 'source_id': c['source_id'], 'error': 'InvalidFactsArray'})
            drafts = oracle.get(c['text'], []) if oracle is not None else [
                f for i in matching if isinstance(i.get('facts'), list) for f in i['facts']]
            for i, draft in enumerate(drafts):
                error = None
                try:
                    FactIndexer()._fact(draft, chunk)
                except (KeyError, ValueError, TypeError) as exc:
                    # Validator errors are fixed local codes; do not include arbitrary provider exceptions.
                    detail = str(exc)
                    error = detail if detail.isascii() and detail.isalpha() else type(exc).__name__
                audit.append({'request': call['key'], 'source_id': c['source_id'], 'draft_index': i,
                              'validator_accepted': error is None, 'error': error})
    return audit


class FixtureClient:
    """Harness smoke only. Never substitutes for a real model in command run."""
    def __init__(self, oracle):
        self.oracle = oracle
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        payload = json.loads(kwargs['messages'][1]['content'])
        if 'chunks' in payload:
            result = {'items': [{'chunk_id': c['chunk_id'], 'complete': c['text'] in self.oracle,
                                 'facts': self.oracle.get(c['text'], [])} for c in payload['chunks']]}
        elif 'messages' in payload:
            result = {'items': [{'source_index': i, 'search_text': m['content']}
                                for i, m in enumerate(payload['messages'])]}
        elif 'options' in payload:
            result = {'expanded_query': ''}
        else:
            result = {'temporal': None}
        return namespace({'choices': [{'finish_reason': 'stop', 'message': {'content': canonical(result)}}]})


class FixtureVectors:
    def embed(self, texts):
        import numpy as np
        import re
        # Deliberate mock. Hash collisions do not matter for harness control-flow testing.
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', text.casefold()):
                out[i, int(sha(token)[:8], 16) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def worker(args):
    # Historical app imports resolve only after sys.path is set. This script itself stays current.
    sys.path.insert(0, str(args.root))
    from app.config import Settings
    from app.embeddings import build_embedder
    from app.llm import LLMConnection, OpenAICompatibleMemoryLLM, build_memory_llm
    from app.schemas import MemoryMessage
    from app.service import MemoryService
    import app.service as service_module
    import app.storage as storage_module
    from app.storage import SQLiteMemoryStore
    manifest = read_json(args.output / 'manifest.json')
    runtime = read_json(args.output / 'runtime.json')
    oracle = read_json(args.output / 'oracle.json') if args.mode == 'oracle' or manifest['selftest'] else None
    freeze_persistence_clock(service_module, storage_module, FIXED_PERSISTENCE)
    record = args.record
    settings_values = dict(manifest['settings'], **manifest['evaluation_overrides'],
        governance_mode='off' if args.mode in ('v3', 'off', 'baseline_off') else 'active',
        warmup_embedding_on_startup=False, data_retention_days=3650)
    settings_values['f1_mode'] = ('shadow' if args.mode in ('f1_shadow', 'baseline_f1_shadow') else
        'active' if (manifest.get('f1_comparison') or manifest.get('f1_units_comparison'))
        and args.mode not in ('legacy', 'v3', 'v4') else 'legacy')
    model = manifest['fingerprints']['llm']['model']
    llm_options = {}
    if 'multihop_retrieval' in inspect.signature(OpenAICompatibleMemoryLLM).parameters:
        llm_options = dict(multihop_retrieval=settings_values['multihop_enabled'],
                           multihop_structured_planner=settings_values['multihop_structured_planner'])
    llm = OpenAICompatibleMemoryLLM(LLMConnection('evaluation-cache', 'https://cache.invalid', model),
        settings_values['llm_timeout_seconds'], 0, settings_values['llm_add_enrichment'],
        settings_values['llm_search_expansion'], **llm_options)
    real_settings = Settings().model_copy(update={**settings_values}) if record and not manifest['selftest'] else None
    client = None
    provider = None
    if record:
        if manifest['selftest']:
            client, provider = FixtureClient(oracle), FixtureVectors()
        else:
            real_llm = build_memory_llm(real_settings)
            if not real_llm.enabled:
                raise ValueError('LLMDisabled')
            client = real_llm._get_client()
            provider = build_embedder(real_settings)
    sdk = CompletionCache(args.output / 'cache/llm', manifest['fingerprints']['llm'], client=client,
        recording=record, oracle=oracle if args.mode == 'oracle' else None)
    vectors = VectorCache(args.output / 'cache/vectors', manifest['fingerprints']['embedding'],
        provider=provider, recording=record)
    llm._client = sdk
    result = {'mode': args.mode, 'status': 'running', 'scenarios': [], 'selftest': manifest['selftest']}
    destination = args.output / ('capture.json' if record else f'runs/{args.mode}.json')
    current_stage = 'initialization'
    try:
        with tempfile.TemporaryDirectory(prefix='memory-e2e-db-') as td:
            settings = Settings(_env_file=None, **settings_values, database_path=Path(td) / 'memory.db')
            if args.mode in ('shadow', 'baseline_shadow'):
                settings = settings.model_copy(update={'governance_mode': 'shadow'})
            kwargs = {}
            auditor = None
            if args.mode not in ('v3', 'off', 'baseline_off'):
                from app.governance.retrieval import GovernanceRetriever

                class AuditRetriever(GovernanceRetriever):
                    trace = None

                    def retrieve(self, snapshot, context):
                        self.trace = {'baseline': encode(context.baseline), 'reference': context.reference,
                                      'window': encode(context.window)}
                        if getattr(context, 'plan', None) is not None:
                            self.trace['plan'] = encode(context.plan)
                        hits = super().retrieve(snapshot, context)
                        return hits

                    def collect(self, snapshot, context, plan):
                        evidence = super().collect(snapshot, context, plan)
                        self.trace['governance_result'] = encode(evidence)
                        return evidence

                    def summarize(self, snapshot, context):
                        from dataclasses import replace
                        from app.governance.summary_planning import SummaryPlan, SummaryRequirement, build_summary_plan
                        from app.governance.summary_retrieval import collect_summary
                        from app.governance.summary_assembly import SummaryOutcome, assemble_summary
                        if args.mode == 'f1_plan':
                            p = build_summary_plan(context.query, snapshot, reference=context.reference, window=context.window)
                            outcome = SummaryOutcome()
                            if p.applicable:
                                # 只替换 F1 计划；证据收集、覆盖标记、装箱仍用冻结前的逻辑。
                                from app.governance.assembly import assemble
                                legacy_plan = replace(context.plan, intent='summary', slots=p.slots,
                                    reference=p.reference, window=p.window, knowledge_cutoff=p.knowledge_cutoff,
                                    scope_keys=frozenset(p.scopes), global_summary=p.global_scope,
                                    operations=('summarize',), requirements=('summarize',))
                                evidence = self.collect(snapshot, context, legacy_plan)
                                outcome = SummaryOutcome(tuple(assemble(evidence, context.baseline, snapshot,
                                    limit=context.limit, token_budget=context.settings.governance_content_tokens)))
                        elif args.mode == 'f1_pack':
                            old = context.plan
                            p = SummaryPlan()
                            outcome = SummaryOutcome()
                            if old.intent == 'summary':
                                # 保持旧触发条件、槽位与时间解释，只替换来源覆盖和压缩。
                                p = SummaryPlan(True, context.query, tuple(sorted(old.scope_keys)), old.slots,
                                    old.global_summary or not old.scope_keys,
                                    (SummaryRequirement('overview', '相关经历', context.query),),
                                    old.reference, old.window, old.knowledge_cutoff)
                                collection = collect_summary(snapshot, p)
                                if not old.global_summary and not old.scope_keys:
                                    # 旧计划没有范围目录时，仍保全其原本会考虑的基础原文和槽位来源。
                                    # 不能把“仅覆盖压缩”消融错误实现成丢弃所有 no_fact 基础结果。
                                    allowed = {h.id for h in context.baseline} | {
                                        f.source_id for f in snapshot.facts if f.slot in old.slots}
                                    collection = replace(collection, evidence=tuple(e for e in collection.evidence
                                        if all(s.source_id in allowed for s in e.spans)))
                                outcome = assemble_summary(collection, p, context.baseline,
                                    snapshot, limit=context.limit, token_budget=context.settings.governance_content_tokens)
                        else:
                            p, outcome = super().summarize(snapshot, context)
                        self.trace['f1_plan'] = encode(p)
                        self.trace['f1_outcome'] = encode(outcome)
                        return p, outcome

                auditor = AuditRetriever(preserve_only=True) if args.mode == 'preserve' else AuditRetriever()
                kwargs['governance_retriever'] = auditor
            service = MemoryService(settings, SQLiteMemoryStore(settings.database_path), vectors, llm, **kwargs)
            service.initialize()
            for scene in runtime['scenarios']:
                progress = {'id': scene['id'], 'split': scene['split'], 'category': scene['category'],
                            'source_ids': {}, 'writes': [], 'queries': [], 'validation_audit': []}
                result['scenarios'].append(progress)
                print(f"{args.mode}: {scene['id']}", file=sys.stderr, flush=True)
                for group_no, (session, sources) in enumerate(itertools.groupby(scene['sources'], key=lambda s: s['session'])):
                    group = list(sources)
                    request_id = f"e2e:{scene['id']}:{group_no}"
                    messages = [MemoryMessage(role=s['role'], content=s['content'], timestamp=date_ms(s['date'])) for s in group]
                    progress['source_ids'].update({s['id']: service._memory_id(request_id, i) for i, s in enumerate(group)})
                    current_stage = 'write_enrichment_embedding_extraction'
                    before = len(sdk.stats['calls'])
                    service.add(request_id, messages, scene['id'], session)
                    progress['writes'].append({'request_id': request_id, 'calls': sdk.stats['calls'][before:]})
                    if sdk.stats['errors'] or vectors.stats['errors']:
                        raise InvalidCompletion('Upstream stage failed; no fallback comparison')
                    if args.mode not in ('v3', 'off', 'baseline_off'):
                        lookup = {service._memory_id(request_id, i): s for i, s in enumerate(group)}
                        progress['validation_audit'] += validation_audit(args.output, sdk.stats['calls'][before:],
                            lookup, oracle if args.mode == 'oracle' else None)
                if args.mode not in ('v3', 'off', 'baseline_off'):
                    snapshot = service.store.fetch_by_user(scene['id'], governance=True)
                    progress['snapshot'] = {name: encode(getattr(snapshot, name)) for name in
                        ('user_id', 'generation', 'revision', 'sources', 'facts', 'relations', 'statuses', 'summaries')}
                    progress['incomplete_sources'] = list(snapshot.incomplete)
                for question in scene['questions']:
                    current_stage = 'query_expansion_embedding_retrieval'
                    before = len(sdk.stats['calls'])
                    if auditor:
                        auditor.trace = None
                    hits = service.search(question['query'], scene['id'], question['top_k'],
                                          query_time_ms=date_ms(question['as_of'], end=True))
                    if sdk.stats['errors'] or vectors.stats['errors']:
                        raise InvalidCompletion('Upstream stage failed; no fallback comparison')
                    all_hits = encode(hits)
                    visible, omitted, count = visible_context(all_hits, manifest['visible_tokens'])
                    progress['queries'].append({'id': question['id'], 'query': question['query'],
                        'hits': all_hits, 'visible_hits': visible, 'omitted_ids': omitted, 'visible_tokens': count,
                        'calls': sdk.stats['calls'][before:], 'trace': auditor.trace if auditor else None})
                write_json(destination, result)
            result['status'] = 'complete'
    except Exception as exc:
        result.update(status='blocked', error_type=type(exc).__name__, failed_stage=current_stage)
    result['completion_stats'] = sdk.stats
    result['embedding_stats'] = vectors.stats
    result['live_llm_calls'] = sdk.stats['remote'] if not manifest['selftest'] else 0
    write_json(destination, result)
    print(canonical({'status': result['status'], 'mode': args.mode}))
    return 0 if result['status'] == 'complete' else 2


def launch_worker(args, mode, *, record=False):
    with tempfile.TemporaryDirectory(prefix='memory-e2e-code-') as td:
        root = ROOT
        if mode in BASELINES:
            data = subprocess.run(['git', 'archive', '--format=zip', BASELINES[mode]], cwd=ROOT,
                                  check=True, capture_output=True).stdout
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                archive.extractall(td)
            root = Path(td)
        cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--root', str(root),
               '--output', str(args.output), '--mode', mode]
        if record:
            cmd.append('--record')
        run = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0'),
                             capture_output=True, text=True, encoding='utf-8')
        # stderr may contain upstream error strings; never print it or persist it wholesale.
        path = args.output / ('capture.json' if record else f'runs/{mode}.json')
        if not path.exists() or run.returncode not in (0, 2):
            write_json(path, {'mode': mode, 'status': 'blocked', 'failed_stage': 'worker_start',
                              'error_type': 'WorkerProcessError', 'exit_code': run.returncode, 'scenarios': []})
        return read_json(path)


def answer(args, manifest, runs):
    from app.config import Settings
    from app.llm import build_memory_llm
    settings = Settings().model_copy(update={'llm_max_retries': 0})
    if fingerprints(settings)['llm'] != manifest['fingerprints']['llm']:
        raise ValueError('Answer model differs from frozen manifest')
    llm = build_memory_llm(settings)
    sdk = CompletionCache(args.output / 'cache/answers', manifest['fingerprints']['llm'],
        client=llm._get_client(), recording=True)
    output = {'status': 'running', 'human_review': 'pending', 'cases': []}
    try:
        # Holdout/development and all algorithms use exactly the same answering prompt/budget.
        for mode in manifest.get('modes', MODES):
            if mode in ('off', 'shadow', 'f1_shadow'):
                continue
            for scene in runs[mode]['scenarios']:
                for question in scene['queries']:
                    payload = {'question': question['query'], 'memories': question['visible_hits']}
                    response = sdk.create(model=manifest['fingerprints']['llm']['model'],
                        messages=[{'role': 'system', 'content': ANSWER_PROMPT},
                                  {'role': 'user', 'content': canonical(payload)}],
                        response_format={'type': 'json_object'}, temperature=0,
                        max_tokens=manifest['answer_max_tokens'], stream=False)
                    choice = response.choices[0]
                    if choice.finish_reason != 'stop':
                        raise InvalidCompletion('Truncated answer')
                    parsed = json.loads(choice.message.content)
                    if (not isinstance(parsed, dict) or not isinstance(parsed.get('answer'), str)
                            or not isinstance(parsed.get('sources'), list)
                            or not all(isinstance(s, str) for s in parsed['sources'])):
                        raise InvalidCompletion('Invalid answer schema')
                    output['cases'].append({'mode': mode, 'scenario': scene['id'], 'question': question['id'],
                                             'response': parsed, 'correct': None})
                    write_json(args.output / 'answers.json', output)
        output['status'] = 'awaiting_semantic_review'
    except Exception as exc:
        output.update(status='blocked', error_type=type(exc).__name__)
    output['stats'] = sdk.stats
    write_json(args.output / 'answers.json', output)
    return output


def build_report(folder):
    manifest = read_json(folder / 'manifest.json')
    gold = read_json(folder / 'gold.json')
    modes = manifest.get('modes', MODES)
    runs = {mode: read_json(folder / f'runs/{mode}.json') for mode in modes if (folder / f'runs/{mode}.json').exists()}
    answers = read_json(folder / 'answers.json') if (folder / 'answers.json').exists() else {'status': 'not_run', 'cases': []}
    answer_map = {(a['mode'], a['scenario'], a['question']): a for a in answers['cases']}
    query_map = {(mode, s['id'], q['id']): q for mode, run in runs.items() for s in run['scenarios'] for q in s['queries']}
    rows = []
    review = []
    for scene in gold['scenarios']:
        if manifest['split'] not in ('all', scene['split']):
            continue
        for question in scene['questions']:
            values = {}
            for mode in modes:
                actual = query_map.get((mode, scene['id'], question['id']))
                if actual is None:
                    values[mode] = {'status': 'not_completed'}
                    continue
                metrics = evidence_metrics(question['gold'], actual['visible_hits'])
                values[mode] = {'status': 'retrieved', 'metrics': metrics,
                               'answer': answer_map.get((mode, scene['id'], question['id'])),
                               'visible_tokens': actual['visible_tokens']}
                review.append({'mode': mode, 'scenario': scene['id'], 'question': question['id'],
                    'answer_correct': None, 'necessary_evidence_valid': None, 'wrong_current_state': None,
                    'qualifiers_preserved': None, 'first_failure_stage': None, 'reason': '',
                    'reviewer': '', 'evidence_locations': []})
            rows.append({'scenario': scene['id'], 'split': scene['split'], 'category': scene['category'],
                         'question': question['id'], 'query': question['query'], 'gold': question['gold'], 'modes': values})
    compatibility = {}
    checks = [('off', 'shadow')]
    if 'v3' in modes:
        checks.append(('v3', 'off'))
    if manifest.get('f1_comparison'):
        checks.append(('legacy', 'f1_shadow'))
    if manifest.get('f1_units_comparison'):
        checks += [('baseline_off', 'off'), ('baseline_shadow', 'shadow'), ('baseline_f1_shadow', 'f1_shadow')]
    for left, right in checks:
        pairs = [(query_map.get((left, r['scenario'], r['question'])), query_map.get((right, r['scenario'], r['question']))) for r in rows]
        complete = bool(pairs) and all(a is not None and b is not None for a, b in pairs)
        compatibility[f'{left}_vs_{right}'] = (all(a['hits'] == b['hits'] and a['calls'] == b['calls']
            for a, b in pairs) if complete else None)
    capture_path = folder / 'capture.json'
    capture = read_json(capture_path) if capture_path.exists() else {}
    captured = {(s['id'], q['id']): q for s in capture.get('scenarios', []) for q in s['queries']}
    if capture.get('status') == 'complete' and runs.get('full', {}).get('status') == 'complete':
        compatibility['capture_vs_replay_full'] = all(
            captured[(r['scenario'], r['question'])]['hits'] == query_map[('full', r['scenario'], r['question'])]['hits']
            for r in rows)
    else:
        compatibility['capture_vs_replay_full'] = None
    report = {'manifest': manifest, 'run_statuses': {m: r['status'] for m, r in runs.items()},
              'answer_status': answers['status'], 'compatibility': compatibility, 'cases': rows,
              'final_answer_accuracy': None, 'overall_quality_conclusion': 'unverified',
              'real_model_evaluation': not manifest['selftest']}
    if manifest.get('f1_comparison') or manifest.get('f1_units_comparison'):
        baseline_mode = 'active_baseline' if manifest.get('f1_units_comparison') else 'legacy'
        paired = []
        untouched = []
        for row in rows:
            key = (row['scenario'], row['question'])
            base, full = query_map.get((baseline_mode, *key)), query_map.get(('full', *key))
            if not base or not full:
                continue
            applicable = full.get('trace', {}).get('f1_plan', {}).get('applicable', False)
            if not applicable:
                untouched.append(base['hits'] == full['hits'] and base['calls'] == full['calls'])
            a = evidence_metrics(row['gold'], base['visible_hits'])['verbatim_atom_coverage']
            b = evidence_metrics(row['gold'], full['visible_hits'])['verbatim_atom_coverage']
            paired.append({'scenario': key[0], 'question': key[1], 'f1_applicable': applicable,
                'baseline_mode': baseline_mode, 'baseline_quote_coverage': a, 'full_quote_coverage': b,
                **({'legacy_quote_coverage': a} if baseline_mode == 'legacy' else {}),
                'hits_identical': base['hits'] == full['hits'], 'calls_identical': base['calls'] == full['calls'],
                'evidence_change': 'improved' if b > a else 'regressed' if b < a else 'unchanged',
                'answer_quality': 'unverified'})
        compatibility[f'{baseline_mode}_vs_full_untouched'] = all(untouched) if untouched else None
        report['f1_evidence_pairs'] = paired
        report['quote_coverage_nonregression'] = all(r['evidence_change'] != 'regressed' for r in paired)
    # Never overwrite a reviewer-edited file.
    if not (folder / 'review.json').exists():
        write_json(folder / 'review.json', {'rubric_version': 1, 'items': review})
    else:
        existing = read_json(folder / 'review.json')
        existing_keys = {(i['mode'], i['scenario'], i['question']) for i in existing['items']}
        existing['items'] += [i for i in review if (i['mode'], i['scenario'], i['question']) not in existing_keys]
        write_json(folder / 'review.json', existing)
    report['semantic_review'] = reviewed_metrics(read_json(folder / 'review.json'), rows, set(answer_map),
                                                selftest=manifest['selftest'])
    write_json(folder / 'comparison.json', report)
    lines = ['# 逐题核验包', '', '**用途：人工语义核验。精确引文覆盖不是最终答案正确率。**', '',
             f"运行类型：{'程序自检（替身模型）' if manifest['selftest'] else '真实模型输出冻结对照'}。",
             f"回答阶段：{answers['status']}。", '',
             '失败环节填写：extraction / planning / version / relation / retrieval / assembly / answer / none / undetermined。',
             '仅在前一环节证据已核实时填写“第一个失败环节”；关键词出现不构成通过。', '',
             '修订 review.json 中的布尔值、理由、核验者和证据位置；保持未验证项为 null。', '']
    lines += ['## 人工核验汇总', '', '| 模式 | 开发集：正确/已评/总数 | 保留集：正确/已评/总数 |',
              '|---|---:|---:|']
    for mode, splits in report['semantic_review']['by_mode_split'].items():
        cells = [f"{splits[s]['correct']}/{splits[s]['reviewed']}/{splits[s]['expected']}" for s in ('development', 'holdout')]
        lines.append(f'| {mode} | ' + ' | '.join(cells) + ' |')
    lines += ['', '0 项已评表示未验证，不是正确率为零。逐题改善/退步见 comparison.json 的 semantic_review。', '']
    for row in rows:
        lines += [f"## {row['scenario']}/{row['question']}（{row['split']}）", '', row['query'], '',
                  f"**人工预设答案：** {row['gold']['answer']}", '', f"**理由：** {row['gold']['reason']}", '',
                  '必要原文：', '']
        for atom in row['gold']['atoms']:
            groups = [atom['support'], *atom.get('support_alternatives', [])]
            lines += [f"- {atom['claim']}：" + ' 或 '.join(' + '.join(f"{s['source_id']}「{s['quote']}」" for s in group) for group in groups)]
        lines += ['', '禁止当作真实结论：' + '；'.join(row['gold']['forbidden_claims']), '']
        lines += ['误导来源：' + '；'.join(f"{d['source_id']}：{d['risk']}" for d in row['gold'].get('distractors', [])), '']
        for mode in modes:
            if mode in ('off', 'shadow', 'f1_shadow'):
                continue
            value = row['modes'][mode]
            lines += [f'### {mode}', '']
            if value['status'] != 'retrieved':
                lines += ['未完成。', '']; continue
            answer_value = value['answer']
            lines += [f"逐字引文覆盖：{value['metrics']['verbatim_atom_coverage']:.1%}；实际可见 token：{value['visible_tokens']}。", '',
                      '模型答案：' + (answer_value['response']['answer'] if answer_value else '未执行'), '', '实际返回：', '']
            actual = query_map[(mode, row['scenario'], row['question'])]
            lines += ['```text', canonical(actual['visible_hits']), '```', '',
                      f"完整来源、事实、版本关系与查询计划：runs/{mode}.json → {row['scenario']}。", '']
    (folder / 'REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('validate', 'run', 'selftest', 'review', 'worker'))
    p.add_argument('--dataset', type=Path, default=DATASET)
    p.add_argument('--output', type=Path, default=ROOT / 'data/governance_e2e')
    p.add_argument('--split', choices=('all', 'development', 'holdout'), default='all')
    p.add_argument('--visible-tokens', type=int, default=8192)
    p.add_argument('--skip-answers', action='store_true', help='Only capture/replay retrieval; answers explicitly remain unverified.')
    p.add_argument('--f1-comparison', action='store_true',
                   help='Compare fe846f7, F1 planning only, F1 coverage only, and combined; preserve D1.')
    p.add_argument('--f1-units-comparison', action='store_true',
                   help='Primary baseline: frozen 0ec1465 with GOV/F1 active; also compare off/shadow paths.')
    p.add_argument('--expect-embedding-model', help='Fail before calls if the configured embedding model differs.')
    p.add_argument('--mode', choices=tuple(dict.fromkeys(MODES + F1_MODES + F1_UNITS_MODES)),
                   default='full', help=argparse.SUPPRESS)
    p.add_argument('--record', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.f1_comparison and args.f1_units_comparison:
        p.error('Select one baseline comparison at a time.')
    args.output = args.output.resolve()
    sys.stdout.reconfigure(encoding='utf-8')
    if args.command == 'worker':
        try:
            return worker(args)
        except Exception as exc:
            destination = args.output / ('capture.json' if args.record else f'runs/{args.mode}.json')
            write_json(destination, {'mode': args.mode, 'status': 'blocked', 'error_type': type(exc).__name__,
                                     'failed_stage': 'worker_initialization', 'scenarios': []})
            return 2
    sys.path.insert(0, str(ROOT))
    if args.command == 'validate':
        dataset = read_json(args.dataset)
        validate_dataset(dataset)
        print(canonical({'status': 'valid', 'dataset_sha256': sha(dataset), 'scenarios': len(dataset['scenarios']),
                         'questions': sum(len(s['questions']) for s in dataset['scenarios'])}))
        return 0
    if args.command == 'review':
        report = build_report(args.output)
        print(canonical({'report': str(args.output / 'REVIEW.md'), 'answer_status': report['answer_status']}))
        return 0
    from app.config import Settings
    settings = Settings(_env_file=None, llm_provider='none') if args.command == 'selftest' else Settings()
    manifest = prepare(args, settings)
    capture = launch_worker(args, 'full', record=True)
    if capture['status'] != 'complete':
        build_report(args.output)
        print(canonical({'status': 'blocked', 'stage': capture.get('failed_stage'),
                         'error_type': capture.get('error_type'), 'report': str(args.output / 'capture.json')}))
        return 2
    runs = {}
    for mode in manifest['modes']:
        runs[mode] = launch_worker(args, mode)
        print(canonical({'mode': mode, 'status': runs[mode]['status']}), flush=True)
        if runs[mode]['status'] != 'complete':
            build_report(args.output)
            return 2
    report = build_report(args.output)
    # Earlier independent multihop can differ from V3. Shadow must preserve current off exactly.
    if (report['compatibility']['off_vs_shadow'] is not True or
            report['compatibility']['capture_vs_replay_full'] is not True or
            report['compatibility'].get('legacy_vs_f1_shadow') is False or
            report['compatibility'].get('legacy_vs_full_untouched') is False or
            any(value is False for key, value in report['compatibility'].items()
                if key.startswith(('baseline_', 'active_baseline_')))):
        print(canonical({'status': 'compatibility_failed', 'checks': report['compatibility']}))
        return 3
    if args.command != 'selftest' and not args.skip_answers:
        result = answer(args, manifest, runs)
        build_report(args.output)
        if result['status'] == 'blocked':
            return 2
    print(canonical({'status': 'harness_passed' if manifest['selftest'] else 'awaiting_semantic_review',
                     'report': str(args.output / 'REVIEW.md')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
