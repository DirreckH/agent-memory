"""在现有合成用例上冻结查询计划，比较旧检索、分步召回与原文桥接。

real 使用配置的真实 LLM；oracle 只用用例自带无答案扩展，不能作为生产效果证据。
所有模式写入一次、共享原文和向量。规划与检索延迟分别记录；本地桥接不调用 LLM。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from app.config import Settings
from app.embeddings import build_embedder
from app.llm import NoOpMemoryLLM, QueryExpansion, build_memory_llm
from app.multihop import parse_retrieval_steps, steps_from_expansion
from app.service import MemoryService
from app.storage import SQLiteMemoryStore
from scripts.benchmark_common import load_multihop_cases
from scripts.run_multihop_eval import (
    MEASUREMENT_NOW_MS, _add_case, evaluate_hits, generate_fillers, summarize,
)


class FrozenPlanner(NoOpMemoryLLM):
    enabled = True

    def __init__(self, plans):
        self.plans = plans

    def expand_query(self, query, options):
        return self.plans[query]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('tests/fixtures/multihop_cases.jsonl'))
    parser.add_argument('--llm', choices=['real', 'oracle'], default='oracle')
    parser.add_argument('--report', type=Path, default=Path('data/multihop_regression.json'))
    parser.add_argument('--cache', type=Path, default=Path('data/multihop_plan_cache.json'))
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=1, help='交替模式顺序重复检索，测量耗时波动')
    parser.add_argument('--governance-mode', choices=['off','shadow'], default='off')
    parser.add_argument('--structured-planner', action='store_true', help='额外生成结构化目标；默认复用 V3 扩展')
    parser.add_argument('--reuse-legacy-cache', type=Path,
                        help='默认 real 模式复用已保存的真实 V3 响应；缺项时停止，不调用模型')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error('--repeats 必须介于 1 和 20')
    cases = load_multihop_cases(args.dataset)
    settings = Settings(multihop_enabled=True, multihop_structured_planner=args.structured_planner,
                        governance_mode=args.governance_mode)
    embedder = build_embedder(settings)
    provider = build_memory_llm(settings) if args.llm == 'real' else None
    legacy_provider = build_memory_llm(settings.model_copy(update={'multihop_enabled': False})) if provider else None
    if provider is not None and not provider.enabled:
        raise SystemExit('real 模式需要已配置的 LLM')

    fingerprint = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    from app.prompts import QUERY_EXPANSION_PROMPT_MULTIHOP, QUERY_EXPANSION_PROMPT_V3
    prompt_hash = hashlib.sha256((QUERY_EXPANSION_PROMPT_MULTIHOP + QUERY_EXPANSION_PROMPT_V3).encode()).hexdigest()
    identity = {'dataset_sha256': fingerprint, 'prompt_sha256': prompt_hash,
                'governance_mode': args.governance_mode,
                'structured_planner': args.structured_planner,
                'planning_schema': 'legacy-expansion-v1',
                'llm_mode': args.llm, 'llm_provider': settings.llm_provider,
                'llm_model': settings.openai_model if settings.llm_provider == 'openai' else settings.deepseek_model}
    reuse = {}
    reuse_identity = None
    if args.reuse_legacy_cache:
        if args.llm != 'real' or args.structured_planner:
            raise SystemExit('--reuse-legacy-cache 仅用于默认 real 模式')
        source = json.loads(args.reuse_legacy_cache.read_text(encoding='utf-8'))
        source_identity = source.get('identity', {})
        if any(source_identity.get(key) != identity[key] for key in (
                'dataset_sha256', 'prompt_sha256', 'llm_mode', 'llm_provider', 'llm_model')):
            raise SystemExit('V3 响应缓存的数据集、提示词或模型不匹配')
        reuse = source.get('legacy_plans', {})
        reuse_identity = {'path': str(args.reuse_legacy_cache),
                          'sha256': hashlib.sha256(args.reuse_legacy_cache.read_bytes()).hexdigest()}
    cached = {}
    legacy_cached = {}
    if args.cache.exists():
        previous = json.loads(args.cache.read_text(encoding='utf-8'))
        if previous.get('identity') == identity:
            cached = previous.get('plans', {})
            legacy_cached = previous.get('legacy_plans', {})
    plans = {}
    legacy_plans = {}
    planning_ms = {}
    legacy_planning_ms = {}
    fresh_planning = {}
    plan_sources = {}
    for case in cases:
        started = time.perf_counter()
        fresh_planning[case.case_id] = (case.case_id not in cached and not args.reuse_legacy_cache)
        if case.case_id in cached:
            item = cached[case.case_id]
            expansion = QueryExpansion(item['text'], None, parse_retrieval_steps(item['steps'], case.query))
            origin = item.get('origin', 'model' if provider else 'oracle')
            plan_sources[case.case_id] = {'mode': 'cache', 'origin': origin}
        elif args.reuse_legacy_cache:
            item = reuse.get(case.case_id)
            if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                raise SystemExit(f'V3 响应缓存缺少 {case.case_id}；未调用模型')
            expansion = QueryExpansion(item['text'], None, steps_from_expansion(case.query, item['text']))
            origin = 'legacy_response_cache'
            plan_sources[case.case_id] = {'mode': 'cache', 'origin': origin}
        elif provider is not None:
            expansion = provider.expand_query(case.query, None)
            origin = 'model'
            plan_sources[case.case_id] = {'mode': 'live', 'origin': origin}
        else:
            steps = [{'query': part.strip(), 'evidence': case.query}
                     for part in case.oracle_expansion.replace('；', ';').split(';') if part.strip()]
            expansion = QueryExpansion(case.oracle_expansion, None, parse_retrieval_steps(steps, case.query))
            origin = 'oracle'
            plan_sources[case.case_id] = {'mode': 'generated', 'origin': origin}
        plans[case.query] = expansion
        planning_ms[case.case_id] = round((time.perf_counter() - started) * 1000, 3)
        started = time.perf_counter()
        if case.case_id in legacy_cached:
            legacy = QueryExpansion(legacy_cached[case.case_id]['text'], None)
        elif legacy_provider is not None and args.structured_planner:
            legacy = legacy_provider.expand_query(case.query, None)
        elif legacy_provider is not None:
            # 默认模式的模型请求与 V3 一致，复用同一响应避免模型随机性混入对照。
            legacy = QueryExpansion(expansion.text, None)
        else:
            legacy = QueryExpansion(case.oracle_expansion, None)
        legacy_plans[case.query] = legacy
        legacy_planning_ms[case.case_id] = round((time.perf_counter() - started) * 1000, 3)
        if provider is not None and not args.structured_planner:
            legacy_planning_ms[case.case_id] = planning_ms[case.case_id]
        legacy_cached[case.case_id] = {'text': legacy.text}
        cached[case.case_id] = {'text': expansion.text,
                               'steps': [asdict(step) for step in expansion.retrieval_steps],
                               'origin': origin}
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps({'identity': identity, 'plans': cached, 'legacy_plans': legacy_cached}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'PLAN {case.case_id} steps={len(expansion.retrieval_steps)}', flush=True)

    wrappers = {name: FrozenPlanner(legacy_plans if name == 'legacy' else plans)
                for name in ('legacy', 'baseline', 'fanout', 'grounded')}
    results = {name: [] for name in wrappers}
    with tempfile.TemporaryDirectory(prefix='mh-regression-') as directory:
        store = SQLiteMemoryStore(Path(directory) / 'eval.db')
        writer = MemoryService(settings, store, embedder, NoOpMemoryLLM())
        writer.initialize()
        services = {}
        for name, llm in wrappers.items():
            config = settings.model_copy(update={
                'multihop_enabled': name not in ('baseline', 'legacy'),
                'multihop_max_rounds': 0 if name == 'fanout' else settings.multihop_max_rounds,
                'multihop_embed_goals': name == 'fanout' or settings.multihop_embed_goals,
                'multihop_context_radius': 0 if name == 'fanout' else settings.multihop_context_radius,
            })
            services[name] = MemoryService(config, store, embedder, llm)
        for case_index, case in enumerate(cases):
            _add_case(writer, case, list(case.memories) + generate_fillers(case), now_ms=MEASUREMENT_NOW_MS)
            case_rows = {}
            for repeat in range(args.repeats):
                order = list(services)
                shift = (case_index + repeat) % len(order)
                for name in order[shift:] + order[:shift]:
                    started = time.perf_counter()
                    hits = services[name].search(case.query, f'mh:{case.case_id}', args.top_k)
                    elapsed = round((time.perf_counter() - started) * 1000, 3)
                    metrics = evaluate_hits([hit.content for hit in hits], case)
                    if name not in case_rows:
                        case_rows[name] = {'case_id': case.case_id, 'hops': len(case.hops),
                                           **metrics, 'latency_samples_ms': [],
                                           'bridge_calls': 0, 'bridge_failures': 0}
                    elif any(case_rows[name][key] != value for key, value in metrics.items()):
                        raise RuntimeError(f'冻结计划重复检索结果不稳定: {case.case_id} {name}')
                    case_rows[name]['latency_samples_ms'].append(elapsed)
            for name, row in case_rows.items():
                row['latency_ms'] = statistics.median(row['latency_samples_ms'])
                results[name].append(row)
                print(f'CASE {case.case_id} {name} chain100={row["chain_at"]["100"]} median_ms={row["latency_ms"]:.1f}', flush=True)

    modes = {}
    for name, rows in results.items():
        latencies = sorted(sample for row in rows for sample in row['latency_samples_ms'])
        summary = summarize(rows)
        summary.update({'median_latency_ms': statistics.median(latencies),
                        'p95_latency_ms': latencies[max(0, int(len(latencies) * 0.95 + 0.999) - 1)],
                        'bridge_calls': sum(row['bridge_calls'] for row in rows),
                        'bridge_failures': sum(row['bridge_failures'] for row in rows)})
        modes[name] = {'summary': summary, 'per_case': rows}
    regressions = {}
    for name in ('fanout', 'grounded'):
        regressions[name] = {str(k): [b['case_id'] for b, a in zip(results['baseline'], results[name], strict=True)
                                     if b['chain_at'][str(k)] and not a['chain_at'][str(k)]]
                             for k in (10, 20, 100)}
    regressions['grounded_vs_legacy'] = {str(k): [b['case_id'] for b, a in zip(results['legacy'], results['grounded'], strict=True)
                                                if b['chain_at'][str(k)] and not a['chain_at'][str(k)]]
                                       for k in (10, 20, 100)}
    hop_regressions = {}
    for reference in ('baseline', 'legacy'):
        hop_regressions[reference] = {str(k): [
            {'case_id': b['case_id'], 'hop': position + 1}
            for b, a in zip(results[reference], results['grounded'], strict=True)
            for position, (old, new) in enumerate(zip(b['hop_ranks'], a['hop_ranks'], strict=True))
            if old is not None and old <= k and (new is None or new > k)
        ] for k in (1, 5, 10, 20, 100)}
    paired_deltas = [new - old
                     for baseline, grounded in zip(results['legacy'], results['grounded'], strict=True)
                     for old, new in zip(baseline['latency_samples_ms'], grounded['latency_samples_ms'], strict=True)]
    report = {'identity': identity, 'dataset': str(args.dataset), 'top_k': args.top_k,
              'repeats': args.repeats, 'latency_sample_count_per_mode': len(cases) * args.repeats,
              'paired_median_latency_delta_ms': statistics.median(paired_deltas),
              'embedding_provider': settings.embedding_provider, 'embedding_model': settings.embedding_model,
              'valid_plan_count': sum(bool(p.retrieval_steps) for p in plans.values()),
              'planning_ms': planning_ms, 'modes': modes, 'regressions': regressions,
              'legacy_planning_ms': legacy_planning_ms, 'fresh_planning': fresh_planning,
              'plan_sources': plan_sources, 'reused_legacy_cache': reuse_identity,
              'hop_regressions': hop_regressions,
              'multihop_settings': {key: value for key, value in settings.model_dump().items() if key.startswith('multihop_')},
              'scope': 'Synthetic fixture; shared raw store/embeddings and frozen plans. Search latency excludes planning; no answer-model scoring.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({name: data['summary'] for name, data in modes.items()}, ensure_ascii=True), flush=True)
    print(f'REPORT={args.report.resolve()}', flush=True)
    return 1 if any(ids for groups in (list(regressions.values()) + list(hop_regressions.values()))
                    for ids in groups.values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
