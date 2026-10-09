"""公开合成用例：固定原文/向量/扩展/参照和返回预算的 D1/F1 对照。

fixture 为人工声明的抽取草稿替身，检验解析与检索，不代表真实模型质量。
real 使用配置的模型独立抽取相同虚构消息；不读取官网评测日志。
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
from types import SimpleNamespace

import numpy as np

from app.config import Settings
from app.embeddings import build_embedder
from app.governance.extraction import token_count
from app.governance.retrieval import GovernanceRetriever
from app.llm import LLMConnection, OpenAICompatibleMemoryLLM, build_memory_llm
from app.schemas import MemoryMessage
from app.service import MemoryService
from app.storage import SQLiteMemoryStore


class FixtureSDK:
    def __init__(self, declarations):
        self.declarations = declarations
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        self.calls += 1
        payload = json.loads(kwargs['messages'][1]['content'])
        items = []
        for chunk in payload['chunks']:
            facts = self.declarations.get(chunk['text'], [])
            items.append({'chunk_id': chunk['chunk_id'], 'complete': True, 'facts': facts})
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps({'items': items}, ensure_ascii=False)))])


def section(text, label):
    marker = f'【{label}】'
    if marker not in text:
        return '' if '【' in text or text.startswith('[抽取式') else text
    return text.split(marker, 1)[1].split('【', 1)[0]


def evaluate(case, hits, budget):
    text = '\n'.join(h.content for h in hits)
    current = section(text, '当前状态')
    flags = all(flag in text for flag in case.get('required_flags', []))
    expected = all(value in current for value in case.get('current', []))
    stale = any(value in current for value in case.get('forbidden_current', []))
    quotes = case.get('required_quotes', [])
    covered = sum(q in text for q in quotes)
    return {'current_value_exact': expected and not stale and flags,
            'invalid_value_pollution': stale,
            'necessary_covered': covered, 'necessary_total': len(quotes),
            'required_flags_present': flags, 'returned_tokens': token_count(text),
            'budget_passed': token_count(text) <= budget}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('tests/fixtures/governance_cases.jsonl'))
    parser.add_argument('--extraction', choices=['fixture','real'], default='fixture')
    parser.add_argument('--report', type=Path, default=Path('data/governance_eval.json'))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--audit-output', type=Path, help='可选的合成数据抽取核验文件，包含已校验的来源片段')
    args = parser.parse_args()
    if not 5 <= args.repeats <= 20:
        parser.error('验收至少需要五轮交替测量，最多二十轮')
    cases = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
    settings = Settings(governance_mode='shadow', warmup_embedding_on_startup=False,
                        llm_add_enrichment=False, llm_search_expansion=False, temporal_extraction_mode='rules')
    embedder = build_embedder(settings)
    if args.extraction == 'real':
        llm = build_memory_llm(settings)
        if not llm.enabled:
            raise SystemExit('real 需要已配置的模型')
        sdk = None
    else:
        declarations = {}
        for case in cases:
            for source in case['sources']:
                declarations[source['content']] = [dict(f, quote=f.get('quote', source['content'])) for f in source.get('facts', [])]
        sdk = FixtureSDK(declarations)
        llm = OpenAICompatibleMemoryLLM(LLMConnection('fixture','https://example.invalid','fixture'),
                                        1, 0, False, False)
        llm._client = sdk
    modes = {'G0': (False, False), 'G1': (True, False), 'G2': (False, True), 'G3': (True, True)}
    rows, writes, audit = [], [], []
    compatible = True
    with tempfile.TemporaryDirectory(prefix='gov-eval-') as directory:
        store = SQLiteMemoryStore(Path(directory) / 'eval.db')
        writer = MemoryService(settings, store, embedder, llm)
        writer.initialize()
        for case in cases:
            user = 'gov-eval:' + case['id']
            raw_tokens = 0
            started = time.perf_counter()
            previous_calls = sdk.calls if sdk else None
            extra_calls, extraction_ms = 0, 0.0
            for index, source in enumerate(case['sources']):
                messages = []
                for repeat in range(source.get('repeat', 1)):
                    timestamp = source.get('timestamp')
                    if timestamp is not None:
                        timestamp += repeat * 86400000
                    messages.append(MemoryMessage(role=source.get('role','user'), content=source['content'], timestamp=timestamp))
                    raw_tokens += token_count(source['content'])
                writer.add(f'{case["id"]}:{index}', messages, user, f'session:{index}')
                extra_calls += writer.last_index_metrics.get('model_calls', 0)
                extraction_ms += writer.last_index_metrics.get('elapsed_ms', 0)
            snapshot = store.fetch_by_user(user, governance=True)
            if args.audit_output:
                audit.append({'case_id':case['id'], 'facts':[asdict(f) for f in snapshot.facts],
                              'statuses':[asdict(s) for s in snapshot.statuses]})
            writes.append({'case_id': case['id'], 'elapsed_ms': (time.perf_counter()-started)*1000,
                           'sources': len(snapshot.sources), 'incomplete': len(snapshot.incomplete),
                           'fact_calls': extra_calls, 'extraction_ms': extraction_ms,
                           'status_counts':{s:sum(x.status == s for x in snapshot.statuses)
                                            for s in ('ready','no_fact','partial','pending','failed')},
                           'error_counts':{e:sum(x.error_type == e for x in snapshot.statuses)
                                           for e in sorted({x.error_type for x in snapshot.statuses if x.error_type})},
                           'source_span_integrity': all(snapshot.sources[e.source_id].content[e.start:e.end] == e.quote
                               and hashlib.sha256(e.quote.encode()).hexdigest() == e.content_hash
                               for f in snapshot.facts for e in f.evidence)})
            services = {name: MemoryService(settings.model_copy(update={'governance_mode': 'off' if name=='G0' else 'active'}),
                        store, embedder, llm, governance_retriever=GovernanceRetriever(versions=v, summaries=s))
                        for name, (v,s) in modes.items()}
            shadow = MemoryService(settings, store, embedder, llm)
            query_ref = case.get('query_time_ms', 1748736000000)
            frozen = services['G0'].search(case['query'], user, 1, query_time_ms=query_ref)
            compatible &= shadow.search(case['query'], user, 1, query_time_ms=query_ref) == frozen
            # 各模式先预热一次，不将模型冷启动混入五轮本地治理开销。
            for service in services.values():
                service.search(case['query'], user, 1, query_time_ms=query_ref)
                service.search('Please recall the information in my notes.', user, 1, query_time_ms=query_ref)
            samples = {name: {'governance': [], 'ordinary': []} for name in modes}
            outputs = {}
            for repeat in range(args.repeats):
                names = list(modes)
                names = names[repeat % 4:] + names[:repeat % 4]
                if repeat % 2:
                    names.reverse()
                ordinary_baseline = services['G0'].search('Please recall the information in my notes.', user, 1, query_time_ms=query_ref)
                for name in names:
                    for kind, query in [('governance',case['query']), ('ordinary','Please recall the information in my notes.')]:
                        tick = time.perf_counter()
                        hits = services[name].search(query, user, 1, query_time_ms=query_ref)
                        samples[name][kind].append((time.perf_counter()-tick)*1000)
                        if kind == 'governance':
                            metrics = evaluate(case, hits, settings.governance_content_tokens)
                            if name not in outputs:
                                outputs[name] = dict(metrics, stable=True)
                            else:
                                stable = outputs[name]['stable'] and all(outputs[name][k] == v for k,v in metrics.items())
                                outputs[name]['current_value_exact'] &= metrics['current_value_exact']
                                outputs[name]['invalid_value_pollution'] |= metrics['invalid_value_pollution']
                                outputs[name]['necessary_covered'] = min(outputs[name]['necessary_covered'],metrics['necessary_covered'])
                                outputs[name]['required_flags_present'] &= metrics['required_flags_present']
                                outputs[name]['budget_passed'] &= metrics['budget_passed']
                                outputs[name]['returned_tokens'] = max(outputs[name]['returned_tokens'],metrics['returned_tokens'])
                                outputs[name]['stable'] = stable
                        else:
                            compatible &= hits == ordinary_baseline
            for name in modes:
                rows.append({'case_id': case['id'], 'group': case['group'], 'mode': name,
                             'raw_tokens': raw_tokens, **outputs[name], 'latency_samples_ms': samples[name]})
            print(f'CASE {case["id"]} sources={len(snapshot.sources)} incomplete={len(snapshot.incomplete)}', flush=True)
    summary = {}
    for name in modes:
        data = [r for r in rows if r['mode'] == name]
        d1 = [r for r in data if r['group'] == 'D1']
        f1 = [r for r in data if r['group'] == 'F1']
        latency = {kind: [x for r in data for x in r['latency_samples_ms'][kind]] for kind in ('ordinary','governance')}
        summary[name] = {'d1_current_exact': sum(r['current_value_exact'] for r in d1)/len(d1),
                         'd1_cases': len(d1), 'invalid_value_pollution': sum(r['invalid_value_pollution'] for r in data),
                         'f1_necessary_fact_coverage': sum(r['necessary_covered'] for r in f1)/sum(r['necessary_total'] for r in f1),
                         'f1_cases': len(f1), 'f1_input_to_output_ratio': sum(r['raw_tokens'] for r in f1)/max(1,sum(r['returned_tokens'] for r in f1)),
                         'all_return_budgets_passed': all(r['budget_passed'] for r in data),
                         'latency': {kind: {'median_ms': statistics.median(values), 'p95_ms': float(np.percentile(values,95))}
                                     for kind,values in latency.items()}}
    base = summary['G0']['latency']
    joint = summary['G3']['latency']
    gates = {'shadow_and_ordinary_equal': compatible,
             'source_integrity': all(w['source_span_integrity'] for w in writes),
             'complete_index': all(w['incomplete'] == 0 for w in writes),
             'ordinary_latency': joint['ordinary']['p95_ms']-base['ordinary']['p95_ms'] <= max(10,base['ordinary']['p95_ms']*.2),
             'governance_latency': joint['governance']['p95_ms']-base['governance']['p95_ms'] <= 50,
             'd1_not_below_baseline': summary['G3']['d1_current_exact'] >= summary['G0']['d1_current_exact'],
             'f1_not_below_baseline': summary['G3']['f1_necessary_fact_coverage'] >= summary['G0']['f1_necessary_fact_coverage']}
    gates['joint_stability'] = all(r['stable'] for r in rows if r['mode']=='G3')
    paired = {r['case_id']:r for r in rows if r['mode']=='G0'}
    gates['no_case_regression'] = all(
        r['current_value_exact'] >= paired[r['case_id']]['current_value_exact']
        and r['necessary_covered'] >= paired[r['case_id']]['necessary_covered']
        and r['invalid_value_pollution'] <= paired[r['case_id']]['invalid_value_pollution']
        for r in rows if r['mode']=='G3')
    gates['improves_baseline_failures'] = (summary['G3']['d1_current_exact'] > summary['G0']['d1_current_exact']
                                          and summary['G3']['f1_necessary_fact_coverage'] > summary['G0']['f1_necessary_fact_coverage'])
    report = {'identity': {'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                          'extraction': args.extraction, 'embedding_model': settings.embedding_model,
                          'repeats': args.repeats, 'validation_mode': 'shadow', 'top_k': 1,
                          'content_tokens': settings.governance_content_tokens,
                          'gold_review': 'independently declared synthetic labels; human review pending'},
              'summary': summary, 'gates': gates, 'writes': writes, 'cases': rows,
              'limitations': {'final_answer_quality': None, 'semantic_fact_accuracy': None,
                              'note': '检索证据代理指标；未运行官网或回答模型，来源片段校验不等于语义正确性。'},
              'activation': 'not performed; feature default remains off'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'summary': summary,'gates': gates}, ensure_ascii=False,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
