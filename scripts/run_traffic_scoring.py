"""真实流量评分模拟：生产 DB 副本 + input.jsonl 真实查询，离线模拟评测平台对 /get 的评分。

证据层（零外部调用）：命中结构、治理/F1 路径前缀、降级标记、空结果、延迟对照（原始 vs 重放）。
答案层（--answers real）：visible_context 装箱 → ANSWER_PROMPT + 配置的 OpenAI 模型
  （CompletionCache 冻结，重跑免费；每查询 1 次调用，注意 ChatAnywhere 配额）。
评分计算（无金标，评测方不给逐题标签）：证据溯源自动校验 grounding_metrics——
  引用必须命中可见 ID、实质句必须被证据词面支持；拒答句（信息不足类）不计为无据。

用法：
  PYTHONPATH=. .venv/bin/python scripts/run_traffic_scoring.py \\
    --database ~/traffic/agent_memory.db.bak-20261016 \\
    --logs ~/traffic/logs/input.jsonl.2026-10-16 \\
    --output data/traffic_scoring_20261016

优化方向对照：同一流量以不同 --governance/--f1 开关跑到不同 output 目录，比较聚合报告。
安全：源库只读复制为副本后使用；旁有 -wal 或与 .env DATABASE_PATH 相同时要求 --allow-live。
零改动 app/：服务在本脚本内装配（与 run_governance_e2e worker 同款模式）。
退出码：0=完成；2=blocked（库/日志/密钥问题）。
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # 同时支持“作为脚本运行”与“作为包模块被测试导入”
    from scripts.governance_e2e_support import (ANSWER_PROMPT, CompletionCache, VectorCache,
                                               canonical, sha)
except ImportError:  # pragma: no cover - 直接以文件方式运行时
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from governance_e2e_support import (ANSWER_PROMPT, CompletionCache, VectorCache,
                                       canonical, sha)

from app.config import Settings
from app.embeddings import build_embedder
from app.llm import LLMConnection, OpenAICompatibleMemoryLLM, build_memory_llm
from app.service import MemoryService
from app.storage import SQLiteMemoryStore

SECTION_MARKERS = ('【当前状态】', '【历史变化】', '【事件与决定】', '【计划】',
                   '【冲突与纠正】', '【待确认原文】')
SENTENCE_SPLIT = re.compile(r'[。！？.!?\n]')
DECLINE_RE = re.compile(r'信息不足|无法|不确定|冲突|没有足够|insufficient|cannot', re.I)
TOKEN_RE = re.compile(r'[A-Za-z0-9]+|[\u4e00-\u9fff]')
TOKEN_SUPPORT_THRESHOLD = 0.85


def ts_to_ms(ts: str) -> int:
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:  # 输入日志一律 UTC
        from datetime import timezone
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def load_traffic(log_paths, *, users=None, limit=None, dedup=False) -> list[dict]:
    """只取 /get 且 200 的记录；顺序：用户过滤 → 去重 → 截断。"""
    records: list[dict] = []
    for log_path in log_paths:
        with Path(log_path).open(encoding='utf-8') as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get('path') != '/get' or rec.get('status') != 200:
                    continue  # /set、bot 探针、422 等一律跳过
                try:
                    payload = json.loads(rec.get('body') or '')
                except ValueError:
                    continue
                if not isinstance(payload, dict):
                    continue
                query = payload.get('query')
                user_id = payload.get('user_id')
                if not isinstance(query, str) or not isinstance(user_id, str):
                    continue
                if users and user_id not in users:
                    continue
                records.append({'ts': rec.get('ts'), 'ts_ms': ts_to_ms(rec['ts']),
                                'query': query, 'user_id': user_id,
                                'top_k': payload.get('top_k') or 5,
                                'original_duration_ms': rec.get('duration_ms'),
                                'source': str(log_path), 'line_no': line_no})
    if dedup:
        seen, unique = set(), []
        for rec in records:
            key = (rec['query'], rec['user_id'], rec['top_k'])
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        records = unique
    if limit is not None:
        records = records[:limit]
    return records


def evidence_metrics_for_hits(hits: list[dict]) -> dict:
    """证据层结构指标：不依赖金标，也不依赖语义判断。"""
    count = len(hits)
    first = hits[0] if hits else None
    prefix = (first['id'].split('_', 1)[0] + '_') if first else '-'
    content = first.get('content', '') if first else ''
    sections = [m for m in SECTION_MARKERS if m in content]
    return {'hits': count, 'first_id_prefix': prefix,
            'first_score': first.get('score') if first else None,
            'content_chars': len(content), 'sections': sections,
            'degraded': '【待确认原文】' in content, 'empty_result': count == 0}


def grounding_metrics(answer_payload: dict, visible_hits: list[dict]) -> dict:
    """证据溯源自动校验：词面诊断，不等于语义正确性。

    - 引用的 source ID 必须出现在可见命中里；
    - 实质句按 token 覆盖率 ≥0.85 判定 supported；
    - 拒答句（信息不足/无法/冲突类）标记 declined，不计入无据。
    """
    answer = answer_payload.get('answer') or ''
    sources = answer_payload.get('sources') or []
    evidence_text = '\n'.join(h.get('content', '') for h in visible_hits)
    evidence_tokens = set(TOKEN_RE.findall(evidence_text.casefold()))
    known_ids = {h.get('id') for h in visible_hits}
    unknown = [s for s in sources if s not in known_ids]

    sentences, substantive, supported, declined = [], 0, 0, 0
    for raw in SENTENCE_SPLIT.split(answer):
        text = raw.strip()
        if len(text) < 4:
            continue
        substantive += 1
        if DECLINE_RE.search(text):
            declined += 1
            sentences.append({'text': text[:80], 'status': 'declined'})
            continue
        tokens = TOKEN_RE.findall(text.casefold())
        if not tokens:
            supported += 1
            sentences.append({'text': text[:80], 'status': 'supported', 'token_support': 1.0})
            continue
        ratio = sum(t in evidence_tokens for t in tokens) / len(tokens)
        ok = ratio >= TOKEN_SUPPORT_THRESHOLD
        supported += ok
        sentences.append({'text': text[:80], 'status': 'supported' if ok else 'unsupported',
                          'token_support': round(ratio, 3)})

    flags = []
    if unknown:
        flags.append('ungrounded_citation')
    if any(s['status'] == 'unsupported' for s in sentences):
        flags.append('unsupported_claims')
    answer_declined = substantive > 0 and declined == substantive
    if not visible_hits and not answer_declined:
        flags.append('empty_evidence')
    checkable = substantive - declined
    return {'cited_unknown_sources': unknown, 'sentences': sentences,
            'substantive_sentences': substantive, 'supported_sentences': supported,
            'declined_sentences': declined, 'declined': answer_declined,
            'support_ratio': round(supported / checkable, 3) if checkable else None,
            'flags': flags,
            'warning': '词面溯源是词法诊断，不等于语义正确性。'}


def check_database_safety(database: Path, env_settings: Settings, *, allow_live: bool) -> None:
    if not database.exists():
        raise SystemExit(f'blocked: database not found: {database}')
    live = env_settings.database_path.resolve() == database.resolve()
    wal = database.with_name(database.name + '-wal')
    if (live or wal.exists()) and not allow_live:
        raise SystemExit('blocked: source database looks live (matches DATABASE_PATH or has a '
                         '-wal sibling). Pass --allow-live only if you accept the risk.')


def copy_database(source: Path, target: Path) -> None:
    """sqlite backup API 复制（含 WAL 内容）；源库零写入。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(source)
    dst = sqlite3.connect(target)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()


def expand_log_files(patterns) -> list[Path]:
    """展开 --logs 参数：绝对/相对路径与通配符均可；glob.glob 支持绝对路径模式。"""
    import glob as globlib
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(match) for match in globlib.glob(str(Path(str(pattern)).expanduser())))
    return sorted(set(files))


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    if len(values) < 2:
        return float(values[0])
    return float(statistics.quantiles(values, n=100, method='inclusive')[p - 1])


def score_traffic(settings: Settings, traffic: list[dict], output: Path, *,
                  embedder, llm, answer_sdk=None, visible_tokens: int = 8192) -> dict:
    """核心评分循环。answer_sdk=None 表示跳过答案层。"""
    output.mkdir(parents=True, exist_ok=True)
    service = MemoryService(settings, SQLiteMemoryStore(settings.database_path), embedder, llm)
    service.initialize()

    queries, answer_errors = [], 0
    for index, rec in enumerate(traffic, 1):
        started = time.perf_counter()
        status, error_type = 'ok', None
        try:
            hits = service.search(rec['query'], rec['user_id'], rec['top_k'],
                                  query_time_ms=rec['ts_ms'])
        except Exception as exc:  # noqa: BLE001 - 单条失败不中断整体
            hits, status, error_type = [], 'error', type(exc).__name__
        replay_ms = round((time.perf_counter() - started) * 1000, 2)
        all_hits = [{'id': h.id, 'content': h.content, 'score': h.score,
                     'created_at': h.created_at} for h in hits]
        entry = {'index': index, **rec, 'status': status, 'error_type': error_type,
                 'replay_ms': replay_ms, 'hits': all_hits,
                 'evidence': evidence_metrics_for_hits(all_hits)}
        if answer_sdk is not None:
            answer, answer_status = None, 'ok'
            if all_hits:
                visible, omitted, tokens = _visible(all_hits, visible_tokens)
                try:
                    answer, answer_status = _generate_answer(answer_sdk, settings, rec, visible)
                    entry['visible_hits'] = visible
                    entry['omitted_ids'] = omitted
                    entry['visible_tokens'] = tokens
                except Exception as exc:  # noqa: BLE001
                    answer_status, answer_errors = f'error:{type(exc).__name__}', answer_errors + 1
            else:
                answer = {'answer': '', 'sources': []}
            if isinstance(answer, dict) and answer_status == 'ok':
                entry['answer'] = answer
                entry['grounding'] = grounding_metrics(answer, entry.get('visible_hits', []))
            else:
                entry['answer_status'] = answer_status
        queries.append(entry)
        print(f"[{index}/{len(traffic)}] {rec['query'][:48]!r} hits={entry['evidence']['hits']} "
              f"prefix={entry['evidence']['first_id_prefix']}", flush=True)
    return {'queries': queries, 'answer_errors': answer_errors}


def _visible(all_hits: list[dict], budget: int):
    import tiktoken
    encoding = tiktoken.get_encoding('cl100k_base')
    visible, omitted = [], []
    for hit in all_hits:
        if len(encoding.encode(canonical(visible + [hit]), disallowed_special=())) <= budget:
            visible.append(hit)
        else:
            omitted.append(hit['id'])
    return visible, omitted, len(encoding.encode(canonical(visible), disallowed_special=()))


def _generate_answer(answer_sdk, settings: Settings, rec: dict, visible: list[dict]):
    response = answer_sdk.create(model=settings.openai_model,
                                 messages=[{'role': 'system', 'content': ANSWER_PROMPT},
                                           {'role': 'user', 'content': canonical(
                                               {'question': rec['query'], 'memories': visible})}],
                                 response_format={'type': 'json_object'}, temperature=0,
                                 max_tokens=768, stream=False)
    choice = response.choices[0]
    if choice.finish_reason != 'stop':
        raise ValueError('Truncated answer')
    parsed = json.loads(choice.message.content)
    if (not isinstance(parsed, dict) or not isinstance(parsed.get('answer'), str)
            or not isinstance(parsed.get('sources'), list)
            or not all(isinstance(s, str) for s in parsed['sources'])):
        raise ValueError('Invalid answer schema')
    return parsed, 'ok'


def aggregate(result: dict, traffic: list[dict]) -> dict:
    queries = result['queries']
    prefix_counts: dict[str, int] = {}
    degraded = empty = errors = grounded_ok = declined = 0
    support_ratios: list[float] = []
    original: list[float] = []
    replay: list[float] = []
    for entry in queries:
        evidence = entry['evidence']
        prefix_counts[evidence['first_id_prefix']] = prefix_counts.get(evidence['first_id_prefix'], 0) + 1
        degraded += bool(evidence['degraded'])
        empty += bool(evidence['empty_result'])
        errors += entry['status'] != 'ok'
        if isinstance(entry.get('original_duration_ms'), (int, float)):
            original.append(float(entry['original_duration_ms']))
        replay.append(entry['replay_ms'])
        grounding = entry.get('grounding')
        if grounding is not None:
            declined += bool(grounding['declined'])
            if grounding['support_ratio'] is not None:
                support_ratios.append(grounding['support_ratio'])
                grounded_ok += not grounding['flags']
    total = len(queries) or 1
    def latency(values):
        return {'p50_ms': round(percentile(values, 50), 1), 'p90_ms': round(percentile(values, 90), 1),
                'max_ms': round(max(values), 1) if values else 0.0}
    return {'queries': len(queries), 'traffic_records': len(traffic),
            'path_prefixes': prefix_counts,
            'degraded_rate': round(degraded / total, 3),
            'empty_result_rate': round(empty / total, 3),
            'query_errors': errors,
            'answer_errors': result['answer_errors'],
            'grounding': {'graded': len(support_ratios), 'no_flag_rate': round(grounded_ok / len(support_ratios), 3)
                          if support_ratios else None,
                          'mean_support_ratio': round(sum(support_ratios) / len(support_ratios), 3)
                          if support_ratios else None, 'declined': declined},
            'latency': {'original_from_log': latency(original), 'replay_local': latency(replay)}}


def write_report(output: Path, manifest: dict, agg: dict, queries: list[dict]) -> None:
    lines = ['# 真实流量评分模拟报告', '',
             '**用途：离线模拟评测方对 /get 的评分。溯源是词面诊断，不等于语义正确率。**', '',
             f"数据库: `{manifest['database']}`（只读副本）",
             f"流量: {manifest['log_count']} 个日志文件 / {agg['traffic_records']} 条记录 / "
             f"评分 {agg['queries']} 条",
             f"治理: GOVERNANCE_MODE={manifest['governance_mode']}  F1_MODE={manifest['f1_mode']}  "
             f"LLM={manifest['llm']}  答案层={manifest['answers']}", '',
             '## 聚合', '',
             f"- 路径前缀分布: {agg['path_prefixes']}",
             f"- 降级率（含【待确认原文】）: {agg['degraded_rate']:.1%}；空结果率: {agg['empty_result_rate']:.1%}",
             f"- 查询异常: {agg['query_errors']}；答案层异常: {agg['answer_errors']}",
             f"- 溯源: 已评分 {agg['grounding']['graded']} 条，无标记率 "
             f"{agg['grounding']['no_flag_rate']}，平均支持率 {agg['grounding']['mean_support_ratio']}，"
             f"拒答 {agg['grounding']['declined']} 条",
             f"- 延迟(原始日志): p50={agg['latency']['original_from_log']['p50_ms']}ms "
             f"p90={agg['latency']['original_from_log']['p90_ms']}ms "
             f"max={agg['latency']['original_from_log']['max_ms']}ms",
             f"- 延迟(本地重放): p50={agg['latency']['replay_local']['p50_ms']}ms "
             f"p90={agg['latency']['replay_local']['p90_ms']}ms "
             f"max={agg['latency']['replay_local']['max_ms']}ms", '',
             '## 逐题', '',
             '| # | 查询（截断） | 用户 | 命中 | 前缀 | 降级 | 支持率 | 标记 |', '|---|---|---|---:|---|---|---|---|']
    for entry in queries:
        evidence = entry['evidence']
        grounding = entry.get('grounding') or {}
        ratio = grounding.get('support_ratio') if grounding else None
        flags = ','.join(grounding.get('flags', [])) if grounding else ''
        lines.append(f"| {entry['index']} | {entry['query'][:44]} | {entry['user_id'][:10]}… "
                     f"| {evidence['hits']} | {evidence['first_id_prefix']} | "
                     f"{'是' if evidence['degraded'] else '-'} | {ratio if ratio is not None else '-'} "
                     f"| {flags or '-'} |")
    (output / 'TRAFFIC_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True, help='生产 DB 副本路径（只读复制）')
    parser.add_argument('--logs', type=Path, nargs='+', required=True, help='input.jsonl 日志文件')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='最多评分条数')
    parser.add_argument('--users', help='逗号分隔的 user_id 过滤')
    parser.add_argument('--dedup', action='store_true', help='按 query+user+top_k 去重')
    parser.add_argument('--llm', choices=('real', 'off'), default='off',
                        help='real=查询扩展/时间抽取走真实 LLM（消耗配额）；off=零外部调用')
    parser.add_argument('--answers', choices=('real', 'skip'), default='skip',
                        help='real=答案层模拟（每查询 1 次 ChatAnywhere 调用）')
    parser.add_argument('--governance', choices=('off', 'shadow', 'active'))
    parser.add_argument('--f1', choices=('legacy', 'shadow', 'active'))
    parser.add_argument('--visible-tokens', type=int, default=8192)
    parser.add_argument('--allow-live', action='store_true',
                        help='源库疑似生产库时仍继续（不推荐）')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    env_settings = Settings(_env_file=str(ROOT / '.env'))
    check_database_safety(args.database.resolve(), env_settings, allow_live=args.allow_live)
    if env_settings.embedding_provider != 'dashscope':
        raise SystemExit('blocked: 查询向量必须与库内向量同源（dashscope:text-embedding-v4）')

    log_files = expand_log_files(args.logs)
    if not log_files:
        raise SystemExit(f'blocked: no log files found: {[str(p) for p in args.logs]}')
    traffic = load_traffic(log_files, users=set(args.users.split(',')) if args.users else None,
                           limit=args.limit, dedup=args.dedup)
    if not traffic:
        raise SystemExit('blocked: no /get traffic found in logs')
    print(f"[INFO] 流量: {len(traffic)} 条 /get 查询（来自 {len(log_files)} 个文件）")

    work_db = args.output / 'work.db'
    copy_database(args.database.resolve(), work_db)
    print(f"[INFO] DB 副本: {work_db}")

    values = dict(_env_file=str(ROOT / '.env'), database_path=work_db,
                  input_log_enabled=False, warmup_embedding_on_startup=False,
                  embedding_cache_dir=args.output / 'cache_dir')
    if args.governance:
        values['governance_mode'] = args.governance
    if args.f1:
        values['f1_mode'] = args.f1
    if args.llm == 'off':
        values['llm_provider'] = 'none'
    settings = Settings(**values)

    embedder = build_embedder(env_settings)
    vectors = VectorCache(args.output / 'cache/vectors',
                          {'provider': 'dashscope', 'model': env_settings.embedding_model},
                          provider=embedder, recording=True)
    if args.llm == 'real':
        real = build_memory_llm(Settings(_env_file=str(ROOT / '.env'), llm_max_retries=0))
        if not real.enabled:
            raise SystemExit('blocked: --llm real 需要 .env 配置可用的 LLM')
        fingerprint = {'provider': 'openai', 'model': env_settings.openai_model,
                       'endpoint': sha(env_settings.openai_base_url)}
        service_llm = OpenAICompatibleMemoryLLM(
            LLMConnection('traffic-cache', 'https://cache.invalid', env_settings.openai_model),
            env_settings.llm_timeout_seconds, 0, env_settings.llm_add_enrichment,
            env_settings.llm_search_expansion,
            multihop_retrieval=env_settings.multihop_enabled,
            multihop_structured_planner=env_settings.multihop_structured_planner)
        service_llm._client = CompletionCache(args.output / 'cache/llm', fingerprint,
                                               client=real._get_client(), recording=True)
    else:
        service_llm = build_memory_llm(settings)

    answer_sdk = None
    if args.answers == 'real':
        if not (env_settings.openai_api_key and
                env_settings.openai_api_key.get_secret_value().strip()):
            raise SystemExit('blocked: --answers real 需要 .env 配置 OPENAI_API_KEY')
        fingerprint = {'provider': 'openai', 'model': env_settings.openai_model,
                       'endpoint': sha(env_settings.openai_base_url)}
        answer_llm = build_memory_llm(Settings(_env_file=str(ROOT / '.env'), llm_max_retries=0))
        answer_sdk = CompletionCache(args.output / 'cache/answers', fingerprint,
                                     client=answer_llm._get_client(), recording=True)
        print('[INFO] 答案层已启用：每条查询 1 次 LLM 调用（CompletionCache 冻结，重跑免费）')

    manifest = {'database': str(args.database), 'logs': [str(p) for p in log_files],
                'log_count': len(log_files), 'governance_mode': settings.governance_mode,
                'f1_mode': settings.f1_mode, 'llm': args.llm, 'answers': args.answers,
                'embedding': {'provider': 'dashscope', 'model': env_settings.embedding_model},
                'answer_model': env_settings.openai_model if answer_sdk else None,
                'visible_tokens': args.visible_tokens, 'dedup': args.dedup, 'limit': args.limit}
    (args.output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    result = score_traffic(settings, traffic, args.output, embedder=vectors, llm=service_llm,
                           answer_sdk=answer_sdk, visible_tokens=args.visible_tokens)
    agg = aggregate(result, traffic)
    report = {'manifest': manifest, 'aggregate': agg, 'queries': result['queries']}
    (args.output / 'traffic_scoring.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(args.output, manifest, agg, result['queries'])
    print(f"[DONE] 报告: {args.output / 'TRAFFIC_REVIEW.md'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
