"""离线测试：真实流量评分模拟的确定性行为；不调用任何真实提供方。"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.governance_e2e_support import CompletionCache, canonical
from scripts.run_traffic_scoring import (aggregate, check_database_safety, copy_database,
                                         evidence_metrics_for_hits, expand_log_files,
                                         grounding_metrics, load_traffic, score_traffic,
                                         ts_to_ms, write_report)
from app.config import Settings
from app.llm import build_memory_llm
from app.schemas import MemoryMessage
from app.service import MemoryService
from app.storage import SQLiteMemoryStore
from tests.test_api import DeterministicTestEmbedder

TS = '2026-10-16T08:00:00.123+00:00'


def get_record(query, user='u1', top_k=5, status=200, ts=TS):
    body = {'query': query, 'user_id': user}
    if top_k is not None:
        body['top_k'] = top_k
    return {'ts': ts, 'method': 'POST', 'path': '/get', 'status': status,
            'duration_ms': 123.45, 'body_chars': len(canonical(body)),
            'query': query, 'user_id': user, 'top_k': top_k, 'body': canonical(body)}


def write_log(path: Path, records) -> Path:
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n',
                    encoding='utf-8')
    return path


def test_ts_to_ms_matches_expected_value():
    expected = int(datetime.fromisoformat(TS).timestamp() * 1000)
    assert ts_to_ms(TS) == expected


def test_expand_log_files_supports_absolute_glob_and_missing(tmp_path, monkeypatch):
    first = write_log(tmp_path / 'input.jsonl.2026-10-25', [get_record('q')])
    second = write_log(tmp_path / 'input.jsonl.2026-10-26', [get_record('q')])
    # 绝对具体路径
    assert expand_log_files([str(first)]) == [first]
    # 绝对通配符（回归：Path('.').glob 不支持绝对模式）
    assert expand_log_files([str(tmp_path / 'input.jsonl.2026-*')]) == [first, second]
    # 相对模式与不存在路径
    monkeypatch.chdir(tmp_path)
    assert [p.resolve() for p in expand_log_files(['input.jsonl.2026-*'])] == [first, second]
    assert expand_log_files(['/nonexistent/x.jsonl']) == []


def test_load_traffic_keeps_only_successful_get(tmp_path):
    log = write_log(tmp_path / 'input.jsonl', [
        get_record('Where do I work?'),
        {'ts': TS, 'method': 'POST', 'path': '/set', 'status': 200, 'body': '{}'},
        {'ts': TS, 'method': 'POST', 'path': '/mcp', 'status': 404, 'body': ''},
        get_record('broken', status=422),
        {'ts': TS, 'method': 'POST', 'path': '/get', 'status': 200, 'body': 'not-json'},
        {'ts': TS, 'method': 'POST', 'path': '/get', 'status': 200, 'body': '{"query": 1}'},
    ])
    traffic = load_traffic([log])
    assert len(traffic) == 1
    rec = traffic[0]
    assert rec['query'] == 'Where do I work?' and rec['user_id'] == 'u1'
    assert rec['top_k'] == 5 and rec['original_duration_ms'] == 123.45
    assert rec['ts_ms'] == ts_to_ms(TS)


def test_load_traffic_without_top_k_defaults_to_five(tmp_path):
    log = write_log(tmp_path / 'input.jsonl', [get_record('q', top_k=None)])
    assert load_traffic([log])[0]['top_k'] == 5


def test_load_traffic_users_dedup_limit(tmp_path):
    log = write_log(tmp_path / 'input.jsonl', [
        get_record('q1', user='u1'), get_record('q1', user='u1'),
        get_record('q2', user='u2'), get_record('q3', user='u1'),
    ])
    assert [r['query'] for r in load_traffic([log], users={'u1'})] == ['q1', 'q1', 'q3']
    assert [r['query'] for r in load_traffic([log], dedup=True)] == ['q1', 'q2', 'q3']
    assert [r['query'] for r in load_traffic([log], dedup=True, limit=2)] == ['q1', 'q2']


HITS = [{'id': 'mem_abc', 'content': 'I work at Acme Corp as a software engineer.', 'score': 0.5}]


def test_grounding_supported_answer_has_no_flags():
    metrics = grounding_metrics(
        {'answer': 'I work at Acme Corp as a software engineer.', 'sources': ['mem_abc']}, HITS)
    assert metrics['support_ratio'] == 1.0
    assert metrics['flags'] == [] and metrics['declined'] is False


def test_grounding_flags_unknown_citation():
    metrics = grounding_metrics(
        {'answer': 'I work at Acme Corp.', 'sources': ['mem_missing']}, HITS)
    assert metrics['cited_unknown_sources'] == ['mem_missing']
    assert 'ungrounded_citation' in metrics['flags']


def test_grounding_declined_answer_is_not_penalized():
    metrics = grounding_metrics(
        {'answer': '根据现有记忆，信息不足，无法确定职位。', 'sources': []}, HITS)
    assert metrics['declined'] is True
    assert metrics['support_ratio'] is None
    assert metrics['flags'] == []


def test_grounding_mixed_answer_reports_partial_support():
    metrics = grounding_metrics(
        {'answer': 'I work at Acme Corp as a software engineer. '
                   'My favorite color is purple unicorn rainbow.',
         'sources': []}, HITS)
    assert metrics['substantive_sentences'] == 2
    assert metrics['support_ratio'] == 0.5
    assert 'unsupported_claims' in metrics['flags']


def test_grounding_empty_evidence_with_answer_is_flagged():
    metrics = grounding_metrics({'answer': 'The answer is Beta Inc.', 'sources': []}, [])
    assert 'empty_evidence' in metrics['flags']


def test_evidence_metrics_prefix_sections_and_degraded():
    metrics = evidence_metrics_for_hits(
        [{'id': 'gov_x', 'content': '头\n【当前状态】\nx\n【待确认原文】\ny', 'score': 1.0}])
    assert metrics['first_id_prefix'] == 'gov_' and metrics['degraded'] is True
    assert '【当前状态】' in metrics['sections'] and '【计划】' not in metrics['sections']
    assert evidence_metrics_for_hits(
        [{'id': 'f1_y', 'content': 'z', 'score': 0.9}])['first_id_prefix'] == 'f1_'
    empty = evidence_metrics_for_hits([])
    assert empty['empty_result'] is True and empty['first_id_prefix'] == '-'


def test_check_database_safety_blocks_live_wal(tmp_path):
    db = tmp_path / 'agent_memory.db'
    db.write_bytes(b'sqlite')
    db.with_name(db.name + '-wal').write_bytes(b'wal')
    settings = Settings(_env_file=None, database_path=db)
    with pytest.raises(SystemExit, match='allow-live'):
        check_database_safety(db, settings, allow_live=False)
    check_database_safety(db, settings, allow_live=True)  # 显式放行不拦截


def test_check_database_safety_blocks_matching_env_path(tmp_path):
    db = tmp_path / 'agent_memory.db'
    db.write_bytes(b'sqlite')
    settings = Settings(_env_file=None, database_path=db)
    with pytest.raises(SystemExit, match='allow-live'):
        check_database_safety(db, settings, allow_live=False)


def test_copy_database_reads_source_without_modifying_it(tmp_path):
    source = tmp_path / 'src.db'
    con = sqlite3.connect(source)
    con.execute('CREATE TABLE t (v TEXT)')
    con.execute("INSERT INTO t VALUES ('x')")
    con.commit()
    con.close()
    before = source.read_bytes()
    target = tmp_path / 'copy.db'
    copy_database(source, target)
    con = sqlite3.connect(target)
    assert con.execute('SELECT v FROM t').fetchone() == ('x',)
    con.close()
    assert source.read_bytes() == before


def _seed_settings(tmp_path):
    return Settings(_env_file=None, database_path=tmp_path / 'work.db', llm_provider='none',
                    governance_mode='off', llm_search_expansion=False,
                    llm_add_enrichment=False, temporal_extraction_mode='rules',
                    min_relevance_score=0.0)


def _seed_memories(settings):
    service = MemoryService(settings, SQLiteMemoryStore(settings.database_path),
                            DeterministicTestEmbedder(), build_memory_llm(settings))
    service.initialize()
    for rid, text in [('r1', 'I work at Acme Corp as a software engineer.'),
                      ('r2', 'I like hiking on weekends.')]:
        service.add(rid, [MemoryMessage(role='user', content=text, timestamp=1791676800000)],
                    'u1', 's1')
    return service


class StubAnswerClient:
    """按可见命中动态作答：答案完全来自证据并引用真实命中 ID。"""

    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        payload = json.loads(kwargs['messages'][1]['content'])
        hit_id = payload['memories'][0]['id']
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop',
            message=SimpleNamespace(content=canonical(
                {'answer': 'I work at Acme Corp as a software engineer.',
                 'sources': [hit_id]})))])


def test_score_traffic_end_to_end_offline(tmp_path):
    settings = _seed_settings(tmp_path)
    _seed_memories(settings)
    log = write_log(tmp_path / 'input.jsonl', [
        get_record('Where do I work?'),
        {'ts': TS, 'method': 'POST', 'path': '/set', 'status': 200, 'body': '{}'},
    ])
    traffic = load_traffic([log])
    result = score_traffic(settings, traffic, tmp_path / 'out',
                           embedder=DeterministicTestEmbedder(), llm=build_memory_llm(settings))
    assert len(result['queries']) == 1 and result['answer_errors'] == 0
    entry = result['queries'][0]
    assert entry['status'] == 'ok' and entry['replay_ms'] >= 0
    assert entry['evidence']['hits'] >= 1
    assert entry['evidence']['first_id_prefix'] == 'mem_'
    assert 'answer' not in entry  # answers=skip 时不做答案层
    agg = aggregate(result, traffic)
    assert agg['traffic_records'] == 1 and agg['queries'] == 1
    assert agg['latency']['replay_local']['p50_ms'] >= 0
    manifest = {'database': 'test', 'log_count': 1, 'governance_mode': 'off',
                 'f1_mode': 'legacy', 'llm': 'off', 'answers': 'skip'}
    write_report(tmp_path / 'out', manifest, agg, result['queries'])
    report = (tmp_path / 'out' / 'TRAFFIC_REVIEW.md').read_text(encoding='utf-8')
    assert 'Where do I work?' in report and '聚合' in report


def test_score_traffic_with_stub_answer_and_grounding(tmp_path):
    settings = _seed_settings(tmp_path)
    _seed_memories(settings)
    log = write_log(tmp_path / 'input.jsonl', [get_record('Where do I work?')])
    traffic = load_traffic([log])
    answer_sdk = CompletionCache(tmp_path / 'cache', {'provider': 'openai', 'model': 'stub'},
                                 client=StubAnswerClient(), recording=True)
    result = score_traffic(settings, traffic, tmp_path / 'out',
                           embedder=DeterministicTestEmbedder(),
                           llm=build_memory_llm(settings), answer_sdk=answer_sdk)
    entry = result['queries'][0]
    assert entry['answer']['sources'] == [entry['hits'][0]['id']]
    grounding = entry['grounding']
    assert grounding['support_ratio'] == 1.0 and grounding['flags'] == []
    agg = aggregate(result, traffic)
    assert agg['grounding']['graded'] == 1 and agg['grounding']['no_flag_rate'] == 1.0
