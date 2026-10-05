"""时间类 A/B 工具测试：用例校验、rank 计算与离线 run_ab 管道。

离线测试复用 test_temporal.py 的 Bag-of-Words 假向量器思路，
验证实验管道本身的数据流，不评价真实模型能力。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.embeddings import normalize_rows
from app.llm import NoOpMemoryLLM, QueryExpansion
from app.service import MemoryService
from app.storage import SQLiteMemoryStore
from app.temporal import parse_temporal
from scripts.benchmark_common import (
    TemporalBenchmarkCase,
    load_temporal_cases,
    rank_of_keyword,
)
from scripts.run_temporal_ab import (
    BenchmarkMemoryService,
    OracleMemoryLLM,
    build_oracle_expansions,
    extraction_report,
    run_ab,
)

FIXTURE = Path(__file__).parent / "fixtures" / "temporal_ab_cases.jsonl"


def _case_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "case_id": "mini-rolling",
        "language": "zh",
        "category": "hobby",
        "query_type": "rolling",
        "query": "最近一周我说的新爱好是什么？",
        "memories": [
            {"content": "最近的新爱好是陆冲板。", "days_before_now": 3},
            {"content": "最近的新爱好是陶艺课。", "days_before_now": 120},
        ],
        "target_index": 0,
        "distractor_index": 1,
        "expected_keyword": "陆冲板",
        "distractor_keyword": "陶艺课",
        "oracle": {
            "windows": [
                {"kind": "rolling", "unit": "day", "amount": 7, "direction": "past"}
            ],
            "event_anchor": None,
            "ordering": None,
        },
    }
    base.update(overrides)
    return base


def test_temporal_fixture_dataset_loads() -> None:
    cases = load_temporal_cases(FIXTURE)
    assert len(cases) == 15
    assert {case.query_type for case in cases} == {
        "rolling",
        "event_before",
        "event_after",
        "ordering_earliest",
        "ordering_latest",
    }
    assert {case.language for case in cases} == {"zh", "en"}
    # 关键词约束由模型校验保证：target/distractor 关键词各只出现一次。
    for case in cases:
        contents = [memory.content.casefold() for memory in case.memories]
        assert sum(
            case.expected_keyword.casefold() in content for content in contents
        ) == 1


def test_valid_rolling_case_passes_validation() -> None:
    case = TemporalBenchmarkCase(**_case_kwargs())
    assert case.memories[case.target_index].days_before_now == 3
    assert parse_temporal(case.oracle) is not None


def test_keyword_must_not_appear_in_query() -> None:
    with pytest.raises(ValueError, match="不能出现在 query 中"):
        TemporalBenchmarkCase(
            **_case_kwargs(query="最近一周我说的新爱好是陆冲板吗？")
        )


def test_keyword_must_exist_in_target_memory() -> None:
    with pytest.raises(ValueError, match="必须原样出现在"):
        TemporalBenchmarkCase(**_case_kwargs(expected_keyword="不存在的爱好"))


def test_keyword_must_not_leak_into_other_memories() -> None:
    memories = [
        {"content": "最近的新爱好是陆冲板，穿的是滑板鞋。", "days_before_now": 3},
        {"content": "以前学陶艺的时候也穿着滑板鞋。", "days_before_now": 120},
    ]
    with pytest.raises(ValueError, match="只能出现在第 1 条记忆中"):
        TemporalBenchmarkCase(
            **_case_kwargs(memories=memories, distractor_keyword="滑板鞋")
        )


def test_query_without_temporal_signal_is_rejected() -> None:
    with pytest.raises(ValueError, match="时间信号"):
        TemporalBenchmarkCase(**_case_kwargs(query="我的新爱好是什么？"))


def test_unparseable_oracle_is_rejected() -> None:
    with pytest.raises(ValueError, match="可解析的时间约束"):
        TemporalBenchmarkCase(
            **_case_kwargs(oracle={"windows": [], "event_anchor": None, "ordering": None})
        )


def test_event_case_requires_independent_event_memory() -> None:
    kwargs = _case_kwargs(
        query_type="event_before",
        query="换工作之前我在哪个团队？",
        memories=[
            {"content": "我正式换工作了，加入了 Acme 的平台组。", "days_before_now": 5},
            {"content": "我在 Innovate 的算法团队做推荐系统。", "days_before_now": 22},
            {"content": "Acme 平台组这边的日常工作是内部工具链。", "days_before_now": 2},
        ],
        target_index=1,
        distractor_index=2,
        event_index=None,
        expected_keyword="算法团队",
        distractor_keyword="内部工具链",
        oracle={
            "windows": [],
            "event_anchor": {"event": "换工作", "direction": "before"},
            "ordering": None,
        },
    )
    with pytest.raises(ValueError, match="事件记忆索引"):
        TemporalBenchmarkCase(**kwargs)

    valid = TemporalBenchmarkCase(**{**kwargs, "event_index": 0})
    assert valid.event_index == 0


def test_ordering_structure_is_enforced() -> None:
    # ordering_earliest 要求 target 更早；mini 数据里 target 比 distractor 近，应被拒绝。
    with pytest.raises(ValueError, match="ordering_earliest 要求 target 更早"):
        TemporalBenchmarkCase(
            **_case_kwargs(
                query_type="ordering_earliest",
                query="我第一次说的新爱好是什么？",
                oracle={"windows": [], "event_anchor": None, "ordering": "earliest"},
            )
        )


def test_rank_of_keyword_is_case_insensitive_and_one_based() -> None:
    contents = ["[user]\nI love padel", "[user]\nwoodworking course", "[user]\npadel again"]
    assert rank_of_keyword(contents, "woodworking") == 2
    assert rank_of_keyword(contents, "PADEL") == 1
    assert rank_of_keyword(contents, "missing") is None


class BagOfWordsEmbedder:
    """离线向量器：按分词构造确定性向量，不下载模型。"""

    dimension = 256

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = "".join(
            character.casefold() if character.isalnum() else " "
            for character in text
        )
        return [token for token in normalized.split() if token]

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                index = self._vocabulary.setdefault(token, len(self._vocabulary))
                matrix[row, index] += 1.0
        return normalize_rows(matrix)


def test_run_ab_temporal_soft_beats_off_on_confusable_case(tmp_path: Path) -> None:
    """同内容、不同时间的两条记忆：off 平局按时间升序错排，soft 时间加权纠正。"""
    case = TemporalBenchmarkCase(**_case_kwargs())
    embedder = BagOfWordsEmbedder()
    llm = OracleMemoryLLM(build_oracle_expansions([case]))

    services: dict[str, MemoryService] = {}
    for mode in ("off", "soft"):
        settings = Settings(
            _env_file=None,
            database_path=tmp_path / f"{mode}.db",
            temporal_mode=mode,
            min_relevance_score=0.05,
        )
        services[mode] = MemoryService(
            settings, SQLiteMemoryStore(settings.database_path), embedder, llm
        )
        services[mode].initialize()

    report = run_ab(services, [case], now_ms=1_800_000_000_000, top_k=5)

    off = report["modes"]["off"]["summary"]
    soft = report["modes"]["soft"]["summary"]
    # off 模式：分数完全打平，created_at 升序平局让旧记录（分心项）排第一。
    assert off["hit_at_1"] == 0.0
    assert off["target_above_distractor"] == 0.0
    # soft 模式：目标在窗口内获得时间加权，排到第一。
    assert soft["hit_at_1"] == 1.0
    assert soft["target_above_distractor"] == 1.0
    assert soft["mrr_at_k"] > off["mrr_at_k"]


def test_extraction_report_measures_signature_agreement() -> None:
    case = TemporalBenchmarkCase(**_case_kwargs())

    perfect = {
        case.query: QueryExpansion(
            text="", temporal=parse_temporal(case.oracle)
        )
    }
    assert extraction_report([case], perfect)["agreement_rate"] == 1.0

    empty = {case.query: QueryExpansion(text="", temporal=None)}
    report = extraction_report([case], empty)
    assert report["agreement_rate"] == 0.0
    assert report["per_case"][0]["got"] == ""


def test_oracle_benchmark_keeps_fixture_constraint_even_when_rules_differ(tmp_path: Path) -> None:
    query = "最近一周我说的新爱好是什么？"
    # 使用不同于规则的一份固定约束，确保 oracle 对比没有偷偷改用规则结果。
    expected = parse_temporal({"windows": [{
        "kind": "rolling", "unit": "day", "amount": 30, "direction": "past",
    }]})
    settings = Settings(_env_file=None, database_path=tmp_path / "oracle.db")
    service = BenchmarkMemoryService(
        settings, SQLiteMemoryStore(settings.database_path), BagOfWordsEmbedder(),
        OracleMemoryLLM({query: QueryExpansion("", expected)}),
    )
    assert service._query_temporal(query) == expected
    assert service.extractions[query].temporal == expected


def test_benchmark_records_constraints_from_rules_without_llm(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, database_path=tmp_path / "rules.db")
    service = BenchmarkMemoryService(
        settings, SQLiteMemoryStore(settings.database_path), BagOfWordsEmbedder(),
        NoOpMemoryLLM(),
    )
    actual = service._query_temporal("今天的项目更新")
    assert actual.relative_window.kind == "calendar"
    assert actual.relative_window.offset == 0
    assert service.extractions["今天的项目更新"].temporal == actual
