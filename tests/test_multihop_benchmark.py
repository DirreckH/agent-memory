"""多跳测量门工具测试：用例校验、fillers 确定性、run_multihop 离线管道。

离线测试使用假向量器与桩 LLM，只验证测量管道的数据流与指标结构，
不评价真实检索能力。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.embeddings import normalize_rows
from app.llm import NoOpMemoryLLM
from app.service import MemoryService
from app.storage import SQLiteMemoryStore
from scripts.benchmark_common import (
    BenchmarkMemory,
    MultihopBenchmarkCase,
    MultihopHop,
    load_multihop_cases,
)
from scripts.run_multihop_eval import (
    OracleExpansionLLM,
    generate_fillers,
    run_multihop,
)

FIXTURE = Path(__file__).parent / "fixtures" / "multihop_cases.jsonl"


def _case_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "case_id": "mini-multihop",
        "language": "en",
        "category": "family",
        "query": "What kind of dog did my cousin end up adopting?",
        "memories": [
            {"content": "My cousin Lena moved into the farmhouse last month.", "days_before_now": 30},
            {"content": "She adopted a corgi from the shelter nearby.", "days_before_now": 10},
        ],
        "hops": [
            {"description": "bridge: the cousin", "keyword": "Lena"},
            {"description": "answer: the adopted dog breed", "keyword": "corgi"},
        ],
        "oracle_expansion": "which cousin the user mentioned; the dog breed that cousin adopted",
        "filler_count": 4,
        "filler_seed": 1,
    }
    base.update(overrides)
    return base


def test_multihop_fixture_dataset_loads() -> None:
    cases = load_multihop_cases(FIXTURE)
    assert len(cases) == 15
    assert {case.language for case in cases} == {"zh", "en"}
    assert all(2 <= len(case.hops) <= 3 for case in cases)
    three_hop = [case.case_id for case in cases if len(case.hops) == 3]
    assert len(three_hop) == 4
    # 模型校验已保证：每跳关键词唯一出现在一条记忆中且不泄漏进查询。
    for case in cases:
        contents = [memory.content.casefold() for memory in case.memories]
        for hop in case.hops:
            assert sum(hop.keyword.casefold() in c for c in contents) == 1
            assert hop.keyword.casefold() not in case.query.casefold()


def test_valid_multihop_case_passes_validation() -> None:
    case = MultihopBenchmarkCase(**_case_kwargs())
    assert case.hops[0].keyword == "Lena"


def test_hop_keyword_must_not_appear_in_query() -> None:
    memories = [
        {"content": "My cousin Lena moved into the farmhouse.", "days_before_now": 30},
        {"content": "She adopted a corgi from the shelter.", "days_before_now": 10},
    ]
    with pytest.raises(ValueError, match="不能出现在 query 中"):
        MultihopBenchmarkCase(
            **_case_kwargs(query="What kind of dog did cousin Lena adopt?", memories=memories)
        )


def test_hop_keyword_must_appear_in_exactly_one_memory() -> None:
    memories = [
        {"content": "My cousin Lena moved into the farmhouse.", "days_before_now": 30},
        {"content": "Lena's brother also likes the farmhouse.", "days_before_now": 100},
        {"content": "She adopted a corgi from the shelter.", "days_before_now": 10},
    ]
    with pytest.raises(ValueError, match="必须且只能出现在一条记忆中"):
        MultihopBenchmarkCase(**_case_kwargs(memories=memories))


def test_hop_keyword_must_not_leak_into_expansion() -> None:
    with pytest.raises(ValueError, match="不能出现在 oracle_expansion 中"):
        MultihopBenchmarkCase(
            **_case_kwargs(oracle_expansion="the cousin Lena; the dog breed she adopted")
        )


def test_duplicate_hop_keywords_are_rejected() -> None:
    hops = [
        {"description": "bridge", "keyword": "Lena"},
        {"description": "answer", "keyword": "lena"},
    ]
    with pytest.raises(ValueError, match="各跳的关键词不能重复"):
        MultihopBenchmarkCase(**_case_kwargs(hops=hops))


def test_multihop_case_requires_at_least_two_hops() -> None:
    hops = [{"description": "bridge", "keyword": "Lena"}]
    with pytest.raises(ValueError, match="at least 2"):
        MultihopBenchmarkCase(**_case_kwargs(hops=hops))


def test_generate_fillers_is_deterministic_and_keyword_safe() -> None:
    case = MultihopBenchmarkCase(**_case_kwargs(language="zh"))
    first = generate_fillers(case, 20)
    second = generate_fillers(case, 20)
    assert [m.content for m in first] == [m.content for m in second]
    assert [m.days_before_now for m in first] == [m.days_before_now for m in second]
    assert len(first) == 20
    assert len({m.content for m in first}) == 20
    lowered = [m.content.casefold() for m in first]
    for hop in case.hops:
        assert all(hop.keyword.casefold() not in c for c in lowered)


def test_generate_fillers_skips_keyword_collisions() -> None:
    # 极端情况：把跳关键词塞进 zh 模板的一个槽位池，验证冲突跳过逻辑。
    case = MultihopBenchmarkCase(
        **_case_kwargs(
            language="zh",
            filler_count=5,
        )
    )
    fillers = generate_fillers(case, 30)
    for hop in case.hops:
        assert all(hop.keyword.casefold() not in f.content.casefold() for f in fillers)


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


def test_run_multihop_measures_chain_recall_on_shared_store(tmp_path: Path) -> None:
    """双模式共享同一记忆库：写入一次，只比较查询表达层。"""
    case = MultihopBenchmarkCase(**_case_kwargs())
    embedder = BagOfWordsEmbedder()
    store = SQLiteMemoryStore(tmp_path / "mh.db")
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "mh.db",
        min_relevance_score=0.01,
    )
    services = {
        "pure": MemoryService(settings, store, embedder, NoOpMemoryLLM()),
        "oracle_expansion": MemoryService(
            settings,
            store,
            embedder,
            OracleExpansionLLM({case.query: case.oracle_expansion}),
        ),
    }
    services["pure"].initialize()

    report = run_multihop(
        services, [case], now_ms=1_800_000_000_000, top_k=10, filler_count=4
    )

    assert set(report["modes"]) == {"pure", "oracle_expansion"}
    for mode in ("pure", "oracle_expansion"):
        per_case = report["modes"][mode]["per_case"][0]
        assert per_case.get("error") is None
        assert all(rank is not None for rank in per_case["hop_ranks"])
        assert per_case["chain_at"]["100"] is True
        summary = report["modes"][mode]["summary"]
        assert summary["chain_at_100"] == 1.0
        assert summary["cases"] == 1
        assert summary["mean_bottleneck_rank"] == per_case["bottleneck_rank"]
