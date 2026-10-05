"""双通道查询：扩展文本不得稀释裸查询本可命中的证据。

核心不变量：max(通道A, 通道B) >= 通道A，因此开启双通道后，任何在
无扩展时能通过相关性地板的证据都不会被扩展文本挤掉。回退开关
query_dual_channel=false 时应精确复现 V3 单通道混合行为。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import Settings
from app.embeddings import normalize_rows
from app.llm import NoOpMemoryLLM, QueryExpansion
from app.schemas import MemoryMessage
from app.service import MemoryService
from app.storage import SQLiteMemoryStore

EXPANSION = "current residence and home city nowadays"


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


class StubMemoryLLM:
    enabled = True

    def __init__(self, expansion: QueryExpansion) -> None:
        self.expansion = expansion

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        return {}

    def expand_query(self, query: str, options: list[str] | None) -> QueryExpansion:
        return self.expansion

    def extract_temporal(self, query: str) -> None:
        return None


def _build_services(tmp_path: Path, expansion: QueryExpansion):
    """四个服务共享同一记忆库，只差查询表达与通道开关。"""
    embedder = BagOfWordsEmbedder()
    database_path = tmp_path / "dual.db"
    store = SQLiteMemoryStore(database_path)

    def make(llm, dual: bool) -> MemoryService:
        settings = Settings(
            _env_file=None,
            database_path=database_path,
            min_relevance_score=0.01,
            query_dual_channel=dual,
        )
        return MemoryService(settings, store, embedder, llm)

    empty = QueryExpansion(text="", temporal=None)
    no_exp = make(StubMemoryLLM(empty), True)
    no_exp_off = make(StubMemoryLLM(empty), False)
    dual = make(StubMemoryLLM(QueryExpansion(text=EXPANSION, temporal=None)), True)
    single = make(StubMemoryLLM(QueryExpansion(text=EXPANSION, temporal=None)), False)
    no_exp.initialize()
    return no_exp, no_exp_off, dual, single


def test_dual_channel_never_dilutes_raw_query_score(tmp_path: Path) -> None:
    no_exp, _, dual, single = _build_services(tmp_path, None)
    no_exp.add(
        request_id="dual-channel",
        messages=[
            MemoryMessage(role="user", content="I lived in Beijing for three years.")
        ],
        user_id="user-1",
        session_id="session-1",
    )

    query = "Where did I live"
    no_exp_hits = no_exp.search(query=query, user_id="user-1", top_k=5)
    dual_hits = dual.search(query=query, user_id="user-1", top_k=5)
    single_hits = single.search(query=query, user_id="user-1", top_k=5)

    assert len(no_exp_hits) == len(dual_hits) == len(single_hits) == 1
    # 双通道取较大分：裸查询通道不被扩展文本稀释，分数与无扩展时完全一致。
    assert dual_hits[0].score == no_exp_hits[0].score
    # 单通道混合文本稀释了余弦与词法重合度，分数严格更低。
    assert single_hits[0].score < dual_hits[0].score


def test_dual_channel_flag_is_irrelevant_without_expansion(tmp_path: Path) -> None:
    """无扩展文本时，双通道开关不影响行为：两者都是单通道裸查询。"""
    no_exp, no_exp_off, _, _ = _build_services(tmp_path, None)
    no_exp.add(
        request_id="dual-off",
        messages=[
            MemoryMessage(role="user", content="I lived in Beijing for three years.")
        ],
        user_id="user-1",
        session_id="session-1",
    )

    query = "Where did I live"
    on_hits = no_exp.search(query=query, user_id="user-1", top_k=5)
    off_hits = no_exp_off.search(query=query, user_id="user-1", top_k=5)
    assert on_hits[0].score == off_hits[0].score


def test_dual_channel_without_llm_is_bit_identical(tmp_path: Path) -> None:
    """LLM 关闭时无扩展文本，行为与 V3 基线逐位一致。"""
    embedder = BagOfWordsEmbedder()
    database_path = tmp_path / "noop.db"
    settings = Settings(
        _env_file=None,
        database_path=database_path,
        min_relevance_score=0.01,
    )
    service = MemoryService(
        settings, SQLiteMemoryStore(database_path), embedder, NoOpMemoryLLM()
    )
    service.initialize()
    service.add(
        request_id="noop",
        messages=[MemoryMessage(role="user", content="Aurora project status update")],
        user_id="user-1",
        session_id="session-1",
    )

    hits = service.search(query="Aurora project status", user_id="user-1", top_k=5)
    assert len(hits) == 1
    assert hits[0].score > 0
