"""DashScope 向量器测试：分批、顺序、归一化、错误语义与工厂选择。

全部离线运行：注入桩客户端替代真实 OpenAI SDK，不访问网络、不使用密钥。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.embeddings import (
    DashScopeEmbedder,
    EmbeddingError,
    FastEmbedder,
    build_embedder,
)


class StubEmbeddingsAPI:
    """模拟百炼 OpenAI 兼容接口的 embeddings.create。"""

    def __init__(self, *, shuffle: bool = False, fail: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.shuffle = shuffle
        self.fail = fail

    def create(self, **kwargs: object) -> SimpleNamespace:
        if self.fail:
            raise RuntimeError("upstream unavailable")
        self.calls.append(kwargs)
        texts = list(kwargs["input"])
        # 每条文本生成一个确定性向量：[序号, 1]，行间可区分。
        data = [
            SimpleNamespace(embedding=[float(i), 1.0], index=i)
            for i in range(len(texts))
        ]
        if self.shuffle:
            data = list(reversed(data))
        return SimpleNamespace(data=data)


def _embedder(**overrides: object) -> tuple[DashScopeEmbedder, StubEmbeddingsAPI]:
    embedder = DashScopeEmbedder(
        model_name="text-embedding-v4",
        api_key="test-only-key",
        base_url="https://example.invalid/compatible-mode/v1",
        **overrides,
    )
    stub = StubEmbeddingsAPI()
    embedder._client = SimpleNamespace(embeddings=stub)
    return embedder, stub


def test_dashscope_embedder_batches_by_ten() -> None:
    embedder, stub = _embedder()
    texts = [f"memory-{i}" for i in range(23)]
    matrix = embedder.embed(texts)

    assert matrix.shape[0] == 23
    # 23 条文本按平台限制分成 10 + 10 + 3 三批。
    assert [len(call["input"]) for call in stub.calls] == [10, 10, 3]
    assert all(call["model"] == "text-embedding-v4" for call in stub.calls)


def test_dashscope_embedder_restores_index_order() -> None:
    embedder, stub = _embedder()
    stub.shuffle = True
    texts = ["alpha", "beta", "gamma"]
    matrix = embedder.embed(texts)

    # 接口乱序返回时按 index 排序还原，行序与输入一致。
    norms = np.linalg.norm(matrix, axis=1)
    expected = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(matrix, expected, atol=1e-6)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_dashscope_embedder_normalizes_rows() -> None:
    embedder, _ = _embedder()
    matrix = embedder.embed(["a", "b", "c"])
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)


def test_dashscope_embedder_empty_input() -> None:
    embedder, stub = _embedder()
    matrix = embedder.embed([])
    assert matrix.shape == (0, 0)
    assert stub.calls == []


def test_dashscope_embedder_wraps_upstream_errors() -> None:
    embedder, stub = _embedder()
    stub.fail = True
    with pytest.raises(EmbeddingError, match="生成文本向量失败"):
        embedder.embed(["hello"])


def test_dashscope_embedder_passes_dimensions_only_when_configured() -> None:
    embedder, stub = _embedder(dimensions=1024)
    embedder.embed(["hello"])
    assert stub.calls[0]["dimensions"] == 1024

    default_embedder, default_stub = _embedder()
    default_embedder.embed(["hello"])
    assert "dimensions" not in default_stub.calls[0]


def test_build_embedder_requires_dashscope_key() -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider="dashscope",
        embedding_model="text-embedding-v4",
        database_path="/tmp/unused.db",
    )
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        build_embedder(settings)


def test_build_embedder_selects_provider() -> None:
    fastembed_settings = Settings(
        _env_file=None,
        embedding_provider="fastembed",
        database_path="/tmp/unused.db",
    )
    assert isinstance(build_embedder(fastembed_settings), FastEmbedder)

    dashscope_settings = Settings(
        _env_file=None,
        embedding_provider="dashscope",
        embedding_model="text-embedding-v4",
        dashscope_api_key="test-only-key",
        database_path="/tmp/unused.db",
    )
    embedder = build_embedder(dashscope_settings)
    assert isinstance(embedder, DashScopeEmbedder)
    assert embedder.model_name == "text-embedding-v4"
    assert embedder.dimensions is None
