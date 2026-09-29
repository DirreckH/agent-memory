from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from app.config import Settings


FloatMatrix = NDArray[np.float32]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> FloatMatrix:
        """把一批文本转换为经过 L2 归一化的二维向量矩阵。"""


class EmbeddingError(RuntimeError):
    pass


def normalize_rows(matrix: NDArray[np.floating]) -> FloatMatrix:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise EmbeddingError("向量模型返回结果不是二维矩阵")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    # 零向量保持为零，避免除零产生 NaN。
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values, dtype=np.float32),
        where=norms > 0,
    )


class FastEmbedder:
    """基于 FastEmbed/ONNX 的懒加载多语言向量模型。"""

    def __init__(self, model_name: str, cache_dir: Path, threads: int = 2) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._model = None
        self._lock = threading.RLock()

    def _load_model(self):
        # 延迟导入使单元测试不需要下载真实模型。
        from fastembed import TextEmbedding

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return TextEmbedding(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            threads=self.threads,
        )

    def embed(self, texts: list[str]) -> FloatMatrix:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        try:
            # ONNX 会话与首次模型下载都在锁内，避免并发重复初始化。
            with self._lock:
                if self._model is None:
                    self._model = self._load_model()
                vectors = list(self._model.embed(texts))
            return normalize_rows(np.asarray(vectors, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 - 在服务边界统一转换为可重试错误
            raise EmbeddingError(f"生成文本向量失败: {type(exc).__name__}") from exc


class DashScopeEmbedder:
    """基于阿里百炼 OpenAI 兼容接口的远程向量器。

    学术榜（开源方法榜）要求统一使用 text-embedding-v4：Qwen3-Embedding 系列
    商业 API，默认 1024 维（可选 64-2048）、单条上限 8192 token、单次请求最多
    10 条文本——分批由本类内部强制执行。输出统一 L2 归一化，与 FastEmbedder
    在 Embedder 协议下行为一致；检索与打分链路对提供方无感知。
    """

    # DashScope 平台限制：单次 embeddings 请求最多 10 条文本。
    BATCH_LIMIT = 10

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        dimensions: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.dimensions = dimensions
        self._client = None
        self._lock = threading.RLock()

    def _get_client(self):
        # 延迟导入与客户端构造，使单元测试不需要真实密钥或网络。
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    max_retries=self.max_retries,
                )
        return self._client

    def embed(self, texts: list[str]) -> FloatMatrix:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        try:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), self.BATCH_LIMIT):
                batch = texts[start : start + self.BATCH_LIMIT]
                request: dict[str, Any] = {"model": self.model_name, "input": batch}
                if self.dimensions is not None:
                    request["dimensions"] = self.dimensions
                response = self._get_client().embeddings.create(**request)
                # 兼容接口按 index 标识对应关系；显式排序防止乱序返回。
                items = sorted(response.data, key=lambda item: item.index)
                if len(items) != len(batch):
                    raise EmbeddingError("向量数量与输入文本数量不一致")
                vectors.extend(item.embedding for item in items)
            return normalize_rows(np.asarray(vectors, dtype=np.float32))
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001 - 在服务边界统一转换为可重试错误
            raise EmbeddingError(f"生成文本向量失败: {type(exc).__name__}") from exc


def build_embedder(settings: Settings) -> Embedder:
    """按配置构造向量器：fastembed=本地 ONNX；dashscope=百炼 API（学术榜要求）。"""
    if settings.embedding_provider == "fastembed":
        return FastEmbedder(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
            threads=settings.embedding_threads,
        )
    key = (
        settings.dashscope_api_key.get_secret_value().strip()
        if settings.dashscope_api_key
        else ""
    )
    if not key:
        raise ValueError(
            "EMBEDDING_PROVIDER=dashscope 时必须通过环境变量设置 DASHSCOPE_API_KEY"
        )
    return DashScopeEmbedder(
        model_name=settings.embedding_model,
        api_key=key,
        base_url=settings.dashscope_base_url,
        timeout_seconds=settings.dashscope_timeout_seconds,
        max_retries=settings.dashscope_max_retries,
        dimensions=settings.embedding_dimensions,
    )
