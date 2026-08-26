from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


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
