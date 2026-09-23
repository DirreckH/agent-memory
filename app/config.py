from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。

    所有敏感信息均从环境变量或 .env 文件读取，代码中不保存任何真实密钥。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Agent Memory API"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    # 本地持久化配置
    database_path: Path = Path("data/agent_memory.db")
    data_retention_days: int = Field(default=30, ge=1, le=3650)
    ingestion_stale_seconds: int = Field(default=300, ge=30, le=86400)

    # FastEmbed 使用 ONNX Runtime，不依赖 PyTorch/GPU。
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_cache_dir: Path = Path("data/model_cache")
    embedding_threads: int = Field(default=2, ge=1, le=64)
    warmup_embedding_on_startup: bool = False

    # 混合检索：语义相似度 + 轻量关键词重合度。
    semantic_weight: float = Field(default=0.80, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    min_relevance_score: float = Field(default=0.20, ge=0.0, le=1.0)
    max_top_k: int = Field(default=100, ge=1, le=1000)
    candidate_pool_size: int = Field(default=400, ge=1, le=10000)

    # 时间感知检索：LLM 只抽取相对时间约束，绝对窗口由确定性代码解析。
    # off/soft/strict：soft 加权参与打分；strict 在窗口内候选足够时截掉窗口外记录。
    temporal_mode: Literal["off", "soft", "strict"] = "soft"
    temporal_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    temporal_decay_half_life_days: float = Field(
        default=30.0, gt=0.0, le=3650.0
    )
    temporal_event_anchor_min_score: float = Field(
        default=0.30, ge=0.0, le=1.0
    )

    # 简化 /set、/get 接口没有 user_id，统一放入此本地隔离空间。
    local_user_id: str = Field(default="local-default", min_length=1, max_length=256)
    max_memory_chars: int = Field(default=200_000, ge=1, le=2_000_000)
    max_query_chars: int = Field(default=20_000, ge=1, le=200_000)

    # 公网部署后建议设置。空值表示不启用鉴权（只适合本地调试/公开 smoke）。
    memory_api_key: SecretStr | None = None

    # LLM 可关闭、使用 DeepSeek，或按当前比赛 Full 规则切换为 OpenAI。
    llm_provider: Literal["none", "deepseek", "openai"] = "none"
    llm_failure_mode: Literal["fallback", "strict"] = "fallback"
    llm_add_enrichment: bool = True
    llm_search_expansion: bool = True
    llm_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    llm_max_retries: int = Field(default=1, ge=0, le=5)

    # DeepSeek 配置：只预留变量，不包含密钥。
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # 官网当前 Full 检查项要求 gpt-4o-mini，保留合规切换入口。
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    @property
    def expected_memory_api_key(self) -> str:
        if self.memory_api_key is None:
            return ""
        return self.memory_api_key.get_secret_value().strip()

    @property
    def normalized_score_weights(self) -> tuple[float, float]:
        total = self.semantic_weight + self.lexical_weight
        if total <= 0:
            # 避免错误配置导致所有相关性分数失真。
            return 1.0, 0.0
        return self.semantic_weight / total, self.lexical_weight / total

    @property
    def normalized_temporal_weights(self) -> tuple[float, float, float]:
        """(语义, 词法, 时间) 三路权重，仅在检索存在有效时间窗口时使用。"""
        total = (
            self.semantic_weight
            + self.lexical_weight
            + self.temporal_weight
        )
        if total <= 0:
            return 1.0, 0.0, 0.0
        return (
            self.semantic_weight / total,
            self.lexical_weight / total,
            self.temporal_weight / total,
        )
