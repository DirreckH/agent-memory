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

    # 向量提供方：fastembed=本地 ONNX（离线）；dashscope=阿里百炼 API。
    # 学术榜（开源方法榜）当前要求统一使用 text-embedding-v4（DashScope）：
    # 提交学术榜时设置 EMBEDDING_PROVIDER=dashscope、EMBEDDING_MODEL=text-embedding-v4。
    # 注意：更换向量模型后维度不一致，必须使用全新数据库，不能混用分数。
    embedding_provider: Literal["fastembed", "dashscope"] = "fastembed"
    # fastembed 模式下是 ONNX 模型名；dashscope 模式下是百炼 API 模型名。
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_cache_dir: Path = Path("data/model_cache")
    embedding_threads: int = Field(default=2, ge=1, le=64)
    warmup_embedding_on_startup: bool = False

    # DashScope 向量：OpenAI 兼容端点。text-embedding-v4 单次请求最多 10 条
    # 文本（由 DashScopeEmbedder 内部分批）、默认 1024 维（可选 64-2048）。
    dashscope_api_key: SecretStr | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    dashscope_max_retries: int = Field(default=2, ge=0, le=5)
    # 可选输出维度：None 走服务端默认（text-embedding-v4 为 1024）。
    embedding_dimensions: int | None = Field(default=None, ge=64, le=2048)

    # 混合检索：语义相似度 + 轻量关键词重合度。
    semantic_weight: float = Field(default=0.80, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    min_relevance_score: float = Field(default=0.20, ge=0.0, le=1.0)
    max_top_k: int = Field(default=100, ge=1, le=1000)
    candidate_pool_size: int = Field(default=400, ge=1, le=10000)

    # 时间感知检索：规则/LLM 抽取相对约束，绝对窗口由确定性代码解析。
    # strict：确认窗内、时间未知、窗外补充依次返回，组内排序后截断。
    temporal_mode: Literal["off", "soft", "strict"] = "soft"
    # replay：以当前用户的最大源消息时间回放；realtime：以检索开始时刻为准。
    # 显式 query_time_ms 优先于模式；replay 缺少源消息时间时不猜测相对窗口。
    temporal_reference_mode: Literal["replay", "realtime"] = "replay"
    # 独立于查询扩展：off=不抽取；rules=仅规则；hybrid=规则优先，复杂表达用 LLM。
    temporal_extraction_mode: Literal["off", "rules", "hybrid"] = "hybrid"
    temporal_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    temporal_decay_half_life_days: float = Field(
        default=30.0, gt=0.0, le=3650.0
    )
    temporal_event_anchor_min_score: float = Field(
        default=0.30, ge=0.0, le=1.0
    )

    # 双通道查询：扩展文本存在时，裸查询与扩展查询各自嵌入打分并取较大值，
    # 防止扩展文本把弱相关证据（多跳链中的桥接/答案跳）挤到相关性地板之下。
    # false 时回退为 V3 的单通道混合查询（扩展文本并入主查询文本）。
    query_dual_channel: bool = True

    # 接口输入日志：把进入 /set、/get 的请求正文写入 JSONL，用于联调与事后复盘。
    # 日志是评测数据在主库之外的第二份副本：文件按天轮转，保留份数对齐
    # DATA_RETENTION_DAYS，到期自动删除。只记录请求体，永不记录请求头。
    input_log_enabled: bool = True
    input_log_path: Path = Path("data/logs/input.jsonl")
    input_log_max_chars: int = Field(default=20_000, ge=100, le=2_000_000)

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
