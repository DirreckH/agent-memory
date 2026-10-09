from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.multihop import RetrievalStep, parse_retrieval_steps, steps_from_expansion
from app.prompts import (
    ADD_ENRICHMENT_PROMPT_V2,
    QUERY_EXPANSION_PROMPT_MULTIHOP,
    QUERY_EXPANSION_PROMPT_V3,
    TEMPORAL_EXTRACTION_PROMPT,
)
from app.temporal import TemporalConstraints
from app.temporal_extraction import parse_grounded_temporal


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryExpansion:
    """检索辅助文本；temporal 保留兼容评测数据，生产时间抽取走独立入口。"""

    text: str
    temporal: TemporalConstraints | None
    retrieval_steps: tuple[RetrievalStep, ...] = ()


class MemoryLLM(Protocol):
    enabled: bool

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        """为原始消息生成仅用于索引的检索增强文本。"""

    def expand_query(self, query: str, options: list[str] | None) -> QueryExpansion:
        """扩展查询表达，但不能生成问题最终答案。"""

    def extract_temporal(self, query: str) -> TemporalConstraints | None:
        """仅从原查询抽取并校验带原文证据的时间约束。"""


class NoOpMemoryLLM:
    enabled = False

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        return {}

    def expand_query(self, query: str, options: list[str] | None) -> QueryExpansion:
        return QueryExpansion(text="", temporal=None)

    def extract_temporal(self, query: str) -> TemporalConstraints | None:
        return None


@dataclass(frozen=True)
class LLMConnection:
    api_key: str
    base_url: str
    model: str


class OpenAICompatibleMemoryLLM:
    """DeepSeek 与 OpenAI 均可通过 OpenAI 兼容 SDK 调用。"""

    enabled = True

    def __init__(
        self,
        connection: LLMConnection,
        timeout_seconds: float,
        max_retries: int,
        add_enrichment: bool,
        search_expansion: bool,
        multihop_retrieval: bool = False,
        multihop_structured_planner: bool = False,
    ) -> None:
        self.connection = connection
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.add_enrichment = add_enrichment
        self.search_expansion = search_expansion
        self.multihop_retrieval = multihop_retrieval
        self.multihop_structured_planner = multihop_structured_planner
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.connection.api_key,
                    base_url=self.connection.base_url,
                    timeout=self.timeout_seconds,
                    max_retries=self.max_retries,
                )
        return self._client

    def _json_completion(
        self, system_prompt: str, user_payload: dict[str, object], max_tokens: int
    ) -> dict[str, object]:
        try:
            response = self._get_client().chat.completions.create(
                model=self.connection.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=max_tokens,
                stream=False,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM 返回空内容")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("LLM JSON 顶层必须是对象")
            return parsed
        except Exception as exc:  # noqa: BLE001 - 不把上游响应或密钥泄漏给调用方
            raise LLMError(f"LLM 调用失败: {type(exc).__name__}") from exc

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        if not self.add_enrichment:
            return {}

        parsed = self._json_completion(
            ADD_ENRICHMENT_PROMPT_V2,
            {"messages": messages},
            max_tokens=min(4096, max(512, len(messages) * 160)),
        )
        items = parsed.get("items")
        if not isinstance(items, list):
            raise LLMError("LLM 索引增强结果缺少 items 数组")

        output: dict[int, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            index = item.get("source_index")
            text = item.get("search_text")
            if (
                isinstance(index, int)
                and 0 <= index < len(messages)
                and isinstance(text, str)
                and text.strip()
            ):
                output[index] = text.strip()[:4000]
        return output

    def expand_query(self, query: str, options: list[str] | None) -> QueryExpansion:
        if not self.search_expansion:
            return QueryExpansion(text="", temporal=None)

        structured = self.multihop_retrieval and self.multihop_structured_planner
        parsed = self._json_completion(
            QUERY_EXPANSION_PROMPT_MULTIHOP if structured else QUERY_EXPANSION_PROMPT_V3,
            {"query": query, "options": options or []},
            max_tokens=1024 if structured else 512,
        )
        expanded = parsed.get("expanded_query")
        if not isinstance(expanded, str):
            raise LLMError("LLM 查询扩展结果缺少 expanded_query")
        steps = ()
        if self.multihop_retrieval:
            steps = parse_retrieval_steps(parsed.get('retrieval_steps'), query)
            if parsed.get('retrieval_steps') is None and not structured:
                steps = steps_from_expansion(query, expanded)
        return QueryExpansion(
            text=expanded.strip()[:4000], temporal=None,
            retrieval_steps=steps,
        )

    def extract_temporal(self, query: str) -> TemporalConstraints | None:
        parsed = self._json_completion(
            TEMPORAL_EXTRACTION_PROMPT, {"query": query}, max_tokens=512,
        )
        return parse_grounded_temporal(parsed.get("temporal"), query)


def build_memory_llm(settings: Settings) -> MemoryLLM:
    if settings.llm_provider == "none":
        return NoOpMemoryLLM()

    if settings.llm_provider == "deepseek":
        key = (
            settings.deepseek_api_key.get_secret_value().strip()
            if settings.deepseek_api_key
            else ""
        )
        if not key:
            raise ValueError(
                "LLM_PROVIDER=deepseek 时必须通过环境变量设置 DEEPSEEK_API_KEY"
            )
        connection = LLMConnection(
            api_key=key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )
    else:
        key = (
            settings.openai_api_key.get_secret_value().strip()
            if settings.openai_api_key
            else ""
        )
        if not key:
            raise ValueError(
                "LLM_PROVIDER=openai 时必须通过环境变量设置 OPENAI_API_KEY"
            )
        connection = LLMConnection(
            api_key=key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
        )

    return OpenAICompatibleMemoryLLM(
        connection=connection,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        add_enrichment=settings.llm_add_enrichment,
        search_expansion=settings.llm_search_expansion,
        multihop_retrieval=settings.multihop_enabled,
        multihop_structured_planner=settings.multihop_structured_planner,
    )
