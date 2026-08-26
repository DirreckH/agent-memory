from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings


class LLMError(RuntimeError):
    pass


class MemoryLLM(Protocol):
    enabled: bool

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        """为原始消息生成仅用于索引的检索增强文本。"""

    def expand_query(self, query: str, options: list[str] | None) -> str:
        """扩展查询表达，但不能生成问题最终答案。"""


class NoOpMemoryLLM:
    enabled = False

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        return {}

    def expand_query(self, query: str, options: list[str] | None) -> str:
        return ""


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
    ) -> None:
        self.connection = connection
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.add_enrichment = add_enrichment
        self.search_expansion = search_expansion
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

        prompt = (
            "你是记忆检索索引器。根据输入消息，为每条消息生成简洁的检索增强文本。"
            "只能改写输入中明确出现的事实，不得推断、补全、回答未来问题或写入评测答案。"
            "保留人名、地名、时间、数字、偏好、否定和变更关系。"
            "必须输出 JSON 对象，格式为 "
            '{"items":[{"source_index":0,"search_text":"..."}]}。'
        )
        parsed = self._json_completion(
            prompt,
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

    def expand_query(self, query: str, options: list[str] | None) -> str:
        if not self.search_expansion:
            return ""

        prompt = (
            "你是记忆检索查询改写器。把原始问题改写成适合召回记忆证据的关键词和同义表达。"
            "不得回答问题，不得判断选项，不得生成输入中不存在的事实。"
            "必须输出 JSON 对象，格式为 {\"expanded_query\":\"...\"}。"
        )
        parsed = self._json_completion(
            prompt,
            {"query": query, "options": options or []},
            max_tokens=512,
        )
        expanded = parsed.get("expanded_query")
        if not isinstance(expanded, str):
            raise LLMError("LLM 查询扩展结果缺少 expanded_query")
        return expanded.strip()[:4000]


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
    )
