from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.config import Settings
from app.embeddings import Embedder, EmbeddingError
from app.llm import LLMError, MemoryLLM
from app.schemas import MemoryMessage
from app.storage import MemoryToStore, SQLiteMemoryStore, StoredMemory


logger = logging.getLogger(__name__)


class MemoryServiceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    id: str
    content: str
    score: float
    created_at: str


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _timestamp_to_text(timestamp_ms: int | None, fallback: str) -> str:
    if timestamp_ms is None:
        return fallback
    try:
        return datetime.fromtimestamp(
            timestamp_ms / 1000, timezone.utc
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return fallback


def _tokenize(text: str) -> set[str]:
    lowered = text.casefold()
    latin_tokens = re.findall(r"[a-z0-9_]+", lowered)
    tokens = set(latin_tokens)
    # 不引入重量级分词器，仅处理常见英文词尾，提升 prefer/prefers、
    # launch/launches 等精确词法召回。原 token 同时保留，避免过度词干化。
    for token in latin_tokens:
        if len(token) > 5 and token.endswith("ing"):
            tokens.add(token[:-3])
        if len(token) > 4 and token.endswith("ed"):
            tokens.add(token[:-2])
        if len(token) > 4 and token.endswith("es"):
            tokens.add(token[:-2])
        elif len(token) > 4 and token.endswith("s"):
            tokens.add(token[:-1])
    cjk_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", lowered)
    for run in cjk_runs:
        tokens.update(run)
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
        if len(run) == 1:
            tokens.add(run)
    return tokens


def _lexical_similarity(query: str, document: str) -> float:
    query_tokens = _tokenize(query)
    document_tokens = _tokenize(document)
    if not query_tokens or not document_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(document_tokens))


class MemoryService:
    def __init__(
        self,
        settings: Settings,
        store: SQLiteMemoryStore,
        embedder: Embedder,
        llm: MemoryLLM,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm

    def initialize(self) -> None:
        self.store.initialize()
        deleted = self.store.prune_older_than(self.settings.data_retention_days)
        if deleted:
            logger.info("已按数据保留策略清理 %s 个过期写入批次", deleted)
        if self.settings.warmup_embedding_on_startup:
            self.embedder.embed(["agent memory embedding warmup"])

    def _llm_enrichment(
        self, messages: list[dict[str, object]]
    ) -> dict[int, str]:
        if not self.llm.enabled:
            return {}
        try:
            return self.llm.enrich_messages(messages)
        except LLMError:
            if self.settings.llm_failure_mode == "strict":
                raise
            # 不记录原始消息或上游响应，避免评测数据出现在日志中。
            logger.warning("LLM 写入增强失败，已回退到原始文本索引")
            return {}

    def _llm_query_expansion(
        self, query: str, options: list[str] | None
    ) -> str:
        if not self.llm.enabled:
            return ""
        try:
            return self.llm.expand_query(query, options)
        except LLMError:
            if self.settings.llm_failure_mode == "strict":
                raise
            logger.warning("LLM 查询扩展失败，已回退到原始查询")
            return ""

    @staticmethod
    def _payload_hash(
        messages: list[dict[str, object]], user_id: str, session_id: str
    ) -> str:
        canonical = json.dumps(
            {"messages": messages, "user_id": user_id, "session_id": session_id},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _memory_id(request_id: str, ordinal: int) -> str:
        digest = hashlib.sha256(f"{request_id}:{ordinal}".encode()).hexdigest()[:24]
        return f"mem_{digest}"

    @staticmethod
    def _render_content(message: dict[str, object], simple: bool) -> str:
        raw = str(message["content"]).strip()
        if simple:
            return raw
        role = str(message["role"])
        timestamp = message.get("timestamp")
        if isinstance(timestamp, int):
            when = _timestamp_to_text(timestamp, "")
            return f"[{role} | {when}]\n{raw}" if when else f"[{role}]\n{raw}"
        return f"[{role}]\n{raw}"

    def add(
        self,
        request_id: str,
        messages: list[MemoryMessage],
        user_id: str,
        session_id: str,
        *,
        simple: bool = False,
    ) -> int:
        raw_messages = [message.model_dump(exclude_none=True) for message in messages]
        if any(
            len(str(message["content"])) > self.settings.max_memory_chars
            for message in raw_messages
        ):
            raise ValueError("单条 memory content 超过 MAX_MEMORY_CHARS 限制")

        payload_hash = self._payload_hash(raw_messages, user_id, session_id)
        persistence_time = utc_now_text()
        should_process = self.store.claim_request(
            request_id=request_id,
            payload_hash=payload_hash,
            user_id=user_id,
            session_id=session_id,
            created_at=persistence_time,
        )
        if not should_process:
            return 0

        try:
            enrichment = self._llm_enrichment(raw_messages)
            contents = [
                self._render_content(message, simple=bool(simple))
                for message in raw_messages
            ]
            search_texts = [
                (
                    f"{content}\n索引增强：{enrichment[index]}"
                    if enrichment.get(index)
                    else content
                )
                for index, content in enumerate(contents)
            ]
            embeddings = self.embedder.embed(search_texts)
            if embeddings.shape[0] != len(search_texts):
                raise EmbeddingError("向量数量与待写入记忆数量不一致")

            records: list[MemoryToStore] = []
            for index, (message, content, search_text) in enumerate(
                zip(raw_messages, contents, search_texts, strict=True)
            ):
                timestamp = message.get("timestamp")
                source_timestamp = timestamp if isinstance(timestamp, int) else None
                records.append(
                    MemoryToStore(
                        id=self._memory_id(request_id, index),
                        ordinal=index,
                        user_id=user_id,
                        session_id=session_id,
                        role=str(message["role"]),
                        content=content,
                        search_text=search_text,
                        embedding=embeddings[index],
                        created_at=_timestamp_to_text(
                            source_timestamp, persistence_time
                        ),
                        source_timestamp=source_timestamp,
                    )
                )
            self.store.complete_request(request_id, records)
            return len(records)
        except (LLMError, EmbeddingError) as exc:
            self.store.mark_failed(request_id, type(exc).__name__)
            raise MemoryServiceUnavailable(str(exc)) from exc
        except Exception as exc:
            self.store.mark_failed(request_id, type(exc).__name__)
            raise

    def add_simple(self, memory_text: str) -> int:
        request_id = f"local:{uuid.uuid4().hex}"
        session_id = f"local-session:{uuid.uuid4().hex}"
        return self.add(
            request_id=request_id,
            messages=[MemoryMessage(role="user", content=memory_text)],
            user_id=self.settings.local_user_id,
            session_id=session_id,
            simple=True,
        )

    def _score_record(
        self,
        query_text: str,
        query_vector: np.ndarray,
        record: StoredMemory,
    ) -> float | None:
        if record.embedding.shape != query_vector.shape:
            # 更换向量模型后旧向量维度可能不同；跳过而非返回错误结果。
            return None
        cosine = float(np.dot(query_vector, record.embedding))
        semantic = max(0.0, min(1.0, cosine))
        lexical = _lexical_similarity(query_text, record.search_text)
        semantic_weight, lexical_weight = self.settings.normalized_score_weights
        return semantic_weight * semantic + lexical_weight * lexical

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int,
        options: list[str] | None = None,
    ) -> list[SearchHit]:
        if len(query) > self.settings.max_query_chars:
            raise ValueError("query 超过 MAX_QUERY_CHARS 限制")

        try:
            expanded = self._llm_query_expansion(query, options)
            parts = [query]
            if options:
                parts.append("候选项：\n" + "\n".join(options))
            if expanded:
                parts.append("查询扩展：" + expanded)
            query_text = "\n".join(parts)
            query_vector = self.embedder.embed([query_text])[0]
        except (LLMError, EmbeddingError) as exc:
            raise MemoryServiceUnavailable(str(exc)) from exc

        records = self.store.fetch_by_user(user_id)
        scored: list[tuple[float, StoredMemory]] = []
        for record in records:
            score = self._score_record(query_text, query_vector, record)
            if score is not None and score >= self.settings.min_relevance_score:
                scored.append((score, record))

        # 确定性排序便于复核：分数优先，相同分数按时间与稳定 ID。
        scored.sort(key=lambda item: (-item[0], item[1].created_at, item[1].id))
        limit = min(top_k, self.settings.max_top_k, self.settings.candidate_pool_size)
        return [
            SearchHit(
                id=record.id,
                content=record.content,
                score=round(float(score), 6),
                created_at=record.created_at,
            )
            for score, record in scored[:limit]
        ]

    def search_simple(self, query: str) -> list[SearchHit]:
        return self.search(
            query=query,
            user_id=self.settings.local_user_id,
            top_k=min(5, self.settings.max_top_k),
        )
