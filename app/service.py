from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.config import Settings
from app.embeddings import Embedder, EmbeddingError
from app.llm import LLMError, MemoryLLM, QueryExpansion
from app.schemas import MemoryMessage
from app.storage import MemoryToStore, SQLiteMemoryStore, StoredMemory
from app.temporal import (
    EventAnchorSpec,
    TemporalConstraints,
    TemporalWindow,
    effective_time_ms,
    query_mentions_temporal,
    resolve_anchor,
    resolve_relative_window,
    time_match,
)


logger = logging.getLogger(__name__)


class MemoryServiceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    id: str
    content: str
    score: float
    created_at: str


@dataclass(frozen=True)
class _TemporalContext:
    """一次检索中解析出的有效时间上下文；ordering 可与 window 同时存在。"""

    window: TemporalWindow | None
    ordering: str | None


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
    ) -> QueryExpansion:
        if not self.llm.enabled:
            return QueryExpansion(text="", temporal=None)
        try:
            return self.llm.expand_query(query, options)
        except LLMError:
            if self.settings.llm_failure_mode == "strict":
                raise
            logger.warning("LLM 查询扩展失败，已回退到原始查询")
            return QueryExpansion(text="", temporal=None)

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

    @staticmethod
    def _record_time_ms(record: StoredMemory) -> int | None:
        return effective_time_ms(record.source_timestamp, record.created_at)

    def _resolve_temporal_context(
        self,
        query: str,
        constraints: TemporalConstraints | None,
        records: list[StoredMemory],
    ) -> _TemporalContext | None:
        """把 LLM 抽取的时间约束解析为有效检索上下文；不可靠时返回 None。"""
        if self.settings.temporal_mode == "off" or constraints is None:
            return None
        # 程序化防御：LLM 声称的时间约束必须能在查询原文中回查到时间信号，
        # 否则视为幻觉约束直接丢弃。
        if not query_mentions_temporal(query):
            return None

        effective_times = [self._record_time_ms(record) for record in records]
        anchor_ms = resolve_anchor(
            effective_times, now_ms=int(time.time() * 1000)
        )

        window: TemporalWindow | None = None
        if constraints.event_anchor is not None:
            window = self._resolve_event_anchor_window(
                constraints.event_anchor, records
            )
        if window is None and constraints.relative_window is not None:
            window = resolve_relative_window(
                constraints.relative_window, anchor_ms
            )

        if window is None and constraints.ordering is None:
            return None
        return _TemporalContext(window=window, ordering=constraints.ordering)

    def _resolve_event_anchor_window(
        self, anchor: EventAnchorSpec, records: list[StoredMemory]
    ) -> TemporalWindow | None:
        """两阶段检索的锚点定位：用事件短语找最佳命中记录，以其事件时间为窗口边界。"""
        try:
            event_vector = self.embedder.embed([anchor.event])[0]
        except EmbeddingError:
            return None
        weights = self.settings.normalized_score_weights + (0.0,)
        best_score = -1.0
        best_time: int | None = None
        for record in records:
            score = self._score_record(
                anchor.event, event_vector, record, weights=weights, time_match=None
            )
            if score is not None and score > best_score:
                best_score = score
                best_time = self._record_time_ms(record)
        if (
            best_time is None
            or best_score < self.settings.temporal_event_anchor_min_score
        ):
            # 找不到高置信事件记录时放弃约束，退回常规检索而不是猜测边界。
            return None
        if anchor.direction == "before":
            return TemporalWindow(
                start_ms=None, end_ms=best_time, hard_boundary=True
            )
        return TemporalWindow(
            start_ms=best_time, end_ms=None, hard_boundary=True
        )

    def _score_record(
        self,
        query_text: str,
        query_vector: np.ndarray,
        record: StoredMemory,
        *,
        weights: tuple[float, float, float],
        time_match: float | None,
    ) -> float | None:
        if record.embedding.shape != query_vector.shape:
            # 更换向量模型后旧向量维度可能不同；跳过而非返回错误结果。
            return None
        cosine = float(np.dot(query_vector, record.embedding))
        semantic = max(0.0, min(1.0, cosine))
        lexical = _lexical_similarity(query_text, record.search_text)
        # 无时间约束时 time_match 为 None，加权路径与旧实现逐位一致。
        score = weights[0] * semantic + weights[1] * lexical
        if time_match is not None:
            score += weights[2] * time_match
        return score

    @staticmethod
    def _apply_strict_temporal_filter(
        scored: list[tuple[float, StoredMemory, int | None]],
        window: TemporalWindow | None,
        limit: int,
    ) -> list[tuple[float, StoredMemory, int | None]]:
        """严格模式：窗口内候选足够时才截掉窗口外记录；时间未知的记录始终保留。"""
        if window is None:
            return scored
        in_window = [
            item for item in scored if item[2] is None or window.contains(item[2])
        ]
        if len(in_window) >= limit:
            return in_window
        return scored

    @staticmethod
    def _sort_scored(
        scored: list[tuple[float, StoredMemory, int | None]],
        ordering: str | None,
    ) -> None:
        if ordering is not None:
            # 首末次查询：在通过相关度门槛的候选内按时间排序，时间未知排最后。
            if ordering == "earliest":
                scored.sort(
                    key=lambda item: (item[2] is None, item[2] or 0, -item[0])
                )
            else:
                scored.sort(
                    key=lambda item: (item[2] is None, -(item[2] or 0), -item[0])
                )
            return
        # 确定性排序便于复核：分数优先，相同分数按时间与稳定 ID。
        scored.sort(key=lambda item: (-item[0], item[1].created_at, item[1].id))

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
            expansion = self._llm_query_expansion(query, options)
            parts = [query]
            if options:
                parts.append("候选项：\n" + "\n".join(options))
            if expansion.text:
                parts.append("查询扩展：" + expansion.text)
            query_text = "\n".join(parts)
            query_vector = self.embedder.embed([query_text])[0]
        except (LLMError, EmbeddingError) as exc:
            raise MemoryServiceUnavailable(str(exc)) from exc

        records = self.store.fetch_by_user(user_id)
        context = self._resolve_temporal_context(query, expansion.temporal, records)

        window = context.window if context is not None else None
        if window is not None:
            weights = self.settings.normalized_temporal_weights
        else:
            # 无时间窗口时退回二元权重，行为与旧实现一致。
            weights = self.settings.normalized_score_weights + (0.0,)

        scored: list[tuple[float, StoredMemory, int | None]] = []
        for record in records:
            record_time = self._record_time_ms(record)
            bonus = (
                time_match(
                    record_time,
                    window,
                    half_life_days=self.settings.temporal_decay_half_life_days,
                )
                if window is not None
                else None
            )
            score = self._score_record(
                query_text, query_vector, record, weights=weights, time_match=bonus
            )
            if score is not None and score >= self.settings.min_relevance_score:
                scored.append((score, record, record_time))

        limit = min(top_k, self.settings.max_top_k, self.settings.candidate_pool_size)
        if self.settings.temporal_mode == "strict":
            scored = self._apply_strict_temporal_filter(scored, window, limit)
        self._sort_scored(scored, context.ordering if context else None)
        return [
            SearchHit(
                id=record.id,
                content=record.content,
                score=round(float(score), 6),
                created_at=record.created_at,
            )
            for score, record, _ in scored[:limit]
        ]

    def search_simple(self, query: str) -> list[SearchHit]:
        return self.search(
            query=query,
            user_id=self.settings.local_user_id,
            top_k=min(5, self.settings.max_top_k),
        )
