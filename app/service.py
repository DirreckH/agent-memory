from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.config import Settings
from app.embeddings import Embedder, EmbeddingError
from app.llm import LLMError, MemoryLLM, QueryExpansion
from app.multihop import parse_retrieval_steps, supplement_multihop
from app.schemas import MemoryMessage
from app.storage import MemoryToStore, SQLiteMemoryStore, StoredMemory
from app.temporal import (
    EventAnchorSpec,
    TemporalConstraints,
    TemporalReference,
    TemporalWindow,
    effective_time_ms,
    query_mentions_temporal,
    resolve_anchor,
    resolve_relative_window,
    time_match,
)
from app.temporal_extraction import extract_temporal_rules


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
    reference: TemporalReference


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
    return _lexical_token_similarity(_tokenize(query), _tokenize(document))


def _lexical_token_similarity(query_tokens: set[str], document_tokens: set[str]) -> float:
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
        *, governance_retriever=None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.governance_retriever = governance_retriever
        self._governance_diagnostics = threading.local()

    def initialize(self) -> None:
        self.store.initialize()
        if self.settings.governance_mode != "off":
            from app.governance.extraction import encoding
            encoding()  # 预算计时前预热固定 tokenizer。
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
        if not self.llm.enabled or not self.settings.llm_search_expansion:
            return QueryExpansion(text="", temporal=None)
        try:
            return self.llm.expand_query(query, options)
        except LLMError:
            if self.settings.llm_failure_mode == "strict":
                raise
            logger.warning("LLM 查询扩展失败，已回退到原始查询")
            return QueryExpansion(text="", temporal=None)

    def _query_temporal(self, query: str) -> TemporalConstraints | None:
        """独立于扩展文本的时间入口；LLM 关闭时仍运行受控规则。"""
        mode = self.settings.temporal_extraction_mode
        if self.settings.temporal_mode == "off" or mode == "off":
            return None
        rules = extract_temporal_rules(query)
        if rules.blocked:
            return None
        if rules.complete:
            return rules.constraints
        if mode == "rules" or not self.llm.enabled or not query_mentions_temporal(query):
            return None
        try:
            return self.llm.extract_temporal(query)
        except LLMError:
            if self.settings.llm_failure_mode == "strict":
                raise
            logger.warning("LLM 时间抽取失败，已放弃无法可靠解析的时间约束")
            return None

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
                        raw_content=str(message["content"]),
                    )
                )
            batch = None
            if self.settings.governance_mode != "off":
                from app.governance.extraction import FactIndexer, FactIndexLLM
                from app.governance.models import Source
                sources = [Source(r.id, r.raw_content, r.role, r.session_id, r.ordinal,
                                  r.source_timestamp) for r in records]
                batch = FactIndexer().extract(sources, FactIndexLLM(self.llm), self.settings)
                self._record_index_metrics(batch)
                if self.settings.governance_mode == "active" and self.settings.governance_strict_write and not batch.complete:
                    raise LLMError("治理事实索引不完整")
            self.store.complete_request(should_process, records, batch)
            if batch is not None:
                self.rebuild_summaries(user_id)
            return len(records)
        except (LLMError, EmbeddingError) as exc:
            self.store.mark_failed(should_process, type(exc).__name__)
            raise MemoryServiceUnavailable(str(exc)) from exc
        except Exception as exc:
            self.store.mark_failed(should_process, type(exc).__name__)
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

    def rebuild_summaries(self, user_id: str, generation: int | None = None) -> bool:
        from app.governance.summaries import SummaryBuilder
        try:
            snapshot = self.store.fetch_by_user(user_id, governance=True, generation=generation)
            reference = resolve_anchor([s.timestamp for s in snapshot.sources.values()],
                                       now_ms=int(time.time() * 1000), mode=self.settings.temporal_reference_mode)
            summary = SummaryBuilder().build(snapshot, budget_ms=self.settings.governance_summary_budget_ms,
                                            reference=reference.timestamp_ms)
            return self.store.publish_summaries(user_id, summary)
        except Exception as exc:
            logger.warning("治理摘要未发布 error_type=%s", type(exc).__name__)
            return False

    def index_sources(self, user_id: str, *, generation: int | None = None, source_ids=None) -> dict:
        """维护入口读取存储原文；不调用 Add 或向量器。缺失/失败来源形成持久修复队列。"""
        from app.governance.extraction import FactIndexer, FactIndexLLM
        snapshot = self.store.fetch_by_user(user_id, governance=True, generation=generation)
        ids = set(source_ids) if source_ids is not None else set(snapshot.incomplete)
        sources = [s for sid, s in snapshot.sources.items() if sid in ids]
        if not sources:
            return {"generation": snapshot.generation, "processed": 0, "complete": not snapshot.incomplete, "committed": True}
        # 新库的首个维护任务先建立独立代次，避免隐式激活未完成的索引。
        if snapshot.revision == 0 and generation is None:
            generation = self.store.create_index_generation(user_id)
            snapshot = self.store.fetch_by_user(user_id, governance=True, generation=generation)
        batch = FactIndexer().extract(sources, FactIndexLLM(self.llm), self.settings)
        self._record_index_metrics(batch)
        committed = self.store.commit_index_batch(user_id, snapshot.generation, snapshot.revision, batch)
        if committed:
            self.rebuild_summaries(user_id, snapshot.generation)
        latest = self.store.fetch_by_user(user_id, governance=True, generation=snapshot.generation)
        return {"generation": snapshot.generation, "processed": len(sources),
                "complete": committed and not latest.incomplete, "committed": committed,
                "model_calls": batch.model_calls, "extraction_elapsed_ms": batch.elapsed_ms,
                "statuses": {name: sum(s.status == name for s in batch.statuses)
                             for name in ('ready','no_fact','partial','pending','failed')}}

    @property
    def last_index_metrics(self) -> dict:
        return getattr(self._governance_diagnostics, 'index', {})

    def _record_index_metrics(self, batch) -> None:
        metrics = {'model_calls': batch.model_calls, 'elapsed_ms': batch.elapsed_ms,
                   'sources': len(batch.statuses), 'facts': len(batch.facts),
                   'incomplete': sum(s.status not in ('ready','no_fact') for s in batch.statuses)}
        self._governance_diagnostics.index = metrics
        logger.info('治理写入 model_calls=%s sources=%s facts=%s incomplete=%s elapsed_ms=%.3f',
                    metrics['model_calls'], metrics['sources'], metrics['facts'], metrics['incomplete'], metrics['elapsed_ms'])

    @staticmethod
    def _record_time_ms(record: StoredMemory) -> int | None:
        return effective_time_ms(record.source_timestamp)

    def _resolve_temporal_context(
        self,
        query: str,
        constraints: TemporalConstraints | None,
        records: list[StoredMemory],
        *,
        request_time_ms: int | None = None,
        query_time_ms: int | None = None,
    ) -> _TemporalContext | None:
        """把已经核验的时间约束解析为检索上下文；无法确定边界时回退。"""
        if self.settings.temporal_mode == "off" or constraints is None:
            return None
        # 独立抽取入口已完成规则/原文校验；这里不再用粗粒度守卫覆盖其结果。

        source_times = [self._record_time_ms(record) for record in records]
        reference = resolve_anchor(
            source_times,
            now_ms=(
                request_time_ms
                if request_time_ms is not None
                else int(time.time() * 1000)
            ),
            mode=self.settings.temporal_reference_mode,
            query_time_ms=query_time_ms,
        )
        logger.debug(
            "Temporal reference: mode=%s source=%s timestamp_ms=%s",
            self.settings.temporal_reference_mode,
            reference.source,
            reference.timestamp_ms,
        )

        window: TemporalWindow | None = None
        if constraints.event_anchor is not None:
            window = self._resolve_event_anchor_window(
                constraints.event_anchor, records
            )
        if (
            window is None
            and constraints.relative_window is not None
            and reference.timestamp_ms is not None
        ):
            try:
                window = resolve_relative_window(
                    constraints.relative_window, reference.timestamp_ms
                )
            except (OverflowError, OSError, ValueError):
                # 合法时间戳平移后仍可能超出日历范围；放弃窗口而不使检索失败。
                logger.debug("Temporal window exceeds supported calendar range")

        # 全部时间未知时沿用普通排序，避免把入库先后解释为事件先后。
        ordering = (
            constraints.ordering
            if any(t is not None for t in source_times)
            else None
        )
        if window is None and ordering is None:
            return None
        return _TemporalContext(window=window, ordering=ordering, reference=reference)

    def _resolve_event_anchor_window(
        self, anchor: EventAnchorSpec, records: list[StoredMemory]
    ) -> TemporalWindow | None:
        """用事件短语找最佳记录，以源消息时间代理事件边界；尚未抽取事件发生时间。"""
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
            start_ms=best_time, end_ms=None, hard_boundary=True,
            start_inclusive=False,
        )

    def _score_record(
        self,
        query_text: str,
        query_vector: np.ndarray,
        record: StoredMemory,
        *,
        weights: tuple[float, float, float],
        time_match: float | None,
        lexical_score: float | None = None,
    ) -> float | None:
        if record.embedding.shape != query_vector.shape:
            # 更换向量模型后旧向量维度可能不同；跳过而非返回错误结果。
            return None
        cosine = float(np.dot(query_vector, record.embedding))
        semantic = max(0.0, min(1.0, cosine))
        lexical = (_lexical_similarity(query_text, record.search_text)
                   if lexical_score is None else lexical_score)
        # 无时间约束时 time_match 为 None，加权路径与旧实现逐位一致。
        score = weights[0] * semantic + weights[1] * lexical
        if time_match is not None:
            score += weights[2] * time_match
        return score

    @classmethod
    def _rank_strict_temporal_candidates(
        cls,
        scored: list[tuple[float, StoredMemory, int | None]],
        window: TemporalWindow,
        ordering: str | None,
    ) -> list[tuple[float, StoredMemory, int | None]]:
        """确认窗内、时间未知、窗外补充分组排序；分组与 top_k 无关。"""
        groups: list[list[tuple[float, StoredMemory, int | None]]] = [[], [], []]
        for item in scored:
            timestamp = item[2]
            tier = 1 if timestamp is None else (0 if window.contains(timestamp) else 2)
            groups[tier].append(item)
        for group in groups:
            cls._sort_scored(group, ordering)
        return [item for group in groups for item in group]

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
        *,
        query_time_ms: int | None = None,
    ) -> list[SearchHit]:
        """检索记忆；query_time_ms 仅覆盖相对窗口的参考时间，不改变记录时间。"""
        # 必须在 LLM 和向量调用之前捕获，避免调用跨日/月改变本次查询的参照。
        request_time_ms = int(time.time() * 1000)
        if len(query) > self.settings.max_query_chars:
            raise ValueError("query 超过 MAX_QUERY_CHARS 限制")
        limit = min(top_k, self.settings.max_top_k, self.settings.candidate_pool_size)

        try:
            temporal = self._query_temporal(query)
            expansion = self._llm_query_expansion(query, options)
            base_parts = [query]
            if options:
                base_parts.append("候选项：\n" + "\n".join(options))
            base_text = "\n".join(base_parts)
            # 双通道：裸查询与扩展查询分别嵌入、每条记录取较大分，扩展只能
            # 提升分数、不能把弱相关证据挤下相关性地板。无扩展文本或关闭
            # 双通道时保持单通道，打分路径与 V3 逐位一致。
            query_texts = [base_text]
            if expansion.text:
                expanded_text = base_text + "\n查询扩展：" + expansion.text
                if self.settings.query_dual_channel:
                    query_texts.append(expanded_text)
                else:
                    query_texts = [expanded_text]
            steps = ()
            if (self.settings.multihop_enabled and self.llm.enabled and temporal is None
                    and limit >= self.settings.multihop_min_top_k):
                steps = parse_retrieval_steps(getattr(expansion, 'retrieval_steps', ()), query)
            # 与原查询共用一个批次；新增文本只含问题目标，不包含记忆原文。
            embed_goals = bool(steps and self.settings.multihop_embed_goals)
            embedding_texts = query_texts + ([step.query for step in steps] if embed_goals else [])
            try:
                vectors = self.embedder.embed(embedding_texts)
                if len(vectors) != len(embedding_texts):
                    raise EmbeddingError('查询向量数量不一致')
            except EmbeddingError:
                if not steps:
                    raise
                logger.warning('多跳目标向量失败，已回退到基础查询向量')
                steps = ()
                vectors = self.embedder.embed(query_texts)
            query_vectors = vectors[:len(query_texts)]
            step_vectors = (vectors[len(query_texts):] if embed_goals
                            else np.repeat(query_vectors[:1], len(steps), axis=0))
        except (LLMError, EmbeddingError) as exc:
            raise MemoryServiceUnavailable(str(exc)) from exc

        snapshot = None
        wants_governance = self.settings.governance_mode != "off"
        if wants_governance:
            from app.governance.retrieval import needs_governance
            wants_governance = needs_governance(query)
        if wants_governance:
            snapshot = self.store.fetch_by_user(user_id, governance=True)
            records = list(snapshot.records)
        else:
            records = self.store.fetch_by_user(user_id)
        context = self._resolve_temporal_context(
            query,
            temporal,
            records,
            request_time_ms=request_time_ms,
            query_time_ms=query_time_ms,
        )

        window = context.window if context is not None else None
        if window is not None:
            weights = self.settings.normalized_temporal_weights
        else:
            # 无时间窗口时退回二元权重，行为与旧实现一致。
            weights = self.settings.normalized_score_weights + (0.0,)

        # 一次查询中复用分词结果，避免每个目标重复处理全部文档；公式保持一致。
        document_tokens = {record.id: _tokenize(record.search_text) for record in records}
        query_tokens: dict[str, set[str]] = {}

        def score_record(text, vector, record, time_match=None):
            if text not in query_tokens:
                query_tokens[text] = _tokenize(text)
            return self._score_record(
                text, vector, record, weights=weights, time_match=time_match,
                lexical_score=_lexical_token_similarity(query_tokens[text], document_tokens[record.id]),
            )

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
            best: float | None = None
            for index, text in enumerate(query_texts):
                score = score_record(
                    text,
                    query_vectors[index],
                    record,
                    time_match=bonus,
                )
                if score is not None and (best is None or score > best):
                    best = score
            if best is not None and best >= self.settings.min_relevance_score:
                scored.append((best, record, record_time))

        ordering = context.ordering if context else None
        if self.settings.temporal_mode == "strict" and window is not None:
            scored = self._rank_strict_temporal_candidates(scored, window, ordering)
        else:
            self._sort_scored(scored, ordering)
        governance_plan = None
        if snapshot is not None:
            from app.governance.retrieval import plan
            governance_plan = plan(query, snapshot)
        active_governance = (self.settings.governance_mode == "active" and governance_plan is not None
                             and governance_plan.intent != "ordinary")
        if not active_governance and steps and context is None and len(scored) < limit:
            scored = supplement_multihop(
                steps=steps, records=records, baseline=scored,
                step_vectors=step_vectors, score_record=score_record,
                min_score=self.settings.min_relevance_score,
                max_rounds=self.settings.multihop_max_rounds,
                seed_limit=self.settings.multihop_seed_limit,
                bridge_limit=self.settings.multihop_bridge_limit,
                supplement_limit=self.settings.multihop_supplement_limit,
                budget_seconds=self.settings.multihop_budget_seconds,
                protected_prefix=limit,
                context_radius=self.settings.multihop_context_radius,
            )
        hits = [
            SearchHit(
                id=record.id,
                content=record.content,
                score=round(float(score), 6),
                created_at=record.created_at,
            )
            for score, record, _ in scored[:limit]
        ]
        if snapshot is not None:
            from app.governance.retrieval import GovernanceRetriever, GovernanceSearchContext
            reference = resolve_anchor([s.timestamp for s in snapshot.sources.values()], now_ms=request_time_ms,
                                       mode=self.settings.temporal_reference_mode, query_time_ms=query_time_ms)
            gov_steps = parse_retrieval_steps(getattr(expansion, "retrieval_steps", ()), query)
            started = time.perf_counter()
            retriever = self.governance_retriever or GovernanceRetriever()
            governed = retriever.retrieve(snapshot, GovernanceSearchContext(
                query, hits, reference.timestamp_ms, self.settings, limit, gov_steps,
                tuple(query_texts), query_vectors,
                lambda t, v, r: self._score_record(t, v, r, weights=weights, time_match=None), window))
            logger.info("治理检索 mode=%s intent=%s incomplete=%s elapsed_ms=%.3f",
                        self.settings.governance_mode, governance_plan.intent, len(snapshot.incomplete),
                        (time.perf_counter() - started) * 1000)
            if self.settings.governance_mode == "active":
                return governed[:limit]
        return hits

    def search_simple(self, query: str) -> list[SearchHit]:
        return self.search(
            query=query,
            user_id=self.settings.local_user_id,
            top_k=min(5, self.settings.max_top_k),
        )
