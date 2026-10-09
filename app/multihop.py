"""有界多跳补全：检索目标来自问题，桥接实体来自已召回原文。"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.storage import StoredMemory

ScoredMemory = tuple[float, StoredMemory, int | None]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalStep:
    query: str
    evidence: str


def parse_retrieval_steps(value: object, query: str) -> tuple[RetrievalStep, ...]:
    if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= 4:
        return ()
    output: list[RetrievalStep] = []
    for item in value:
        if isinstance(item, RetrievalStep):
            text, evidence = item.query, item.evidence
        elif isinstance(item, dict):
            text, evidence = item.get('query'), item.get('evidence')
        else:
            return ()
        if not isinstance(text, str) or not isinstance(evidence, str):
            return ()
        text, evidence = text.strip(), evidence.strip()
        if not text or len(text) > 300 or not evidence or evidence not in query:
            return ()
        output.append(RetrievalStep(text, evidence))
    if len({step.query.casefold() for step in output}) != len(output):
        return ()
    return tuple(output)


def steps_from_expansion(query: str, expansion: str) -> tuple[RetrievalStep, ...]:
    """复用 V3 已有的分号目标；结构线索只是启用检索补充的启发式。"""
    indirect = query.count('的') >= 2 or re.search(r'(?:上次|之前|此前|刚才).+(?:的|那)', query)
    body = re.sub(r'^\s*(?:what|which|who|where|when|why|how)\b', '', query.casefold())
    relations = re.findall(r'\b(?:who|which|that|whose|where|of|behind|by|from|at)\b', body)
    possessives = [word for word in re.findall(r"\b([\w-]+)['’]s\b", body)
                   if word not in {'what', 'who', 'where', 'that', 'it'}]
    indirect = bool(indirect or len(relations) + len(possessives) >= 2
                    or (relations and re.search(r'\b(?:i|my|we|our)\b', body)))
    if not indirect:
        return ()
    parts = [part.strip().rstrip('。.!?') for part in re.split(r'[;；]', expansion) if part.strip()]
    return parse_retrieval_steps([RetrievalStep(part, query) for part in parts], query)


def _contains(text: str, term: str) -> bool:
    if term not in text:
        return False
    if term.isascii() and re.search(r'[A-Za-z0-9]', term):
        return re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text) is not None
    return True


@dataclass(frozen=True)
class _Seed:
    record: StoredMemory
    step: int
    path: tuple[str, ...]
    score: float


def _anchor_candidates(content: str) -> list[tuple[int, str]]:
    """名称只是候选片段；共现不能证明关系、归属或因果。"""
    text = content[:2000]
    candidates: dict[str, int] = {}
    for match in re.finditer(r'《([^《》\n]{2,160})》|“([^“”\n]{2,160})”|"([^"\n]{2,160})"', text):
        phrase = next(group for group in match.groups() if group is not None)
        candidates[phrase] = 2
    stop = {'My', 'The', 'This', 'That', 'We', 'She', 'He', 'It', 'They', 'You', 'I'}
    for match in re.finditer(r'\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,4}\b', text):
        phrase = match.group()
        if 2 <= len(phrase) <= 160 and phrase not in stop:
            candidates[phrase] = 2
    # 中文无大小写；使用共享片段而不把它们宣称为已消歧的命名实体。
    for run in re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]+', text):
        for size in range(min(8, len(run)), 1, -1):
            for offset in range(len(run) - size + 1):
                candidates.setdefault(run[offset:offset + size], 1)
                if len(candidates) >= 256:
                    break
            if len(candidates) >= 256:
                break
        if len(candidates) >= 256:
            break
    return sorted(((priority, text) for text, priority in candidates.items()),
                  key=lambda item: (-item[0], -len(item[1]), item[1]))[:256]


def _local_bridges(
    seeds: list[_Seed], records: list[StoredMemory], bridge_limit: int, deadline: float,
    bridge_anchors: dict[str, tuple[str, ...]] | None = None,
    clock: Callable = time.monotonic,
) -> list[tuple[_Seed, str, list[StoredMemory]]]:
    proposals: list[tuple[int, _Seed, str, list[StoredMemory]]] = []
    # 频繁片段通常不能区分对象。上限限制宽泛共现导致的扇出。
    frequency_cap = max(2, min(8, len(records) // 10))
    for seed in seeds:
        candidates = ([(3, a) for a in bridge_anchors.get(seed.record.id, ())]
                      if bridge_anchors is not None else _anchor_candidates(seed.record.content))
        for priority, anchor in candidates:
            if clock() >= deadline:
                break
            matches: list[StoredMemory] = []
            for index, record in enumerate(records):
                if index % 64 == 0 and clock() >= deadline:
                    return []
                if record.id not in seed.path and _contains(record.content, anchor):
                    matches.append(record)
                    if len(matches) > frequency_cap:
                        break
            if not matches or len(matches) > frequency_cap:
                continue
            # 较长片段已覆盖相同后继时，不再为其子串重复检索。
            ids = {record.id for record in matches}
            if any(s.record.id == seed.record.id and anchor in previous
                   and ids == {record.id for record in targets}
                   for _, s, previous, targets in proposals):
                continue
            proposals.append((priority, seed, anchor, matches))
    proposals.sort(key=lambda item: (-item[0], len(item[3]), -len(item[2]), -item[1].score, item[2]))
    return [(seed, anchor, targets) for _, seed, anchor, targets in proposals[:bridge_limit]]


def supplement_multihop(
    *, steps: tuple[RetrievalStep, ...], records: list[StoredMemory],
    baseline: list[ScoredMemory], step_vectors: np.ndarray, score_record: Callable,
    min_score: float, max_rounds: int,
    seed_limit: int, bridge_limit: int, supplement_limit: int,
    budget_seconds: float, protected_prefix: int, context_radius: int,
    bridge_anchors: dict[str, tuple[str, ...]] | None = None,
) -> list[ScoredMemory]:
    if not steps or not baseline:
        return baseline
    clock = time.perf_counter if bridge_anchors is not None else time.monotonic
    deadline = clock() + budget_seconds
    fused = {record.id: (score, record, stamp) for score, record, stamp in baseline}
    ranked_steps: list[list[ScoredMemory]] = []
    for step, vector in zip(steps, step_vectors, strict=True):
        ranked: list[ScoredMemory] = []
        for index, record in enumerate(records):
            if index % 64 == 0 and clock() >= deadline:
                return baseline
            score = score_record(step.query, vector, record)
            if score is None or score < min_score:
                continue
            candidate = (score, record, record.source_timestamp)
            ranked.append(candidate)
            if record.id not in fused or score > fused[record.id][0]:
                fused[record.id] = candidate
        ranked.sort(key=lambda item: (-item[0], item[1].created_at, item[1].id))
        ranked_steps.append(ranked)
    seeds = [_Seed(record, 0, (record.id,), score)
             for score, record, _ in ranked_steps[0][:seed_limit]]
    if context_radius:
        context_index = {
            (record.request_id, record.session_id, record.ordinal): record
            for record in records
            if record.request_id and record.session_id and record.ordinal is not None
        }
        for seed in seeds:
            source = seed.record
            if source.ordinal is None or not source.request_id or not source.session_id:
                continue
            for offset in range(-context_radius, context_radius + 1):
                if clock() >= deadline:
                    break
                neighbor = context_index.get((source.request_id, source.session_id, source.ordinal + offset))
                if neighbor is None or neighbor.id in seed.path:
                    continue
                # 相邻原文是上下文候选，不宣称已经消解代词或证明关联。
                previous = fused.get(neighbor.id)
                if previous is None or seed.score > previous[0]:
                    fused[neighbor.id] = (seed.score, neighbor, neighbor.source_timestamp)
                logger.debug('Multihop context source_id=%s target_id=%s offset=%s',
                             source.id, neighbor.id, offset)
    seen: set[tuple[str, int]] = set()
    for _ in range(max_rounds):
        if clock() >= deadline or not seeds:
            break
        active = [seed for seed in seeds if (seed.record.id, seed.step) not in seen
                  and seed.step + 1 < len(steps)][:seed_limit]
        if not active:
            break
        seen.update((seed.record.id, seed.step) for seed in active)
        next_seeds: list[_Seed] = []
        for seed, bridge, targets in _local_bridges(active, records, bridge_limit, deadline, bridge_anchors, clock):
            if clock() >= deadline:
                break
            next_step = seed.step + 1
            text = bridge + '\n' + steps[next_step].query
            matches: list[tuple[float, StoredMemory]] = []
            for record in targets:
                # 只复用已存储向量与问题目标向量；桥接原文不发送给外部模型。
                scores = [score_record(text, vector, record)
                          for vector in (step_vectors[next_step], seed.record.embedding)]
                valid = [score for score in scores if score is not None]
                if not valid or (score := max(valid)) < min_score:
                    continue
                matches.append((score, record))
                logger.debug('Multihop candidate source_id=%s target_id=%s step=%s',
                             seed.record.id, record.id, next_step)
                candidate = (score, record, record.source_timestamp)
                if record.id not in fused or score > fused[record.id][0]:
                    fused[record.id] = candidate
            matches.sort(key=lambda item: (-item[0], item[1].created_at, item[1].id))
            next_seeds.extend(_Seed(record, next_step, seed.path + (record.id,), score)
                              for score, record in matches[:seed_limit])
        next_seeds.sort(key=lambda seed: (-seed.score, seed.record.id))
        seeds = next_seeds[:seed_limit]
    prefix = baseline[:protected_prefix]
    prefix_ids = {record.id for _, record, _ in prefix}
    tail = [item for key, item in fused.items() if key not in prefix_ids]
    tail.sort(key=lambda item: (-item[0], item[1].created_at, item[1].id))
    tail = tail[:supplement_limit]
    if prefix:
        ceiling = prefix[-1][0]
        tail = [(min(score, ceiling), record, stamp) for score, record, stamp in tail]
    return prefix + tail
