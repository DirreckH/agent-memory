from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Intent = Literal["ordinary", "current", "history", "changes", "summary"]


@dataclass(frozen=True)
class Source:
    id: str
    content: str
    role: str
    session_id: str
    ordinal: int
    timestamp: int | None
    recorded_at: str = ""
    commit_sequence: int | None = None
    raw_available: bool = True


@dataclass(frozen=True)
class Evidence:
    source_id: str
    start: int
    end: int
    quote: str
    content_hash: str
    purpose: str = "assertion"


@dataclass(frozen=True)
class Fact:
    id: str
    subject_id: str
    subject_name: str
    predicate: str
    scope: str
    value: str
    kind: str
    polarity: str
    modality: str
    operation: str
    evidence: tuple[Evidence, ...]
    source_time: int | None = None
    recorded_at: str = ""
    commit_sequence: int | None = None
    ordinal: int = 0
    valid_from: int | None = None
    valid_to: int | None = None
    time_precision: str = "unknown"
    target_value: str | None = None
    object_entities: tuple[str, ...] = ()
    identity_context: str = ""
    current_observation: bool = False
    aliases: tuple[str, ...] = ()
    target_date: int | None = None

    @property
    def slot(self) -> tuple[str, str, str]:
        return self.subject_id, self.predicate, self.scope

    @property
    def source_id(self) -> str:
        return self.evidence[0].source_id

    @property
    def quote(self) -> str:
        return self.evidence[0].quote

    @property
    def observation_time(self) -> int | None:
        return self.valid_from if self.valid_from is not None else self.source_time


@dataclass(frozen=True)
class SourceStatus:
    source_id: str
    status: str
    processed_ranges: tuple[tuple[int, int], ...] = ()
    error_type: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: Source
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.source.content[self.start:self.end]


@dataclass(frozen=True)
class IndexBatch:
    chunks: tuple[Chunk, ...]
    timeout_seconds: float
    output_tokens: int


@dataclass(frozen=True)
class IndexBatchDraft:
    items: tuple[dict[str, Any], ...]
    complete: bool = True


@dataclass(frozen=True)
class ValidatedIndexBatch:
    facts: tuple[Fact, ...] = ()
    statuses: tuple[SourceStatus, ...] = ()
    model_calls: int = 0
    elapsed_ms: float = 0.0

    @property
    def complete(self) -> bool:
        return all(s.status in ("ready", "no_fact") for s in self.statuses)


@dataclass(frozen=True)
class VersionRelation:
    newer: str
    older: str
    relation: str
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class SummaryUnit:
    id: str
    level: str
    key: str
    section: str
    text: str
    fact_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    child_ids: tuple[str, ...] = ()
    coverage: tuple[tuple[str, int, int], ...] = ()
    topic: str = ""
    stage: str = ""
    build_version: str = "extractive-v1"
    reference_ms: int | None = None
    display_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class UserSnapshot:
    user_id: str
    generation: int
    revision: int
    sources: dict[str, Source] = field(default_factory=dict)
    facts: tuple[Fact, ...] = ()
    relations: tuple[VersionRelation, ...] = ()
    statuses: tuple[SourceStatus, ...] = ()
    summaries: tuple[SummaryUnit, ...] = ()
    records: tuple[Any, ...] = ()
    blocked_facts: tuple[tuple[str, str], ...] = ()

    @property
    def incomplete(self) -> tuple[str, ...]:
        ready = {s.source_id for s in self.statuses if s.status in ("ready", "no_fact")}
        return tuple(sorted(set(self.sources) - ready))


@dataclass(frozen=True)
class StateResolution:
    selected: tuple[Fact, ...]
    excluded: dict[str, str]
    update_evidence: tuple[Evidence, ...]
    conflicts: tuple[tuple[str, ...], ...]
    incomplete_sources: tuple[str, ...]
    unknown_slots: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class SummarySnapshot:
    generation: int
    revision: int
    units: tuple[SummaryUnit, ...]
    complete: bool
