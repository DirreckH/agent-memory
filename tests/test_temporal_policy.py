"""时间端点与严格模式回归：真实存储流程，固定向量和抽取替身。"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.llm import QueryExpansion
from app.schemas import MemoryMessage
from app.service import MemoryService
from app.storage import SQLiteMemoryStore
from app.temporal import (
    EventAnchorSpec, RelativeWindowSpec, TemporalConstraints, TemporalWindow,
    resolve_relative_window, time_match,
)


def ms(value: str) -> int:
    return int(datetime.fromisoformat(value + "+00:00").timestamp() * 1000)


class ConstantEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 1), dtype=np.float32)


class StubLLM:
    enabled = True

    def __init__(self, constraints: TemporalConstraints) -> None:
        self.constraints = constraints

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        return {}

    def expand_query(self, query: str, options: list[str] | None) -> QueryExpansion:
        return QueryExpansion("", self.constraints)

    def extract_temporal(self, query: str) -> TemporalConstraints:
        return self.constraints


def service_at(tmp_path: Path, constraints: TemporalConstraints) -> MemoryService:
    settings = Settings(
        _env_file=None, database_path=tmp_path / "policy.db", llm_provider="none",
        temporal_mode="strict", semantic_weight=1, lexical_weight=0,
        input_log_enabled=False, warmup_embedding_on_startup=False,
    )
    service = MemoryService(
        settings, SQLiteMemoryStore(settings.database_path), ConstantEmbedder(),
        StubLLM(constraints),
    )
    service.initialize()
    return service


def add(service: MemoryService, name: str, timestamp: int | None) -> None:
    service.add(
        request_id=name, user_id="u", session_id="s",
        messages=[MemoryMessage(role="user", content=name, timestamp=timestamp)],
    )


@pytest.mark.parametrize("unit", ["day", "week", "month", "year"])
def test_past_window_includes_reference_and_excludes_next_millisecond(unit: str) -> None:
    anchor = ms("2026-10-03T12:00:00")
    window = resolve_relative_window(RelativeWindowSpec("rolling", unit, 3, 0, "past"), anchor)
    assert window.contains(window.start_ms)
    assert window.contains(anchor)
    assert not window.contains(anchor + 1)


def test_calendar_adjacent_periods_do_not_overlap() -> None:
    anchor = ms("2026-10-03T12:00:00")
    previous = resolve_relative_window(RelativeWindowSpec("calendar", "month", 1, -1, "past"), anchor)
    current = resolve_relative_window(RelativeWindowSpec("calendar", "month", 1, 0, "past"), anchor)
    assert previous.end_ms == current.start_ms
    assert not previous.contains(current.start_ms)
    assert current.contains(current.start_ms)


def test_strict_recent_query_keeps_latest_message(tmp_path: Path) -> None:
    constraints = TemporalConstraints(RelativeWindowSpec("rolling", "day", 3, 0, "past"), None, "latest")
    service = service_at(tmp_path, constraints)
    add(service, "older", ms("2026-10-01T12:00:00"))
    add(service, "latest", ms("2026-10-03T12:00:00"))
    assert service.search("latest project in the past 3 days", "u", 1)[0].content.endswith("latest")


@pytest.mark.parametrize("ordering", [None, "earliest", "latest"])
def test_strict_groups_have_stable_prefix_for_every_top_k(tmp_path: Path, ordering: str | None) -> None:
    constraints = TemporalConstraints(RelativeWindowSpec("calendar", "month", 1, -1, "past"), None, ordering)
    service = service_at(tmp_path, constraints)
    add(service, "outside_old", ms("2024-01-01T00:00:00"))
    add(service, "inside_a", ms("2026-09-02T00:00:00"))
    add(service, "inside_b", ms("2026-09-20T00:00:00"))
    add(service, "unknown", None)
    add(service, "outside_new", ms("2026-11-01T00:00:00"))
    query = "project last month" + (f" {ordering}" if ordering else "")
    anchor = ms("2026-10-03T00:00:00")
    all_hits = service.search(query, "u", 10, query_time_ms=anchor)
    names = [hit.content.splitlines()[-1] for hit in all_hits]
    assert set(names[:2]) == {"inside_a", "inside_b"}
    assert names[2] == "unknown"
    assert set(names[3:]) == {"outside_old", "outside_new"}
    for k in range(1, 6):
        assert service.search(query, "u", k, query_time_ms=anchor) == all_hits[:k]


@pytest.mark.parametrize("direction", ["before", "after"])
def test_event_boundary_excludes_event_itself(tmp_path: Path, direction: str) -> None:
    service = service_at(tmp_path, TemporalConstraints(None, EventAnchorSpec("moving", direction), None))
    event_time = ms("2026-09-15T00:00:00")
    add(service, "moving", event_time)
    window = service._resolve_event_anchor_window(EventAnchorSpec("moving", direction), service.store.fetch_by_user("u"))
    assert window is not None
    assert not window.contains(event_time)
    assert time_match(event_time, window, half_life_days=30) == 0
    assert window.contains(event_time - 1 if direction == "before" else event_time + 1)


def test_excluded_boundary_has_nonzero_distance() -> None:
    window = TemporalWindow(1000, 2000)
    assert window.distance_ms(2000) > 0
    assert time_match(2000, window, half_life_days=30) < 1
