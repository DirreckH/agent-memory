"""查询参考时间回归：真实 SQLite 与检索流程，离线向量/LLM 替身。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.llm import QueryExpansion
from app.main import create_app
from app.schemas import MemoryMessage
from app.service import MemoryService
from app.storage import SQLiteMemoryStore
from app.temporal import TemporalConstraints, parse_temporal, resolve_anchor


def _ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


JAN = _ms("2024-01-15T00:00:00Z")
JUL = _ms("2026-07-15T00:00:00Z")
AUG = _ms("2026-08-15T00:00:00Z")
SEP = _ms("2026-09-15T00:00:00Z")
OCT = _ms("2026-10-15T00:00:00Z")
LAST_MONTH = parse_temporal(
    {"windows": [{"kind": "calendar", "unit": "month", "offset": -1}]}
)
QUERY = "Aurora project updates last month"


class ConstantEmbedder:
    """固定相关性以隔离时间选择的行为，不评价模型质量。"""

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 1), dtype=np.float32)


class StubLLM:
    enabled = True

    def __init__(self, temporal: TemporalConstraints | None = LAST_MONTH) -> None:
        self.temporal = temporal

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        return {}

    def expand_query(self, query: str, options: list[str] | None) -> QueryExpansion:
        return QueryExpansion(text="", temporal=self.temporal)

    def extract_temporal(self, query: str) -> TemporalConstraints | None:
        return self.temporal


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "database_path": tmp_path / "reference.db",
        "llm_provider": "none",
        "input_log_enabled": False,
        "memory_api_key": "",
        "warmup_embedding_on_startup": False,
        "semantic_weight": 1.0,
        "lexical_weight": 0.0,
        "temporal_reference_mode": "replay",
    }
    values.update(overrides)
    return Settings(**values)


def _service(tmp_path: Path, **overrides: object) -> MemoryService:
    settings = _settings(tmp_path, **overrides)
    service = MemoryService(
        settings, SQLiteMemoryStore(settings.database_path), ConstantEmbedder(), StubLLM()
    )
    service.initialize()
    return service


def _add(
    service: MemoryService,
    name: str,
    timestamp: int | None,
    user_id: str = "user-1",
) -> None:
    service.add(
        request_id=f"{user_id}:{name}",
        messages=[MemoryMessage(role="user", content=name, timestamp=timestamp)],
        user_id=user_id,
        session_id="session-1",
    )


def _plain(service: MemoryService) -> MemoryService:
    return MemoryService(
        service.settings.model_copy(update={"temporal_mode": "off"}),
        service.store,
        service.embedder,
        service.llm,
    )


@pytest.mark.parametrize("mode", ["replay", "realtime"])
def test_reference_mode_can_be_configured_from_environment(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("TEMPORAL_REFERENCE_MODE", mode)
    assert Settings(_env_file=None).temporal_reference_mode == mode


def test_reference_mode_defaults_to_replay_and_rejects_unknown_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEMPORAL_REFERENCE_MODE", raising=False)
    assert Settings(_env_file=None).temporal_reference_mode == "replay"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, temporal_reference_mode="automatic")


@pytest.mark.parametrize("mode", ["replay", "realtime"])
def test_explicit_query_time_precedes_both_reference_modes(mode: str) -> None:
    reference = resolve_anchor(
        [JAN, None, AUG], now_ms=SEP, mode=mode, query_time_ms=JUL
    )
    assert reference.timestamp_ms == JUL
    assert reference.source == "query_time"


def test_epoch_zero_is_an_explicit_query_time() -> None:
    reference = resolve_anchor([AUG], now_ms=SEP, query_time_ms=0)
    assert reference.timestamp_ms == 0
    assert reference.source == "query_time"


@pytest.mark.parametrize("query_time", [True, -1, 1.5, 10**18])
def test_invalid_explicit_query_time_is_rejected(query_time: object) -> None:
    with pytest.raises(ValueError, match="query_time_ms"):
        resolve_anchor([AUG], now_ms=SEP, query_time_ms=query_time)


@pytest.mark.parametrize("source_time", [None, 10**18], ids=["missing", "out-of-range"])
def test_replay_backfill_does_not_shift_existing_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_time: int | None,
) -> None:
    service = _service(tmp_path)
    _add(service, "January update", JAN)
    _add(service, "July update", JUL)
    _add(service, "August update", AUG)
    before = service.search(QUERY, "user-1", 10)

    monkeypatch.setattr("app.service.utc_now_text", lambda: "2029-10-15T00:00:00.000Z")
    _add(service, "Undated imported history", source_time)
    after = service.search(QUERY, "user-1", 10)
    scores = {hit.id: hit.score for hit in after}
    assert {hit.id: hit.score for hit in before} == {
        hit.id: scores[hit.id] for hit in before
    }
    undated = next(
        record for record in service.store.fetch_by_user("user-1")
        if record.content.endswith("Undated imported history")
    )
    assert undated.source_timestamp == source_time
    assert undated.created_at == "2029-10-15T00:00:00.000Z"
    assert service._record_time_ms(undated) is None


@pytest.mark.parametrize("mode", ["soft", "strict"])
@pytest.mark.parametrize(
    "temporal",
    [
        LAST_MONTH,
        parse_temporal({"ordering": "latest"}),
        parse_temporal({
            "windows": [{"kind": "calendar", "unit": "month", "offset": -1}],
            "ordering": "earliest",
        }),
        parse_temporal({"event_anchor": {"event": "moving", "direction": "before"}}),
    ],
    ids=["window", "ordering", "window-and-ordering", "event"],
)
def test_undated_replay_falls_back_to_plain_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mode: str, temporal: TemporalConstraints | None,
) -> None:
    service = _service(tmp_path, temporal_mode=mode)
    service.llm = StubLLM(temporal)
    monkeypatch.setattr("app.service.utc_now_text", lambda: "2026-09-15T00:00:00.000Z")
    _add(service, "moving project alpha", None)
    monkeypatch.setattr("app.service.utc_now_text", lambda: "2026-10-15T00:00:00.000Z")
    _add(service, "moving project beta", None)
    query = "latest project before moving last month"
    assert service.search(query, "user-1", 10) == _plain(service).search(
        query, "user-1", 10
    )


def test_realtime_uses_search_start_time_before_llm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, temporal_reference_mode="realtime", temporal_mode="strict")
    _add(service, "July update", JUL)
    _add(service, "August update", AUG)
    clock = [SEP]
    monkeypatch.setattr("app.service.time.time", lambda: clock[0] / 1000)

    class DelayedLLM(StubLLM):
        def expand_query(self, query: str, options: list[str] | None) -> QueryExpansion:
            clock[0] = OCT  # 模拟调用跨过月界，不得改变该次查询的参考时刻。
            return super().expand_query(query, options)

    service.llm = DelayedLLM()
    hits = service.search(QUERY, "user-1", 1)
    assert hits[0].content.endswith("August update")


@pytest.mark.parametrize("mode", ["replay", "realtime"])
def test_explicit_query_time_reaches_window_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    service = _service(tmp_path, temporal_reference_mode=mode, temporal_mode="strict")
    _add(service, "January update", JAN)
    _add(service, "August update", AUG)
    monkeypatch.setattr("app.service.time.time", lambda: SEP / 1000)
    hits = service.search(
        QUERY, "user-1", 1, query_time_ms=_ms("2024-02-15T00:00:00Z")
    )
    assert hits[0].content.endswith("January update")


def test_undated_record_cannot_win_latest_ordering(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.llm = StubLLM(parse_temporal({"ordering": "latest"}))
    _add(service, "known update", JAN)
    _add(service, "undated update", None)
    hits = service.search("latest update", "user-1", 2)
    assert hits[0].content.endswith("known update")
    assert hits[1].content.endswith("undated update")


def test_other_user_timestamps_do_not_change_reference(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _add(service, "July update", JUL)
    _add(service, "August update", AUG)
    before = service.search(QUERY, "user-1", 10)
    _add(service, "future record", _ms("2030-01-01T00:00:00Z"), user_id="user-2")
    assert service.search(QUERY, "user-1", 10) == before


@pytest.mark.parametrize("mode", ["replay", "realtime"])
def test_temporal_off_still_bypasses_reference_policy(tmp_path: Path, mode: str) -> None:
    service = _service(tmp_path, temporal_mode="off", temporal_reference_mode=mode)
    _add(service, "July update", JUL)
    _add(service, "undated update", None)
    baseline = service.search(QUERY, "user-1", 10)
    assert service.search(QUERY, "user-1", 10, query_time_ms=JAN) == baseline


def test_calendar_overflow_falls_back_without_failing_search(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _add(service, "valid historical record", _ms("9999-12-15T00:00:00Z"))
    service.llm = StubLLM(parse_temporal({
        "windows": [{"kind": "calendar", "unit": "year", "offset": 1}],
    }))
    query = "project next year"
    assert service.search(query, "user-1", 5) == _plain(service).search(query, "user-1", 5)


@pytest.mark.parametrize("mode", ["replay", "realtime"])
def test_http_contract_and_mode_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    settings = _settings(tmp_path, temporal_reference_mode=mode, temporal_mode="strict")
    app = create_app(settings, embedder=ConstantEmbedder(), llm=StubLLM())
    with TestClient(app) as client:
        payload = {
            "request_id": "api-reference",
            "user_id": "user-1",
            "session_id": "session-1",
            "messages": [
                {"role": "user", "content": "July update", "timestamp": JUL},
                {"role": "user", "content": "August update", "timestamp": AUG},
            ],
        }
        expected = {
            "success": True, "request_id": "api-reference",
            "user_id": "user-1", "session_id": "session-1",
        }
        assert client.post("/set", json=payload).json() == expected
        assert client.post("/set", json=payload).json() == expected
        monkeypatch.setattr("app.service.time.time", lambda: SEP / 1000)
        request = {"query": QUERY, "user_id": "user-1", "top_k": 1}
        response = client.post("/get", json=request)
        assert response.status_code == 200
        hit = response.json()["data"][0]
        assert set(hit) == {"id", "content", "score", "created_at"}
        expected_content = "July update" if mode == "replay" else "August update"
        assert hit["content"].endswith(expected_content)
        # query_time_ms 仅为内部调用参数，HTTP schema 保持原契约。
        assert client.post("/get", json={**request, "query_time_ms": JAN}).status_code == 422
