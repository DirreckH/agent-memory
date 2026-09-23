"""时间感知检索：纯函数单元测试 + 服务级行为测试。

服务级测试使用离线 Bag-of-Words 向量器与桩 LLM，验证软打分、严格模式、
事件锚定与守卫逻辑的端到端数据流，不验证 LLM 的真实语义能力。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.embeddings import normalize_rows
from app.llm import QueryExpansion
from app.schemas import MemoryMessage
from app.service import MemoryService
from app.storage import SQLiteMemoryStore
from app.temporal import (
    EventAnchorSpec,
    RelativeWindowSpec,
    TemporalConstraints,
    TemporalWindow,
    effective_time_ms,
    parse_temporal,
    query_mentions_temporal,
    resolve_anchor,
    resolve_relative_window,
    time_match,
)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


# 固定测试时间轴（毫秒时间戳）。
OLD = _ms(datetime(2024, 1, 15, tzinfo=timezone.utc))
EVENT_TIME = _ms(datetime(2024, 6, 15, tzinfo=timezone.utc))
EARLY = _ms(datetime(2025, 1, 15, tzinfo=timezone.utc))
RECENT = _ms(datetime(2026, 8, 1, tzinfo=timezone.utc))
SENTINEL = _ms(datetime(2026, 8, 15, tzinfo=timezone.utc))


# ---------------------------------------------------------------- 单元测试


def test_parse_temporal_rolling_window() -> None:
    payload = {
        "windows": [
            {"kind": "rolling", "unit": "day", "amount": 3, "direction": "past"}
        ]
    }
    constraints = parse_temporal(payload)
    assert constraints is not None
    assert constraints.relative_window == RelativeWindowSpec(
        kind="rolling", unit="day", amount=3, offset=0, direction="past"
    )
    assert constraints.event_anchor is None
    assert constraints.ordering is None


def test_parse_temporal_event_anchor() -> None:
    payload = {"event_anchor": {"event": "搬到上海", "direction": "before"}}
    constraints = parse_temporal(payload)
    assert constraints is not None
    assert constraints.event_anchor == EventAnchorSpec(
        event="搬到上海", direction="before"
    )
    assert constraints.relative_window is None


def test_parse_temporal_calendar_window_and_ordering() -> None:
    payload = {
        "windows": [{"kind": "calendar", "unit": "month", "offset": -1}],
        "ordering": "latest",
    }
    constraints = parse_temporal(payload)
    assert constraints is not None
    assert constraints.relative_window is not None
    assert constraints.relative_window.offset == -1
    assert constraints.ordering == "latest"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "temporal",
        {},
        {"windows": []},
        {"windows": [{"kind": "rolling"}]},
        {"windows": [{"kind": "rolling", "unit": "epoch", "amount": 1}]},
        {"windows": [{"kind": "rolling", "unit": "day", "amount": 0}]},
        {"windows": [{"kind": "calendar", "unit": "month", "offset": True}]},
        {"event_anchor": {"event": "", "direction": "before"}},
        {"event_anchor": {"event": "搬家", "direction": "during"}},
        {"ordering": "whenever"},
    ],
)
def test_parse_temporal_invalid_payload_returns_none(payload) -> None:
    assert parse_temporal(payload) is None


def test_resolve_calendar_month_window() -> None:
    anchor = _ms(datetime(2026, 9, 13, tzinfo=timezone.utc))
    spec = RelativeWindowSpec(
        kind="calendar", unit="month", amount=1, offset=-1, direction="past"
    )
    window = resolve_relative_window(spec, anchor)
    assert window.start_ms == _ms(datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert window.end_ms == _ms(datetime(2026, 9, 1, tzinfo=timezone.utc))


def test_resolve_calendar_year_window() -> None:
    anchor = _ms(datetime(2026, 9, 13, tzinfo=timezone.utc))
    spec = RelativeWindowSpec(
        kind="calendar", unit="year", amount=1, offset=-1, direction="past"
    )
    window = resolve_relative_window(spec, anchor)
    assert window.start_ms == _ms(datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert window.end_ms == _ms(datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_resolve_rolling_past_days_window() -> None:
    anchor = _ms(datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc))
    spec = RelativeWindowSpec(
        kind="rolling", unit="day", amount=3, offset=0, direction="past"
    )
    window = resolve_relative_window(spec, anchor)
    assert window.end_ms == anchor
    assert window.start_ms == anchor - 3 * 86_400_000


def test_resolve_rolling_month_uses_calendar_math() -> None:
    anchor = _ms(datetime(2026, 3, 31, tzinfo=timezone.utc))
    spec = RelativeWindowSpec(
        kind="rolling", unit="month", amount=1, offset=0, direction="past"
    )
    window = resolve_relative_window(spec, anchor)
    # 2026 年 2 月只有 28 天，月末日期按目标月天数收敛。
    assert window.start_ms == _ms(datetime(2026, 2, 28, tzinfo=timezone.utc))


def test_resolve_anchor_prefers_conversation_frontier() -> None:
    assert resolve_anchor([10, None, 999, 100], now_ms=50_000) == 999


def test_resolve_anchor_falls_back_to_server_clock() -> None:
    assert resolve_anchor([None, None], now_ms=50_000) == 50_000


def test_time_match_unknown_time_is_neutral() -> None:
    window = TemporalWindow(start_ms=0, end_ms=1000)
    assert time_match(None, window, half_life_days=30.0) == 0.5


def test_time_match_in_window_full_score() -> None:
    window = TemporalWindow(start_ms=0, end_ms=1000)
    assert time_match(500, window, half_life_days=30.0) == 1.0


def test_time_match_decays_with_distance() -> None:
    window = TemporalWindow(start_ms=0, end_ms=1000)
    near = time_match(1000 + 86_400_000, window, half_life_days=30.0)
    far = time_match(1000 + 10 * 86_400_000, window, half_life_days=30.0)
    assert 0 < far < near < 1.0
    # 半衰期含义：距离一个半衰期时分数约为 0.5。
    one_half_life = time_match(
        1000 + 30 * 86_400_000, window, half_life_days=30.0
    )
    assert one_half_life == pytest.approx(0.5)


def test_time_match_unbounded_windows() -> None:
    before = TemporalWindow(start_ms=None, end_ms=5000)
    assert time_match(1000, before, half_life_days=30.0) == 1.0
    assert time_match(6000, before, half_life_days=30.0) < 1.0
    after = TemporalWindow(start_ms=5000, end_ms=None)
    assert time_match(6000, after, half_life_days=30.0) == 1.0
    assert time_match(1000, after, half_life_days=30.0) < 1.0


def test_effective_time_prefers_source_timestamp() -> None:
    assert effective_time_ms(1234, "2026-01-01T00:00:00.000Z") == 1234


def test_effective_time_parses_created_at_fallback() -> None:
    value = effective_time_ms(None, "2026-01-01T00:00:00.000Z")
    assert value == _ms(datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_effective_time_invalid_created_at_returns_none() -> None:
    assert effective_time_ms(None, "not-a-date") is None
    assert effective_time_ms(None, None) is None


@pytest.mark.parametrize(
    "query",
    [
        "我上个月说过不吃什么？",
        "搬到上海之前我住在哪里？",
        "最近三天我的日程",
        "我第一次提到 Aurora 是什么时候",
        "Where did I live before moving to Shanghai?",
        "What did I say 3 days ago?",
        "What was the last time I mentioned Aurora?",
    ],
)
def test_query_mentions_temporal_positive(query: str) -> None:
    assert query_mentions_temporal(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "Aurora project status",
        "What coffee does Alice prefer?",
        "数据库连接配置",
    ],
)
def test_query_mentions_temporal_negative(query: str) -> None:
    assert query_mentions_temporal(query) is False


# ------------------------------------------------------------ 服务级测试


class BagOfWordsEmbedder:
    """离线向量器：按分词构造确定性向量，不下载模型。"""

    dimension = 256

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = "".join(
            character.casefold() if character.isalnum() else " "
            for character in text
        )
        return [token for token in normalized.split() if token]

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                index = self._vocabulary.setdefault(
                    token, len(self._vocabulary)
                )
                matrix[row, index] += 1.0
        return normalize_rows(matrix)


class StubMemoryLLM:
    enabled = True

    def __init__(self, expansion: QueryExpansion) -> None:
        self.expansion = expansion

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        return {}

    def expand_query(self, query: str, options: list[str] | None) -> QueryExpansion:
        return self.expansion


def _window_expansion(unit: str = "month", amount: int = 6) -> QueryExpansion:
    return QueryExpansion(
        text="",
        temporal=TemporalConstraints(
            relative_window=RelativeWindowSpec(
                kind="rolling", unit=unit, amount=amount, offset=0, direction="past"
            ),
            event_anchor=None,
            ordering=None,
        ),
    )


def _make_settings(tmp_path: Path, **overrides: object) -> Settings:
    defaults: dict[str, object] = dict(
        _env_file=None,
        database_path=tmp_path / "memory.db",
        embedding_cache_dir=tmp_path / "models",
        llm_provider="none",
        min_relevance_score=0.05,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_service(
    settings: Settings, llm: object, embedder: BagOfWordsEmbedder
) -> MemoryService:
    service = MemoryService(
        settings, SQLiteMemoryStore(settings.database_path), embedder, llm
    )
    service.initialize()
    return service


def _remember(
    service: MemoryService, content: str, timestamp: int, user_id: str = "user-1"
) -> None:
    service.add(
        request_id=f"req:{timestamp}:{content[:24]}",
        messages=[MemoryMessage(role="user", content=content, timestamp=timestamp)],
        user_id=user_id,
        session_id="session-1",
    )


def test_temporal_window_promotes_in_window_records(tmp_path: Path) -> None:
    """软模式：滚动窗口内的同内容记录分数必须高于窗口外记录。"""
    embedder = BagOfWordsEmbedder()
    expansion = _window_expansion()
    service = _make_service(
        _make_settings(tmp_path, temporal_mode="soft"),
        StubMemoryLLM(expansion),
        embedder,
    )
    off_service = _make_service(
        _make_settings(tmp_path, temporal_mode="off"),
        StubMemoryLLM(expansion),
        embedder,
    )
    _remember(service, "Aurora project status update", OLD)
    _remember(service, "Aurora project status update", RECENT)

    query = "最近半年的 Aurora project status"
    hits = service.search(query=query, user_id="user-1", top_k=10)
    assert len(hits) == 2
    assert hits[0].created_at == "2026-08-01T00:00:00.000Z"
    assert hits[0].score > hits[1].score

    # 回归底线：off 模式下无时间项，同内容记录分数相等，按 created_at 升序。
    off_hits = off_service.search(query=query, user_id="user-1", top_k=10)
    assert len(off_hits) == 2
    assert off_hits[0].created_at == "2024-01-15T00:00:00.000Z"
    assert off_hits[0].score == off_hits[1].score


def test_strict_mode_drops_out_of_window_records(tmp_path: Path) -> None:
    """严格模式：窗口内候选足够时，窗口外的高分记录被截掉。"""
    embedder = BagOfWordsEmbedder()
    expansion = _window_expansion()
    soft = _make_service(
        _make_settings(tmp_path, temporal_mode="soft"),
        StubMemoryLLM(expansion),
        embedder,
    )
    strict = _make_service(
        _make_settings(tmp_path, temporal_mode="strict"),
        StubMemoryLLM(expansion),
        embedder,
    )
    exact = "Aurora project status update meeting notes"
    _remember(soft, exact, OLD)
    _remember(soft, "Aurora project", RECENT)
    _remember(soft, "Aurora timeline", RECENT + 3_600_000)
    # 哨兵记录：与查询无重叠，只为把对话前沿锚点推到 2026-08-15。
    _remember(soft, "weekly weather diary", SENTINEL)

    query = "Aurora project status update meeting notes in the last 6 months"
    soft_hits = soft.search(query=query, user_id="user-1", top_k=5)
    # content 是渲染后的 "[user | 时间]\n正文"，断言以原文结尾。
    assert soft_hits[0].content.endswith(exact)

    strict_hits = strict.search(query=query, user_id="user-1", top_k=1)
    assert len(strict_hits) == 1
    assert strict_hits[0].content.endswith("Aurora project")


def test_event_anchor_window_boosts_pre_event_records(tmp_path: Path) -> None:
    """事件锚定：以事件时间为边界，事件前记录必须排在事件后同内容记录之前。"""
    embedder = BagOfWordsEmbedder()
    expansion = QueryExpansion(
        text="",
        temporal=TemporalConstraints(
            relative_window=None,
            event_anchor=EventAnchorSpec(event="moved to Shanghai", direction="before"),
            ordering=None,
        ),
    )
    service = _make_service(
        _make_settings(tmp_path), StubMemoryLLM(expansion), embedder
    )
    _remember(service, "I moved to Shanghai", EVENT_TIME)
    _remember(service, "I lived in Beijing", OLD)
    _remember(service, "I lived in Beijing", RECENT)

    hits = service.search(
        query="Where did I live before moving to Shanghai",
        user_id="user-1",
        top_k=10,
    )
    assert len(hits) == 3
    by_time = {hit.created_at: hit for hit in hits}
    before_hit = by_time["2024-01-15T00:00:00.000Z"]
    after_hit = by_time["2026-08-01T00:00:00.000Z"]
    assert before_hit.score > after_hit.score
    assert hits.index(before_hit) < hits.index(after_hit)


def test_unresolvable_event_anchor_falls_back_to_plain_search(tmp_path: Path) -> None:
    """事件短语找不到高置信记录时，约束被放弃而不是猜测边界。"""
    embedder = BagOfWordsEmbedder()
    expansion = QueryExpansion(
        text="",
        temporal=TemporalConstraints(
            relative_window=None,
            event_anchor=EventAnchorSpec(event="moved to Paris", direction="before"),
            ordering=None,
        ),
    )
    service = _make_service(
        _make_settings(tmp_path), StubMemoryLLM(expansion), embedder
    )
    _remember(service, "I lived in Beijing", OLD)
    _remember(service, "I lived in Beijing", RECENT)

    hits = service.search(
        query="Where did I live before moving to Shanghai",
        user_id="user-1",
        top_k=10,
    )
    assert len(hits) == 2
    # 无时间项参与时，同内容记录分数完全相等。
    assert hits[0].score == hits[1].score


def test_temporal_claim_without_query_signal_is_dropped(tmp_path: Path) -> None:
    """守卫：查询原文没有时间信号时，LLM 声称的时间约束视为幻觉并丢弃。"""
    embedder = BagOfWordsEmbedder()
    expansion = _window_expansion()
    service = _make_service(
        _make_settings(tmp_path), StubMemoryLLM(expansion), embedder
    )
    _remember(service, "Aurora project status update", OLD)
    _remember(service, "Aurora project status update", RECENT)

    hits = service.search(query="Aurora project status update", user_id="user-1", top_k=10)
    assert len(hits) == 2
    assert hits[0].score == hits[1].score
    # 无时间项时按 created_at 升序 tie-break，首条是旧记录。
    assert hits[0].created_at == "2024-01-15T00:00:00.000Z"


def test_ordering_latest_prefers_newest_match(tmp_path: Path) -> None:
    """首末次排序：latest 让同内容的新记录排在最前，覆盖默认的时间升序。"""
    embedder = BagOfWordsEmbedder()
    latest_expansion = QueryExpansion(
        text="",
        temporal=TemporalConstraints(
            relative_window=None, event_anchor=None, ordering="latest"
        ),
    )
    plain_expansion = QueryExpansion(text="", temporal=None)
    latest = _make_service(
        _make_settings(tmp_path), StubMemoryLLM(latest_expansion), embedder
    )
    plain = _make_service(
        _make_settings(tmp_path), StubMemoryLLM(plain_expansion), embedder
    )
    _remember(latest, "I mentioned Aurora in standup", EARLY)
    _remember(latest, "I mentioned Aurora in standup", RECENT)

    query = "What was the last time I mentioned Aurora"
    latest_hits = latest.search(query=query, user_id="user-1", top_k=10)
    assert latest_hits[0].created_at == "2026-08-01T00:00:00.000Z"

    plain_hits = plain.search(query=query, user_id="user-1", top_k=10)
    assert plain_hits[0].created_at == "2025-01-15T00:00:00.000Z"
