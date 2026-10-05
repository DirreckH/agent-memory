"""时间感知检索的纯函数模块。

设计原则：规则或 LLM 抽取相对时间约束（方向、数量、单位、事件短语），
绝对窗口一律由本模块的确定性代码计算。本模块不做网络调用或模型推理，
全部函数可离线单元测试。
"""

from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

_VALID_UNITS = ("day", "week", "month", "year")

# 时间信号用于决定是否请求复杂抽取，并检查未被解析的限定。
# 信号命中本身不能证明约束正确，LLM 结果还需通过逐字段原文校验。
_TEMPORAL_SIGNAL_RE = re.compile(
    r"(?:[\u4e00-\u9fff](?:前|后))|之前|以前|之后|以后|最近|当初|当时|那时|如今|"
    r"现在|目前|当前|上个月|下个月|上月|下月|上周|下周|本周|这周|"
    r"大前天|前天|昨天|今天|明天|后天|本月|这个月|这月|上上周|下下周|"
    r"(?:过去|未来|今后|接下来|近)(?:的)?\s*(?:[0-9零〇一二两三四五六七八九十百千]+|半)\s*(?:天|日|周|星期|个月|月|年)|"
    r"今年|去年|明年|前年|年初|年底|月初|月底|"
    r"第[一二三四五六七八九十百]?次|最早|最晚|最新|最旧|上次|下次|一次|"
    r"[0-9一二三四五六七八九十百千]+(?:多)?(?:天|日|周|星期|月|年|小时|分钟)"
    r"(?:前|后|内|以来)|"
    r"\b\d+\s*(?:days?|weeks?|months?|years?|hours?|minutes?)\s*"
    r"(?:ago|later|within)\b|"
    r"\b(?:yesterday|tomorrow|today|first(?!\s+(?:aid|class|name)\b)|last(?!\s+name\b)|latest|earliest|recently|recent|"
    r"since|until|before|after|ago|next\s+\w+|"
    r"(?:past|coming)\s+\w+|(?:this|previous)\s+(?:past\s+)?(?:day|week|month|year))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RelativeWindowSpec:
    """规则或 LLM 抽取的相对窗口表达，尚未解析为绝对时间。"""

    kind: str  # rolling | calendar
    unit: str  # day | week | month | year
    amount: int  # rolling：窗口长度
    offset: int  # calendar：相对当前周期的偏移，上个月为 -1
    direction: str  # rolling：past | future


@dataclass(frozen=True)
class EventAnchorSpec:
    """以某个事件为参照的时间约束，如“搬家之前”。"""

    event: str
    direction: str  # before | after


@dataclass(frozen=True)
class TemporalConstraints:
    """一次查询解析出的全部时间约束。

    解析器只接受一个相对窗口或事件锚点，不猜测多个窗口的并集/交集；
    ordering 单独出现时（“第一次是什么时候”）不产生窗口，只改变排序。
    """

    relative_window: RelativeWindowSpec | None
    event_anchor: EventAnchorSpec | None
    ordering: str | None  # earliest | latest


@dataclass(frozen=True)
class TemporalReference:
    """相对窗口的查询参考时间及其来源；无法可靠确定时 timestamp_ms 为 None。"""

    timestamp_ms: int | None
    source: Literal["query_time", "conversation_frontier", "request_time", "unavailable"]


@dataclass(frozen=True)
class TemporalWindow:
    """显式端点的毫秒区间；默认 [start_ms, end_ms)，None 表示该侧无界。

    hard_boundary=True 用于事件锚定窗口：边界语义是硬的（“搬家之前”
    明确不含搬家事件本身），窗外记录不因距离小而获得软衰减分。
    """

    start_ms: int | None
    end_ms: int | None
    hard_boundary: bool = False
    start_inclusive: bool = True
    end_inclusive: bool = False

    def contains(self, time_ms: int) -> bool:
        if self.start_ms is not None and (
            time_ms < self.start_ms
            or (time_ms == self.start_ms and not self.start_inclusive)
        ):
            return False
        if self.end_ms is not None and (
            time_ms > self.end_ms
            or (time_ms == self.end_ms and not self.end_inclusive)
        ):
            return False
        return True

    def distance_ms(self, time_ms: int) -> float:
        """到最近可包含的整数毫秒的距离；被排除的端点至少距离 1ms。"""
        if self.contains(time_ms):
            return 0.0
        if self.start_ms is not None and time_ms <= self.start_ms:
            return float(self.start_ms + int(not self.start_inclusive) - time_ms)
        if self.end_ms is not None:
            return float(time_ms - self.end_ms + int(not self.end_inclusive))
        return 0.0


def query_mentions_temporal(query: str) -> bool:
    return bool(_TEMPORAL_SIGNAL_RE.search(query))


def _as_int(value: Any) -> int | None:
    # bool 是 int 的子类，必须显式排除，否则 True 会被当作 1。
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _parse_relative_window(windows: Any) -> RelativeWindowSpec | None:
    if not isinstance(windows, list) or len(windows) != 1:
        return None
    item = windows[0]
    if not isinstance(item, dict):
        return None
    kind, unit = item.get("kind"), item.get("unit")
    if kind not in ("rolling", "calendar") or unit not in _VALID_UNITS:
        return None
    if kind == "rolling":
        if set(item) - {"kind", "unit", "amount", "direction", "evidence"}:
            return None
        direction, amount = item.get("direction"), _as_int(item.get("amount"))
        if direction not in ("past", "future") or amount is None or not 1 <= amount <= 1000:
            return None
        return RelativeWindowSpec(kind, unit, amount, 0, direction)
    if set(item) - {"kind", "unit", "offset", "evidence"}:
        return None
    offset = _as_int(item.get("offset"))
    if offset is None or not -1200 <= offset <= 1200:
        return None
    return RelativeWindowSpec(kind, unit, 1, offset, "past")


def _parse_event_anchor(payload: Any) -> EventAnchorSpec | None:
    if not isinstance(payload, dict):
        return None
    event = payload.get("event")
    direction = payload.get("direction")
    if not isinstance(event, str) or not 1 <= len(event.strip()) <= 200:
        return None
    if direction not in ("before", "after"):
        return None
    if set(payload) - {"event", "direction", "evidence"}:
        return None
    return EventAnchorSpec(event=event.strip(), direction=direction)


def parse_temporal(payload: Any) -> TemporalConstraints | None:
    """把 LLM 输出的 temporal 字段解析为受控约束；不合法时返回 None。"""
    if not isinstance(payload, dict):
        return None
    if set(payload) - {"windows", "event_anchor", "ordering", "ordering_evidence"}:
        return None
    relative = _parse_relative_window(payload.get("windows"))
    event = _parse_event_anchor(payload.get("event_anchor"))
    ordering = payload.get("ordering")
    if payload.get("windows") not in (None, []) and relative is None:
        return None
    if payload.get("event_anchor") is not None and event is None:
        return None
    if ordering is not None and ordering not in ("earliest", "latest"):
        return None
    if relative is not None and event is not None:
        return None
    if relative is None and event is None and ordering is None:
        return None
    return TemporalConstraints(
        relative_window=relative, event_anchor=event, ordering=ordering
    )


def _shift_months(moment: datetime, months: int) -> datetime:
    total = moment.month - 1 + months
    year = moment.year + total // 12
    month = total % 12 + 1
    # 月末日期平移时按目标月天数收敛，如 5 月 31 日 -1 月得到 4 月 30 日。
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def _period_start(moment: datetime, unit: str) -> datetime:
    moment = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    if unit == "day":
        return moment
    if unit == "week":
        return moment - timedelta(days=moment.weekday())  # 周一为一周起点
    if unit == "month":
        return moment.replace(day=1)
    return moment.replace(month=1, day=1)  # year


def _shift_periods(moment: datetime, unit: str, count: int) -> datetime:
    if unit == "day":
        return moment + timedelta(days=count)
    if unit == "week":
        return moment + timedelta(weeks=count)
    if unit == "month":
        return _shift_months(moment, count)
    return _shift_months(moment, 12 * count)  # year


def _to_ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def resolve_relative_window(
    spec: RelativeWindowSpec, anchor_ms: int
) -> TemporalWindow:
    """把相对窗口表达解析为绝对时间窗口；anchor_ms 是锚点毫秒时间戳。"""
    anchor = datetime.fromtimestamp(anchor_ms / 1000, tz=timezone.utc)
    if spec.kind == "rolling":
        if spec.unit in ("month", "year"):
            # 月/年必须用日历运算，不能用固定天数近似。
            months = spec.amount * (12 if spec.unit == "year" else 1)
            if spec.direction == "past":
                return TemporalWindow(
                    start_ms=_to_ms(_shift_months(anchor, -months)),
                    end_ms=anchor_ms,
                    end_inclusive=True,
                )
            return TemporalWindow(
                start_ms=anchor_ms,
                end_ms=_to_ms(_shift_months(anchor, months)),
            )
        delta = timedelta(
            days=spec.amount if spec.unit == "day" else spec.amount * 7
        )
        if spec.direction == "past":
            return TemporalWindow(
                start_ms=_to_ms(anchor - delta), end_ms=anchor_ms,
                end_inclusive=True,
            )
        return TemporalWindow(start_ms=anchor_ms, end_ms=_to_ms(anchor + delta))

    # calendar：以锚点所在周期为基准平移 offset 个周期。
    base = _period_start(anchor, spec.unit)
    start = _shift_periods(base, spec.unit, spec.offset)
    end = _shift_periods(start, spec.unit, 1)
    return TemporalWindow(start_ms=_to_ms(start), end_ms=_to_ms(end))


def resolve_anchor(
    source_times_ms: list[int | None],
    *,
    now_ms: int,
    mode: Literal["replay", "realtime"] = "replay",
    query_time_ms: int | None = None,
) -> TemporalReference:
    """显式查询时间优先；回放用源消息前沿，实时用检索开始时刻。

    now_ms 由调用方在检索开始时捕获。回放缺少可信源时间时返回 unavailable，
    不使用服务器时钟或入库时间填补历史时间线。
    """
    if mode not in ("replay", "realtime"):
        raise ValueError("unknown temporal reference mode")
    if query_time_ms is not None:
        if effective_time_ms(query_time_ms) is None:
            raise ValueError(
                "query_time_ms must be a valid non-negative millisecond timestamp"
            )
        return TemporalReference(query_time_ms, "query_time")
    if mode == "realtime":
        if effective_time_ms(now_ms) is None:
            raise ValueError("now_ms must be a valid non-negative millisecond timestamp")
        return TemporalReference(now_ms, "request_time")
    known = [
        timestamp
        for value in source_times_ms
        if (timestamp := effective_time_ms(value)) is not None
    ]
    if known:
        return TemporalReference(max(known), "conversation_frontier")
    return TemporalReference(None, "unavailable")


def effective_time_ms(
    source_timestamp: int | None, created_at_text: str | None = None
) -> int | None:
    """只采用可解析的源消息时间；缺失或无效时保持未知。

    created_at_text 仅保留调用兼容性，不参与时间解析：该字段可能是入库时间，
    不能据此恢复消息时间或事件发生时间。
    """
    timestamp = _as_int(source_timestamp)
    if timestamp is None or timestamp < 0:
        return None
    try:
        datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return timestamp


def time_match(
    time_ms: int | None, window: TemporalWindow, *, half_life_days: float
) -> float:
    """时间匹配分：窗口内 1.0；窗口外按距离半衰衰减；时间未知取中性值 0.5。

    未知时间固定取 0.5，仍受加权后的相关度阈值约束。
    硬边界窗口（事件锚定）窗外一律 0 分，不享受近距离软衰减。
    """
    if time_ms is None:
        return 0.5
    if window.contains(time_ms):
        return 1.0
    if window.hard_boundary:
        return 0.0
    distance_ms = window.distance_ms(time_ms)
    if distance_ms <= 0:
        return 1.0
    distance_days = distance_ms / 86_400_000
    return math.exp(-math.log(2) * distance_days / half_life_days)
