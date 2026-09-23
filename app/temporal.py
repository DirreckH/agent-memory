"""时间感知检索的纯函数模块。

设计原则：LLM 只负责从查询中抽取相对时间约束（方向、数量、单位、事件短语），
绝对窗口一律由本模块的确定性代码计算。本模块不做网络调用或模型推理，
全部函数可离线单元测试。
"""

from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

_VALID_UNITS = ("day", "week", "month", "year")

# 程序化防御：LLM 声称存在时间约束时，查询原文中必须能回查到时间信号，
# 否则丢弃约束。正则宁可误放（约束继续生效），不可误杀真实时间查询。
_TEMPORAL_SIGNAL_RE = re.compile(
    r"(?:[\u4e00-\u9fff](?:前|后))|之前|以前|之后|以后|最近|当初|当时|那时|如今|"
    r"现在|目前|当前|上个月|下个月|上月|下月|上周|下周|本周|这周|"
    r"今年|去年|明年|前年|年初|年底|月初|月底|"
    r"第[一二三四五六七八九十百]?次|最早|最晚|最新|最旧|上次|下次|一次|"
    r"[0-9一二三四五六七八九十百千]+(?:多)?(?:天|日|周|星期|月|年|小时|分钟)"
    r"(?:前|后|内|以来)|"
    r"\b\d+\s*(?:days?|weeks?|months?|years?|hours?|minutes?)\s*"
    r"(?:ago|later|within)\b|"
    r"\b(?:yesterday|tomorrow|today|first|last|latest|earliest|recently|recent|"
    r"since|until|before|after|ago|last\s+\w+|next\s+\w+|"
    r"this\s+(?:week|month|year))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RelativeWindowSpec:
    """LLM 抽取的相对窗口表达，尚未解析为绝对时间。"""

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

    windows 与 event_anchor 同时出现时，服务层优先使用 event_anchor；
    ordering 单独出现时（“第一次是什么时候”）不产生窗口，只改变排序。
    """

    relative_window: RelativeWindowSpec | None
    event_anchor: EventAnchorSpec | None
    ordering: str | None  # earliest | latest


@dataclass(frozen=True)
class TemporalWindow:
    """半开时间区间 [start_ms, end_ms)；None 表示该侧无界。"""

    start_ms: int | None
    end_ms: int | None

    def contains(self, time_ms: int) -> bool:
        if self.start_ms is not None and time_ms < self.start_ms:
            return False
        if self.end_ms is not None and time_ms >= self.end_ms:
            return False
        return True

    def distance_ms(self, time_ms: int) -> float:
        """窗口外记录到最近窗口边界的距离；窗口内为 0。"""
        if self.contains(time_ms):
            return 0.0
        if self.start_ms is not None and time_ms < self.start_ms:
            return float(self.start_ms - time_ms)
        if self.end_ms is not None:
            return float(time_ms - self.end_ms)
        return 0.0


def query_mentions_temporal(query: str) -> bool:
    return bool(_TEMPORAL_SIGNAL_RE.search(query))


def _as_int(value: Any) -> int | None:
    # bool 是 int 的子类，必须显式排除，否则 True 会被当作 1。
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _parse_relative_window(windows: Any) -> RelativeWindowSpec | None:
    if not isinstance(windows, list):
        return None
    for item in windows:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        unit = item.get("unit")
        if kind not in ("rolling", "calendar") or unit not in _VALID_UNITS:
            continue
        direction = item.get("direction", "past")
        if direction not in ("past", "future"):
            direction = "past"
        if kind == "rolling":
            amount = _as_int(item.get("amount"))
            if amount is None or not 1 <= amount <= 1000:
                continue
            return RelativeWindowSpec(kind, unit, amount, 0, direction)
        offset = _as_int(item.get("offset"))
        if offset is None or not -1200 <= offset <= 1200:
            continue
        return RelativeWindowSpec(kind, unit, 1, offset, direction)
    return None


def _parse_event_anchor(payload: Any) -> EventAnchorSpec | None:
    if not isinstance(payload, dict):
        return None
    event = payload.get("event")
    direction = payload.get("direction")
    if not isinstance(event, str) or not event.strip():
        return None
    if direction not in ("before", "after"):
        return None
    return EventAnchorSpec(event=event.strip()[:200], direction=direction)


def parse_temporal(payload: Any) -> TemporalConstraints | None:
    """把 LLM 输出的 temporal 字段解析为受控约束；不合法时返回 None。"""
    if not isinstance(payload, dict):
        return None
    relative = _parse_relative_window(payload.get("windows"))
    event = _parse_event_anchor(payload.get("event_anchor"))
    ordering = payload.get("ordering")
    if ordering not in ("earliest", "latest"):
        ordering = None
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
                start_ms=_to_ms(anchor - delta), end_ms=anchor_ms
            )
        return TemporalWindow(start_ms=anchor_ms, end_ms=_to_ms(anchor + delta))

    # calendar：以锚点所在周期为基准平移 offset 个周期。
    base = _period_start(anchor, spec.unit)
    start = _shift_periods(base, spec.unit, spec.offset)
    end = _shift_periods(start, spec.unit, 1)
    return TemporalWindow(start_ms=_to_ms(start), end_ms=_to_ms(end))


def resolve_anchor(
    effective_times_ms: list[int | None], *, now_ms: int
) -> int:
    """三级锚点：对话前沿（最大已知事件时间）优先，全库无时间才用服务器时钟。"""
    known = [value for value in effective_times_ms if value is not None]
    if known:
        return max(known)
    return now_ms


def effective_time_ms(
    source_timestamp: int | None, created_at_text: str | None
) -> int | None:
    """记录的有效事件时间：消息自带时间戳优先，否则回退解析入库时间。"""
    if source_timestamp is not None:
        return source_timestamp
    if not created_at_text:
        return None
    try:
        text = created_at_text.replace("Z", "+00:00")
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return int(moment.timestamp() * 1000)
    except ValueError:
        return None


def time_match(
    time_ms: int | None, window: TemporalWindow, *, half_life_days: float
) -> float:
    """时间匹配分：窗口内 1.0；窗口外按距离半衰衰减；时间未知取中性值 0.5。

    未知时间取 0.5 而不是惩罚值，避免无时间戳记忆在时间查询中被错杀。
    """
    if time_ms is None:
        return 0.5
    if window.contains(time_ms):
        return 1.0
    distance_ms = window.distance_ms(time_ms)
    if distance_ms <= 0:
        return 1.0
    distance_days = distance_ms / 86_400_000
    return math.exp(-math.log(2) * distance_days / half_life_days)
