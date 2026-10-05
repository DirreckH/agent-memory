"""常见时间表达的离线规则及 LLM 输出的原文证据校验。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.temporal import (
    RelativeWindowSpec,
    TemporalConstraints,
    parse_temporal,
    query_mentions_temporal,
)

_ZH_UNITS = {
    "天": "day", "日": "day", "周": "week", "星期": "week",
    "个月": "month", "月": "month", "年": "year",
}
_EN_NUMBERS = dict(zip(
    ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
     "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split(),
    range(21), strict=True,
))
_ZH_ROLLING = re.compile(
    r"(?P<direction>最近|过去|未来|今后|接下来|近)(?:的)?\s*"
    r"(?P<amount>[0-9零〇一二两三四五六七八九十百千]+)\s*"
    r"(?P<unit>个月|星期|天|日|周|月|年)"
)
_EN_ROLLING = re.compile(
    r"\b(?P<direction>past|last|next|coming)\s+"
    r"(?P<amount>\d+|" + "|".join(_EN_NUMBERS) + r")\s+"
    r"(?P<unit>days?|weeks?|months?|years?)\b", re.IGNORECASE,
)
_HALF_YEAR = re.compile(r"(?P<direction>最近|过去|未来|今后|接下来|近)(?:的)?半年")
_CALENDAR_ZH: dict[str, tuple[str, int]] = {
    "大前天": ("day", -3), "前天": ("day", -2), "昨天": ("day", -1),
    "今天": ("day", 0), "明天": ("day", 1), "后天": ("day", 2), "大后天": ("day", 3),
    "上上周": ("week", -2), "上周": ("week", -1), "本周": ("week", 0), "这周": ("week", 0),
    "下周": ("week", 1), "下下周": ("week", 2),
    "上上个月": ("month", -2), "上个月": ("month", -1), "上月": ("month", -1),
    "本月": ("month", 0), "这个月": ("month", 0), "这月": ("month", 0),
    "下个月": ("month", 1), "下月": ("month", 1), "下下个月": ("month", 2),
    "前年": ("year", -2), "去年": ("year", -1), "今年": ("year", 0),
    "明年": ("year", 1), "后年": ("year", 2),
}
_CALENDAR_EN = {
    "the day before yesterday": ("day", -2), "yesterday": ("day", -1),
    "today": ("day", 0), "tomorrow": ("day", 1), "the day after tomorrow": ("day", 2),
    **{f"{prefix} {unit}": (unit, offset)
       for prefix, offset in (("last", -1), ("previous", -1), ("this", 0), ("next", 1))
       for unit in ("day", "week", "month", "year")},
}
_ZH_CALENDAR_RE = re.compile("|".join(sorted(_CALENDAR_ZH, key=len, reverse=True)))
_EN_CALENDAR_RE = re.compile(
    r"(?<![a-z])(?:" + "|".join(
        r"\s+".join(phrase.split()) for phrase in sorted(_CALENDAR_EN, key=len, reverse=True)
    ) + r")(?![a-z])", re.IGNORECASE,
)
_ORDERING = {
    "earliest": re.compile(r"第一次|最早|最初|\b(?:earliest|first\s+(?:time|mention|mentioned|record))\b", re.IGNORECASE),
    "latest": re.compile(r"最后一次|最近一次|最新|最晚|\b(?:latest|last\s+(?:time|mention|mentioned|record))\b", re.IGNORECASE),
}
_NEGATED_PREFIX = re.compile(
    r"(?:不是|并非|不在|除了|除去|不要|不含|"
    r"\bnot(?:\s+(?:in|during|within|over))?|\bexcept(?:\s+for)?|\bexcluding|\boutside(?:\s+of)?)"
    r"\s*(?:the\s+)?$", re.IGNORECASE,
)
_UNCERTAIN_PREFIX = re.compile(
    r"(?:大约|大概|超过|至少|至多|不到|少于|多于|不满|约|"
    r"\babout|\baround|\broughly|\bapproximately|\bmore than|\bless than|"
    r"\bat least|\bat most)\s*(?:在|the\s+)?$", re.IGNORECASE,
)
_UNSUPPORTED_SUFFIX = re.compile(
    r"\s*(?:半|左右|以上|以下|有余|前|后|起|以来|之前|之后|以前|以后|"
    r"[0-9零〇一二两三四五六七八九十百千]+\s*(?:天|日|周|星期|个月|月|年)|"
    r"or\s+(?:more|less)\b|and\s+(?:\d+|" + "|".join(_EN_NUMBERS)
    + r")\s+(?:days?|weeks?|months?|years?)\b)", re.IGNORECASE,
)


@dataclass(frozen=True)
class RuleExtraction:
    constraints: TemporalConstraints | None
    complete: bool = False
    blocked: bool = False


def _number(text: str) -> int | None:
    if text.isascii() and text.isdigit():
        return int(text) if len(text) <= 4 else None
    if text.casefold() in _EN_NUMBERS:
        return _EN_NUMBERS[text.casefold()]
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "两": 2})
    if all(c in digits for c in text):
        return int("".join(str(digits[c]) for c in text)) if len(text) <= 4 else None
    # 受控的十/百/千表达，拒绝倒序单位和连续数字等不明确形式。
    total, digit, last_unit = 0, None, 10000
    zero_after_unit = False
    for char in text:
        if char in digits:
            if digits[char] == 0:
                zero_after_unit = True
            if digit is not None:
                if digit == 0:
                    digit = digits[char]
                    continue
                return None
            digit = digits[char]
        elif char in "十百千":
            unit = {"十": 10, "百": 100, "千": 1000}[char]
            if unit >= last_unit:
                return None
            total += (digit if digit is not None else 1) * unit
            digit, last_unit = None, unit
            zero_after_unit = False
        else:
            return None
    if digit and last_unit > 10 and not zero_after_unit:
        return None  # “一百二”有省略歧义，不猜成 102 或 120。
    return total + (digit or 0)


def _is_excluded(query: str, start: int, end: int) -> bool:
    return bool(
        _NEGATED_PREFIX.search(query[:start])
        or re.match(r"(?:以外|之外|除外)", query[end:])
    )


def extract_temporal_rules(query: str) -> RuleExtraction:
    """单个明确相对窗口及首末次排序；多个不同窗口或否定窗口保守拒绝。"""
    windows: list[RelativeWindowSpec] = []
    spans: list[tuple[int, int]] = []
    blocked = False

    def add_window(match: re.Match[str], spec: RelativeWindowSpec | None) -> None:
        nonlocal blocked
        spans.append(match.span())
        if (
            spec is None
            or _is_excluded(query, *match.span())
            or _UNCERTAIN_PREFIX.search(query[:match.start()])
            or _UNSUPPORTED_SUFFIX.match(query[match.end():])
        ):
            blocked = True
        elif spec not in windows:
            windows.append(spec)

    for pattern in (_ZH_ROLLING, _EN_ROLLING):
        for match in pattern.finditer(query):
            amount = _number(match["amount"])
            direction = (
                "future" if match["direction"].casefold()
                in ("未来", "今后", "接下来", "next", "coming") else "past"
            )
            raw_unit = match["unit"].casefold()
            unit = _ZH_UNITS.get(raw_unit, raw_unit.rstrip("s"))
            spec = (
                RelativeWindowSpec("rolling", unit, amount, 0, direction)
                if amount is not None and 1 <= amount <= 1000 else None
            )
            add_window(match, spec)
    for match in _HALF_YEAR.finditer(query):
        direction = "future" if match["direction"] in ("未来", "今后", "接下来") else "past"
        add_window(match, RelativeWindowSpec("rolling", "month", 6, 0, direction))
    for pattern, mapping in ((_ZH_CALENDAR_RE, _CALENDAR_ZH), (_EN_CALENDAR_RE, _CALENDAR_EN)):
        for match in pattern.finditer(query):
            unit, offset = mapping[re.sub(r"\s+", " ", match[0].casefold())]
            add_window(match, RelativeWindowSpec("calendar", unit, 1, offset, "past"))

    orderings: set[str] = set()
    for direction, pattern in _ORDERING.items():
        for match in pattern.finditer(query):
            spans.append(match.span())
            if _is_excluded(query, *match.span()):
                blocked = True
            orderings.add(direction)
    if blocked or len(windows) > 1 or len(orderings) > 1:
        return RuleExtraction(None, blocked=True)
    constraints = TemporalConstraints(
        windows[0] if windows else None, None, next(iter(orderings), None),
    ) if windows or orderings else None
    residual = list(query)
    for start, end in spans:
        residual[start:end] = " " * (end - start)
    complete = constraints is not None and not query_mentions_temporal("".join(residual))
    return RuleExtraction(constraints, complete=complete)


def _same_window(left: RelativeWindowSpec, right: RelativeWindowSpec) -> bool:
    def canonical(spec: RelativeWindowSpec) -> tuple[str, str, int, str]:
        if spec.kind == "rolling":
            if spec.unit in ("day", "week"):
                amount = spec.amount * (7 if spec.unit == "week" else 1)
                return spec.kind, "day", amount, spec.direction
            amount = spec.amount * (12 if spec.unit == "year" else 1)
            return spec.kind, "month", amount, spec.direction
        return spec.kind, spec.unit, spec.offset, ""
    return canonical(left) == canonical(right)


def _evidence(value: Any, query: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    start = query.find(value)
    if start < 0 or _is_excluded(query, start, start + len(value)):
        return None
    return value


def parse_grounded_temporal(payload: Any, query: str) -> TemporalConstraints | None:
    """不采信无原文依据的 LLM 约束；不猜测模糊数量或不支持的组合窗口。"""
    parsed = parse_temporal(payload)
    if parsed is None or not query_mentions_temporal(query):
        return None
    rules = extract_temporal_rules(query)
    if rules.blocked:
        return None
    proven: list[tuple[int, int]] = []

    def remember(evidence: str, start: int = 0, end: int | None = None) -> None:
        base = query.find(evidence)
        proven.append((base + start, base + (len(evidence) if end is None else end)))

    if parsed.relative_window is not None:
        evidence = _evidence(payload["windows"][0].get("evidence"), query)
        if evidence is None:
            return None
        evidence_rule = extract_temporal_rules(evidence)
        window = (
            evidence_rule.constraints.relative_window
            if evidence_rule.constraints else None
        )
        if (
            not evidence_rule.complete or window is None
            or not _same_window(window, parsed.relative_window)
        ):
            return None
        # 部分规则意味着还有事件/其他时间限定，不能只采用其中一个窗口。
        if not rules.complete:
            return None
        remember(evidence)
    if parsed.event_anchor is not None:
        anchor = parsed.event_anchor
        evidence = _evidence(payload["event_anchor"].get("evidence"), query)
        if evidence is None or anchor.event not in evidence:
            return None
        prefix, suffix = evidence.split(anchor.event, 1)
        before_prefix = re.search(r"\b(?:before|prior to)\s*$", prefix, re.I)
        before_suffix = re.match(r"\s*(?:之前|以前|前)", suffix)
        after_prefix = re.search(r"\bafter\s*$", prefix, re.I)
        after_suffix = re.match(r"\s*(?:之后|以后|后)", suffix)
        before = bool(before_prefix or before_suffix)
        after = bool(after_prefix or after_suffix)
        if before == after or (anchor.direction == "before") != before:
            return None
        # calendar+event 需要窗口交集，当前 schema 的单锚点语义不支持。
        if rules.constraints and rules.constraints.relative_window is not None:
            return None
        remember(evidence, len(prefix), len(prefix) + len(anchor.event))
        marker_prefix = before_prefix or after_prefix
        marker_suffix = before_suffix or after_suffix
        if marker_prefix is not None:
            remember(evidence, marker_prefix.start(), marker_prefix.end())
        if marker_suffix is not None:
            base = len(prefix) + len(anchor.event)
            remember(evidence, base + marker_suffix.start(), base + marker_suffix.end())
    if parsed.ordering is not None:
        evidence = _evidence(payload.get("ordering_evidence"), query)
        if evidence is None:
            return None
        ordering_rule = extract_temporal_rules(evidence)
        found = ordering_rule.constraints.ordering if ordering_rule.constraints else None
        if found != parsed.ordering:
            # 首个实体/属性的复杂表达交给 LLM，但必须保留 first/last 的原文方向。
            word = "first" if parsed.ordering == "earliest" else "last"
            match = re.search(rf"\b{word}\b", evidence, re.I)
            if match is None or not query_mentions_temporal(evidence):
                return None
            if ordering_rule.constraints and ordering_rule.constraints.relative_window:
                return None
            remember(evidence, match.start(), match.end())
        else:
            for match in _ORDERING[parsed.ordering].finditer(evidence):
                remember(evidence, match.start(), match.end())
    if rules.constraints:
        expected = rules.constraints
        if expected.relative_window and (
            parsed.relative_window is None
            or not _same_window(expected.relative_window, parsed.relative_window)
        ):
            return None
        if expected.ordering and expected.ordering != parsed.ordering:
            return None
    residual = list(query)
    for start, end in proven:
        residual[start:end] = " " * (end - start)
    if query_mentions_temporal("".join(residual)):
        return None
    return parsed
