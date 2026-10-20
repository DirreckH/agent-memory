"""Verify day-granularity effective dates against source-bound expressions.

Only complete calendar days and yesterday/today/tomorrow are normalized. Month
and year expressions, uncertain dates and unbound dates remain unsupported.
Relative days use the source message's UTC date, never the request/server clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re

from app.governance.attribution import has_self_support, quotations


_MONTHS = {name: index for index, names in enumerate((
    ('jan', 'january'), ('feb', 'february'), ('mar', 'march'), ('apr', 'april'),
    ('may',), ('jun', 'june'), ('jul', 'july'), ('aug', 'august'),
    ('sep', 'sept', 'september'), ('oct', 'october'), ('nov', 'november'), ('dec', 'december'),
), 1) for name in names}
_MONTH = '|'.join(sorted(_MONTHS, key=len, reverse=True))
_PATTERNS = (
    re.compile(r'(?<!\d)(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?!\d)'),
    re.compile(r'(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日'),
    re.compile(rf'\b(?P<m>{_MONTH})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(?P<y>\d{{4}})\b', re.I),
    re.compile(rf'\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m>{_MONTH})\.?\s*,?\s+(?P<y>\d{{4}})\b', re.I),
)
_RELATIVE = re.compile(r'\b(?:yesterday|today|tomorrow)\b|昨天|今天|明天', re.I)
_OFFSETS = {'yesterday': -1, 'today': 0, 'tomorrow': 1, '昨天': -1, '今天': 0, '明天': 1}
_BOUNDARY = re.compile(
    r'[.!?。！？;；\n]|\b(?:while|whereas|but)\b|'
    r'(?:\band\s+|[,，]\s*)(?=(?:I|we|he|she|they)\b|'
    r'[A-Z][A-Za-z]+\s+(?:visited|joined|left|worked|works|started|ended|said|was|is)\b)|但是|不过', re.I)
_UNCERTAIN_PREFIX = re.compile(
    r'(?:\b(?:about|around|approximately|roughly|before|after|maybe|possibly|either|not)\s*(?:on\s+)?|'
    r'(?:大约|大概|可能|不是|不在)\s*(?:在\s*)?)$', re.I)
_UNCERTAIN_SUFFIX = re.compile(r'^\s*(?:(?:or|before|after)\b|左右|之前|之后|或者|或是)', re.I)


@dataclass(frozen=True)
class DayExpression:
    start: int
    end: int
    day: datetime
    relative: bool = False


def day_expressions(text: str, source_time: int | None = None) -> tuple[DayExpression, ...]:
    found = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            raw_month = match['m'].casefold()
            month = _MONTHS.get(raw_month) if not raw_month.isdigit() else int(raw_month)
            try:
                day = datetime(int(match['y']), month, int(match['d']), tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            found.append(DayExpression(match.start(), match.end(), day))
    if type(source_time) is int:
        try:
            anchor = datetime.fromtimestamp(source_time / 1000, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            for match in _RELATIVE.finditer(text):
                day = anchor + timedelta(days=_OFFSETS[match[0].casefold()])
                found.append(DayExpression(match.start(), match.end(), day, True))
        except (ValueError, OverflowError, OSError):
            pass
    return tuple(sorted(set(found), key=lambda expr: (expr.start, expr.end)))


def mask_calendar_dates(text: str) -> str:
    """Keep month May and abbreviation dots out of modal/sentence heuristics."""
    chars = list(text)
    for expr in day_expressions(text):
        chars[expr.start:expr.end] = ' ' * (expr.end - expr.start)
    return ''.join(chars)


def supported_day(*, raw_date: str, key: str, content: str, quote_start: int,
                  quote_end: int, time_start: int, time_end: int, value: str,
                  subject: str, source_time: int | None) -> int:
    """Return a verified UTC day or reject support that cannot bind to this fact."""
    if not isinstance(raw_date, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', raw_date):
        raise ValueError('UngroundedDate')
    try:
        expected = datetime.strptime(raw_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError('UngroundedDate') from None
    expressions = day_expressions(content, source_time)
    candidates = [expr for expr in expressions if time_start <= expr.start and expr.end <= time_end and expr.day == expected]
    if not candidates:
        raise ValueError('UngroundedDate')
    masked = list(content)
    for expr in expressions:
        masked[expr.start:expr.end] = ' ' * (expr.end - expr.start)
    masked = ''.join(masked)
    boundaries = list(_BOUNDARY.finditer(masked))
    value_matches = list(re.finditer(re.escape(value), content[quote_start:quote_end], re.I))
    for expr in candidates:
        left = max((m.end() for m in boundaries if m.end() <= expr.start), default=0)
        right = min((m.start() for m in boundaries if m.start() >= expr.end), default=len(content))
        if not any(left <= quote_start + match.start() and quote_start + match.end() <= right for match in value_matches):
            continue
        if subject == 'self':
            if not has_self_support(content, max(left, quote_start), min(right, quote_end), value):
                continue
        elif subject.casefold() not in content[left:right].casefold():
            continue
        clause = content[left:right]
        clause_dates = [other for other in expressions if left <= other.start and other.end <= right]
        # Qualifiers must modify this date. "The previous record was wrong" in a
        # correction is not a before-date constraint on the corrected assertion.
        date_prefix = content[left:expr.start]
        date_suffix = content[expr.end:right]
        if _UNCERTAIN_PREFIX.search(date_prefix) or _UNCERTAIN_SUFFIX.search(date_suffix):
            raise ValueError('AmbiguousDate')
        # Through explicitly includes its end day, whereas valid_to is exclusive.
        # Keep it unsupported instead of silently dropping the final day.
        if key == 'valid_to' and re.search(r'\bthrough\s*$', date_prefix, re.I):
            raise ValueError('AmbiguousDate')
        if expr.relative:
            if re.search(r'\b(?:UTC|GMT)\s*[+-]|\b(?:PST|PDT|EST|EDT|CST|CDT|Pacific|Eastern)\b|北京时间|时区', content, re.I):
                raise ValueError('UngroundedDate')
            if any(span.start <= expr.start < span.end for span in quotations(content)):
                raise ValueError('UngroundedDate')
            if re.search(r'\b(?:said|says|wrote|told)\b|说|表示|写道', clause, re.I):
                raise ValueError('UngroundedDate')
        if len(clause_dates) > 1:
            if len(clause_dates) != 2:
                raise ValueError('AmbiguousDate')
            first, last = clause_dates
            prefix = content[left:first.start]
            between = content[first.end:last.start]
            if not (re.search(r'\b(?:from|between)\s*$|从\s*$', prefix, re.I)
                    and re.fullmatch(r'\s*(?:to|until|and|到|至)\s*', between, re.I)):
                raise ValueError('AmbiguousDate')
            correct = first if key == 'valid_from' else last
            if expr != correct:
                raise ValueError('UngroundedDate')
        else:
            prefix = content[left:expr.start]
            if key == 'valid_from' and re.search(r'\b(?:until|through|ended|left)\s*$|截至\s*$|直到\s*$', prefix, re.I):
                raise ValueError('UngroundedDate')
            if key == 'valid_to' and re.search(r'\b(?:from|since|starting|joined)\s*$|从\s*$|自\s*$', prefix, re.I):
                raise ValueError('UngroundedDate')
        return int(expected.timestamp() * 1000)
    raise ValueError('UnboundDate')
