"""Conservative first-person support checks, independent of model confidence.

This is a bounded quotation/report parser, not general entity coreference. It
requires at least one value-bearing clause whose first person belongs to the
message author. Unknown or third-party quotation ownership cannot establish self.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


_REPORT = re.compile(r"\b(?:said|says|say|wrote|writes|write|told|tells|tell|stated|states)\b|说道|写道|告诉|表示|说", re.I)
_FIRST = re.compile(r"\b(?:I|my|me|we|our)\b|我", re.I)
_SELF_SPEAKER = re.compile(r"\b(?:I|we)\s*$|(?:我|我们)\s*$", re.I)
_BOUNDARY = re.compile(r"[.!?。！？;；\n]|\b(?:but|while|whereas)\b|\band\s+(?=I\b|we\b|he\b|she\b|they\b)|但是|不过|而(?=我)", re.I)
_SPEAKER_LABEL = re.compile(r"^\s*(?P<speaker>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|I|我|小[\u4e00-\u9fff]{1,3})\s*[:：]")
_HEADINGS = frozenset(('note', 'summary', 'reminder', 'context', 'update', 'correction'))
_CONTEXT_HEADING = re.compile(r'^\s*(?:project|topic|task|phase|session|meeting)\s+[^:\n]{1,80}[:：]', re.I)
_ROLE_LABEL = re.compile(r'^\s*(?:(?:project|team|session|meeting|task)\s+)?(?:lead|leader|manager|owner|speaker|representative|director)\s*[:：]', re.I)


@dataclass(frozen=True)
class Quotation:
    start: int
    end: int
    self_owned: bool


def _report_owners(text: str) -> list[bool]:
    return [bool(_SELF_SPEAKER.search(text[:match.start()])) for match in _REPORT.finditer(text)]


def quotations(text: str) -> tuple[Quotation, ...]:
    """Track nested direct speech, including attribution after a closing quote."""
    pairs = {'"': '"', "'": "'", '“': '”', '‘': '’', '「': '」', '『': '』'}
    stack: list[tuple[int, str, bool | None]] = []
    result = []
    last_closed = 0
    for pos, char in enumerate(text):
        # An apostrophe inside a word is not a speech delimiter.
        if char in ("'", '’') and pos and pos + 1 < len(text) and text[pos - 1].isalnum() and text[pos + 1].isalnum():
            continue
        if stack and char == stack[-1][1]:
            start, _, owner = stack.pop()
            suffix = re.split(r"[.!?。！？;；\n]", text[pos + 1:])[0]
            # Only a report immediately following speech supplies postfix ownership;
            # a later independent clause must not reassign the quoted speaker.
            report = _REPORT.search(suffix)
            if report and re.fullmatch(r"\s*[,，]?\s*(?:[A-Za-z]+\s+|[\u4e00-\u9fff]{1,8})", suffix[:report.start()]):
                postfix_self = bool(_SELF_SPEAKER.search(suffix[:report.start()]))
                owner = postfix_self if owner is None else owner and postfix_self
                owner = owner and all(frame[2] is True for frame in stack)
            result.append(Quotation(start + 1, pos, owner is True))
            last_closed = pos + 1
        elif char in pairs:
            boundary = max((m.end() for m in re.finditer(r"[.!?。！？;；\n]", text[:pos])), default=0)
            context_start = max(boundary, stack[-1][0] + 1 if stack else last_closed)
            owners = _report_owners(text[context_start:pos])
            parent_self = all(frame[2] is True for frame in stack)
            owner = parent_self and all(owners) if owners else None
            if stack and not owners:
                owner = parent_self
            stack.append((pos, pairs[char], owner))
    # Unclosed speech is retained as a conservative ownership boundary.
    result.extend(Quotation(start + 1, len(text), False) for start, _, _ in stack)
    return tuple(result)


def has_self_support(text: str, start: int, end: int, value: str) -> bool:
    """Check the clauses containing the actual value, not any I in the quote."""
    spans = quotations(text)
    for match in re.finditer(re.escape(value), text[start:end], re.I):
        pos = start + match.start()
        # A quoted literal value (I work at "Acme") is not reported speech.
        containing = [span for span in spans if span.start <= pos < span.end
                      and text[span.start:span.end].strip().casefold() != value.casefold()]
        if any(not span.self_owned for span in containing):
            continue
        left = max((m.end() for m in _BOUNDARY.finditer(text[:pos])), default=0)
        right_match = _BOUNDARY.search(text, pos + len(value))
        right = right_match.start() if right_match else len(text)
        if containing:
            left = max(left, max(span.start for span in containing))
            right = min(right, min(span.end for span in containing))
        # Completed quotes before a new clause are not its attribution context.
        closed = [span.end + 1 for span in spans if left <= span.end < pos]
        if closed:
            left = max(left, max(closed))
        if not _FIRST.search(text[left:right]):
            continue
        prefix = text[left:pos]
        if _ROLE_LABEL.match(prefix):
            continue
        label = _SPEAKER_LABEL.match(prefix)
        if (label and not _CONTEXT_HEADING.match(prefix)
                and label['speaker'].casefold() not in _HEADINGS
                and label['speaker'].casefold() not in ('i', '我')):
            continue
        owners = _report_owners(prefix)
        if owners and not all(owners):
            continue
        if owners:
            last_report = list(_REPORT.finditer(prefix))[-1]
            if not _FIRST.search(text[left + last_report.end():right]):
                continue  # I said/told is the reporter, not the reported actor.
        return True
    return False
