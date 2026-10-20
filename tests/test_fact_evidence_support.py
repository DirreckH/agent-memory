"""Source-bound attribution and date normalization; all inputs are synthetic."""
from datetime import datetime, timezone

import pytest

from app.governance.extraction import FactIndexer
from app.governance.models import Chunk, Source


def timestamp(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def extract(text, *, quote=None, value="Acme", source_time=timestamp("2026-01-06"), **fields):
    source = Source("source", text, "user", "session", 0, source_time)
    item = dict(subject="self", predicate="employer", value=value, quote=quote or text)
    item.update(fields)
    return FactIndexer()._fact(item, Chunk("chunk", source, 0, len(text)))


@pytest.mark.parametrize("text,quote", [
    ('Bob said, "I work at Acme."', None),
    ('Bob said, "I work at Acme."', "I work at Acme."),
    ('I said, "Bob said, \'I work at Acme.\'"', None),
    ('Bob said, "I said I work at Acme."', None),
    ('"I work at Acme," Bob said.', None),
    ('Bob said I work at Acme.', None),
    ('小李说：“我在Acme工作。”', None),
    ('我说：“小李说：‘我在Acme工作。’”', None),
    ('Bob said, "I work at Acme." I work at Beta.', None),
    ('Bob: I work at Acme.', None),
    ('I said, "Bob works at Acme."', None),
    ('Project lead: I work at Acme.', None),
    ('Project Lead: I work at Acme.', None),
    ('I said Bob works at Acme.', None),
    ('I told Bob he works at Acme.', None),
])
def test_reported_speaker_does_not_become_self(text, quote):
    with pytest.raises(ValueError, match="ReportedSpeakerIsNotSelf"):
        extract(text, quote=quote)


@pytest.mark.parametrize("text,quote", [
    ("I work at Acme.", None),
    ('I said, "I work at Acme."', None),
    ('I told Bob, "I work at Acme."', "I work at Acme."),
    ('Bob said, "I work at Beta." I work at Acme.', None),
    ('Bob said, "I work at Beta," but I work at Acme.', None),
    ('我说：“我在Acme工作。”', None),
    ('小李说：“我在Beta工作。”我在Acme工作。', None),
    ('"I work at Acme," I said.', None),
    ('Note: I work at Acme.', None),
    ('小李说：“我在Beta工作。”而我在Acme工作。', None),
    ('I work at "Acme".', None),
    ('I told Bob I work at Acme.', None),
])
def test_direct_or_explicit_self_speech_survives(text, quote):
    assert extract(text, quote=quote).subject_id == "self"


@pytest.mark.parametrize("date_text,date", [
    ("January 5, 2026", "2026-01-05"),
    ("5 January 2026", "2026-01-05"),
    ("Jan. 5, 2026", "2026-01-05"),
    ("May 5, 2026", "2026-05-05"),
    ("2026年1月5日", "2026-01-05"),
    ("2026-01-05", "2026-01-05"),
    ("yesterday", "2026-01-05"),
    ("today", "2026-01-06"),
    ("tomorrow", "2026-01-07"),
    ("昨天", "2026-01-05"),
    ("今天", "2026-01-06"),
    ("明天", "2026-01-07"),
])
def test_supported_day_normalization_keeps_literal_evidence(date_text, date):
    text = f"I work at Acme from {date_text}."
    fact = extract(text, valid_from=date, time_quote=date_text)
    assert fact.valid_from == timestamp(date)
    assert fact.time_precision == "day"
    evidence = next(e for e in fact.evidence if e.purpose == "time")
    assert evidence.quote == date_text
    assert text[evidence.start:evidence.end] == date_text


@pytest.mark.parametrize("text,quote,time_quote,date,source_time", [
    ("I work at Acme. Bob visited on January 5, 2026.", "I work at Acme.", "January 5, 2026", "2026-01-05", timestamp("2026-01-06")),
    ("I work at Acme; Bob visited on 2026-01-05.", None, "2026-01-05", "2026-01-05", timestamp("2026-01-06")),
    ("I work at Acme, while Bob visited on January 5, 2026.", None, "January 5, 2026", "2026-01-05", timestamp("2026-01-06")),
    ("I work at Acme and Bob visited on January 5, 2026.", None, "January 5, 2026", "2026-01-05", timestamp("2026-01-06")),
    ("I work at Acme, Bob visited on January 5, 2026.", None, "January 5, 2026", "2026-01-05", timestamp("2026-01-06")),
    ('Bob said, "I joined Acme on January 5, 2026." I joined Acme on January 6, 2026.', None, "January 5, 2026", "2026-01-05", timestamp("2026-01-06")),
    ("I joined Acme on January 5, 2026.", None, "January 5, 2026", "2026-01-06", timestamp("2026-01-06")),
    ("I joined Acme yesterday.", None, "yesterday", "2026-01-05", None),
    ("I joined Acme yesterday.", None, "yesterday", "2026-01-04", timestamp("2026-01-06")),
    ("I joined Acme in January 2026.", None, "January 2026", "2026-01-01", timestamp("2026-01-06")),
    ("I joined Acme in 2026.", None, "2026", "2026-01-01", timestamp("2026-01-06")),
    ("I joined Acme around January 5, 2026.", None, "January 5, 2026", "2026-01-05", timestamp("2026-01-06")),
    ("I joined Acme before 2026-01-05.", None, "2026-01-05", "2026-01-05", timestamp("2026-01-06")),
    ("I joined Acme on January 5, 2026 or January 6, 2026.", None, "January 5, 2026", "2026-01-05", timestamp("2026-01-06")),
    ("Correction: 2025-01-01 was wrong. I work at Acme.", "I work at Acme.", "2025-01-01", "2025-01-01", timestamp("2026-01-06")),
    ('I said, "I joined Acme yesterday."', None, "yesterday", "2026-01-05", timestamp("2026-01-06")),
    ('I joined Acme yesterday in Pacific time.', None, "yesterday", "2026-01-05", timestamp("2026-01-06")),
])
def test_date_requires_unambiguous_fact_bound_support(text, quote, time_quote, date, source_time):
    with pytest.raises(ValueError, match="UngroundedDate|AmbiguousDate|UnboundDate"):
        extract(text, quote=quote, source_time=source_time, valid_from=date, time_quote=time_quote)


def test_future_plan_is_not_promoted_by_normalized_date():
    text = "I plan to join Acme tomorrow."
    with pytest.raises(ValueError, match="PlanMisclassified"):
        extract(text, valid_from="2026-01-07", time_quote="tomorrow")
    fact = extract(text, modality="planned", valid_from="2026-01-07", time_quote="tomorrow")
    assert fact.modality == "planned"


def test_range_endpoints_cannot_be_swapped():
    text = "I work at Acme from January 5, 2026 until January 9, 2026."
    fact = extract(text, valid_from="2026-01-05", valid_to="2026-01-09", time_quote=text)
    assert (fact.valid_from, fact.valid_to) == (timestamp("2026-01-05"), timestamp("2026-01-09"))
    with pytest.raises(ValueError, match="UngroundedDate|AmbiguousDate"):
        extract(text, valid_from="2026-01-09", time_quote="January 9, 2026")


def test_missing_effective_date_stays_unknown():
    fact = extract("I work at Acme.")
    assert fact.valid_from is None and fact.valid_to is None and fact.time_precision == "unknown"


def test_relative_day_at_epoch_zero_and_midnight_uses_source_utc_day():
    fact = extract("I joined Acme today.", source_time=0, valid_from="1970-01-01", time_quote="today")
    assert fact.valid_from == 0
    fact = extract("I joined Acme yesterday.", source_time=timestamp("2026-01-06") + 23 * 3600000,
                   valid_from="2026-01-05", time_quote="yesterday")
    assert fact.valid_from == timestamp("2026-01-05")


def test_named_fact_cannot_borrow_another_persons_date():
    text = "Alice works at Acme, while Bob joined Beta on January 5, 2026."
    with pytest.raises(ValueError, match="UnboundDate"):
        extract(text, subject="Alice", valid_from="2026-01-05", time_quote="January 5, 2026")


def test_supported_valid_to_preserves_existing_exclusive_endpoint():
    from app.governance.models import UserSnapshot
    from app.governance.resolution import StateResolver

    text = "I work at Acme until January 9, 2026."
    fact = extract(text, valid_to="2026-01-09", time_quote="January 9, 2026")
    snapshot = UserSnapshot("user", 1, 1, facts=(fact,))
    assert StateResolver().state_at(snapshot, reference=timestamp("2026-01-09") - 1).selected
    assert not StateResolver().state_at(snapshot, reference=timestamp("2026-01-09")).selected


def test_project_heading_is_not_a_reported_speaker():
    text = "Project Sequoia: I approved launch on January 4, 2026."
    fact = extract(text, predicate="event", scope="Sequoia", value="launch", kind="event",
                   valid_from="2026-01-04", time_quote="January 4, 2026")
    assert fact.subject_id == "self" and fact.valid_from == timestamp("2026-01-04")


@pytest.mark.parametrize("text", [
    "I work at Acme through January 9, 2026.",
    "I work at Acme from January 5, 2026 through January 9, 2026.",
])
def test_inclusive_through_is_not_an_exclusive_end_day(text):
    with pytest.raises(ValueError, match="AmbiguousDate|UngroundedDate"):
        extract(text, valid_to="2026-01-09", time_quote="January 9, 2026")


def test_correction_explanation_is_not_a_date_inequality():
    from app.governance.models import UserSnapshot
    from app.governance.resolution import StateResolver, derive_relations

    old = extract("我从2025-01-01起的职位是助理。", value="助理", predicate="job_role",
                  valid_from="2025-01-01", time_quote="2025-01-01", source_time=timestamp("2025-01-02"))
    text = "更正：我从2025-01-01起的职位实际是研究员，之前写成助理属于笔误。"
    new = extract(text, value="研究员", predicate="job_role", valid_from="2025-01-01",
                  time_quote=text, operation="correct", target_value="助理", update_quote=text,
                  source_time=timestamp("2025-01-06"))
    snapshot = UserSnapshot("user", 1, 1, facts=(old, new), relations=derive_relations((old, new)))
    state = StateResolver().history_until(snapshot, reference=timestamp("2025-01-10"))
    assert [fact.value for fact in state.selected] == ["研究员"]
    assert state.excluded[old.id] == "corrected_claim"
