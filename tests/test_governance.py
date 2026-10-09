from __future__ import annotations

import json
from types import SimpleNamespace
import pytest

from app.config import Settings
from app.llm import LLMConnection, OpenAICompatibleMemoryLLM
from app.schemas import MemoryMessage
from app.service import MemoryService
from app.storage import SQLiteMemoryStore
from app.governance.resolution import StateResolver
from app.governance.extraction import token_count
from app.governance.summaries import SummaryBuilder
from tests.test_api import DeterministicTestEmbedder


class FactSDK:
    """仅替代外部 SDK。来源金标由用例明确给出，不由解析器生成。"""
    def __init__(self, facts):
        self.facts = facts
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        if "chunks" not in payload:
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                message=SimpleNamespace(content=json.dumps(self.query_output)))])
        items = []
        for chunk in payload["chunks"]:
            drafts = self.facts.get(chunk["text"], [])
            items.append({"chunk_id": chunk["chunk_id"], "complete": True, "facts": drafts})
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps({"items": items})))])


def draft(quote, value, **kwargs):
    result = dict(subject="self", predicate="employer", scope="work", value=value, quote=quote)
    result.update(kwargs)
    return result


def make_service(tmp_path, facts, mode="active", **updates):
    values = dict(_env_file=None, database_path=tmp_path / "gov.db",
                  governance_mode=mode, llm_provider="none", llm_add_enrichment=False,
                  llm_search_expansion=False, temporal_extraction_mode="rules", min_relevance_score=0.0)
    values.update(updates)
    config = Settings(**values)
    llm = OpenAICompatibleMemoryLLM(LLMConnection("test", "https://example.invalid", "test"),
                                    1, 0, False, False)
    sdk = FactSDK(facts)
    llm._client = sdk
    service = MemoryService(config, SQLiteMemoryStore(config.database_path),
                            DeterministicTestEmbedder(), llm)
    service.initialize()
    return service, sdk


def add(service, rid, text, timestamp=None, user="u", session=None, role="user"):
    service.add(rid, [MemoryMessage(role=role, content=text, timestamp=timestamp)],
                user, session or rid)


def test_current_state_reads_complete_versions_even_at_top_one(tmp_path):
    old = "I work at Acme."
    new = "I now work at Beta."
    service, sdk = make_service(tmp_path, {old: [draft(old, "Acme")], new: [draft(new, "Beta")]})
    add(service, "old", old, 1704067200000)
    add(service, "new", new, 1706745600000)
    hits = service.search("Where do I currently work?", "u", 1)
    assert len(hits) == 1
    assert "Beta" in hits[0].content
    assert "Acme" not in hits[0].content
    assert len(sdk.calls) == 2


def test_correction_invalidates_historical_claim_without_inventing_a_change(tmp_path):
    old = "I work at Acme."
    corrected = "Correction: I never worked at Acme. I work at Beta."
    service, _ = make_service(tmp_path, {
        old: [draft(old, "Acme")],
        corrected: [draft("I work at Beta.", "Beta", operation="correct", target_value="Acme",
                          update_quote=corrected)]})
    add(service, "old", old, 1704067200000)
    add(service, "corrected", corrected, 1706745600000)
    snapshot = service.store.fetch_by_user("u", governance=True)
    history = StateResolver().resolve(snapshot, intent="history")
    assert [f.value for f in history.selected] == ["Beta"]
    assert set(history.excluded.values()) == {"corrected_claim"}
    assert any(e.quote == corrected for e in history.update_evidence)
    hits = service.search("What is my employment history?", "u", 1)
    assert "Beta" in hits[0].content
    assert "纠正" in hits[0].content


@pytest.mark.parametrize("text,role,kind", [
    ("I plan to work at Gamma.", "user", "state"),
    ("I do not work at Gamma.", "user", "state"),
    ("I work at Gamma if the contract is signed.", "user", "state"),
    ("I work at Gamma.", "assistant", "state"),
    ("I plan to work at Gamma.", "user", "rule"),
])
def test_invalid_attribution_or_modality_cannot_overwrite_current_value(tmp_path, text, role, kind):
    old = "I work at Beta."
    service, _ = make_service(tmp_path, {old: [draft(old, "Beta")], text: [draft(text, "Gamma", kind=kind)]})
    add(service, "old", old, 1704067200000)
    add(service, "unsafe", text, 1706745600000, role=role)
    snapshot = service.store.fetch_by_user("u", governance=True)
    assert any(s.status == "partial" for s in snapshot.statuses)
    resolution = StateResolver().resolve(snapshot)
    assert [f.value for f in resolution.selected] == ["Beta"]
    hits = service.search("Where do I currently work?", "u", 1)
    assert "待确认" in hits[0].content
    assert "Gamma" not in hits[0].content.split("【待确认原文】")[0]


def test_long_source_is_batched_with_full_coverage_and_bounded_input(tmp_path):
    text = "\n".join(f"Discussion note {i}: we considered several possible approaches today."
                     for i in range(350))
    service, sdk = make_service(tmp_path, {}, governance_input_tokens=1200)
    add(service, "long", text)
    assert len(sdk.calls) > 1
    for call in sdk.calls:
        assert sum(token_count(m["content"]) for m in call["messages"]) <= 1200
    snapshot = service.store.fetch_by_user("u", governance=True)
    assert snapshot.statuses[0].status == "no_fact"
    assert snapshot.statuses[0].processed_ranges == ((0, len(text)),)
    assert not snapshot.incomplete


def test_future_effective_state_does_not_replace_current_and_late_history_stays_old(tmp_path):
    old, new = "I work at Acme.", "I work at Beta from 2025-01-01."
    service, _ = make_service(tmp_path, {
        old: [draft(old, "Acme")],
        new: [draft(new, "Beta", valid_from="2025-01-01", time_quote="2025-01-01")]})
    add(service, "new", new, 1706745600000)
    add(service, "late", old, 1704067200000)
    hits = service.search("Where do I currently work?", "u", 1, query_time_ms=1711929600000)
    assert "Acme" in hits[0].content and "Beta" not in hits[0].content
    hits = service.search("Where do I currently work?", "u", 1, query_time_ms=1740787200000)
    assert "Beta" in hits[0].content and "Acme" not in hits[0].content


def test_summary_covers_early_decision_and_rebuilds_current_without_losing_history(tmp_path):
    old, new = "I work at Acme.", "I now work at Beta."
    decision = "I decided to retain the safety exception."
    service, sdk = make_service(tmp_path, {
        old: [draft(old, "Acme")], new: [draft(new, "Beta")],
        decision: [dict(subject="self", predicate="event", scope="general",
                        value="safety exception", quote=decision, kind="decision")]})
    add(service, "early", decision)
    add(service, "old", old, 1704067200000)
    first = service.search("Summarize my entire history and key decisions.", "u", 1)
    assert "safety exception" in first[0].content
    add(service, "new", new, 1706745600000)
    final = service.search("Summarize my entire history and key decisions.", "u", 1)
    assert "safety exception" in final[0].content
    assert "时间未知" in final[0].content
    current = final[0].content.split("【当前状态】")[1].split("【")[0]
    assert "Beta" in current and "Acme" not in current
    assert "【历史变化】" in final[0].content and "Acme" in final[0].content
    snapshot = service.store.fetch_by_user("u", governance=True)
    assert {u.level for u in snapshot.summaries} == {"session", "topic", "overview"}
    assert len(sdk.calls) == 3


def test_deleting_update_source_invalidates_summary_and_does_not_resurrect_old_state(tmp_path):
    old, new = "I work at Acme.", "I now work at Beta."
    service, _ = make_service(tmp_path, {old: [draft(old, "Acme")], new: [draft(new, "Beta")]})
    add(service, "old", old, 1704067200000)
    first = service.store.fetch_by_user("u", governance=True)
    stale_summary = SummaryBuilder().build(first)
    add(service, "new", new, 1706745600000)
    assert not service.store.publish_summaries("u", stale_summary)
    newer = next(s.id for s in service.store.fetch_by_user("u", governance=True).sources.values()
                 if s.content == new)
    assert service.store.delete_sources("other", [newer]) == 0
    assert service.store.delete_sources("u", [newer]) == 1
    snapshot = service.store.fetch_by_user("u", governance=True)
    assert not snapshot.summaries
    hits = service.search("Where do I currently work?", "u", 1)
    assert "待确认" in hits[0].content and "Acme" not in hits[0].content and "Beta" not in hits[0].content
    assert service.rebuild_summaries("u")
    text = service.search("Summarize my work history.", "u", 1)[0].content
    assert "Beta" not in text


def test_same_name_in_different_sessions_is_not_silently_merged(tmp_path):
    a, b = "Alex works at Acme.", "Alex works at Beta."
    service, _ = make_service(tmp_path, {
        a: [dict(subject="Alex", predicate="employer", value="Acme", quote=a)],
        b: [dict(subject="Alex", predicate="employer", value="Beta", quote=b)]})
    add(service, "a", a, 1704067200000)
    add(service, "b", b, 1706745600000)
    snapshot = service.store.fetch_by_user("u", governance=True)
    assert len({f.subject_id for f in snapshot.facts}) == 2
    text = service.search("Where does Alex currently work?", "u", 1)[0].content
    assert "身份待确认" in text
    assert "Acme" in text and "Beta" in text


def test_supported_alias_in_same_identity_context_can_update_the_named_entity(tmp_path):
    old = "Robert at LabA works at Acme."
    new = "Bob at LabA, also known as Robert, now works at Beta."
    service, _ = make_service(tmp_path, {
        old: [dict(subject="Robert", identity_context="LabA", predicate="employer", value="Acme", quote=old)],
        new: [dict(subject="Bob", aliases=["Robert"], identity_context="LabA", predicate="employer", value="Beta", quote=new)]})
    add(service, "old", old, 1704067200000)
    add(service, "new", new, 1706745600000)
    text = service.search("Where does Robert at LabA currently work?", "u", 1)[0].content
    assert "Beta" in text and "Acme" not in text and "身份待确认" not in text


def test_dated_correction_preserves_earlier_genuine_same_value(tmp_path):
    first, middle, last = "I work at Acme.", "I work at Beta.", "I currently work at Acme."
    correction = "Correction: my 2024-03-01 claim that I work at Acme was wrong. I work at Beta."
    service, _ = make_service(tmp_path, {
        first: [draft(first, "Acme")], middle: [draft(middle, "Beta")], last: [draft(last, "Acme")],
        correction: [draft("I work at Beta.", "Beta", operation="correct", target_value="Acme",
                           target_date="2024-03-01", update_quote=correction)]})
    for rid, text, stamp in [("first", first, 1704067200000), ("middle", middle, 1706745600000),
                              ("last", last, 1709251200000), ("correction", correction, 1711929600000)]:
        add(service, rid, text, stamp)
    snapshot = service.store.fetch_by_user("u", governance=True)
    history = StateResolver().resolve(snapshot, intent="history")
    assert any(f.quote == first for f in history.selected)
    assert not any(f.quote == last for f in history.selected)
    text = service.search("Where do I currently work?", "u", 1)[0].content
    assert "Beta" in text and "Acme" not in text


def test_quoted_other_person_first_person_is_not_user_current_state(tmp_path):
    own = "I work at Beta."
    reported = 'Chris said: "I now work at Gamma."'
    service, _ = make_service(tmp_path, {own: [draft(own, "Beta")],
                                       reported: [draft("I now work at Gamma.", "Gamma")]})
    add(service, "own", own, 1704067200000)
    add(service, "reported", reported, 1706745600000)
    snapshot = service.store.fetch_by_user("u", governance=True)
    assert [f.value for f in StateResolver().resolve(snapshot).selected] == ["Beta"]
    assert snapshot.incomplete


def test_legacy_display_timestamp_cannot_be_effective_time_evidence(tmp_path):
    from app.storage import MemoryToStore
    legacy = "[user | 2024-01-01T00:00:00.000Z]\nI work at Acme."
    service, _ = make_service(tmp_path, {legacy: [draft("I work at Acme.", "Acme",
        valid_from="2024-01-01", time_quote="2024-01-01")]}, mode="off")
    lease = service.store.claim_request("legacy", "hash", "u", "s", "2026-10-01T00:00:00Z")
    service.store.complete_request(lease, [MemoryToStore("legacy-source", 0, "u", "s", "user", legacy,
        legacy, service.embedder.embed([legacy])[0], "2024-01-01T00:00:00Z", 1704067200000)])
    result = service.index_sources("u")
    snapshot = service.store.fetch_by_user("u", governance=True)
    assert not result["complete"] and snapshot.incomplete
    assert not any(f.valid_from is not None for f in snapshot.facts)


@pytest.mark.parametrize("top_k", [1, 10])
def test_current_multihop_uses_active_entity_and_cannot_restore_old_neighbor(tmp_path, top_k):
    old = "I work at Acme."
    new = "I changed from Acme to Beta."
    a, b = "Acme founder is Ada.", "Beta founder is Bea."
    misleading = "Network founder is Ada and Network recommends Acme."
    service, sdk = make_service(tmp_path, {
        old: [draft(old, "Acme")],
        new: [draft(new, "Beta", operation="change", target_value="Acme", update_quote=new,
                    object_entities=["Acme", "Beta"])],
        a: [dict(subject="Acme", predicate="founder", value="Ada", quote=a, kind="other")],
        b: [dict(subject="Beta", predicate="founder", value="Bea", quote=b, kind="other")],
        misleading: [dict(subject="Network", predicate="founder", value="Ada", quote=misleading,
                          kind="other", object_entities=["Acme"])],
    }, llm_search_expansion=True, temporal_mode="off")
    service.llm.search_expansion = True
    service.llm.multihop_retrieval = True
    query = "Who founded my current employer?"
    sdk.query_output = {"expanded_query": "my current employer; Who founded", "retrieval_steps": [
        {"query": "my current employer", "evidence": "my current employer"},
        {"query": "Who founded", "evidence": "Who founded"}]}
    add(service, "old", old, 1704067200000)
    service.add("mixed", [MemoryMessage(role="user", content=t, timestamp=1706745600000)
                           for t in (new, a, b, misleading)], "u", "second")
    hits = service.search(query, "u", top_k)
    text = "\n".join(h.content for h in hits)
    assert "Beta" in text and "Bea" in text
    assert "Ada" not in text
    assert len([c for c in sdk.calls if "chunks" not in json.loads(c["messages"][1]["content"])]) == 1


def test_late_message_describing_earlier_effective_state_is_not_a_current_conflict(tmp_path):
    current = "I work at Beta."
    past = "I worked at Acme from 2023-01-01."
    service, _ = make_service(tmp_path, {
        current: [draft(current, "Beta")],
        past: [draft(past, "Acme", valid_from="2023-01-01", time_quote="2023-01-01")]})
    add(service, "current", current, 1706745600000)
    add(service, "late", past, 1735689600000)
    result = StateResolver().resolve(service.store.fetch_by_user("u", governance=True))
    assert [f.value for f in result.selected] == ["Beta"]
    assert not result.conflicts


def test_backfill_and_generation_activation_never_replay_add_or_duplicate_sources(tmp_path):
    old, new = "I work at Acme.", "I now work at Beta."
    service, sdk = make_service(tmp_path, {old: [draft(old, "Acme")], new: [draft(new, "Beta")]}, mode="off")
    add(service, "old", old, 1704067200000)
    add(service, "new", new, 1706745600000)
    legacy = service.search("Where do I currently work?", "u", 1)
    assert len(sdk.calls) == 0
    generation = service.store.create_index_generation("u")
    first = service.store.fetch_by_user("u", governance=True, generation=generation)
    assert not service.store.activate_index_generation("u", generation, first.revision)
    assert service.index_sources("u", generation=generation)["complete"]
    assert service.store.count_by_user("u") == 2
    assert service.search("Where do I currently work?", "u", 1) == legacy
    staged = service.store.fetch_by_user("u", governance=True, generation=generation)
    assert service.store.activate_index_generation("u", generation, staged.revision)
    service.settings = service.settings.model_copy(update={"governance_mode": "active"})
    assert "Beta" in service.search("Where do I currently work?", "u", 1)[0].content
    assert service.index_sources("u")["processed"] == 0


def test_historical_date_query_returns_dated_observations_with_unknown_boundaries(tmp_path):
    old, new = "I work at Acme.", "I now work at Beta."
    service, _ = make_service(tmp_path, {old: [draft(old, "Acme")], new: [draft(new, "Beta")]}, temporal_mode="off")
    add(service, "old", old, 1704067200000)
    add(service, "new", new, 1706745600000)
    text = service.search("Where did I work on 2024-01-15?", "u", 1)[0].content
    assert "Acme" in text and "Beta" not in text
    assert "区间" in text and "提及日:2024-01-01" in text


def test_summary_keeps_negative_rule_exception_and_low_frequency_early_event_under_budget(tmp_path):
    decision = "I decided to preserve the audit record."
    rule = "My policy does not allow deployment unless the owner approves."
    sources = [f"My employer for project Task{i} is Company{i}." for i in range(20)]
    facts = {t: [draft(t, f"Company{i}", scope=f"Task{i}")] for i, t in enumerate(sources)}
    facts[decision] = [dict(subject="self", predicate="event", value="audit record", quote=decision, kind="decision")]
    facts[rule] = [dict(subject="self", predicate="rules", value="deployment", quote=rule,
                        kind="rule", polarity="negative")]
    service, _ = make_service(tmp_path, facts, governance_content_tokens=768)
    add(service, "early", decision)
    add(service, "rule", rule)
    service.add("many", [MemoryMessage(role="user", content=t) for t in sources], "u", "later")
    text = service.search("Summarize my entire history and rules.", "u", 1)[0].content
    assert "audit record" in text and rule in text
    assert "未覆盖" in text
    assert token_count(text) <= 768


def test_negative_observation_is_visible_as_conflict_instead_of_confirming_old_positive(tmp_path):
    old, neg = "I work at Beta.", "I do not work at Beta."
    service, _ = make_service(tmp_path, {old: [draft(old, "Beta")], neg: [draft(neg, "Beta", polarity="negative")]})
    add(service, "old", old, 1704067200000)
    add(service, "neg", neg, 1706745600000)
    text = service.search("Where do I currently work?", "u", 1)[0].content
    assert "冲突" in text and neg in text


@pytest.mark.parametrize("query", ["Where do I currently work?", "Summarize my entire history.", "Tell me about Acme."])
def test_shadow_returns_frozen_legacy_results_and_spends_no_extra_query_calls(tmp_path, query):
    old, new = "I work at Acme.", "I now work at Beta."
    service, sdk = make_service(tmp_path, {old: [draft(old, "Acme")], new: [draft(new, "Beta")]},
                                mode="shadow", llm_search_expansion=True, temporal_mode="off")
    service.llm.search_expansion = True
    sdk.query_output = {"expanded_query": "", "retrieval_steps": []}
    add(service, "old", old, 1704067200000)
    add(service, "new", new, 1706745600000)
    baseline = MemoryService(service.settings.model_copy(update={"governance_mode": "off"}), service.store,
                             service.embedder, service.llm)
    frozen = baseline.search(query, "u", 10)
    count = len(sdk.calls)
    assert service.search(query, "u", 10) == frozen
    assert len(sdk.calls) - count == 1


@pytest.mark.parametrize("policy", ["multiple", "plan", "unknown_time", "retract", "unknown_property"])
def test_version_policy_does_not_apply_last_write_wins_to_every_slot(tmp_path, policy):
    first, second = "I work at Alpha.", "I now work at Beta."
    a, b = draft(first, "Alpha"), draft(second, "Beta")
    expected = ["Beta"]
    if policy == "multiple":
        first, second = "My preference is Alpha.", "My preference also includes Beta."
        a = draft(first, "Alpha", predicate="preference", scope="general", kind="preference")
        b = draft(second, "Beta", predicate="preference", scope="general", kind="preference", operation="add")
        expected = ["Alpha", "Beta"]
    elif policy == "plan":
        second = "I plan to work at Beta."
        b = draft(second, "Beta", modality="planned")
        expected = ["Alpha"]
    elif policy == "unknown_time":
        expected = ["Alpha", "Beta"]
    elif policy == "retract":
        second = "I withdraw my earlier Alpha statement."
        b = draft(second, "Alpha", operation="retract", target_value="Alpha", update_quote=second)
        expected = []
    elif policy == "unknown_property":
        first, second = "My badge is Alpha.", "My badge is Beta."
        a, b = draft(first, "Alpha", predicate="badge"), draft(second, "Beta", predicate="badge")
        expected = ["Alpha", "Beta"]
    service, _ = make_service(tmp_path, {first: [a], second: [b]})
    add(service, "first", first, None if policy == "unknown_time" else 1704067200000)
    add(service, "second", second, None if policy == "unknown_time" else 1706745600000)
    result = StateResolver().resolve(service.store.fetch_by_user("u", governance=True))
    assert sorted(f.value for f in result.selected) == sorted(expected)


@pytest.mark.parametrize("fault", ["missing", "truncated", "network", "budget"])
def test_incomplete_extraction_keeps_original_success_and_can_be_repaired(tmp_path, fault):
    text = "I work at Beta."
    service, sdk = make_service(tmp_path, {text: [draft(text, "Beta")]}, mode="shadow")
    good_create = sdk.create
    def broken_create(**kwargs):
        if fault == "network":
            raise RuntimeError("synthetic transport error")
        response = good_create(**kwargs)
        if fault == "missing":
            response.choices[0].message.content = '{"items":[]}'
        elif fault == "truncated":
            response.choices[0].finish_reason = "length"
        return response
    sdk.create = broken_create
    if fault == "budget":
        service.settings = service.settings.model_copy(update={"governance_add_budget_seconds": 0.000001})
    add(service, "partial", text)
    assert service.store.count_by_user("u") == 1
    snapshot = service.store.fetch_by_user("u", governance=True)
    assert snapshot.incomplete
    assert snapshot.statuses[0].status != "no_fact"
    sdk.create = good_create
    service.settings = service.settings.model_copy(update={"governance_add_budget_seconds": 60})
    assert service.index_sources("u")["complete"]
    assert not service.store.fetch_by_user("u", governance=True).incomplete
    assert service.store.count_by_user("u") == 1
