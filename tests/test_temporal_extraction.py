"""时间抽取回归：规则、原文校验、独立开关与真实服务调用路径。"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from app.llm import LLMConnection, NoOpMemoryLLM, OpenAICompatibleMemoryLLM
from app.temporal import RelativeWindowSpec, parse_temporal, query_mentions_temporal
from app.temporal_extraction import extract_temporal_rules, parse_grounded_temporal
from test_temporal_policy import ConstantEmbedder, add, ms
from app.config import Settings
from app.service import MemoryService, MemoryServiceUnavailable
from app.storage import SQLiteMemoryStore
from app.llm import LLMError, QueryExpansion


@pytest.mark.parametrize("phrase,unit,offset", [
    ("前天", "day", -2), ("昨天", "day", -1), ("今天", "day", 0),
    ("明天", "day", 1), ("本月", "month", 0), ("上个月", "month", -1),
    ("上上周", "week", -2), ("去年", "year", -1),
    ("yesterday", "day", -1), ("today", "day", 0),
    ("this month", "month", 0), ("last week", "week", -1), ("previous month", "month", -1),
])
def test_common_calendar_rules(phrase: str, unit: str, offset: int) -> None:
    result = extract_temporal_rules(f"{phrase}的 project updates")
    assert result.complete
    assert result.constraints.relative_window == RelativeWindowSpec("calendar", unit, 1, offset, "past")


@pytest.mark.parametrize("phrase,unit,amount,direction", [
    ("过去三天", "day", 3, "past"), ("最近两个月", "month", 2, "past"),
    ("未来七天", "day", 7, "future"), ("最近半年", "month", 6, "past"),
    ("过去二十一天", "day", 21, "past"), ("in the past three days", "day", 3, "past"),
    ("过去 三 天", "day", 3, "past"), ("过去一百零二天", "day", 102, "past"),
    ("last 6 months", "month", 6, "past"), ("next two weeks", "week", 2, "future"),
])
def test_common_rolling_rules(phrase: str, unit: str, amount: int, direction: str) -> None:
    result = extract_temporal_rules(f"project updates {phrase}")
    assert result.complete
    assert result.constraints.relative_window == RelativeWindowSpec("rolling", unit, amount, 0, direction)


@pytest.mark.parametrize("query", ["first aid kit", "last name", "Aurora project updates"])
def test_non_temporal_queries_do_not_get_rules(query: str) -> None:
    assert extract_temporal_rules(query).constraints is None
    assert not query_mentions_temporal(query)


@pytest.mark.parametrize("query", [
    "昨天和今天的安排", "last month or this month", "不是上个月的更新",
    "updates not in the last 3 days", "过去零天的安排", "过去1001天的安排",
    "过去一百二天的安排",
    "过去一年半的更新", "过去三天左右的更新", "more than the last 3 days",
    "过去一年三个月的更新", "past 3 years and 2 months", "近三个月前的更新",
])
def test_ambiguous_or_invalid_windows_are_not_silently_narrowed(query: str) -> None:
    result = extract_temporal_rules(query)
    assert result.blocked
    assert result.constraints is None


@pytest.mark.parametrize("item", [
    {"kind": "rolling", "unit": "day", "amount": 3, "direction": "sideways"},
    {"kind": "rolling", "unit": "day", "amount": 3},
    {"kind": "rolling", "unit": "day", "amount": True, "direction": "past"},
    {"kind": "calendar", "unit": "month", "offset": True},
])
def test_invalid_fields_are_rejected(item: dict) -> None:
    assert parse_temporal({"windows": [item]}) is None


def test_multiple_model_windows_are_not_reduced_to_the_first() -> None:
    assert parse_temporal({"windows": [
        {"kind": "calendar", "unit": "month", "offset": -1},
        {"kind": "calendar", "unit": "month", "offset": 0},
    ]}) is None


def test_window_evidence_must_support_every_field() -> None:
    query = "过去七天的项目更新"
    item = {"kind": "rolling", "unit": "day", "amount": 7, "direction": "past", "evidence": "过去七天"}
    assert parse_grounded_temporal({"windows": [item]}, query) is not None
    for override in ({"amount": 999}, {"unit": "year"}, {"direction": "future"}, {"evidence": "未来七天"}):
        assert parse_grounded_temporal({"windows": [{**item, **override}]}, query) is None
    assert parse_grounded_temporal({"windows": [{k: v for k, v in item.items() if k != "evidence"}]}, query) is None


def test_event_evidence_checks_event_text_and_direction() -> None:
    query = "搬到上海之前我住哪里"
    item = {"event": "搬到上海", "direction": "before", "evidence": "搬到上海之前"}
    assert parse_grounded_temporal({"event_anchor": item}, query) is not None
    for override in ({"event": "搬到北京"}, {"direction": "after"}, {"evidence": "搬到上海之后"}):
        assert parse_grounded_temporal({"event_anchor": {**item, **override}}, query) is None


@pytest.mark.parametrize("evidence", ["before moving", "before moving and after graduating"])
def test_partial_event_constraints_do_not_drop_other_time_conditions(evidence: str) -> None:
    query = "Where did I live before moving and after graduating?"
    assert parse_grounded_temporal({"event_anchor": {
        "event": "moving", "direction": "before", "evidence": evidence,
    }}, query) is None


def test_ordering_evidence_must_have_the_same_direction() -> None:
    query = "我最后一次提到项目是什么时候"
    assert parse_grounded_temporal({"ordering": "latest", "ordering_evidence": "最后一次"}, query) is not None
    assert parse_grounded_temporal({"ordering": "earliest", "ordering_evidence": "最后一次"}, query) is None
    assert parse_grounded_temporal({"ordering": "latest"}, query) is None


def test_window_and_ordering_are_both_preserved() -> None:
    result = extract_temporal_rules("上个月最早的 project update")
    assert result.complete
    assert result.constraints.relative_window.offset == -1
    assert result.constraints.ordering == "earliest"


def build_service(tmp_path: Path, llm=None, **overrides) -> MemoryService:
    settings = Settings(**{
        "_env_file": None, "database_path": tmp_path / "extraction.db",
        "llm_provider": "none", "temporal_mode": "strict", "semantic_weight": 1,
        "lexical_weight": 0, "input_log_enabled": False, **overrides,
    })
    service = MemoryService(settings, SQLiteMemoryStore(settings.database_path), ConstantEmbedder(), llm or NoOpMemoryLLM())
    service.initialize()
    return service


def test_rules_work_without_llm_or_query_expansion(tmp_path: Path) -> None:
    service = build_service(tmp_path, llm_search_expansion=False)
    add(service, "yesterday", ms("2026-10-02T12:00:00"))
    add(service, "today", ms("2026-10-03T12:00:00"))
    assert service.search("昨天的更新", "u", 1)[0].content.endswith("yesterday")
    assert service.search("今天的更新", "u", 1)[0].content.endswith("today")


def test_valid_rule_is_not_discarded_by_legacy_signal_guard(tmp_path: Path) -> None:
    service = build_service(tmp_path, llm_search_expansion=False)
    add(service, "old", ms("2020-01-01T00:00:00"))
    add(service, "new", ms("2026-10-03T12:00:00"))
    assert service.search("过去 三 天的更新", "u", 1)[0].content.endswith("new")


def test_extraction_can_be_disabled_without_disabling_expansion(tmp_path: Path) -> None:
    llm = Mock(enabled=True)
    llm.enrich_messages.return_value = {}
    llm.expand_query.return_value = QueryExpansion("project context", None)
    service = build_service(tmp_path, llm, temporal_extraction_mode="off")
    add(service, "old", ms("2020-01-01T00:00:00"))
    add(service, "new", ms("2026-10-03T12:00:00"))
    hits = service.search("今天的更新", "u", 2)
    assert hits[0].score == hits[1].score
    llm.expand_query.assert_called_once()
    llm.extract_temporal.assert_not_called()


def test_rules_do_not_call_temporal_llm(tmp_path: Path) -> None:
    llm = Mock(enabled=True)
    llm.enrich_messages.return_value = {}
    service = build_service(tmp_path, llm, llm_search_expansion=False)
    add(service, "update", ms("2026-10-03T12:00:00"))
    service.search("今天的更新", "u", 1)
    llm.extract_temporal.assert_not_called()
    llm.expand_query.assert_not_called()


def test_expansion_failure_does_not_discard_valid_rules(tmp_path: Path) -> None:
    llm = Mock(enabled=True)
    llm.enrich_messages.return_value = {}
    llm.expand_query.side_effect = LLMError("simulated")
    service = build_service(tmp_path, llm, llm_failure_mode="fallback")
    add(service, "yesterday", ms("2026-10-02T12:00:00"))
    add(service, "today", ms("2026-10-03T12:00:00"))
    assert service.search("今天的更新", "u", 1)[0].content.endswith("today")
    llm.extract_temporal.assert_not_called()


@pytest.mark.parametrize("query", ["before moving to Shanghai", "不是上个月的更新"])
def test_rules_only_mode_does_not_call_llm_for_unresolved_or_blocked_query(tmp_path: Path, query: str) -> None:
    llm = Mock(enabled=True)
    llm.enrich_messages.return_value = {}
    service = build_service(tmp_path, llm, llm_search_expansion=False, temporal_extraction_mode="rules")
    add(service, "record", ms("2026-10-03T12:00:00"))
    assert len(service.search(query, "u", 1)) == 1
    llm.extract_temporal.assert_not_called()


def test_http_uses_rules_with_query_expansion_disabled(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from app.main import create_app

    service = build_service(tmp_path, llm_search_expansion=False)
    app = create_app(service.settings, embedder=service.embedder, llm=service.llm)
    with TestClient(app) as client:
        payload = {
            "request_id": "rules-http", "user_id": "u", "session_id": "s",
            "messages": [
                {"role": "user", "content": "yesterday", "timestamp": ms("2026-10-02T12:00:00")},
                {"role": "user", "content": "today", "timestamp": ms("2026-10-03T12:00:00")},
            ],
        }
        assert client.post("/set", json=payload).status_code == 200
        response = client.post("/get", json={"query": "今天的更新", "user_id": "u", "top_k": 1})
        assert response.status_code == 200
        hit = response.json()["data"][0]
        assert hit["content"].endswith("today")
        assert set(hit) == {"id", "content", "score", "created_at"}


def test_complex_sdk_extraction_does_not_read_options_or_expansion(tmp_path: Path) -> None:
    from app.prompts import QUERY_EXPANSION_PROMPT_V3, TEMPORAL_EXTRACTION_PROMPT

    llm = OpenAICompatibleMemoryLLM(LLMConnection("test", "https://example.invalid", "test"), 1, 0, False, True)
    llm._json_completion = Mock(side_effect=[
        {"temporal": {"event_anchor": {"event": "moving", "direction": "before", "evidence": "before moving"}}},
        {"expanded_query": "residence before moving", "temporal": {"ordering": "latest"}},
    ])
    service = build_service(tmp_path, llm)
    add(service, "moving", ms("2026-10-03T12:00:00"))
    add(service, "old residence", ms("2026-10-01T12:00:00"))
    assert service.search("Where did I live before moving?", "u", 1, ["tomorrow", "next year"])[0].content.endswith("old residence")
    temporal_call, expansion_call = llm._json_completion.call_args_list
    assert temporal_call.args == (TEMPORAL_EXTRACTION_PROMPT, {"query": "Where did I live before moving?"})
    assert expansion_call.args[0] == QUERY_EXPANSION_PROMPT_V3
    assert expansion_call.args[1]["options"] == ["tomorrow", "next year"]


def test_complex_extraction_has_independent_llm_entry() -> None:
    llm = OpenAICompatibleMemoryLLM(LLMConnection("test", "https://example.invalid", "test"), 1, 0, False, False)
    llm._json_completion = Mock(return_value={"temporal": {"event_anchor": {
        "event": "moving to Shanghai", "direction": "before", "evidence": "before moving to Shanghai",
    }}})
    temporal = llm.extract_temporal("Where did I live before moving to Shanghai?")
    assert temporal.event_anchor.event == "moving to Shanghai"
    assert llm.expand_query("anything", None).text == ""
    assert llm._json_completion.call_count == 1


def test_temporal_prompt_satisfies_json_mode_provider_requirement() -> None:
    """真实接口拒绝未明确要求 JSON 的请求；在 SDK 边界复现该条件。"""
    llm = OpenAICompatibleMemoryLLM(
        LLMConnection("test", "https://example.invalid", "test"), 1, 0, False, False,
    )
    client = Mock()

    def completion(**kwargs: Any) -> Mock:
        if kwargs["response_format"] == {"type": "json_object"}:
            if not any("json" in item["content"].casefold() for item in kwargs["messages"]):
                raise ValueError("JSON mode requires an explicit JSON instruction")
        return Mock(choices=[Mock(message=Mock(content=(
            '{"temporal":{"event_anchor":{"event":"搬到上海",'
            '"direction":"before","evidence":"搬到上海之前"}}}'
        )))])

    client.chat.completions.create.side_effect = completion
    llm._client = client
    actual = llm.extract_temporal("搬到上海之前，我住在哪里？")
    assert actual.event_anchor.event == "搬到上海"
    client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("mode,status", [("fallback", "ok"), ("strict", "error")])
def test_complex_extraction_failure_respects_failure_policy(tmp_path: Path, mode: str, status: str) -> None:
    llm = Mock(enabled=True)
    llm.enrich_messages.return_value = {}
    llm.extract_temporal.side_effect = LLMError("simulated")
    service = build_service(tmp_path, llm, llm_search_expansion=False, llm_failure_mode=mode)
    add(service, "record", ms("2026-10-03T12:00:00"))
    if status == "error":
        with pytest.raises(MemoryServiceUnavailable):
            service.search("before moving to Shanghai", "u", 1)
    else:
        assert len(service.search("before moving to Shanghai", "u", 1)) == 1
