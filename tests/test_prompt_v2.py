"""V2 接口回归：模拟 SDK 响应，只证明数据流，不证明 LLM 的语义能力。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import LLMConnection, OpenAICompatibleMemoryLLM
from app.main import create_app
from app.prompts import ADD_ENRICHMENT_PROMPT_V2, QUERY_EXPANSION_PROMPT_V3


CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "prompt_v2_cases.json").read_text(
        encoding="utf-8"
    )
)["cases"]


class RecordingEmbedder:
    """记录真实送入向量器的文本，不评价检索排序。"""

    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.inputs.append(list(texts))
        return np.ones((len(texts), 1), dtype=np.float32)


def make_llm(responses: list[object]) -> tuple[OpenAICompatibleMemoryLLM, Mock]:
    def sdk_response(value: object) -> SimpleNamespace:
        content = value if isinstance(value, str) else json.dumps(value)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    completion = Mock(side_effect=[sdk_response(value) for value in responses])
    llm = OpenAICompatibleMemoryLLM(
        LLMConnection("test-only", "https://example.invalid/v1", "test-model"),
        timeout_seconds=1,
        max_retries=0,
        add_enrichment=True,
        search_expansion=True,
    )
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
    )
    return llm, completion


def make_app(tmp_path: Path, llm: OpenAICompatibleMemoryLLM, mode: str = "strict"):
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "memory.db",
        llm_provider="none",
        llm_failure_mode=mode,
        memory_api_key="",
        warmup_embedding_on_startup=False,
        semantic_weight=1,
        lexical_weight=0,
    )
    embedder = RecordingEmbedder()
    return create_app(settings, llm=llm, embedder=embedder), embedder


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_v2_add_search_data_flow(case: dict, tmp_path: Path) -> None:
    llm, completion = make_llm([case["add_output"], case["query_output"]])
    app, embedder = make_app(tmp_path, llm)
    payload = {
        "request_id": case["id"],
        "messages": case["messages"],
        "user_id": "v2-user",
        "session_id": "v2-session",
    }
    with TestClient(app) as client:
        added = client.post("/set", json=payload)
        assert added.status_code == 200
        assert added.json() == {
            "success": True,
            "request_id": case["id"],
            "user_id": "v2-user",
            "session_id": "v2-session",
        }
        # 重试不得再次调用 LLM 或重复写入。
        assert client.post("/set", json=payload).status_code == 200
        assert completion.call_count == 1
        assert len(embedder.inputs) == 1

        originals = [f"[{m['role']}]\n{m['content']}" for m in case["messages"]]
        by_index = {
            item["source_index"]: item["search_text"]
            for item in case["add_output"]["items"]
        }
        indexed = [
            original + ("\n索引增强：" + by_index[i] if i in by_index else "")
            for i, original in enumerate(originals)
        ]
        assert embedder.inputs[0] == indexed
        stored = app.state.memory_service.store.fetch_by_user("v2-user")
        assert {r.content: r.search_text for r in stored} == dict(zip(originals, indexed))

        search = {"query": case["query"], "user_id": "v2-user", "top_k": 100}
        if "options" in case:
            search["options"] = case["options"]
        found = client.post("/get", json=search)
        assert found.status_code == 200
        assert {hit["content"] for hit in found.json()["data"]} == set(originals)

    base_parts = [case["query"]]
    if case.get("options"):
        base_parts.append("候选项：\n" + "\n".join(case["options"]))
    # 双通道：裸查询与扩展文本在同一次 embed 调用中作为两个通道；
    # 无扩展文本时只有裸查询一个通道。
    expected_query_texts = ["\n".join(base_parts)]
    expanded_query = case["query_output"]["expanded_query"]
    if expanded_query:
        expected_query_texts.append(
            "\n".join(base_parts) + "\n查询扩展：" + expanded_query
        )
    assert embedder.inputs[1] == expected_query_texts

    add_call, search_call = [call.kwargs for call in completion.call_args_list]
    assert add_call["messages"][0] == {"role": "system", "content": ADD_ENRICHMENT_PROMPT_V2}
    assert search_call["messages"][0] == {"role": "system", "content": QUERY_EXPANSION_PROMPT_V3}
    assert json.loads(add_call["messages"][1]["content"]) == {"messages": case["messages"]}
    assert json.loads(search_call["messages"][1]["content"]) == {
        "query": case["query"], "options": case.get("options", [])
    }
    assert add_call["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("mode", ["strict", "fallback"])
@pytest.mark.parametrize("phase", ["add", "search"])
@pytest.mark.parametrize("bad_response", ["not valid JSON", {"wrong_field": []}])
def test_invalid_llm_response_handling(
    tmp_path: Path, mode: str, phase: str, bad_response: object
) -> None:
    responses = [bad_response] if phase == "add" else [{"items": []}, bad_response]
    llm, _ = make_llm(responses)
    app, embedder = make_app(tmp_path, llm, mode)
    with TestClient(app) as client:
        response = client.post("/set", json={"memory_text": "你好"})
        if phase == "search":
            assert response.status_code == 200
            response = client.post("/get", json={"query": "你好"})
        assert response.status_code == (503 if mode == "strict" else 200)
        if mode == "fallback":
            assert embedder.inputs[-1] == ["你好"]
        elif phase == "add":
            assert embedder.inputs == []
            assert app.state.memory_service.store.fetch_by_user("local-default") == []
        else:
            assert len(embedder.inputs) == 1


def test_disabled_enhancement_does_not_call_sdk() -> None:
    llm, completion = make_llm([])
    llm.add_enrichment = False
    llm.search_expansion = False
    assert llm.enrich_messages([{"role": "user", "content": "你好"}]) == {}
    expansion = llm.expand_query("你好", None)
    assert expansion.text == ""
    assert expansion.temporal is None
    completion.assert_not_called()
