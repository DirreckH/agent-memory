from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import normalize_rows
from app.main import create_app


class DeterministicTestEmbedder:
    """离线测试向量器：保证测试不下载模型、不访问任何外部 API。"""

    dimension = 512

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = "".join(
            character.casefold() if character.isalnum() else " "
            for character in text
        )
        raw_tokens = [token for token in normalized.split() if token]
        tokens = list(raw_tokens)
        for token in raw_tokens:
            if len(token) > 5 and token.endswith("ing"):
                tokens.append(token[:-3])
            if len(token) > 4 and token.endswith("ed"):
                tokens.append(token[:-2])
            if len(token) > 4 and token.endswith("es"):
                tokens.append(token[:-2])
            elif len(token) > 4 and token.endswith("s"):
                tokens.append(token[:-1])
        return tokens

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                # 测试语料很小，顺序词表可完全避免哈希碰撞造成的假阳性。
                index = self._vocabulary.setdefault(token, len(self._vocabulary))
                if index >= self.dimension:
                    raise AssertionError("测试词表超过离线向量器容量")
                matrix[row, index] += 1.0
        return normalize_rows(matrix)


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "test.db",
        embedding_cache_dir=tmp_path / "models",
        llm_provider="none",
        min_relevance_score=0.20,
        semantic_weight=0.8,
        lexical_weight=0.2,
    )
    test_app = create_app(settings, embedder=DeterministicTestEmbedder())
    with TestClient(test_app) as test_client:
        yield test_client


def test_set_then_get_returns_related_memory(client: TestClient) -> None:
    set_response = client.post(
        "/set", json={"memory_text": "Project Aurora launches on Friday."}
    )
    assert set_response.status_code == 200
    assert set_response.json() == {
        "success": True,
        "message": "记忆写入成功，现已可检索",
        "memory_count": 1,
    }

    get_response = client.post(
        "/get", json={"query": "When does Project Aurora launch?"}
    )
    assert get_response.status_code == 200
    body = get_response.json()
    assert "Project Aurora launches on Friday." in body["memory_text"]
    assert body["results"][0]["score"] >= 0.20


def test_unrelated_query_returns_empty_result(client: TestClient) -> None:
    client.post("/set", json={"memory_text": "Project Aurora launches Friday."})

    response = client.post(
        "/get", json={"query": "How do whales communicate underwater?"}
    )
    assert response.status_code == 200
    assert response.json() == {"memory_text": "", "results": []}


def test_competition_add_and_search_contract(client: TestClient) -> None:
    payload = {
        "request_id": "eval:run-1:chunk-0",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "Alice prefers Ethiopian coffee.",
            },
            {
                "role": "assistant",
                "content": "Alice bought coffee beans on Monday.",
            },
        ],
        "user_id": "eval:run-1:user-1",
        "session_id": "eval:run-1:session-1",
    }
    add_response = client.post("/set", json=payload)
    assert add_response.status_code == 200
    assert add_response.json() == {
        "success": True,
        "request_id": payload["request_id"],
        "user_id": payload["user_id"],
        "session_id": payload["session_id"],
    }

    search_response = client.post(
        "/get",
        json={
            "query": "What coffee does Alice prefer?",
            "user_id": payload["user_id"],
            "top_k": 100,
        },
    )
    assert search_response.status_code == 200
    data = search_response.json()["data"]
    assert 1 <= len(data) <= 100
    assert "Ethiopian coffee" in data[0]["content"]
    assert {"id", "content", "score", "created_at"} == set(data[0])


def test_competition_search_is_strictly_isolated_by_user_id(
    client: TestClient,
) -> None:
    client.post(
        "/set",
        json={
            "request_id": "request-isolation",
            "messages": [{"role": "user", "content": "Secret codename Aurora."}],
            "user_id": "user-a",
            "session_id": "session-a",
        },
    )

    response = client.post(
        "/get",
        json={"query": "Aurora codename", "user_id": "user-b", "top_k": 100},
    )
    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_competition_add_is_idempotent(client: TestClient) -> None:
    payload = {
        "request_id": "same-request",
        "messages": [{"role": "user", "content": "The launch code is Aurora."}],
        "user_id": "idempotent-user",
        "session_id": "idempotent-session",
    }
    assert client.post("/set", json=payload).status_code == 200
    assert client.post("/set", json=payload).status_code == 200

    response = client.post(
        "/get",
        json={
            "query": "Aurora launch code",
            "user_id": "idempotent-user",
            "top_k": 100,
        },
    )
    assert len(response.json()["data"]) == 1


def test_same_request_id_with_different_payload_is_rejected(
    client: TestClient,
) -> None:
    base = {
        "request_id": "conflicting-request",
        "messages": [{"role": "user", "content": "First payload."}],
        "user_id": "conflict-user",
        "session_id": "conflict-session",
    }
    assert client.post("/set", json=base).status_code == 200
    changed = {**base, "messages": [{"role": "user", "content": "Changed payload."}]}
    response = client.post("/set", json=changed)
    assert response.status_code == 409
    assert "request_id" in response.json()["detail"]["reason"]


def test_optional_api_key_supports_bearer_and_rejects_missing_key(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "auth.db",
        embedding_cache_dir=tmp_path / "models",
        llm_provider="none",
        memory_api_key="test-secret",
    )
    test_app = create_app(settings, embedder=DeterministicTestEmbedder())
    with TestClient(test_app) as auth_client:
        assert auth_client.get("/health").status_code == 200
        assert (
            auth_client.post("/set", json={"memory_text": "protected"}).status_code
            == 401
        )
        response = auth_client.post(
            "/set",
            json={"memory_text": "protected memory"},
            headers={"Authorization": "Bearer test-secret"},
        )
        assert response.status_code == 200
