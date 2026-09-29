"""接口输入日志测试：记录内容、截断、开关、密钥不落盘、422 也记录。

全部离线运行（假向量器 + 无 LLM），日志写到临时目录逐行校验。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
        return [token for token in normalized.split() if token]

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                index = self._vocabulary.setdefault(token, len(self._vocabulary))
                matrix[row, index] += 1.0
        return normalize_rows(matrix)


def _settings(tmp_path: Path, log_path: Path, **overrides: object) -> Settings:
    defaults: dict[str, object] = dict(
        _env_file=None,
        database_path=tmp_path / "test.db",
        embedding_cache_dir=tmp_path / "models",
        llm_provider="none",
        input_log_path=log_path,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _read_records(log_path: Path) -> list[dict[str, object]]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_input_log_records_set_and_get_bodies(tmp_path: Path) -> None:
    log_path = tmp_path / "input.jsonl"
    test_app = create_app(
        _settings(tmp_path, log_path), embedder=DeterministicTestEmbedder()
    )
    with TestClient(test_app) as client:
        assert (
            client.post(
                "/set",
                json={
                    "request_id": "eval:demo:chunk-0",
                    "messages": [{"role": "user", "content": "Alice prefers tea."}],
                    "user_id": "eval:demo:user-1",
                    "session_id": "eval:demo:session-1",
                },
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/get",
                json={
                    "query": "What does Alice prefer?",
                    "user_id": "eval:demo:user-1",
                    "top_k": 100,
                },
            ).status_code
            == 200
        )

    records = _read_records(log_path)
    assert len(records) == 2

    set_record, get_record = records
    assert set_record["path"] == "/set"
    assert set_record["status"] == 200
    assert set_record["method"] == "POST"
    assert set_record["request_id"] == "eval:demo:chunk-0"
    assert set_record["user_id"] == "eval:demo:user-1"
    assert set_record["session_id"] == "eval:demo:session-1"
    assert "Alice prefers tea." in set_record["body"]
    assert "ts" in set_record and "duration_ms" in set_record

    assert get_record["path"] == "/get"
    assert get_record["query"] == "What does Alice prefer?"
    assert get_record["top_k"] == 100


def test_input_log_skips_get_requests(tmp_path: Path) -> None:
    log_path = tmp_path / "input.jsonl"
    test_app = create_app(
        _settings(tmp_path, log_path), embedder=DeterministicTestEmbedder()
    )
    with TestClient(test_app) as client:
        assert client.get("/health").status_code == 200

    # 健康检查是 GET，不入输入日志；无 POST 时文件不创建（delay 懒打开）。
    assert not log_path.exists()


def test_input_log_truncates_long_body(tmp_path: Path) -> None:
    log_path = tmp_path / "input.jsonl"
    test_app = create_app(
        _settings(tmp_path, log_path, input_log_max_chars=200),
        embedder=DeterministicTestEmbedder(),
    )
    long_text = "A" * 5_000
    with TestClient(test_app) as client:
        assert (
            client.post("/set", json={"memory_text": long_text}).status_code == 200
        )

    record = _read_records(log_path)[0]
    assert record["body_chars"] > 5_000
    # 正文截断到配置上限，原始长度由 body_chars 保留。
    assert len(record["body"]) == 200


def test_input_log_disabled_writes_nothing(tmp_path: Path) -> None:
    log_path = tmp_path / "input.jsonl"
    test_app = create_app(
        _settings(tmp_path, log_path, input_log_enabled=False),
        embedder=DeterministicTestEmbedder(),
    )
    with TestClient(test_app) as client:
        assert (
            client.post("/set", json={"memory_text": "hello"}).status_code == 200
        )
    assert not log_path.exists()


def test_input_log_never_records_auth_headers(tmp_path: Path) -> None:
    log_path = tmp_path / "input.jsonl"
    test_app = create_app(
        _settings(tmp_path, log_path, memory_api_key="super-secret-key"),
        embedder=DeterministicTestEmbedder(),
    )
    with TestClient(test_app) as client:
        assert (
            client.post(
                "/set",
                json={"memory_text": "hello"},
                headers={"Authorization": "Bearer super-secret-key"},
            ).status_code
            == 200
        )

    content = log_path.read_text(encoding="utf-8")
    assert "super-secret-key" not in content
    record = _read_records(log_path)[0]
    assert record["path"] == "/set"


def test_input_log_records_rejected_payloads(tmp_path: Path) -> None:
    log_path = tmp_path / "input.jsonl"
    test_app = create_app(
        _settings(tmp_path, log_path), embedder=DeterministicTestEmbedder()
    )
    with TestClient(test_app) as client:
        response = client.post("/set", json={"unexpected_field": "oops"})
        assert response.status_code == 422

    records = _read_records(log_path)
    assert len(records) == 1
    # 未通过 schema 校验的请求体同样被记录——排查契约问题时最需要的就是它。
    assert records[0]["status"] == 422
    assert "unexpected_field" in records[0]["body"]
