from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts.benchmark_common import BenchmarkCase, load_cases
from scripts.generate_benchmark import generate_cases
from scripts.run_benchmark import run_benchmark


@contextmanager
def benchmark_api_server():
    memories: dict[str, list[str]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - 标准库固定接口名
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if self.path == "/set":
                memories.setdefault(payload["user_id"], []).extend(
                    item["content"] for item in payload["messages"]
                )
                body = {
                    "success": True,
                    "request_id": payload["request_id"],
                    "user_id": payload["user_id"],
                    "session_id": payload["session_id"],
                }
                self._json_response(200, body)
                return
            if self.path == "/get":
                query = payload["query"]
                candidates = memories.get(payload["user_id"], [])
                data = []
                if "喜欢喝什么" in query:
                    data = [
                        {"id": "mem-1", "content": text, "score": 0.9}
                        for text in candidates
                        if "桂花乌龙" in text
                    ]
                self._json_response(200, {"data": data})
                return
            self._json_response(404, {"detail": "not found"})

        def _json_response(self, status: int, payload: object) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_benchmark_dataset_requires_answer_to_exist_only_in_positive_memory(
    tmp_path,
) -> None:
    dataset = tmp_path / "cases.jsonl"
    rows = [
        {
            "case_id": "case-0001",
            "kind": "positive",
            "user_group": 0,
            "category": "preference",
            "language": "zh",
            "memory_text": "林澈最喜欢的饮料是桂花乌龙。",
            "query": "林澈平时最喜欢喝什么？",
            "expected_keyword": "桂花乌龙",
        },
        {
            "case_id": "case-0002",
            "kind": "negative",
            "user_group": 0,
            "category": "unrelated",
            "language": "zh",
            "memory_text": "周岚每周三晚上练习小提琴。",
            "query": "木星最大的卫星叫什么？",
            "expected_keyword": "",
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    cases = load_cases(dataset)

    assert cases == [BenchmarkCase(**row) for row in rows]

    invalid = rows[0] | {"expected_keyword": "不存在的答案"}
    with pytest.raises(ValueError, match="expected_keyword 必须原样出现在 memory_text"):
        BenchmarkCase(**invalid)


def test_benchmark_runner_uses_public_set_get_and_scores_positive_and_negative() -> None:
    cases = [
        BenchmarkCase(
            case_id="case-0001",
            kind="positive",
            user_group=0,
            category="preference",
            language="zh",
            memory_text="林澈最喜欢的饮料是桂花乌龙。",
            query="林澈平时喜欢喝什么？",
            expected_keyword="桂花乌龙",
        ),
        BenchmarkCase(
            case_id="case-0002",
            kind="negative",
            user_group=0,
            category="unrelated",
            language="zh",
            memory_text="周岚每周三晚上练习小提琴。",
            query="木星最大的卫星叫什么？",
            expected_keyword="",
        ),
    ]

    with benchmark_api_server() as base_url:
        report = run_benchmark(
            cases,
            base_url=base_url,
            memory_api_key="",
            run_id="unit-test",
            top_k=5,
            timeout_seconds=2,
            concurrency=1,
            show_progress=False,
        )

    assert report["add"]["succeeded"] == 2
    assert report["search"]["succeeded"] == 2
    assert report["quality"]["recall_at_1"] == 1.0
    assert report["quality"]["recall_at_5"] == 1.0
    assert report["quality"]["mrr_at_5"] == 1.0
    assert report["quality"]["negative_false_positive_rate"] == 0.0


def test_deepseek_generation_builds_exact_valid_dataset_and_isolates_negatives() -> None:
    calls: list[tuple[str, int]] = []

    def fake_batch(kind: str, count: int, _: list[str]) -> list[dict[str, object]]:
        offset = sum(previous_count for previous_kind, previous_count in calls if previous_kind == kind)
        calls.append((kind, count))
        if kind == "positive":
            return [
                {
                    "kind": "positive",
                    "category": "preference",
                    "language": "zh",
                    "memory_text": f"合成人物{i}最喜欢的代号是青鸟-{i}。",
                    "query": f"合成人物{i}最喜欢哪个代号？",
                    "expected_keyword": f"青鸟-{i}",
                }
                for i in range(offset, offset + count)
            ]
        return [
            {
                "kind": "negative",
                "category": "unrelated",
                "language": "zh",
                "memory_text": f"合成人物N{i}每周一练习陶艺。",
                "query": f"第{i + 3}颗行星有哪些卫星？",
                "expected_keyword": "",
            }
            for i in range(count)
        ]

    cases = generate_cases(
        fake_batch,
        count=5,
        negative_count=2,
        batch_size=2,
        group_size=2,
        max_attempts=10,
        show_progress=False,
    )

    assert len(cases) == 5
    assert [case.case_id for case in cases] == [
        "case-0001",
        "case-0002",
        "case-0003",
        "case-0004",
        "case-0005",
    ]
    assert calls == [("positive", 2), ("positive", 1), ("negative", 2)]
    positive_groups = {case.user_group for case in cases if case.kind == "positive"}
    negative_groups = [case.user_group for case in cases if case.kind == "negative"]
    assert positive_groups == {0, 1}
    assert len(negative_groups) == len(set(negative_groups))
    assert min(negative_groups) > max(positive_groups)
