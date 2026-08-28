from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from app.config import Settings
from scripts.benchmark_common import BenchmarkCase, load_cases


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 2)


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": round(max(values), 2) if values else 0.0,
    }


def _user_id(run_id: str, case: BenchmarkCase) -> str:
    return f"benchmark:{run_id}:group-{case.user_group:03d}"


def _run_parallel(
    cases: list[BenchmarkCase],
    operation: Callable[[BenchmarkCase], dict[str, object]],
    *,
    concurrency: int,
    phase: str,
    show_progress: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    if concurrency == 1:
        for index, case in enumerate(cases, start=1):
            results.append(operation(case))
            if show_progress and (index % 10 == 0 or index == len(cases)):
                print(f"[{phase}] {index}/{len(cases)}")
        return results

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(operation, case): case for case in cases}
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if show_progress and (index % 10 == 0 or index == len(cases)):
                print(f"[{phase}] {index}/{len(cases)}")
    return results


def run_benchmark(
    cases: list[BenchmarkCase],
    *,
    base_url: str,
    memory_api_key: str,
    run_id: str,
    top_k: int,
    timeout_seconds: float,
    concurrency: int,
    show_progress: bool = True,
) -> dict[str, object]:
    """只通过公开 /set、/get 执行批量评测，不读取内部数据库。"""

    if not cases:
        raise ValueError("cases 不能为空")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k 必须在 1..100")
    if not 1 <= concurrency <= 32:
        raise ValueError("concurrency 必须在 1..32")

    api_root = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if memory_api_key:
        headers["Authorization"] = f"Bearer {memory_api_key}"

    failures: list[dict[str, object]] = []
    limits = httpx.Limits(
        max_connections=max(4, concurrency * 2),
        max_keepalive_connections=max(2, concurrency),
    )
    with httpx.Client(headers=headers, timeout=timeout_seconds, limits=limits) as client:

        def add_one(case: BenchmarkCase) -> dict[str, object]:
            started = time.perf_counter()
            try:
                response = client.post(
                    f"{api_root}/set",
                    json={
                        "request_id": f"benchmark:{run_id}:{case.case_id}",
                        "messages": [{"role": "user", "content": case.memory_text}],
                        "user_id": _user_id(run_id, case),
                        "session_id": (
                            f"benchmark:{run_id}:session-{case.user_group:03d}"
                        ),
                    },
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("success") is not True:
                    raise ValueError("/set 响应缺少 success=true")
                return {
                    "case_id": case.case_id,
                    "ok": True,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            except Exception as exc:  # noqa: BLE001 - 逐条记录，继续完成整批测试
                status = exc.response.status_code if isinstance(
                    exc, httpx.HTTPStatusError
                ) else None
                return {
                    "case_id": case.case_id,
                    "ok": False,
                    "status": status,
                    "error": type(exc).__name__,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }

        add_results = _run_parallel(
            cases,
            add_one,
            concurrency=concurrency,
            phase="set",
            show_progress=show_progress,
        )

        def search_one(case: BenchmarkCase) -> dict[str, object]:
            started = time.perf_counter()
            try:
                response = client.post(
                    f"{api_root}/get",
                    json={
                        "query": case.query,
                        "user_id": _user_id(run_id, case),
                        "top_k": top_k,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, list):
                    raise ValueError("/get 响应缺少 data 数组")
                contents = [
                    item.get("content", "")
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("content"), str)
                ]
                rank = None
                if case.kind == "positive":
                    expected = case.expected_keyword.casefold()
                    rank = next(
                        (
                            index
                            for index, content in enumerate(contents, start=1)
                            if expected in content.casefold()
                        ),
                        None,
                    )
                return {
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "ok": True,
                    "result_count": len(contents),
                    "rank": rank,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            except Exception as exc:  # noqa: BLE001 - 逐条记录，继续完成整批测试
                status = exc.response.status_code if isinstance(
                    exc, httpx.HTTPStatusError
                ) else None
                return {
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "ok": False,
                    "status": status,
                    "error": type(exc).__name__,
                    "result_count": 0,
                    "rank": None,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }

        search_results = _run_parallel(
            cases,
            search_one,
            concurrency=concurrency,
            phase="get",
            show_progress=show_progress,
        )

    for phase, phase_results in (("set", add_results), ("get", search_results)):
        for result in phase_results:
            if not result["ok"]:
                failures.append(
                    {
                        "phase": phase,
                        "case_id": result["case_id"],
                        "status": result.get("status"),
                        "error": result.get("error"),
                    }
                )

    positives = [item for item in search_results if item["kind"] == "positive"]
    negatives = [item for item in search_results if item["kind"] == "negative"]
    positive_count = len(positives)
    negative_count = len(negatives)
    top1_hits = sum(item["ok"] and item["rank"] == 1 for item in positives)
    topk_hits = sum(
        item["ok"]
        and isinstance(item["rank"], int)
        and int(item["rank"]) <= top_k
        for item in positives
    )
    reciprocal_rank = sum(
        1.0 / int(item["rank"])
        for item in positives
        if item["ok"]
        and isinstance(item["rank"], int)
        and int(item["rank"]) <= top_k
    )
    negative_false_positives = sum(
        (not item["ok"]) or int(item["result_count"]) > 0 for item in negatives
    )

    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": api_root,
        "case_count": len(cases),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "top_k": top_k,
        "concurrency": concurrency,
        "add": {
            "attempted": len(add_results),
            "succeeded": sum(bool(item["ok"]) for item in add_results),
            "failed": sum(not bool(item["ok"]) for item in add_results),
            "latency_ms": _latency_summary(
                [float(item["latency_ms"]) for item in add_results]
            ),
        },
        "search": {
            "attempted": len(search_results),
            "succeeded": sum(bool(item["ok"]) for item in search_results),
            "failed": sum(not bool(item["ok"]) for item in search_results),
            "latency_ms": _latency_summary(
                [float(item["latency_ms"]) for item in search_results]
            ),
        },
        "quality": {
            "recall_at_1": round(top1_hits / positive_count, 4)
            if positive_count
            else 0.0,
            f"recall_at_{top_k}": round(topk_hits / positive_count, 4)
            if positive_count
            else 0.0,
            f"mrr_at_{top_k}": round(reciprocal_rank / positive_count, 4)
            if positive_count
            else 0.0,
            "negative_false_positive_rate": round(
                negative_false_positives / negative_count, 4
            )
            if negative_count
            else 0.0,
        },
        "failures": failures,
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="通过公开 /set、/get 对 Agent Memory 服务执行批量评测"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmark_200.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--report", type=Path, default=Path("data/benchmark_report.json"))
    parser.add_argument("--run-id", default=f"run-{uuid.uuid4().hex[:10]}")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    args = parser.parse_args()

    settings = Settings()
    memory_api_key = os.getenv("BENCHMARK_MEMORY_API_KEY", "").strip()
    if not memory_api_key:
        memory_api_key = settings.expected_memory_api_key

    cases = load_cases(args.dataset)
    report = run_benchmark(
        cases,
        base_url=args.base_url,
        memory_api_key=memory_api_key,
        run_id=args.run_id,
        top_k=args.top_k,
        timeout_seconds=args.timeout,
        concurrency=args.concurrency,
    )
    _write_report(args.report, report)

    quality = report["quality"]
    recall_key = f"recall_at_{args.top_k}"
    passed = (
        report["add"]["failed"] == 0
        and report["search"]["failed"] == 0
        and quality[recall_key] >= args.min_recall
        and quality["negative_false_positive_rate"] <= args.max_negative_fpr
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"BENCHMARK_STATUS={'PASS' if passed else 'FAIL'}")
    print(f"REPORT={args.report.resolve()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
