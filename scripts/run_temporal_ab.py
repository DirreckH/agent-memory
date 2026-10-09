"""时间感知检索 A/B 实验。

同一份时间歧义数据集在 TEMPORAL_MODE=off 与 soft 两个服务实例上各跑一遍，
对比目标记忆的排名指标（Hit@1、Hit@K、MRR、目标压过分心项比例）。
实验为进程内运行：真实 FastEmbed 向量 + MemoryService 全链路，不依赖 HTTP。

抽取层两种模式：
- oracle（默认）：时间约束直接取自数据集内置的期望值，衡量“约束抽取
  正确的前提下，时间打分带来的检索增益”，即打分层上界；
- real：使用生产规则和 .env 配置的真实 LLM 抽取，记录实际采用的约束。
  现有一致率仅比较类型签名，不代表字段级正确率。
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from app.config import Settings
from app.embeddings import Embedder, FastEmbedder
from app.llm import MemoryLLM, QueryExpansion, build_memory_llm
from app.schemas import MemoryMessage
from app.service import MemoryService, MemoryServiceUnavailable
from app.storage import SQLiteMemoryStore
from app.temporal import TemporalConstraints, parse_temporal
from scripts.benchmark_common import (
    TemporalBenchmarkCase,
    load_temporal_cases,
    rank_of_keyword,
)

MODES = ("off", "soft")
DAY_MS = 86_400_000
# 固定测量基准时间：记录内容渲染包含时间戳（[user | 时间]）且会参与向量化，
# 用墙钟会让每次运行的嵌入漂移、近平局用例随机翻转。钉死常量保证可复现。
MEASUREMENT_NOW_MS = 1_800_000_000_000

# query_type 到期望时间约束签名的映射，用于抽取层一致率诊断。
_EXPECTED_SIGNATURE = {
    "rolling": "window",
    "event_before": "before",
    "event_after": "after",
    "ordering_earliest": "earliest",
    "ordering_latest": "latest",
}


class OracleMemoryLLM:
    """以数据集内置期望约束充当抽取器，隔离打分层与抽取层。"""

    enabled = True

    def __init__(self, expansions: dict[str, QueryExpansion]) -> None:
        self._expansions = expansions

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        return {}

    def expand_query(self, query: str, options: list[str] | None) -> QueryExpansion:
        return self._expansions.get(query, QueryExpansion(text="", temporal=None))

    def extract_temporal(self, query: str) -> TemporalConstraints | None:
        return self._expansions.get(query, QueryExpansion(text="", temporal=None)).temporal


class BenchmarkMemoryService(MemoryService):
    """记录实际采用的时间约束；oracle 模式固定约束以隔离抽取层变化。"""

    def __init__(
        self, settings: Settings, store: SQLiteMemoryStore,
        embedder: Embedder, llm: MemoryLLM,
    ) -> None:
        super().__init__(settings, store, embedder, llm)
        self.extractions: dict[str, QueryExpansion] = {}

    def _query_temporal(self, query: str) -> TemporalConstraints | None:
        if isinstance(self.llm, OracleMemoryLLM):
            temporal = (
                self.llm.extract_temporal(query)
                if self.settings.temporal_mode != "off" else None
            )
        else:
            temporal = super()._query_temporal(query)
        self.extractions[query] = QueryExpansion("", temporal)
        return temporal


def build_oracle_expansions(
    cases: list[TemporalBenchmarkCase],
) -> dict[str, QueryExpansion]:
    return {
        case.query: QueryExpansion(text="", temporal=parse_temporal(case.oracle))
        for case in cases
    }


def constraint_signature(temporal: TemporalConstraints | None) -> str:
    """把解析出的时间约束映射为可比较的类型签名。"""
    if temporal is None:
        return ""
    if temporal.event_anchor is not None:
        return temporal.event_anchor.direction
    if temporal.relative_window is not None:
        return "window"
    return temporal.ordering or ""


def _metrics(items: list[dict[str, object]]) -> dict[str, object]:
    total = len(items)
    if total == 0:
        return {"cases": 0, "hit_at_1": 0.0, "hit_at_k": 0.0, "mrr_at_k": 0.0, "target_above_distractor": 0.0}
    hit1 = sum(1 for item in items if item.get("target_rank") == 1)
    hitk = sum(1 for item in items if item.get("target_rank") is not None)
    mrr = sum(
        1.0 / float(item["target_rank"])
        for item in items
        if item.get("target_rank") is not None
    )
    above = sum(
        1
        for item in items
        if item.get("target_rank") is not None
        and (
            item.get("distractor_rank") is None
            or item["target_rank"] < item["distractor_rank"]
        )
    )
    return {
        "cases": total,
        "hit_at_1": round(hit1 / total, 4),
        "hit_at_k": round(hitk / total, 4),
        "mrr_at_k": round(mrr / total, 4),
        "target_above_distractor": round(above / total, 4),
    }


def summarize(per_case: list[dict[str, object]]) -> dict[str, object]:
    by_type: dict[str, list[dict[str, object]]] = {}
    for item in per_case:
        if item.get("error") is None:
            by_type.setdefault(str(item["query_type"]), []).append(item)
    return {
        **_metrics([item for item in per_case if item.get("error") is None]),
        "errors": sum(1 for item in per_case if item.get("error") is not None),
        "by_query_type": {key: _metrics(items) for key, items in sorted(by_type.items())},
    }


def run_mode(
    service: MemoryService,
    cases: list[TemporalBenchmarkCase],
    *,
    now_ms: int,
    top_k: int,
    mode_label: str,
) -> dict[str, object]:
    per_case: list[dict[str, object]] = []
    for case in cases:
        user_id = f"ab-{mode_label}:{case.case_id}"
        item: dict[str, object] = {
            "case_id": case.case_id,
            "query_type": case.query_type,
            "language": case.language,
        }
        try:
            service.add(
                request_id=f"ab-{mode_label}:{case.case_id}",
                messages=[
                    MemoryMessage(
                        role="user",
                        content=memory.content,
                        timestamp=now_ms - memory.days_before_now * DAY_MS,
                    )
                    for memory in case.memories
                ],
                user_id=user_id,
                session_id=f"session:{case.case_id}",
            )
            hits = service.search(query=case.query, user_id=user_id, top_k=top_k)
            contents = [hit.content for hit in hits]
            item["target_rank"] = rank_of_keyword(contents, case.expected_keyword)
            item["distractor_rank"] = rank_of_keyword(
                contents, case.distractor_keyword
            )
        except (MemoryServiceUnavailable, ValueError) as exc:
            item["error"] = str(exc)
        per_case.append(item)
    return {"summary": summarize(per_case), "per_case": per_case}


def run_ab(
    services: dict[str, MemoryService],
    cases: list[TemporalBenchmarkCase],
    *,
    now_ms: int,
    top_k: int,
) -> dict[str, object]:
    return {
        "top_k": top_k,
        "now_ms": now_ms,
        "modes": {
            mode: run_mode(
                service, cases, now_ms=now_ms, top_k=top_k, mode_label=mode
            )
            for mode, service in services.items()
        },
    }


def extraction_report(
    cases: list[TemporalBenchmarkCase],
    recorded: dict[str, QueryExpansion],
) -> dict[str, object]:
    per_case: list[dict[str, object]] = []
    matched = 0
    for case in cases:
        got = recorded.get(case.query)
        signature = constraint_signature(got.temporal) if got is not None else "<missing>"
        expected = _EXPECTED_SIGNATURE[case.query_type]
        ok = signature == expected
        matched += int(ok)
        per_case.append(
            {
                "case_id": case.case_id,
                "query_type": case.query_type,
                "expected": expected,
                "got": signature,
                "match": ok,
            }
        )
    return {
        "agreement_rate": round(matched / len(cases), 4),
        "per_case": per_case,
    }


def print_summary(report: dict[str, object]) -> None:
    modes = report["modes"]
    print(f"\n{'mode':<8}{'hit@1':>8}{'hit@k':>8}{'mrr@k':>8}{'above':>8}{'errors':>8}")
    for mode in MODES:
        summary = modes[mode]["summary"]
        print(
            f"{mode:<8}"
            f"{summary['hit_at_1']:>8.2%}"
            f"{summary['hit_at_k']:>8.2%}"
            f"{summary['mrr_at_k']:>8.4f}"
            f"{summary['target_above_distractor']:>8.2%}"
            f"{summary['errors']:>8}"
        )
    extraction = report.get("extraction")
    if isinstance(extraction, dict) and "agreement_rate" in extraction:
        print(f"\n抽取层一致率: {extraction['agreement_rate']:.2%}")
    elif isinstance(extraction, dict):
        print(f"\n抽取层: {extraction.get('note', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="时间感知检索 A/B 实验（off vs soft）"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/fixtures/temporal_ab_cases.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm", choices=["oracle", "real"], default="oracle")
    parser.add_argument("--governance-mode", choices=["off", "shadow"], default="off")
    parser.add_argument(
        "--report", type=Path, default=Path("data/temporal_ab_report.json")
    )
    args = parser.parse_args()

    cases = load_temporal_cases(args.dataset)
    settings = Settings(governance_mode=args.governance_mode)  # 读取 .env 的 LLM/向量配置
    embedder = FastEmbedder(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        threads=settings.embedding_threads,
    )

    if args.llm == "real":
        inner = build_memory_llm(settings)
        if not inner.enabled:
            print("real 模式需要 .env 配置可用的 LLM_PROVIDER 与密钥")
            return 1
        llm: MemoryLLM = inner
    else:
        llm = OracleMemoryLLM(build_oracle_expansions(cases))

    services: dict[str, MemoryService] = {}
    for mode in MODES:
        mode_settings = Settings(
            governance_mode=args.governance_mode,
            temporal_mode=mode,
            llm_failure_mode="strict",
            database_path=Path(
                tempfile.mkdtemp(prefix=f"temporal-ab-{mode}-")
            )
            / "ab.db",
        )
        services[mode] = BenchmarkMemoryService(
            mode_settings,
            SQLiteMemoryStore(mode_settings.database_path),
            embedder,
            llm,
        )
        services[mode].initialize()

    now_ms = MEASUREMENT_NOW_MS
    report = run_ab(services, cases, now_ms=now_ms, top_k=args.top_k)
    if args.llm == "real":
        soft_service = services["soft"]
        assert isinstance(soft_service, BenchmarkMemoryService)
        report["extraction"] = extraction_report(cases, soft_service.extractions)
    else:
        report["extraction"] = {
            "mode": "oracle",
            "note": "抽取层由数据集内置期望值代替，本报告只衡量打分层增益。",
        }
    report["llm_mode"] = args.llm
    report["governance_mode"] = args.governance_mode
    report["embedding_model"] = settings.embedding_model

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(report)
    print(f"\nREPORT={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
