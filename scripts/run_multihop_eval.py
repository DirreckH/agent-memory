"""多跳链式召回测量门：评估当前系统在桥接实体问题上的证据链完整性。

协议事实：AML 由平台回答模型串联多跳推理，参赛系统只负责返回证据。
因此本实验的核心指标是“全链召回率”（top-k 中同时包含每一跳证据的
比例），而非答案正确率。

两种查询条件对比（共享同一记忆库，只差查询表达层）：
- pure：裸查询（无扩展），即当前系统的检索下界；
- oracle_expansion：附上仅由查询可推导的理想扩展文本，预演
  “查询表达层做到完美”能带来的增益，为 Tier 1/2 决策提供依据。

干扰记忆（fillers）由模板库按 case 种子确定性生成，模拟真实规模的
历史（默认每 user 180 条），使 top-100 的选择具有排序压力。

判定阈值：
- chain@100 >= 85%        -> 证据链基本完整，暂缓 Tier 1/2；
- pure 低而 oracle 高     -> 病根在查询表达层，做轻量扩展优化即可；
- 两者都低                -> 桥接实体排名不足，考虑子查询扇出 / 实体层。
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

from app.config import Settings
from app.embeddings import FastEmbedder
from app.llm import MemoryLLM, NoOpMemoryLLM, QueryExpansion
from app.schemas import MemoryMessage
from app.service import MemoryService, MemoryServiceUnavailable
from app.storage import SQLiteMemoryStore
from scripts.benchmark_common import (
    BenchmarkMemory,
    MultihopBenchmarkCase,
    load_multihop_cases,
    rank_of_keyword,
)

DAY_MS = 86_400_000
ADD_BATCH = 100
THRESHOLDS = (10, 20, 100)

# ---------------------------- 干扰记忆模板库（与任何用例的跳关键词无交集）


def _bank(language: str) -> tuple[dict[str, list[str]], list[str]]:
    if language == "zh":
        pools: dict[str, list[str]] = {
            "project": [
                "报表中心", "登录重构", "缓存层", "工单系统",
                "搜索服务", "官网改版", "计费模块", "推送管道",
            ],
            "person": ["小赵", "阿磊", "周姐", "老范", "苏苏", "大熊"],
            "food": ["湘菜", "日料", "泰餐", "烧烤", "粤菜", "火锅"],
            "sport": ["羽毛球", "夜跑", "游泳", "骑行", "攀岩", "飞盘"],
            "topic": ["季度 OKR", "预算方案", "招聘计划", "数据迁移", "排期", "故障复盘"],
        }
        templates = [
            "这周主要在推进{project}，联调占用了很多时间。",
            "和{person}对了一下{topic}，结论下周同步给全组。",
            "周末去吃了{food}，排队四十分钟但值得。",
            "最近在坚持{sport}，状态比上个月好多了。",
            "{project}的灰度放量到一半，监控没有报错。",
            "帮{person}看了一个线上问题，是配置写反了。",
            "家里换了新的路由器，是{person}推荐的那款。",
            "看完了一部讲{topic}的纪录片，节奏有点慢。",
            "把阳台的旧{sport}装备挂上了二手平台。",
            "公司楼下的咖啡车换了个配方，{person}说好喝多了。",
            "{person}请了年假去爬山，组里安静了一周。",
            "给爸妈订了体检套餐，加项里有颈动脉彩超。",
            "小区的快递柜扩容了，取件不再排队。",
            "整理了一波浏览器书签，删掉了{topic}相关的死链。",
            "地铁新线开通后通勤少了二十分钟。",
            "把{sport}的护具清单补齐了，最贵的是护腕。",
            "试了新的笔记软件，用来记录{topic}很顺手。",
            "楼下便利店进了新的饭团口味，海苔味不错。",
            "把常穿的几双鞋送去清洗，价格又涨了。",
            "{topic}的文档终于补完了，附了三张架构图。",
            "和朋友拼单买了{food}的年卡，打算每月打卡一次。",
            "重读了大学时的{topic}笔记，原来当时没看懂的是目录。",
            "{project}的评审会改到周五，和体检撞期了。",
            "{person}养的绿萝放在工位，长势压过所有人。",
        ]
    else:
        pools = {
            "project": [
                "billing service", "auth refactor", "search stack",
                "landing page", "notification pipeline", "data export job",
            ],
            "person": ["Nadia", "Tom", "Marcus", "Elena", "Jules", "Priya"],
            "food": ["ramen", "tacos", "Thai curry", "brunch", "dumplings", "BBQ"],
            "sport": ["bouldering", "cycling", "swimming", "chess", "sketching", "trail running"],
            "topic": [
                "the quarterly plan", "the migration doc", "the hiring loop",
                "the on-call rota", "the design review", "the retro notes",
            ],
        }
        templates = [
            "Spent most of the week unblocking the {project}; mostly config noise.",
            "Synced with {person} about {topic}; decisions land next sprint.",
            "Tried the new {food} place with the team; worth the wait.",
            "Keeping up with {sport} twice a week; shoulders feel fine now.",
            "The {project} rollout hit fifty percent traffic with no alerts.",
            "Helped {person} debug a weird prod issue; it was a stale env var.",
            "Swapped the home router for the model {person} recommended.",
            "Finished a documentary about {topic}; slower than expected.",
            "Listed the old {sport} gear on the marketplace this weekend.",
            "The office coffee cart changed the recipe; {person} approves.",
            "{person} is on leave hiking; the standups are quieter.",
            "Booked my parents' health checkup; added the carotid ultrasound.",
            "The parcel lockers got expanded; no more Saturday queue.",
            "Cleaned up browser bookmarks; half of {topic} links were dead.",
            "The new metro line saves me twenty minutes each way.",
            "Completed the {sport} gear checklist; the chalk was overpriced.",
            "Tried a new notes app for tracking {topic}; backlinks are handy.",
            "The convenience store restocked seaweed rice balls; solid.",
            "Took my running shoes in for a wash; prices went up.",
            "Finally wrote up {topic}; three architecture diagrams attached.",
            "Split a yearly {food} pass with a friend; monthly visits planned.",
            "Reread college notes on {topic}; the outline finally makes sense.",
            "The {project} review moved to Friday and clashed with my checkup.",
            "{person} keeps a pothos at the desk; it outgrows everyone else's.",
        ]
    return pools, templates


def generate_fillers(
    case: MultihopBenchmarkCase, count: int | None = None
) -> list[BenchmarkMemory]:
    """按 case 种子确定性生成与跳关键词无冲突的干扰记忆。"""

    wanted = case.filler_count if count is None else count
    pools, templates = _bank(case.language)
    rng = random.Random(f"mh:{case.case_id}:{case.filler_seed}")
    forbidden = [hop.keyword.casefold() for hop in case.hops]
    seen: set[str] = set()
    fillers: list[BenchmarkMemory] = []
    attempts = 0
    index = 0
    while len(fillers) < wanted and attempts < wanted * 6:
        attempts += 1
        template = templates[index % len(templates)]
        index += 1
        choices = {name: rng.choice(values) for name, values in pools.items()}
        content = template.format(**choices)
        if any(word in content.casefold() for word in forbidden):
            continue
        if content.casefold() in seen:
            continue
        seen.add(content.casefold())
        fillers.append(
            BenchmarkMemory(content=content, days_before_now=rng.randint(1, 400))
        )
    return fillers


class OracleExpansionLLM:
    """按数据集内置的理想扩展文本充当抽取器，隔离查询表达层与检索层。"""

    enabled = True

    def __init__(self, expansions: dict[str, str]) -> None:
        self._expansions = expansions

    def enrich_messages(self, messages: list[dict[str, object]]) -> dict[int, str]:
        return {}

    def expand_query(self, query: str, options: list[str] | None) -> QueryExpansion:
        return QueryExpansion(
            text=self._expansions.get(query, ""), temporal=None
        )


# ---------------------------- 评测管道


def _add_case(
    writer: MemoryService,
    case: MultihopBenchmarkCase,
    memories: list[BenchmarkMemory],
    *,
    now_ms: int,
) -> None:
    messages = [
        MemoryMessage(
            role="user",
            content=memory.content,
            timestamp=now_ms - memory.days_before_now * DAY_MS,
        )
        for memory in memories
    ]
    for chunk_index, start in enumerate(range(0, len(messages), ADD_BATCH)):
        writer.add(
            request_id=f"mh:{case.case_id}:{chunk_index}",
            messages=messages[start : start + ADD_BATCH],
            user_id=f"mh:{case.case_id}",
            session_id=f"session:{case.case_id}",
        )


def evaluate_hits(
    contents: list[str], case: MultihopBenchmarkCase
) -> dict[str, object]:
    """对一次检索结果计算每跳排名与全链召回。"""
    hop_ranks = [
        rank_of_keyword(contents, hop.keyword) for hop in case.hops
    ]
    chain: dict[str, bool] = {}
    for k in THRESHOLDS:
        chain[str(k)] = all(rank is not None and rank <= k for rank in hop_ranks)
    complete_ranks = [rank for rank in hop_ranks if rank is not None]
    return {
        "hop_ranks": hop_ranks,
        "chain_at": chain,
        "bottleneck_rank": max(complete_ranks) if complete_ranks else None,
        "pool_size": len(contents),
    }


def run_multihop(
    services: dict[str, MemoryService],
    cases: list[MultihopBenchmarkCase],
    *,
    now_ms: int,
    top_k: int = 100,
    filler_count: int | None = None,
) -> dict[str, object]:
    """所有模式共享同一记忆库（写入一次），只比较查询表达层。"""
    writer = next(iter(services.values()))
    per_mode: dict[str, list[dict[str, object]]] = {mode: [] for mode in services}

    for case in cases:
        memories = list(case.memories) + generate_fillers(case, filler_count)
        _add_case(writer, case, memories, now_ms=now_ms)
        for mode, service in services.items():
            item: dict[str, object] = {"case_id": case.case_id, "hops": len(case.hops)}
            try:
                hits = service.search(
                    query=case.query, user_id=f"mh:{case.case_id}", top_k=top_k
                )
                item.update(
                    evaluate_hits([hit.content for hit in hits], case)
                )
            except (MemoryServiceUnavailable, ValueError) as exc:
                item["error"] = str(exc)
            per_mode[mode].append(item)

    return {
        "top_k": top_k,
        "modes": {
            mode: {"summary": summarize(items), "per_case": items}
            for mode, items in per_mode.items()
        },
    }


def summarize(per_case: list[dict[str, object]]) -> dict[str, object]:
    valid = [item for item in per_case if item.get("error") is None]
    total = len(valid)
    if total == 0:
        return {"cases": 0, "errors": len(per_case)}

    chain = {
        f"chain_at_{k}": round(
            sum(
                1
                for item in valid
                if item["chain_at"][str(k)]
            )
            / total,
            4,
        )
        for k in THRESHOLDS
    }
    # 按跳位置统计召回，定位是桥接跳（hop1）还是答案跳先断链。
    hop_position: dict[str, float] = {}
    max_hops = max(len(item["hop_ranks"]) for item in valid)
    for position in range(max_hops):
        covered = sum(
            1 for item in valid if len(item["hop_ranks"]) > position
        )
        found = sum(
            1
            for item in valid
            if len(item["hop_ranks"]) > position
            and item["hop_ranks"][position] is not None
        )
        if covered:
            hop_position[f"hop{position + 1}_recall"] = round(found / covered, 4)
    bottlenecks = [
        item["bottleneck_rank"]
        for item in valid
        if item.get("bottleneck_rank") is not None
    ]
    return {
        "cases": total,
        "errors": len(per_case) - total,
        **chain,
        "hop_position_recall": hop_position,
        "mean_bottleneck_rank": (
            round(sum(bottlenecks) / len(bottlenecks), 2) if bottlenecks else None
        ),
        "mean_pool_size": round(
            sum(int(item["pool_size"]) for item in valid) / total, 2
        ),
    }


def print_summary(report: dict[str, object]) -> None:
    print(f"\n{'mode':<18}{'chain@10':>10}{'chain@20':>10}{'chain@100':>11}"
          f"{'pool':>8}{'errors':>8}")
    for mode, payload in report["modes"].items():
        summary = payload["summary"]
        print(
            f"{mode:<18}"
            f"{summary.get('chain_at_10', 0):>10.2%}"
            f"{summary.get('chain_at_20', 0):>10.2%}"
            f"{summary.get('chain_at_100', 0):>11.2%}"
            f"{summary.get('mean_pool_size', 0):>8.1f}"
            f"{summary.get('errors', 0):>8}"
        )
    print("\n按跳位置的召回率（定位断链位置）：")
    for mode, payload in report["modes"].items():
        positions = payload["summary"].get("hop_position_recall", {})
        readable = ", ".join(f"{key}={value:.2f}" for key, value in positions.items())
        print(f"  {mode:<16}{readable}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="多跳链式召回测量门（pure vs oracle_expansion）"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/fixtures/multihop_cases.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--report", type=Path, default=Path("data/multihop_report.json")
    )
    args = parser.parse_args()

    cases = load_multihop_cases(args.dataset)
    settings = Settings()  # 读取 .env 的向量配置；LLM 由本脚本注入
    embedder = FastEmbedder(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        threads=settings.embedding_threads,
    )
    database_path = (
        Path(tempfile.mkdtemp(prefix="multihop-eval-")) / "eval.db"
    )
    store = SQLiteMemoryStore(database_path)

    expansions = {case.query: case.oracle_expansion for case in cases}
    services: dict[str, MemoryService] = {}
    for mode, llm in (
        ("pure", NoOpMemoryLLM()),
        ("oracle_expansion", OracleExpansionLLM(expansions)),
    ):
        services[mode] = MemoryService(
            Settings(database_path=database_path), store, embedder, llm
        )
    services["pure"].initialize()

    now_ms = int(time.time() * 1000)
    report = run_multihop(
        services, cases, now_ms=now_ms, top_k=args.top_k
    )
    report["dataset"] = str(args.dataset)
    report["embedding_model"] = settings.embedding_model
    report["decision_thresholds"] = {
        "hold_if_chain_at_100_gte": 0.85,
        "expression_layer_if_oracle_much_higher": 0.10,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(report)
    print(f"\nREPORT={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
