"""学术榜合规端到端自检：gpt-4o-mini（ChatAnywhere 中转）+ text-embedding-v4（DashScope）。

按比赛规则走真实通路（无桩、无 mock）：
- 结构化 /set 与 /get：竞赛 schema、响应原样回显标识、严格字段校验
- request_id 幂等、user_id 隔离、401 鉴权边界
- V3 时间约束检索 + 写入端增强（依赖真实 gpt-4o-mini 的 JSON 抽取）
- 输入日志记录且鉴权头不落盘
- 向量全程走真实 text-embedding-v4（1024 维）

缺 OPENAI_API_KEY 时：LLM 相关检查标记 BLOCKED，其余照常执行。
退出码：0=全部通过；2=存在 BLOCKED；1=存在 FAIL。

用法：PYTHONPATH=. .venv/bin/python scripts/run_compliance_e2e.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

AUTH_HEADER = {"Authorization": "Bearer e2e-local-key"}


class Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def add(self, name: str, state: str | bool, evidence: str = "") -> None:
        if isinstance(state, bool):
            state = "PASS" if state else "FAIL"
        self.results.append((name, state, evidence))
        marker = {"PASS": "PASS", "FAIL": "FAIL", "BLOCKED": "BLCK"}[state]
        suffix = f" — {evidence}" if evidence else ""
        print(f"[{marker}] {name}{suffix}")

    def summary(self) -> int:
        states = [state for _, state, _ in self.results]
        failed = states.count("FAIL")
        blocked = states.count("BLOCKED")
        print(
            f"\n=== 合规自检总结：{states.count('PASS')} passed / "
            f"{blocked} blocked / {failed} failed ==="
        )
        if failed:
            return 1
        if blocked:
            return 2
        return 0


def build_settings(report: Report) -> tuple[Settings, bool]:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    base = Settings(_env_file=str(env_path))
    llm_ready = bool(
        base.openai_api_key and base.openai_api_key.get_secret_value().strip()
    )
    tmp = tempfile.mkdtemp(prefix="aml-compliance-")
    settings = Settings(
        _env_file=str(env_path),
        # 比赛规则：学术榜向量 text-embedding-v4 + Add/Search 阶段 gpt-4o-mini。
        embedding_provider="dashscope",
        embedding_model="text-embedding-v4",
        llm_provider="openai" if llm_ready else "none",
        llm_failure_mode="fallback",
        llm_add_enrichment=True,
        llm_search_expansion=True,
        database_path=Path(tmp) / "compliance.db",
        input_log_path=Path(tmp) / "input.jsonl",
        embedding_cache_dir=Path(tmp) / "cache",
        memory_api_key="e2e-local-key",
        warmup_embedding_on_startup=False,
    )
    dashscope_ready = bool(
        settings.dashscope_api_key
        and settings.dashscope_api_key.get_secret_value().strip()
    )
    report.add(
        "学术榜配置解析",
        "PASS" if dashscope_ready else "FAIL",
        f"向量={settings.embedding_provider}:{settings.embedding_model}"
        f" @ {settings.dashscope_base_url}；LLM={settings.llm_provider}"
        f":{settings.openai_model} @ {settings.openai_base_url}",
    )
    if llm_ready:
        report.add(
            "gpt-4o-mini 通路",
            "PASS" if settings.llm_provider == "openai" else "FAIL",
            "OPENAI_API_KEY 已配置，走 ChatAnywhere 中转",
        )
    else:
        report.add(
            "gpt-4o-mini 通路",
            "BLOCKED",
            "OPENAI_API_KEY 为空——填写 .env 后重跑即可解锁 LLM 检查",
        )
    return settings, llm_ready


def main() -> int:
    report = Report()
    settings, llm_ready = build_settings(report)
    now_ms = int(time.time() * 1000)
    day_ms = 86_400_000

    app = create_app(settings)
    with TestClient(app) as client:
        # ---- 基础边界 ----
        health = client.get("/health")
        report.add(
            "/health 无需鉴权",
            health.status_code == 200 and health.json() == {"status": "ok"},
            f"status={health.status_code}",
        )
        unauth = client.post(
            "/set", json={"memory_text": "x"}, headers={}
        )
        report.add(
            "缺失鉴权被拒绝",
            unauth.status_code == 401,
            f"status={unauth.status_code}",
        )

        # ---- 结构化写入（竞赛 schema，含真实时间戳） ----
        alice_sets = [
            {
                "request_id": "e2e:alice:chunk-0",
                "messages": [
                    {
                        "role": "user",
                        "content": "我在考虑换一辆山地自行车。",
                        "timestamp": now_ms - 40 * day_ms,
                    }
                ],
                "user_id": "e2e:alice",
                "session_id": "e2e:alice:s1",
            },
            {
                "request_id": "e2e:alice:chunk-1",
                "messages": [
                    {
                        "role": "user",
                        "content": "我下个月要搬到滨江区的新公寓。",
                        "timestamp": now_ms - 7 * day_ms,
                    },
                    {
                        "role": "assistant",
                        "content": "听起来很棒，搬家需要我帮你记点什么吗？",
                        "timestamp": now_ms - 7 * day_ms,
                    },
                ],
                "user_id": "e2e:alice",
                "session_id": "e2e:alice:s1",
            },
            {
                "request_id": "e2e:alice:chunk-2",
                "messages": [
                    {
                        "role": "user",
                        "content": "我每个月的预算大概是三千元。",
                        "timestamp": now_ms - 2 * day_ms,
                    }
                ],
                "user_id": "e2e:alice",
                "session_id": "e2e:alice:s2",
            },
        ]
        bob_set = {
            "request_id": "e2e:bob:chunk-0",
            "messages": [
                {
                    "role": "user",
                    "content": "Bob 喜欢在周末爬山。",
                    "timestamp": now_ms - day_ms,
                }
            ],
            "user_id": "e2e:bob",
            "session_id": "e2e:bob:s1",
        }

        for payload in [*alice_sets, bob_set]:
            started = time.perf_counter()
            response = client.post("/set", json=payload, headers=AUTH_HEADER)
            duration = round((time.perf_counter() - started) * 1000)
            if response.status_code != 200:
                report.add(
                    f"结构化写入 {payload['request_id']}",
                    "FAIL",
                    f"status={response.status_code} body={response.text[:120]}",
                )
                continue
            body = response.json()
            echoed = (
                body.get("request_id") == payload["request_id"]
                and body.get("user_id") == payload["user_id"]
                and body.get("session_id") == payload["session_id"]
                and body.get("success") is True
                and set(body) == {"success", "request_id", "user_id", "session_id"}
            )
            report.add(
                f"结构化写入 {payload['request_id']}",
                "PASS" if echoed else "FAIL",
                f"回显三标识且无多余字段，{duration}ms",
            )

        # ---- 幂等 ----
        duplicate = client.post("/set", json=alice_sets[1], headers=AUTH_HEADER)
        moved_hits = client.post(
            "/get",
            json={
                "query": "搬家 滨江区 新公寓",
                "user_id": "e2e:alice",
                "top_k": 100,
            },
            headers=AUTH_HEADER,
        ).json()["data"]
        moved_rows = [h for h in moved_hits if "滨江区" in h["content"]]
        report.add(
            "request_id 幂等",
            duplicate.status_code == 200 and len(moved_rows) == 1,
            f"重复提交 status={duplicate.status_code}，滨江区记录数={len(moved_rows)}",
        )

        # ---- 严格 schema ----
        rejected = client.post(
            "/set",
            json={
                "request_id": "e2e:bad",
                "messages": [{"role": "user", "content": "x"}],
                "user_id": "e2e:alice",
                "session_id": "e2e:alice:s1",
                "unexpected_field": 1,
            },
            headers=AUTH_HEADER,
        )
        report.add(
            "未知字段被拒绝（extra=forbid）",
            rejected.status_code == 422,
            f"status={rejected.status_code}",
        )

        # ---- user_id 隔离 ----
        cross = client.post(
            "/get",
            json={
                "query": "搬家 预算 自行车",
                "user_id": "e2e:bob",
                "top_k": 100,
            },
            headers=AUTH_HEADER,
        ).json()["data"]
        # bob 只应看到自己的爬山记忆；alice 内容标记一个都不能出现
        # （内容含 [role | 时间戳] 前缀，不能做全等比较）。
        alice_markers = ("滨江区", "预算", "三千", "自行车")
        leaked = [
            h
            for h in cross
            if any(marker in h["content"] for marker in alice_markers)
        ]
        report.add(
            "user_id 隔离",
            not leaked,
            f"bob 视角检索到 {len(cross)} 条，泄漏 {len(leaked)} 条 alice 记忆",
        )

        # ---- LLM 直连探测（真实 gpt-4o-mini） ----
        if llm_ready:
            service = app.state.memory_service
            llm = service.llm
            try:
                enriched = llm.enrich_messages(
                    [{"role": "user", "content": "我下个月要搬到滨江区的新公寓。"}]
                )
                expansion = llm.expand_query("我最近一周说过什么计划？", None)
                # 时间约束是独立于扩展文本的抽取通路：
                # 简单窗口由规则层确定性负责（LLM 对其正确输出 null），
                # LLM 层只负责事件锚定表达——各测各的，不互相冒充。
                rules_temporal = service._query_temporal("我最近一周说过什么计划？")
                anchor_temporal = llm.extract_temporal("搬到上海之前，我住在哪里？")
                ok = (
                    bool(enriched)
                    and expansion.text.strip() != ""
                    and rules_temporal is not None
                    and anchor_temporal is not None
                )
                evidence = (
                    f"enrich={list(enriched.values())[0][:40]}…；"
                    f"expand={expansion.text[:40]}…；"
                    f"规则窗口={'有' if rules_temporal else '无'}；"
                    f"LLM事件锚={'有' if anchor_temporal else '无'}"
                )
                report.add(
                    "gpt-4o-mini 增强/扩展/时间抽取（直连探测）",
                    "PASS" if ok else "FAIL",
                    evidence,
                )
            except Exception as exc:  # noqa: BLE001 - 探测失败必须显式暴露
                report.add(
                    "gpt-4o-mini 增强/扩展/时间抽取（直连探测）",
                    "FAIL",
                    f"{type(exc).__name__}: {str(exc)[:120]}",
                )
        else:
            report.add(
                "gpt-4o-mini 增强/扩展/时间抽取（直连探测）",
                "BLOCKED",
                "等待 OPENAI_API_KEY",
            )

        # ---- 时间约束检索（V3：gpt-4o-mini 抽取 + 窗口打分） ----
        plan_hits = client.post(
            "/get",
            json={
                "query": "我最近一周说过什么计划？",
                "user_id": "e2e:alice",
                "top_k": 100,
            },
            headers=AUTH_HEADER,
        ).json()["data"]
        ranks = {}
        for rank, hit in enumerate(plan_hits, start=1):
            if "滨江区" in hit["content"]:
                ranks.setdefault("搬家", rank)
            if "自行车" in hit["content"]:
                ranks.setdefault("自行车", rank)
        moved_rank = ranks.get("搬家")
        bike_rank = ranks.get("自行车")
        if moved_rank is None:
            report.add("时间窗口检索（最近一周）", "FAIL", "一周内的搬家记忆未被召回")
        elif bike_rank is None or moved_rank < bike_rank:
            report.add(
                "时间窗口检索（最近一周）",
                "PASS",
                f"搬家 rank={moved_rank}，自行车 rank={bike_rank if bike_rank else '未召回（被窗口压出）'}",
            )
        else:
            report.add(
                "时间窗口检索（最近一周）", "FAIL", f"搬家 rank={moved_rank} 落后于自行车 rank={bike_rank}"
            )

        # ---- 改述检索（写入端增强 + 双通道） ----
        budget_hits = client.post(
            "/get",
            json={
                "query": "我每个月大概能花多少钱？",
                "user_id": "e2e:alice",
                "top_k": 100,
            },
            headers=AUTH_HEADER,
        ).json()["data"]
        budget_top = budget_hits[0]["content"] if budget_hits else ""
        hit = any("三千" in h["content"] for h in budget_hits[:3])
        report.add(
            "改述检索（预算）",
            "PASS" if hit else "FAIL",
            f"top-1: {budget_top[:50]}",
        )

        # ---- 响应字段严格性 ----
        expected_fields = {"id", "content", "score", "created_at"}
        fields_ok = all(set(h) == expected_fields for h in plan_hits) and all(
            set(h) == expected_fields for h in budget_hits
        )
        report.add(
            "响应字段恰为 id/content/score/created_at",
            "PASS" if fields_ok else "FAIL",
            f"抽查 {len(plan_hits) + len(budget_hits)} 条命中记录",
        )

        # ---- 输入日志 ----
        log_path = settings.input_log_path
        log_text = (
            log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        )
        lines = [line for line in log_text.splitlines() if line.strip()]
        parsed = all(_safe_json(line) for line in lines)
        no_secret = "e2e-local-key" not in log_text
        report.add(
            "输入日志记录且鉴权头不落盘",
            bool(lines) and parsed and no_secret,
            f"{len(lines)} 条 JSONL，鉴权密钥{'未' if no_secret else ''}泄漏",
        )

    return report.summary()


def _safe_json(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    sys.exit(main())
