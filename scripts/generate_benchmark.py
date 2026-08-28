from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Literal

from openai import OpenAI

from app.config import Settings
from scripts.benchmark_common import BenchmarkCase, write_cases


BatchGenerator = Callable[
    [Literal["positive", "negative"], int, list[str]],
    list[dict[str, object]],
]


def generate_cases(
    batch_generator: BatchGenerator,
    *,
    count: int = 200,
    negative_count: int = 20,
    batch_size: int = 20,
    group_size: int = 20,
    max_attempts: int = 40,
    show_progress: bool = True,
) -> list[BenchmarkCase]:
    """分批生成并严格筛选样例，直到得到准确数量的数据。"""

    if count < 1:
        raise ValueError("count 必须大于 0")
    if not 0 <= negative_count < count:
        raise ValueError("negative_count 必须满足 0 <= negative_count < count")
    if batch_size < 1 or group_size < 1 or max_attempts < 1:
        raise ValueError("batch_size、group_size、max_attempts 必须大于 0")

    targets = {
        "positive": count - negative_count,
        "negative": negative_count,
    }
    accepted: dict[str, list[BenchmarkCase]] = {"positive": [], "negative": []}
    seen_memories: set[str] = set()
    seen_queries: set[str] = set()
    seen_keywords: set[str] = set()
    attempts = 0
    last_error = ""

    for kind in ("positive", "negative"):
        while len(accepted[kind]) < targets[kind]:
            if attempts >= max_attempts:
                detail = f"；最后错误: {last_error}" if last_error else ""
                raise RuntimeError(
                    f"生成尝试达到上限，{kind} 仅得到 "
                    f"{len(accepted[kind])}/{targets[kind]} 条{detail}"
                )
            attempts += 1
            needed = min(batch_size, targets[kind] - len(accepted[kind]))
            try:
                raw_cases = batch_generator(
                    kind,
                    needed,
                    sorted(seen_keywords)[-100:],
                )
            except Exception as exc:  # noqa: BLE001 - 允许上游偶发失败后继续补批次
                last_error = type(exc).__name__
                if show_progress:
                    print(f"[generate] attempt={attempts} error={last_error}")
                continue

            for raw_case in raw_cases:
                if len(accepted[kind]) >= targets[kind]:
                    break
                try:
                    candidate = BenchmarkCase.model_validate(
                        {
                            **raw_case,
                            "case_id": "pending",
                            "user_group": 0,
                        }
                    )
                except ValueError:
                    continue
                if candidate.kind != kind:
                    continue
                memory_key = candidate.memory_text.casefold()
                query_key = candidate.query.casefold()
                keyword_key = candidate.expected_keyword.casefold()
                if memory_key in seen_memories or query_key in seen_queries:
                    continue
                if keyword_key and keyword_key in seen_keywords:
                    continue
                seen_memories.add(memory_key)
                seen_queries.add(query_key)
                if keyword_key:
                    seen_keywords.add(keyword_key)
                accepted[kind].append(candidate)

            if show_progress:
                print(
                    f"[generate:{kind}] "
                    f"{len(accepted[kind])}/{targets[kind]}"
                )

    positive_group_count = math.ceil(targets["positive"] / group_size)
    output: list[BenchmarkCase] = []
    ordered = accepted["positive"] + accepted["negative"]
    for index, candidate in enumerate(ordered):
        if candidate.kind == "positive":
            user_group = index // group_size
        else:
            # 每个负例单独隔离：只衡量无关 query 对一条无关记忆是否误召回。
            negative_index = index - targets["positive"]
            user_group = positive_group_count + negative_index
        output.append(
            candidate.model_copy(
                update={
                    "case_id": f"case-{index + 1:04d}",
                    "user_group": user_group,
                }
            )
        )
    return output


class DeepSeekBatchGenerator:
    """使用 DeepSeek OpenAI 兼容接口生成不包含真实个人信息的合成样例。"""

    def __init__(self, settings: Settings) -> None:
        key = (
            settings.deepseek_api_key.get_secret_value().strip()
            if settings.deepseek_api_key
            else ""
        )
        if not key:
            raise ValueError("请先在 .env 设置 DEEPSEEK_API_KEY")
        self.model = settings.deepseek_model
        self.client = OpenAI(
            api_key=key,
            base_url=settings.deepseek_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def __call__(
        self,
        kind: Literal["positive", "negative"],
        count: int,
        avoid_keywords: list[str],
    ) -> list[dict[str, object]]:
        if kind == "positive":
            rules = (
                "每条 memory_text 只陈述一个清晰且可检索的虚构事实；query 必须用不同措辞询问该事实；"
                "expected_keyword 必须是答案中的独特短语，逐字出现在 memory_text 中，但绝不能出现在 query 中。"
                "答案应有区分度，避免只用‘是/否’、普通日期或过短通用词。"
            )
        else:
            rules = (
                "memory_text 与 query 必须主题完全无关，expected_keyword 必须为空字符串；"
                "query 不应能从 memory_text 得到任何答案。"
            )

        system_prompt = (
            "你是 Agent Memory 检索系统的合成测试数据生成器。只生成虚构、无敏感信息的数据，"
            "不得使用真实个人资料、评测集内容或现有基准答案。必须输出一个 JSON 对象，顶层只有 cases 数组。"
            "每个元素只能包含 kind、category、language、memory_text、query、expected_keyword。"
            "language 只能是 zh 或 en，整体约 60% 中文、40% 英文；category 应覆盖偏好、计划、地点、人物关系、"
            "数字、工作事项、状态变更等。" + rules
        )
        payload = {
            "kind": kind,
            "count": count,
            "avoid_expected_keywords": avoid_keywords,
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "请严格按要求生成以下 JSON 请求：\n"
                    + json.dumps(payload, ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=max(1024, min(8192, count * 320)),
            stream=False,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("DeepSeek 返回空内容")
        parsed = json.loads(content)
        cases = parsed.get("cases") if isinstance(parsed, dict) else None
        if not isinstance(cases, list):
            raise ValueError("DeepSeek 输出缺少 cases 数组")
        return [item for item in cases if isinstance(item, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用 DeepSeek 分批生成 Agent Memory 合成 JSONL 测试集"
    )
    parser.add_argument("--output", type=Path, default=Path("data/benchmark_200.jsonl"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--negative-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=40)
    args = parser.parse_args()

    settings = Settings()
    generator = DeepSeekBatchGenerator(settings)
    cases = generate_cases(
        generator,
        count=args.count,
        negative_count=args.negative_count,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_attempts=args.max_attempts,
    )
    write_cases(args.output, cases)
    positives = sum(case.kind == "positive" for case in cases)
    negatives = sum(case.kind == "negative" for case in cases)
    print(f"DATASET={args.output.resolve()}")
    print(f"MODEL={settings.deepseek_model}")
    print(f"CASES={len(cases)} POSITIVE={positives} NEGATIVE={negatives}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
