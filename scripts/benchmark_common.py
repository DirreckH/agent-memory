from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.temporal import parse_temporal, query_mentions_temporal


class BenchmarkCase(BaseModel):
    """一条可通过公开 /set、/get 接口验证的合成样例。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(min_length=1, max_length=100)
    kind: Literal["positive", "negative"]
    user_group: int = Field(ge=0, le=9999)
    category: str = Field(min_length=1, max_length=100)
    language: Literal["zh", "en"]
    memory_text: str = Field(min_length=1, max_length=4000)
    query: str = Field(min_length=1, max_length=1000)
    expected_keyword: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def validate_expected_keyword(self) -> "BenchmarkCase":
        keyword = self.expected_keyword.strip()
        if self.kind == "positive":
            if not keyword:
                raise ValueError("positive 样例必须提供 expected_keyword")
            if keyword.casefold() not in self.memory_text.casefold():
                raise ValueError("expected_keyword 必须原样出现在 memory_text")
            if keyword.casefold() in self.query.casefold():
                raise ValueError("query 不应直接包含 expected_keyword")
        elif keyword:
            raise ValueError("negative 样例的 expected_keyword 必须为空")
        return self


def load_cases(path: Path) -> list[BenchmarkCase]:
    """从 JSONL 读取并严格校验样例，避免错误标签污染评测。"""

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
                case = BenchmarkCase.model_validate(value)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number} 样例无效: {exc}") from exc
            if case.case_id in seen_ids:
                raise ValueError(f"{path}:{line_number} case_id 重复: {case.case_id}")
            seen_ids.add(case.case_id)
            cases.append(case)

    if not cases:
        raise ValueError(f"数据集为空: {path}")
    return cases


def write_cases(path: Path, cases: list[BenchmarkCase]) -> None:
    """以 JSONL 写入数据集；输出目录可用于本地 data/，不会包含密钥。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")
    temporary.replace(path)


class BenchmarkMemory(BaseModel):
    """一条带相对时间戳的记忆；days_before_now 越大表示越久远。

    时间类与多跳类评测共用：时间类用它承载时间戳，多跳类用它承载
    证据与分心记忆在历史中的位置。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=4000)
    days_before_now: int = Field(ge=0, le=3650)


class TemporalBenchmarkCase(BaseModel):
    """时间歧义场景：同一主题、不同时间的多条记忆 + 带时间限定的查询。

    target 是唯一正确答案；distractor 是同主题但时间不匹配的记录；
    oracle 是数据集内置的“理想时间约束”，供 A/B 实验的打分层使用，
    同时作为真实 LLM 抽取质量的对照答案。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(min_length=1, max_length=100)
    language: Literal["zh", "en"]
    category: str = Field(min_length=1, max_length=100)
    query_type: Literal[
        "rolling",
        "event_before",
        "event_after",
        "ordering_earliest",
        "ordering_latest",
    ]
    query: str = Field(min_length=1, max_length=1000)
    memories: list[BenchmarkMemory] = Field(min_length=2, max_length=20)
    target_index: int = Field(ge=0)
    distractor_index: int = Field(ge=0)
    event_index: int | None = None
    expected_keyword: str = Field(min_length=1, max_length=300)
    distractor_keyword: str = Field(min_length=1, max_length=300)
    oracle: dict[str, object]

    @model_validator(mode="after")
    def validate_case(self) -> "TemporalBenchmarkCase":
        size = len(self.memories)
        if self.target_index >= size or self.distractor_index >= size:
            raise ValueError("target_index/distractor_index 超出 memories 范围")
        if self.target_index == self.distractor_index:
            raise ValueError("target 与 distractor 不能指向同一条记忆")
        if self.expected_keyword.casefold() == self.distractor_keyword.casefold():
            raise ValueError("expected_keyword 与 distractor_keyword 不能相同")
        self._ensure_unique_keyword(
            self.expected_keyword, self.target_index, "expected_keyword"
        )
        self._ensure_unique_keyword(
            self.distractor_keyword, self.distractor_index, "distractor_keyword"
        )
        if not query_mentions_temporal(self.query):
            raise ValueError("query 必须包含可回查的时间信号，否则时间打分不激活")

        parsed = parse_temporal(self.oracle)
        if parsed is None:
            raise ValueError("oracle 必须是可解析的时间约束")
        target_days = self.memories[self.target_index].days_before_now
        distractor_days = self.memories[self.distractor_index].days_before_now

        if self.query_type == "rolling":
            if target_days >= distractor_days:
                raise ValueError("rolling 用例要求 target 比 distractor 更近")
            if parsed.relative_window is None:
                raise ValueError("rolling 用例的 oracle 必须提供窗口")
        elif self.query_type in ("event_before", "event_after"):
            if (
                self.event_index is None
                or self.event_index >= size
                or self.event_index
                in (self.target_index, self.distractor_index)
            ):
                raise ValueError("event 用例必须提供独立的事件记忆索引")
            if parsed.event_anchor is None:
                raise ValueError("event 用例的 oracle 必须提供事件锚点")
            event_days = self.memories[self.event_index].days_before_now
            if self.query_type == "event_before":
                if not (target_days > event_days and distractor_days < event_days):
                    raise ValueError(
                        "event_before 要求 target 早于事件、distractor 晚于事件"
                    )
                if parsed.event_anchor.direction != "before":
                    raise ValueError("oracle 的 direction 必须是 before")
            else:
                if not (target_days < event_days and distractor_days > event_days):
                    raise ValueError(
                        "event_after 要求 target 晚于事件、distractor 早于事件"
                    )
                if parsed.event_anchor.direction != "after":
                    raise ValueError("oracle 的 direction 必须是 after")
        else:
            if parsed.ordering is None:
                raise ValueError("ordering 用例的 oracle 必须提供排序")
            if self.query_type == "ordering_earliest":
                if target_days <= distractor_days:
                    raise ValueError("ordering_earliest 要求 target 更早")
                if parsed.ordering != "earliest":
                    raise ValueError("oracle 的 ordering 必须是 earliest")
            else:
                if target_days >= distractor_days:
                    raise ValueError("ordering_latest 要求 target 更晚")
                if parsed.ordering != "latest":
                    raise ValueError("oracle 的 ordering 必须是 latest")
        return self

    def _ensure_unique_keyword(
        self, keyword: str, owner_index: int, field_name: str
    ) -> None:
        lowered = keyword.casefold()
        if lowered in self.query.casefold():
            raise ValueError(f"{field_name} 不能出现在 query 中")
        for index, memory in enumerate(self.memories):
            contained = lowered in memory.content.casefold()
            if index == owner_index and not contained:
                raise ValueError(f"{field_name} 必须原样出现在第 {owner_index} 条记忆中")
            if index != owner_index and contained:
                raise ValueError(f"{field_name} 只能出现在第 {owner_index} 条记忆中")


def load_temporal_cases(path: Path) -> list[TemporalBenchmarkCase]:
    """从 JSONL 读取并严格校验时间类样例；query 全局唯一以支持期望映射。"""

    cases: list[TemporalBenchmarkCase] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
                case = TemporalBenchmarkCase.model_validate(value)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number} 时间类样例无效: {exc}") from exc
            if case.case_id in seen_ids:
                raise ValueError(f"{path}:{line_number} case_id 重复: {case.case_id}")
            if case.query in seen_queries:
                raise ValueError(
                    f"{path}:{line_number} query 重复，oracle 映射需要唯一查询: {case.query}"
                )
            seen_ids.add(case.case_id)
            seen_queries.add(case.query)
            cases.append(case)

    if not cases:
        raise ValueError(f"时间类数据集为空: {path}")
    return cases


def rank_of_keyword(contents: list[str], keyword: str) -> int | None:
    """返回 1-based 首个包含关键词的排名；未命中返回 None。"""

    lowered = keyword.casefold()
    for index, content in enumerate(contents, start=1):
        if lowered in content.casefold():
            return index
    return None


class MultihopHop(BaseModel):
    """多跳链中的一跳：description 说明该跳作用，keyword 唯一标识证据。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    description: str = Field(min_length=1, max_length=200)
    keyword: str = Field(min_length=1, max_length=300)


class MultihopBenchmarkCase(BaseModel):
    """多跳链式场景：桥接实体问题 + 证据链 + 同主题分心记忆。

    考核指标是“全链召回率”：top-k 返回结果必须同时包含每一跳的证据，
    平台的回答模型才能完成串联。oracle_expansion 是仅由查询可推导的
    理想扩展文本（不得泄漏记忆内容），用于预演查询表达层的贡献。
    fillers 由评测器按 category 程序化生成，模拟真实规模的历史。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(min_length=1, max_length=100)
    language: Literal["zh", "en"]
    category: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=1000)
    memories: list[BenchmarkMemory] = Field(min_length=2, max_length=20)
    hops: list[MultihopHop] = Field(min_length=2, max_length=3)
    oracle_expansion: str = Field(min_length=1, max_length=1000)
    filler_count: int = Field(default=180, ge=0, le=1000)
    filler_seed: int = Field(default=1, ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_case(self) -> "MultihopBenchmarkCase":
        keywords = [hop.keyword for hop in self.hops]
        lowered_keywords = [keyword.casefold() for keyword in keywords]
        if len(set(lowered_keywords)) != len(lowered_keywords):
            raise ValueError("各跳的关键词不能重复")
        query_lowered = self.query.casefold()
        expansion_lowered = self.oracle_expansion.casefold()
        for hop, lowered in zip(self.hops, lowered_keywords):
            if lowered in query_lowered:
                raise ValueError(f"跳的关键词不能出现在 query 中: {hop.keyword}")
            if lowered in expansion_lowered:
                raise ValueError(
                    f"跳的关键词不能出现在 oracle_expansion 中: {hop.keyword}"
                )
            owners = [
                index
                for index, memory in enumerate(self.memories)
                if lowered in memory.content.casefold()
            ]
            if len(owners) != 1:
                raise ValueError(
                    f"关键词必须且只能出现在一条记忆中: {hop.keyword}（出现于 {owners}）"
                )
        return self


def load_multihop_cases(path: Path) -> list[MultihopBenchmarkCase]:
    """从 JSONL 读取并严格校验多跳样例；query 全局唯一以支持期望映射。"""

    cases: list[MultihopBenchmarkCase] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
                case = MultihopBenchmarkCase.model_validate(value)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number} 多跳样例无效: {exc}") from exc
            if case.case_id in seen_ids:
                raise ValueError(f"{path}:{line_number} case_id 重复: {case.case_id}")
            if case.query in seen_queries:
                raise ValueError(
                    f"{path}:{line_number} query 重复，期望映射需要唯一查询: {case.query}"
                )
            seen_ids.add(case.case_id)
            seen_queries.add(case.query)
            cases.append(case)

    if not cases:
        raise ValueError(f"多跳数据集为空: {path}")
    return cases
