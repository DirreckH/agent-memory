from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
