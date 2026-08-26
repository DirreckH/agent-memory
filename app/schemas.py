from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SimpleSetRequest(StrictModel):
    memory_text: str = Field(min_length=1)


class MemoryMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    timestamp: int | None = Field(default=None, ge=0)


class CompetitionAddRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=1024)
    messages: list[MemoryMessage] = Field(min_length=1, max_length=100)
    user_id: str = Field(min_length=1, max_length=1024)
    session_id: str = Field(min_length=1, max_length=1024)


class SimpleGetRequest(StrictModel):
    query: str = Field(min_length=1)


class CompetitionSearchRequest(StrictModel):
    query: str = Field(min_length=1)
    options: list[str] | None = Field(default=None, max_length=100)
    user_id: str = Field(min_length=1, max_length=1024)
    top_k: int = Field(ge=1, le=1000)

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [option.strip() for option in value]
        if any(not option for option in cleaned):
            raise ValueError("options 中不能包含空字符串")
        return cleaned


class MemoryResult(StrictModel):
    id: str
    content: str
    score: float
    created_at: str


class SimpleSetResponse(StrictModel):
    success: bool
    message: str
    memory_count: int


class CompetitionAddResponse(StrictModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SimpleGetResponse(StrictModel):
    memory_text: str
    results: list[MemoryResult]


class CompetitionSearchResponse(StrictModel):
    data: list[MemoryResult]
