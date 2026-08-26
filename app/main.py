from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import Union

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.config import Settings
from app.embeddings import Embedder, FastEmbedder
from app.llm import MemoryLLM, build_memory_llm
from app.schemas import (
    CompetitionAddRequest,
    CompetitionSearchRequest,
    MemoryResult,
    SimpleGetRequest,
    SimpleGetResponse,
    SimpleSetRequest,
    SimpleSetResponse,
)
from app.service import MemoryService, MemoryServiceUnavailable
from app.storage import (
    SQLiteMemoryStore,
    StorageBusyError,
    StorageConflictError,
)


SetPayload = Union[SimpleSetRequest, CompetitionAddRequest]
GetPayload = Union[SimpleGetRequest, CompetitionSearchRequest]


def _extract_auth_token(authorization: str | None, x_api_key: str | None) -> list[str]:
    candidates: list[str] = []
    if x_api_key:
        candidates.append(x_api_key.strip())
    if authorization:
        scheme, separator, value = authorization.strip().partition(" ")
        if separator and scheme.casefold() in {"bearer", "token"} and value.strip():
            candidates.append(value.strip())
    return candidates


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    llm: MemoryLLM | None = None,
) -> FastAPI:
    settings = settings or Settings()
    store = SQLiteMemoryStore(
        settings.database_path, stale_seconds=settings.ingestion_stale_seconds
    )
    embedder = embedder or FastEmbedder(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        threads=settings.embedding_threads,
    )
    llm = llm or build_memory_llm(settings)
    service = MemoryService(settings, store, embedder, llm)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # SQLite 初始化和可选模型预热必须在接收流量前同步完成。
        service.initialize()
        yield

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
        # 减少公网攻击面；比赛只需要 set/get 和健康检查。
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.memory_service = service

    def require_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    ) -> None:
        expected = settings.expected_memory_api_key
        if not expected:
            return
        candidates = _extract_auth_token(authorization, x_api_key)
        if not any(secrets.compare_digest(expected, value) for value in candidates):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"reason": "invalid memory system key"},
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        # 官网要求 Health 为无需鉴权的 GET，任意 2xx 表示健康。
        return {"status": "ok"}

    @app.post("/set", include_in_schema=False)
    def set_memory(
        payload: SetPayload,
        _: None = Depends(require_api_key),
    ) -> dict[str, object]:
        try:
            if isinstance(payload, SimpleSetRequest):
                count = service.add_simple(payload.memory_text)
                response = SimpleSetResponse(
                    success=True,
                    message="记忆写入成功，现已可检索",
                    memory_count=count,
                )
                return response.model_dump()

            service.add(
                request_id=payload.request_id,
                messages=payload.messages,
                user_id=payload.user_id,
                session_id=payload.session_id,
            )
            # 评测响应必须原样回显三个标识，不增加异步 task ID。
            return {
                "success": True,
                "request_id": payload.request_id,
                "user_id": payload.user_id,
                "session_id": payload.session_id,
            }
        except StorageConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": str(exc)},
            ) from exc
        except StorageBusyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": str(exc)},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"reason": str(exc)},
            ) from exc
        except MemoryServiceUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"reason": str(exc)},
            ) from exc

    @app.post("/get", include_in_schema=False)
    def get_memory(
        payload: GetPayload,
        _: None = Depends(require_api_key),
    ) -> dict[str, object]:
        try:
            if isinstance(payload, SimpleGetRequest):
                hits = service.search_simple(payload.query)
                results = [MemoryResult(**hit.__dict__) for hit in hits]
                response = SimpleGetResponse(
                    memory_text=results[0].content if results else "",
                    results=results,
                )
                return response.model_dump()

            hits = service.search(
                query=payload.query,
                options=payload.options,
                user_id=payload.user_id,
                top_k=payload.top_k,
            )
            return {
                "data": [MemoryResult(**hit.__dict__).model_dump() for hit in hits]
            }
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"reason": str(exc)},
            ) from exc
        except MemoryServiceUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"reason": str(exc)},
            ) from exc

    return app


app = create_app()
