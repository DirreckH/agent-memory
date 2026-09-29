"""接口输入日志：把进入 /set、/get 的请求正文逐条写入 JSONL 文件。

设计动机与边界（与 app/config.py 的配置注释一致）：
- 这是评测数据在主库之外的一份副本。为守住“评测数据 N 天内删除”的承诺，
  文件按天轮转，保留份数在装配时对齐 DATA_RETENTION_DAYS，到期自动删除。
- 只记录请求体，永不记录请求头——鉴权凭据不落盘。
- 单条正文超过 INPUT_LOG_MAX_CHARS 截断；body_chars 字段保留原始长度。
- INPUT_LOG_ENABLED=false 时不挂载中间件，零开销。

中间件采用纯 ASGI 透传实现（不依赖 BaseHTTPMiddleware 的请求体缓存行为）：
下游处理器收到的仍是原始消息流，校验失败的 422 请求体同样会被记录。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

INPUT_LOGGER_NAME = "agent_memory.input"


def build_input_logger(settings: Any) -> logging.Logger:
    """为当前进程装配输入日志器；重复调用会重置处理器，便于测试与重复装配。"""
    logger = logging.getLogger(INPUT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    settings.input_log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        settings.input_log_path,
        when="midnight",
        # 保留份数对齐数据保留期：轮转出的旧文件到期自动删除。
        backupCount=max(settings.data_retention_days, 1),
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


class InputLogMiddleware:
    """纯 ASGI 中间件：记录 POST 请求的正文与响应状态；GET（健康检查）不入日志。"""

    def __init__(self, app: ASGIApp, logger: logging.Logger, max_chars: int) -> None:
        self.app = app
        self.logger = logger
        self.max_chars = max_chars

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        chunks: list[bytes] = []
        status: int | None = None

        async def receive_proxy() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
            return message

        async def send_proxy(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive_proxy, send_proxy)
        finally:
            self._emit(scope, b"".join(chunks), status, started)

    def _emit(self, scope: Scope, body: bytes, status: int | None, started: float) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "method": scope.get("method"),
            "path": scope.get("path"),
            "status": status if status is not None else 500,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "body_chars": len(body),
        }
        try:
            payload = json.loads(body) if body else None
        except (ValueError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict):
            for key in ("request_id", "user_id", "session_id", "query", "top_k"):
                if key in payload:
                    record[key] = payload[key]
        record["body"] = body.decode("utf-8", errors="replace")[: self.max_chars]
        try:
            self.logger.info(json.dumps(record, ensure_ascii=False))
        except Exception:  # noqa: BLE001 - 日志失败不得影响主请求流程
            pass
