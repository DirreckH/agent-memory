from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class StorageConflictError(RuntimeError):
    """相同 request_id 被用于不同请求体。"""


class StorageBusyError(RuntimeError):
    """相同 request_id 正在由另一个请求处理。"""


@dataclass(frozen=True)
class MemoryToStore:
    id: str
    ordinal: int
    user_id: str
    session_id: str
    role: str
    content: str
    search_text: str
    embedding: NDArray[np.float32]
    created_at: str
    source_timestamp: int | None


@dataclass(frozen=True)
class StoredMemory:
    id: str
    content: str
    search_text: str
    embedding: NDArray[np.float32]
    created_at: str
    # 源消息时间（毫秒时间戳），不等同于事件发生时间；缺失时保持 None。
    source_timestamp: int | None = None


class SQLiteMemoryStore:
    """SQLite 持久化存储。

    向量以 float32 BLOB 保存，检索时仅加载指定 user_id 的记录并在内存中
    计算余弦相似度。比赛样本天然按 user_id 隔离且单样本规模较小，这种实现
    比部署独立向量数据库更适合轻量原型。
    """

    def __init__(self, path: Path, stale_seconds: int = 300) -> None:
        self.path = path
        self.stale_seconds = stale_seconds

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ingestion_requests (
                    request_id TEXT PRIMARY KEY,
                    payload_hash TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('processing', 'completed', 'failed')),
                    created_at TEXT NOT NULL,
                    updated_at_epoch REAL NOT NULL,
                    error_type TEXT
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    source_timestamp INTEGER,
                    UNIQUE(request_id, ordinal),
                    FOREIGN KEY(request_id) REFERENCES ingestion_requests(request_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_memories_user_id
                    ON memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_memories_user_created
                    ON memories(user_id, created_at DESC);
                """
            )

    def prune_older_than(self, retention_days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_text = cutoff.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "DELETE FROM ingestion_requests WHERE created_at < ?",
                (cutoff_text,),
            )
            connection.commit()
            return max(cursor.rowcount, 0)

    def claim_request(
        self,
        request_id: str,
        payload_hash: str,
        user_id: str,
        session_id: str,
        created_at: str,
    ) -> bool:
        """占用写请求。

        返回 True 表示调用方应执行写入；返回 False 表示相同请求已成功完成，
        可直接返回幂等成功。
        """

        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM ingestion_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()

            if row is None:
                connection.execute(
                    """
                    INSERT INTO ingestion_requests (
                        request_id, payload_hash, user_id, session_id, status,
                        created_at, updated_at_epoch, error_type
                    ) VALUES (?, ?, ?, ?, 'processing', ?, ?, NULL)
                    """,
                    (
                        request_id,
                        payload_hash,
                        user_id,
                        session_id,
                        created_at,
                        now,
                    ),
                )
                connection.commit()
                return True

            if (
                row["payload_hash"] != payload_hash
                or row["user_id"] != user_id
                or row["session_id"] != session_id
            ):
                connection.rollback()
                raise StorageConflictError(
                    "request_id 已存在，但请求内容或隔离标识与原请求不一致"
                )

            if row["status"] == "completed":
                connection.commit()
                return False

            is_stale = now - float(row["updated_at_epoch"]) >= self.stale_seconds
            if row["status"] == "processing" and not is_stale:
                connection.rollback()
                raise StorageBusyError("该 request_id 正在处理，请稍后重试")

            # 失败请求或进程异常留下的超时 processing 请求允许安全重试。
            connection.execute(
                """
                UPDATE ingestion_requests
                SET status = 'processing', updated_at_epoch = ?, error_type = NULL
                WHERE request_id = ?
                """,
                (now, request_id),
            )
            connection.commit()
            return True

    def complete_request(
        self, request_id: str, memories: list[MemoryToStore]
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM ingestion_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None or row["status"] != "processing":
                connection.rollback()
                raise StorageConflictError("写请求不处于可提交状态")

            for memory in memories:
                vector = np.asarray(memory.embedding, dtype=np.float32).reshape(-1)
                connection.execute(
                    """
                    INSERT INTO memories (
                        id, request_id, ordinal, user_id, session_id, role,
                        content, search_text, embedding, embedding_dim,
                        created_at, source_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.id,
                        request_id,
                        memory.ordinal,
                        memory.user_id,
                        memory.session_id,
                        memory.role,
                        memory.content,
                        memory.search_text,
                        vector.tobytes(order="C"),
                        int(vector.shape[0]),
                        memory.created_at,
                        memory.source_timestamp,
                    ),
                )

            connection.execute(
                """
                UPDATE ingestion_requests
                SET status = 'completed', updated_at_epoch = ?, error_type = NULL
                WHERE request_id = ?
                """,
                (time.time(), request_id),
            )
            connection.commit()

    def mark_failed(self, request_id: str, error_type: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE ingestion_requests
                SET status = 'failed', updated_at_epoch = ?, error_type = ?
                WHERE request_id = ? AND status = 'processing'
                """,
                (time.time(), error_type[:128], request_id),
            )
            connection.commit()

    def fetch_by_user(self, user_id: str) -> list[StoredMemory]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, content, search_text, embedding, embedding_dim,
                       created_at, source_timestamp
                FROM memories
                WHERE user_id = ?
                ORDER BY created_at DESC, id ASC
                """,
                (user_id,),
            ).fetchall()

        output: list[StoredMemory] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if vector.shape[0] != int(row["embedding_dim"]):
                # 单条损坏记录不应污染整个检索响应。
                continue
            output.append(
                StoredMemory(
                    id=row["id"],
                    content=row["content"],
                    search_text=row["search_text"],
                    embedding=vector,
                    created_at=row["created_at"],
                    source_timestamp=row["source_timestamp"],
                )
            )
        return output

    def count_by_user(self, user_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM memories WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return int(row["count"])
