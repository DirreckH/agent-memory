from __future__ import annotations

import sqlite3
import time
import uuid
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
class ClaimLease:
    request_id: str
    attempt_token: str


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
    raw_content: str | None = None


@dataclass(frozen=True)
class StoredMemory:
    id: str
    content: str
    search_text: str
    embedding: NDArray[np.float32]
    created_at: str
    # 源消息时间（毫秒时间戳），不等同于事件发生时间；缺失时保持 None。
    source_timestamp: int | None = None

    # 原始写入上下文，只在内部检索使用；ordinal 是输入顺序，不是事件时间。
    request_id: str = ''
    session_id: str = ''
    ordinal: int | None = None


class SQLiteMemoryStore:
    """SQLite 持久化存储。

    向量以 float32 BLOB 保存，检索时仅加载指定 user_id 的记录并在内存中
    计算余弦相似度。比赛样本天然按 user_id 隔离且单样本规模较小，这种实现
    比部署独立向量数据库更适合轻量原型。
    """

    def __init__(self, path: Path, stale_seconds: int = 300) -> None:
        self.path = path
        self.stale_seconds = stale_seconds
        self.last_migration_backup: Path | None = None

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
        if self.path.exists() and self.path.stat().st_size:
            with self._connect() as existing:
                tables = {r[0] for r in existing.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                needs_migration = 'ingestion_requests' in tables and 'governance_schema' not in tables
                if needs_migration and existing.execute("SELECT 1 FROM ingestion_requests WHERE status='processing' LIMIT 1").fetchone():
                    raise StorageBusyError("迁移前须停止旧写入进程并处理遗留 processing 请求")
            if needs_migration:
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                backup = self.path.with_name(f'{self.path.stem}.pre-governance-{stamp}-{uuid.uuid4().hex[:8]}.db')
                self.backup_to(backup)
                self.last_migration_backup = backup
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

            columns = {row["name"] for row in connection.execute("PRAGMA table_info(ingestion_requests)")}
            if "attempt_token" not in columns:
                connection.execute("ALTER TABLE ingestion_requests ADD COLUMN attempt_token TEXT")
            memory_columns = {row["name"] for row in connection.execute("PRAGMA table_info(memories)")}
            if "raw_content" not in memory_columns:
                connection.execute("ALTER TABLE memories ADD COLUMN raw_content TEXT")
            connection.execute("""CREATE TABLE IF NOT EXISTS ingestion_commits (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL UNIQUE REFERENCES ingestion_requests(request_id) ON DELETE CASCADE,
                committed_at TEXT NOT NULL
            )""")
            from app.governance.persistence import initialize
            initialize(connection)
            connection.execute("CREATE TABLE IF NOT EXISTS governance_schema (version INTEGER PRIMARY KEY, installed_at TEXT NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO governance_schema VALUES (1, ?)", (datetime.now(timezone.utc).isoformat(),))

    def prune_older_than(self, retention_days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_text = cutoff.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            from app.governance.persistence import before_delete
            affected = connection.execute("""SELECT m.user_id, m.id FROM memories m
                JOIN ingestion_requests r ON r.request_id=m.request_id WHERE r.created_at < ?""", (cutoff_text,)).fetchall()
            for user_id in {r["user_id"] for r in affected}:
                before_delete(connection, user_id, [r["id"] for r in affected if r["user_id"] == user_id])
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
    ) -> ClaimLease | None:
        """占用写请求。

        返回带执行令牌的租约；已完成请求返回 None。
        """

        now = time.time()
        token = uuid.uuid4().hex
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
                        created_at, updated_at_epoch, error_type, attempt_token
                    ) VALUES (?, ?, ?, ?, 'processing', ?, ?, NULL, ?)
                    """,
                    (
                        request_id,
                        payload_hash,
                        user_id,
                        session_id,
                        created_at,
                        now,
                        token,
                    ),
                )
                connection.commit()
                return ClaimLease(request_id, token)

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
                return None

            is_stale = now - float(row["updated_at_epoch"]) >= self.stale_seconds
            if row["status"] == "processing" and not is_stale:
                connection.rollback()
                raise StorageBusyError("该 request_id 正在处理，请稍后重试")

            # 失败请求或进程异常留下的超时 processing 请求允许安全重试。
            connection.execute(
                """
                UPDATE ingestion_requests
                SET status = 'processing', updated_at_epoch = ?, error_type = NULL, attempt_token = ?
                WHERE request_id = ?
                """,
                (now, token, request_id),
            )
            connection.commit()
            return ClaimLease(request_id, token)

    def complete_request(
        self, lease: ClaimLease, memories: list[MemoryToStore], governance_batch=None
    ) -> None:
        request_id = lease.request_id
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM ingestion_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None or row["status"] != "processing" or row["attempt_token"] != lease.attempt_token:
                connection.rollback()
                raise StorageConflictError("写请求不处于可提交状态")
            if governance_batch is not None:
                status_ids = [s.source_id for s in governance_batch.statuses]
                if len(set(status_ids)) != len(status_ids) or set(status_ids) != {m.id for m in memories}:
                    raise StorageConflictError("治理处理状态必须完整对应本次写入来源")

            for memory in memories:
                if memory.user_id != row["user_id"] or memory.session_id != row["session_id"]:
                    raise StorageConflictError("消息不属于写入租约的用户和会话")
                vector = np.asarray(memory.embedding, dtype=np.float32).reshape(-1)
                connection.execute(
                    """
                    INSERT INTO memories (
                        id, request_id, ordinal, user_id, session_id, role,
                        content, search_text, embedding, embedding_dim,
                        created_at, source_timestamp, raw_content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        memory.raw_content,
                    ),
                )

            connection.execute(
                "INSERT INTO ingestion_commits(request_id, committed_at) VALUES (?, ?)",
                (request_id, datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")),
            )
            if governance_batch is not None:
                from app.governance.persistence import put_batch
                commit = connection.execute("SELECT sequence, committed_at FROM ingestion_commits WHERE request_id=?", (request_id,)).fetchone()
                put_batch(connection, row["user_id"], governance_batch, commit["committed_at"], commit["sequence"])
            else:
                from app.governance.persistence import invalidate
                invalidate(connection, row["user_id"])
            connection.execute(
                """
                UPDATE ingestion_requests
                SET status = 'completed', updated_at_epoch = ?, error_type = NULL
                WHERE request_id = ?
                """,
                (time.time(), request_id),
            )
            connection.commit()

    def mark_failed(self, lease: ClaimLease, error_type: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE ingestion_requests
                SET status = 'failed', updated_at_epoch = ?, error_type = ?
                WHERE request_id = ? AND status = 'processing' AND attempt_token = ?
                """,
                (time.time(), error_type[:128], lease.request_id, lease.attempt_token),
            )
            connection.commit()

    def fetch_by_user(self, user_id: str, *, governance: bool = False, generation: int | None = None):
        with self._connect() as connection:
            connection.execute("BEGIN")
            rows = connection.execute(
                """
                SELECT id, content, search_text, embedding, embedding_dim,
                       created_at, source_timestamp, request_id, session_id, ordinal
                FROM memories
                WHERE user_id = ?
                ORDER BY created_at DESC, id ASC
                """,
                (user_id,),
            ).fetchall()

            output = self._decode_memories(rows)
            if governance:
                from app.governance.persistence import load
                snapshot = load(connection, user_id, output, generation=generation)
                connection.commit()
                return snapshot
            connection.commit()
            return output

    @staticmethod
    def _decode_memories(rows) -> list[StoredMemory]:
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
                    request_id=row['request_id'],
                    session_id=row['session_id'],
                    ordinal=row['ordinal'],
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

    def publish_summaries(self, user_id: str, summary) -> bool:
        from app.governance.persistence import publish
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            published = publish(connection, user_id, summary)
            connection.commit()
            return published

    def delete_sources(self, user_id: str, source_ids: list[str]) -> int:
        from app.governance.persistence import before_delete
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            requested = set(source_ids)
            actual = [r[0] for r in connection.execute("SELECT id FROM memories WHERE user_id=?", (user_id,)) if r[0] in requested]
            if actual:
                before_delete(connection, user_id, actual)
                connection.executemany("DELETE FROM memories WHERE user_id=? AND id=?", [(user_id, sid) for sid in actual])
            connection.commit()
            return len(actual)

    def create_index_generation(self, user_id: str) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT OR IGNORE INTO governance_users(user_id) VALUES (?)", (user_id,))
            active = connection.execute("SELECT active_generation FROM governance_users WHERE user_id=?", (user_id,)).fetchone()[0]
            connection.execute("INSERT OR IGNORE INTO governance_generations VALUES (?, ?, 'active')", (user_id, active))
            generation = connection.execute("SELECT MAX(generation)+1 FROM governance_generations WHERE user_id=?", (user_id,)).fetchone()[0]
            connection.execute("INSERT INTO governance_generations VALUES (?, ?, 'building')", (user_id, generation))
            connection.commit()
            return generation

    def commit_index_batch(self, user_id: str, generation: int, expected_revision: int, batch) -> bool:
        from app.governance.persistence import put_batch
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = connection.execute("SELECT revision FROM governance_users WHERE user_id=?", (user_id,)).fetchone()
            gen = connection.execute("SELECT status FROM governance_generations WHERE user_id=? AND generation=?", (user_id, generation)).fetchone()
            if state is None or state[0] != expected_revision or gen is None or gen[0] not in ('building','active'):
                connection.rollback()
                return False
            put_batch(connection, user_id, batch, generation=generation, replace_sources=True)
            connection.commit()
            return True

    def activate_index_generation(self, user_id: str, generation: int, expected_revision: int) -> bool:
        from app.governance.persistence import load, invalidate
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            snapshot = load(connection, user_id, (), generation=generation)
            gen = connection.execute("SELECT status FROM governance_generations WHERE user_id=? AND generation=?", (user_id, generation)).fetchone()
            if snapshot.revision != expected_revision or snapshot.incomplete or gen is None or gen[0] not in ('building','active'):
                connection.rollback()
                return False
            connection.execute("UPDATE governance_generations SET status='retired' WHERE user_id=? AND status='active'", (user_id,))
            connection.execute("UPDATE governance_generations SET status='active' WHERE user_id=? AND generation=?", (user_id, generation))
            connection.execute("UPDATE governance_users SET active_generation=? WHERE user_id=?", (generation, user_id))
            invalidate(connection, user_id)
            connection.commit()
            return True

    def backup_to(self, destination: Path) -> None:
        destination = destination.resolve()
        if destination == self.path.resolve() or destination.exists():
            raise ValueError("备份目标必须是尚不存在的独立文件")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source, sqlite3.connect(str(destination)) as target:
            source.backup(target)
