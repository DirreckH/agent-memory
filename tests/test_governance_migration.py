from __future__ import annotations

import sqlite3
import time

import numpy as np

from app.storage import SQLiteMemoryStore


def test_legacy_migration_backs_up_and_keeps_unknown_commit_order(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE ingestion_requests (request_id TEXT PRIMARY KEY, payload_hash TEXT,
            user_id TEXT, session_id TEXT, status TEXT, created_at TEXT, updated_at_epoch REAL, error_type TEXT);
        CREATE TABLE memories (id TEXT PRIMARY KEY, request_id TEXT, ordinal INTEGER,
            user_id TEXT, session_id TEXT, role TEXT, content TEXT, search_text TEXT,
            embedding BLOB, embedding_dim INTEGER, created_at TEXT, source_timestamp INTEGER,
            UNIQUE(request_id,ordinal), FOREIGN KEY(request_id) REFERENCES ingestion_requests(request_id) ON DELETE CASCADE);
        """)
        conn.execute("INSERT INTO ingestion_requests VALUES ('r','h','u','s','completed','2026-10-01',?,NULL)", (time.time(),))
        conn.execute("INSERT INTO memories VALUES ('m','r',0,'u','s','user',?,?,?,1,'2005-01-01',NULL)",
                     ("[user]\nI work at Alpha.", "[user]\nI work at Alpha.", np.array([1], np.float32).tobytes()))
    store = SQLiteMemoryStore(path)
    store.initialize()
    assert store.last_migration_backup is not None and store.last_migration_backup.exists()
    assert SQLiteMemoryStore(store.last_migration_backup).count_by_user("u") == 1
    store.initialize()
    assert len(list(tmp_path.glob("*.pre-governance-*.db"))) == 1
    snapshot = store.fetch_by_user("u", governance=True)
    assert snapshot.sources["m"].commit_sequence is None
    assert snapshot.sources["m"].timestamp is None
    assert snapshot.sources["m"].recorded_at == ""
    assert snapshot.incomplete == ("m",)
