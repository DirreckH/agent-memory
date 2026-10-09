from __future__ import annotations

import numpy as np
import pytest
import sqlite3

from app.governance.models import SourceStatus, ValidatedIndexBatch

from app.storage import MemoryToStore, SQLiteMemoryStore, StorageConflictError


def memory(user_id="u", id="m"):
    return MemoryToStore(id, 0, user_id, "s", "user", "evidence", "evidence",
                         np.array([1.0], dtype=np.float32), "2026-10-01T00:00:00Z", None)


def test_stale_writer_cannot_commit_or_fail_the_retry(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "lease.db", stale_seconds=0)
    store.initialize()
    old = store.claim_request("r", "hash", "u", "s", "2026-10-01T00:00:00Z")
    retry = store.claim_request("r", "hash", "u", "s", "2026-10-01T00:00:00Z")
    with pytest.raises(StorageConflictError):
        store.complete_request(old, [memory()])
    store.mark_failed(old, "Timeout")
    store.complete_request(retry, [memory()])
    assert store.count_by_user("u") == 1
    assert store.claim_request("r", "hash", "u", "s", "2026-10-01T00:00:00Z") is None


@pytest.mark.parametrize("source_user", ["other", "u"])
def test_foreign_source_rolls_back_original_and_derived_commit(tmp_path, source_user):
    store = SQLiteMemoryStore(tmp_path / "isolated.db")
    store.initialize()
    private = store.claim_request("private", "h1", source_user, "s", "2026-10-01T00:00:00Z")
    store.complete_request(private, [memory(source_user, "secret")])
    own = store.claim_request("own", "h2", "u", "s", "2026-10-01T00:00:00Z")
    forged = ValidatedIndexBatch(statuses=(SourceStatus("secret", "no_fact", ((0, 8),)),))
    with pytest.raises((StorageConflictError, sqlite3.IntegrityError)):
        store.complete_request(own, [memory()], forged)
    assert store.count_by_user("u") == int(source_user == "u")
    assert store.count_by_user("other") == int(source_user == "other")
    assert not store.fetch_by_user("u", governance=True).facts
