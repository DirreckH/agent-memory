from __future__ import annotations

import json
from dataclasses import asdict, replace

from app.governance.models import Evidence, Fact, Source, SourceStatus, UserSnapshot, VersionRelation, SummaryUnit


def initialize(connection):
    connection.executescript("""
    CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_user_id ON memories(user_id, id);
    CREATE TABLE IF NOT EXISTS governance_users (
        user_id TEXT PRIMARY KEY, active_generation INTEGER NOT NULL DEFAULT 1,
        revision INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS governance_generations (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'building',
        PRIMARY KEY(user_id, generation)
    );
    CREATE TABLE IF NOT EXISTS governance_entities (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, id TEXT NOT NULL,
        name TEXT NOT NULL, identity_context TEXT NOT NULL, source_id TEXT NOT NULL,
        PRIMARY KEY(user_id, generation, id),
        FOREIGN KEY(user_id, generation) REFERENCES governance_generations(user_id, generation) ON DELETE CASCADE,
        FOREIGN KEY(user_id, source_id) REFERENCES memories(user_id, id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS governance_facts (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, id TEXT NOT NULL,
        subject_id TEXT NOT NULL, predicate_key TEXT NOT NULL, scope_key TEXT NOT NULL,
        value TEXT NOT NULL, source_id TEXT NOT NULL, data TEXT NOT NULL,
        PRIMARY KEY(user_id, generation, id),
        FOREIGN KEY(user_id, generation, subject_id) REFERENCES governance_entities(user_id, generation, id),
        FOREIGN KEY(user_id, source_id) REFERENCES memories(user_id, id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS governance_aliases (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, entity_id TEXT NOT NULL,
        alias TEXT NOT NULL, source_id TEXT NOT NULL,
        PRIMARY KEY(user_id, generation, entity_id, alias, source_id),
        FOREIGN KEY(user_id, generation, entity_id) REFERENCES governance_entities(user_id, generation, id) ON DELETE CASCADE,
        FOREIGN KEY(user_id, source_id) REFERENCES memories(user_id, id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_governance_slots ON governance_facts(user_id, generation, subject_id, predicate_key, scope_key);
    CREATE TABLE IF NOT EXISTS governance_evidence (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, fact_id TEXT NOT NULL,
        source_id TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
        content_hash TEXT NOT NULL, purpose TEXT NOT NULL,
        PRIMARY KEY(user_id, generation, fact_id, source_id, start, end, purpose),
        FOREIGN KEY(user_id, generation, fact_id) REFERENCES governance_facts(user_id, generation, id) ON DELETE CASCADE,
        FOREIGN KEY(user_id, source_id) REFERENCES memories(user_id, id) ON DELETE CASCADE,
        CHECK(start >= 0 AND end > start)
    );
    CREATE TABLE IF NOT EXISTS governance_source_status (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, source_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('ready','no_fact','partial','pending','failed')),
        data TEXT NOT NULL,
        PRIMARY KEY(user_id, generation, source_id),
        FOREIGN KEY(user_id, generation) REFERENCES governance_generations(user_id, generation) ON DELETE CASCADE,
        FOREIGN KEY(user_id, source_id) REFERENCES memories(user_id, id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS governance_relations (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, newer TEXT NOT NULL, older TEXT NOT NULL,
        relation TEXT NOT NULL CHECK(relation IN ('changes','corrects','retracts','supports','conflicts')),
        data TEXT NOT NULL, PRIMARY KEY(user_id, generation, newer, older, relation),
        FOREIGN KEY(user_id, generation, newer) REFERENCES governance_facts(user_id, generation, id) ON DELETE CASCADE,
        FOREIGN KEY(user_id, generation, older) REFERENCES governance_facts(user_id, generation, id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS governance_blocked_facts (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, fact_id TEXT NOT NULL, reason TEXT NOT NULL,
        PRIMARY KEY(user_id, generation, fact_id),
        FOREIGN KEY(user_id, generation, fact_id) REFERENCES governance_facts(user_id, generation, id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS governance_summaries (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, id TEXT NOT NULL,
        revision INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('ready','dirty')),
        data TEXT NOT NULL, PRIMARY KEY(user_id, generation, id),
        FOREIGN KEY(user_id, generation) REFERENCES governance_generations(user_id, generation) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS governance_summary_facts (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, summary_id TEXT NOT NULL, fact_id TEXT NOT NULL,
        PRIMARY KEY(user_id, generation, summary_id, fact_id),
        FOREIGN KEY(user_id, generation, summary_id) REFERENCES governance_summaries(user_id, generation, id) ON DELETE CASCADE,
        FOREIGN KEY(user_id, generation, fact_id) REFERENCES governance_facts(user_id, generation, id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS governance_summary_sources (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, summary_id TEXT NOT NULL, source_id TEXT NOT NULL,
        PRIMARY KEY(user_id, generation, summary_id, source_id),
        FOREIGN KEY(user_id, generation, summary_id) REFERENCES governance_summaries(user_id, generation, id) ON DELETE CASCADE,
        FOREIGN KEY(user_id, source_id) REFERENCES memories(user_id, id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS governance_summary_children (
        user_id TEXT NOT NULL, generation INTEGER NOT NULL, summary_id TEXT NOT NULL, child_id TEXT NOT NULL,
        PRIMARY KEY(user_id, generation, summary_id, child_id),
        FOREIGN KEY(user_id, generation, summary_id) REFERENCES governance_summaries(user_id, generation, id) ON DELETE CASCADE,
        FOREIGN KEY(user_id, generation, child_id) REFERENCES governance_summaries(user_id, generation, id) ON DELETE CASCADE
    );
    """)


def put_batch(connection, user_id, batch, recorded_at=None, sequence=None, *, generation=None, replace_sources=False):
    from app.storage import StorageConflictError
    from app.governance.extraction import digest
    source_rows = {r['id']: r for r in connection.execute(
        "SELECT id, content, raw_content, source_timestamp FROM memories WHERE user_id=?", (user_id,))}
    status_ids = [s.source_id for s in batch.statuses]
    if len(set(status_ids)) != len(status_ids) or not set(status_ids).issubset(source_rows):
        raise StorageConflictError("治理来源不属于用户或处理状态重复")
    for fact in batch.facts:
        if not fact.evidence or fact.source_id not in status_ids or fact.source_time != source_rows[fact.source_id]['source_timestamp']:
            raise StorageConflictError("治理断言缺少本批来源或时间不符")
        for ev in fact.evidence:
            if ev.source_id not in status_ids:
                raise StorageConflictError("治理证据不属于本批来源")
            row = source_rows[ev.source_id]
            text = row['raw_content'] if row['raw_content'] is not None else row['content']
            if (type(ev.start) is not int or type(ev.end) is not int or not 0 <= ev.start < ev.end <= len(text)
                    or text[ev.start:ev.end] != ev.quote or digest(ev.quote) != ev.content_hash):
                raise StorageConflictError("治理证据范围或哈希校验失败")
    for status in batch.statuses:
        row = source_rows[status.source_id]
        text = row['raw_content'] if row['raw_content'] is not None else row['content']
        if any(type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text) for a,b in status.processed_ranges):
            raise StorageConflictError("治理处理范围无效")
        if status.status in ('ready','no_fact') and status.processed_ranges != ((0,len(text)),):
            raise StorageConflictError("完整处理状态缺少完整来源范围")
        present = any(f.source_id == status.source_id for f in batch.facts)
        if status.status == 'ready' and not present or status.status == 'no_fact' and present:
            raise StorageConflictError("治理事实与处理状态不一致")
    connection.execute("INSERT OR IGNORE INTO governance_users(user_id) VALUES (?)", (user_id,))
    state = connection.execute("SELECT * FROM governance_users WHERE user_id=?", (user_id,)).fetchone()
    generation = generation if generation is not None else state["active_generation"]
    connection.execute("INSERT OR IGNORE INTO governance_generations VALUES (?, ?, 'active')", (user_id, generation))
    previous = load(connection, user_id, (), generation=generation)
    active = load(connection, user_id, ()) if generation != state["active_generation"] else previous
    blocks = {}
    for prior in (previous, active):
        by_id = {f.id: f for f in prior.facts}
        for fid, reason in prior.blocked_facts:
            if fid in by_id:
                f = by_id[fid]
                blocks[(f.source_id, f.slot, f.value)] = reason
    if replace_sources:
        replaced = {s.source_id for s in batch.statuses if s.status in ('ready', 'no_fact')}
        by_id = {f.id: f for f in previous.facts}
        for rel in previous.relations:
            old, new = by_id[rel.older], by_id[rel.newer]
            if rel.relation in ('changes', 'corrects', 'retracts') and new.source_id in replaced and old.source_id not in replaced:
                blocks[(old.source_id, old.slot, old.value)] = 'reindexed_' + rel.relation
        for sid in replaced:
            connection.execute("DELETE FROM governance_facts WHERE user_id=? AND generation=? AND source_id=?", (user_id, generation, sid))
            connection.execute("DELETE FROM governance_aliases WHERE user_id=? AND generation=? AND source_id=?", (user_id, generation, sid))
    catalog = [f for f in previous.facts if not replace_sources or f.source_id not in replaced]
    for fact in batch.facts:
        if fact.subject_id != 'self':
            names = {fact.subject_name.casefold(), *(a.casefold() for a in fact.aliases)}
            candidates = {f.subject_id for f in catalog
                          if f.identity_context and fact.identity_context
                          and f.identity_context.casefold() == fact.identity_context.casefold()
                          and names.intersection({f.subject_name.casefold(), *(a.casefold() for a in f.aliases)})}
            if len(candidates) == 1:
                fact = replace(fact, subject_id=next(iter(candidates)))
        fact = replace(fact, recorded_at=recorded_at if recorded_at is not None else fact.recorded_at,
                       commit_sequence=sequence if sequence is not None else fact.commit_sequence)
        catalog.append(fact)
        connection.execute("INSERT OR IGNORE INTO governance_entities VALUES (?, ?, ?, ?, ?, ?)",
                           (user_id, generation, fact.subject_id, fact.subject_name, fact.identity_context, fact.source_id))
        for alias in fact.aliases:
            connection.execute("INSERT OR IGNORE INTO governance_aliases VALUES (?, ?, ?, ?, ?)",
                               (user_id, generation, fact.subject_id, alias, fact.source_id))
        connection.execute("INSERT OR REPLACE INTO governance_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (user_id, generation, fact.id, fact.subject_id, fact.predicate, fact.scope,
                            fact.value, fact.source_id, json.dumps(asdict(fact), ensure_ascii=False)))
        for ev in fact.evidence:
            connection.execute("INSERT OR REPLACE INTO governance_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (user_id, generation, fact.id, ev.source_id, ev.start, ev.end, ev.content_hash, ev.purpose))
    for status in batch.statuses:
        connection.execute("INSERT OR REPLACE INTO governance_source_status VALUES (?, ?, ?, ?, ?)",
                           (user_id, generation, status.source_id, status.status, json.dumps(asdict(status))))
    from app.governance.resolution import derive_relations
    snapshot = load(connection, user_id, (), generation=generation)
    for fact in snapshot.facts:
        reason = blocks.get((fact.source_id, fact.slot, fact.value))
        if reason:
            connection.execute("INSERT OR REPLACE INTO governance_blocked_facts VALUES (?, ?, ?, ?)",
                               (user_id, generation, fact.id, reason))
    connection.execute("DELETE FROM governance_relations WHERE user_id=? AND generation=?", (user_id, generation))
    for rel in derive_relations(snapshot.facts):
        connection.execute("INSERT OR REPLACE INTO governance_relations VALUES (?, ?, ?, ?, ?, ?)",
                           (user_id, generation, rel.newer, rel.older, rel.relation, json.dumps(asdict(rel))))
    invalidate(connection, user_id)


def invalidate(connection, user_id):
    connection.execute("UPDATE governance_users SET revision=revision+1 WHERE user_id=?", (user_id,))
    connection.execute("UPDATE governance_summaries SET status='dirty' WHERE user_id=?", (user_id,))


def before_delete(connection, user_id, source_ids):
    """删除更新来源后只保留失效标记，既不保存被删文本也不恢复旧值。"""
    removed = set(source_ids)
    rows = connection.execute("SELECT generation, id, source_id FROM governance_facts WHERE user_id=?", (user_id,)).fetchall()
    source_by_fact = {(r["generation"], r["id"]): r["source_id"] for r in rows}
    for rel in connection.execute("SELECT * FROM governance_relations WHERE user_id=?", (user_id,)).fetchall():
        if (rel["relation"] in ("changes", "corrects", "retracts")
                and source_by_fact.get((rel["generation"], rel["newer"])) in removed
                and source_by_fact.get((rel["generation"], rel["older"])) not in removed):
            connection.execute("INSERT OR REPLACE INTO governance_blocked_facts VALUES (?, ?, ?, ?)",
                               (user_id, rel["generation"], rel["older"], "deleted_" + rel["relation"]))
    # 实体身份来源可以有多条支持，删除最初来源前迁移到存活断言的支持来源。
    for ent in connection.execute("SELECT * FROM governance_entities WHERE user_id=?", (user_id,)).fetchall():
        if ent["source_id"] not in removed:
            continue
        supporting = connection.execute("SELECT source_id FROM governance_facts WHERE user_id=? AND generation=? AND subject_id=? ORDER BY id", (user_id, ent["generation"], ent["id"])).fetchall()
        replacement = next((r[0] for r in supporting if r[0] not in removed), None)
        if replacement:
            connection.execute("UPDATE governance_entities SET source_id=? WHERE user_id=? AND generation=? AND id=?", (replacement, user_id, ent["generation"], ent["id"]))
    # 一次移除所有主断言，避免多来源删除时实体的立即外键检查中途失败。
    for sid in removed:
        connection.execute("DELETE FROM governance_facts WHERE user_id=? AND source_id=?", (user_id, sid))
    invalidate(connection, user_id)


def publish(connection, user_id, summary):
    state = connection.execute("SELECT * FROM governance_users WHERE user_id=?", (user_id,)).fetchone()
    generation_exists = connection.execute("SELECT status FROM governance_generations WHERE user_id=? AND generation=?", (user_id, summary.generation)).fetchone()
    if (not summary.complete or state is None or state["revision"] != summary.revision
            or generation_exists is None or generation_exists[0] not in ('building', 'active')):
        return False
    gen = summary.generation
    connection.execute("DELETE FROM governance_summaries WHERE user_id=? AND generation=?", (user_id, gen))
    for unit in summary.units:
        connection.execute("INSERT INTO governance_summaries VALUES (?, ?, ?, ?, 'ready', ?)",
                           (user_id, gen, unit.id, summary.revision, json.dumps(asdict(unit), ensure_ascii=False)))
    for unit in summary.units:
        for fid in unit.fact_ids:
            connection.execute("INSERT INTO governance_summary_facts VALUES (?, ?, ?, ?)", (user_id, gen, unit.id, fid))
        for sid in unit.source_ids:
            connection.execute("INSERT INTO governance_summary_sources VALUES (?, ?, ?, ?)", (user_id, gen, unit.id, sid))
        for cid in unit.child_ids:
            connection.execute("INSERT INTO governance_summary_children VALUES (?, ?, ?, ?)", (user_id, gen, unit.id, cid))
    return True


def load(connection, user_id, records, *, generation=None):
    state = connection.execute("SELECT * FROM governance_users WHERE user_id=?", (user_id,)).fetchone()
    active_generation, revision = (state["active_generation"], state["revision"]) if state else (1, 0)
    generation = generation if generation is not None else active_generation
    sources = {}
    for row in connection.execute("""SELECT m.*, c.sequence, c.committed_at FROM memories m
                 LEFT JOIN ingestion_commits c ON c.request_id=m.request_id WHERE m.user_id=?""", (user_id,)):
        # 旧库无法证明显示前缀是否来自用户，保留存储内容原样；不以增强文本为来源。
        raw = row["raw_content"]
        if raw is None:
            raw = row["content"]
        sources[row["id"]] = Source(row["id"], raw, row["role"], row["session_id"], row["ordinal"],
                                    row["source_timestamp"], row["committed_at"] or "", row["sequence"],
                                    row["raw_content"] is not None)
    facts = []
    for row in connection.execute("SELECT data FROM governance_facts WHERE user_id=? AND generation=?", (user_id, generation)):
        data = json.loads(row[0])
        data["evidence"] = tuple(Evidence(**e) for e in data["evidence"])
        data["object_entities"] = tuple(data["object_entities"])
        data["aliases"] = tuple(data.get("aliases", ()))
        facts.append(Fact(**data))
    statuses = []
    for row in connection.execute("SELECT data FROM governance_source_status WHERE user_id=? AND generation=?", (user_id, generation)):
        data = json.loads(row[0])
        data["processed_ranges"] = tuple(tuple(r) for r in data["processed_ranges"])
        statuses.append(SourceStatus(**data))
    relations = []
    for row in connection.execute("SELECT data FROM governance_relations WHERE user_id=? AND generation=?", (user_id, generation)):
        data = json.loads(row[0])
        data["evidence"] = tuple(Evidence(**e) for e in data["evidence"])
        relations.append(VersionRelation(**data))
    summaries = []
    for row in connection.execute("SELECT data FROM governance_summaries WHERE user_id=? AND generation=? AND revision=? AND status='ready'", (user_id, generation, revision)):
        data = json.loads(row[0])
        for key in ("fact_ids", "source_ids", "child_ids"):
            data[key] = tuple(data[key])
        data["display_fact_ids"] = tuple(data.get("display_fact_ids", ()))
        data["coverage"] = tuple(tuple(r) for r in data["coverage"])
        summaries.append(SummaryUnit(**data))
    blocked = tuple((r[0], r[1]) for r in connection.execute("SELECT fact_id, reason FROM governance_blocked_facts WHERE user_id=? AND generation=?", (user_id, generation)))
    return UserSnapshot(user_id, generation, revision, sources, tuple(facts), tuple(relations), tuple(statuses), tuple(summaries), tuple(records), blocked)
