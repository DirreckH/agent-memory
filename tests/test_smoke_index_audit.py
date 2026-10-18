"""The diagnostic must not mistake changed ordering or absent data for regression."""
import json
from pathlib import Path
import sqlite3
import struct

import pytest

from scripts.audit_smoke_indices import (
    IndexReader, compare, load_log, memory_id, pair_queries, read_only_database, sha,
)


def user(letter):
    return 'u_' + letter * 64


def add(who, request, text, status=200):
    return {'path': '/set', 'status': status, 'body': json.dumps({'user_id': who,
        'request_id': request, 'session_id': 'session-' + who, 'messages': [
            {'role': 'user', 'content': text, 'timestamp': 1704067200000}]})}


def get(who, query='What is the current location?'):
    return {'path': '/get', 'status': 200, 'body': json.dumps({'user_id': who, 'query': query, 'top_k': 5})}


def log(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    return load_log(path)


def schema(path):
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE memories(id TEXT, user_id TEXT, raw_content TEXT, role TEXT,
              source_timestamp INTEGER, search_text TEXT, embedding BLOB, embedding_dim INTEGER);
            CREATE TABLE governance_users(user_id TEXT, active_generation INTEGER, revision INTEGER);
            CREATE TABLE governance_source_status(user_id TEXT, generation INTEGER, source_id TEXT, status TEXT, data TEXT);
            CREATE TABLE governance_facts(user_id TEXT, generation INTEGER, id TEXT, subject_id TEXT, source_id TEXT, data TEXT);
            CREATE TABLE governance_relations(user_id TEXT, generation INTEGER, newer TEXT, older TEXT, relation TEXT);
            CREATE TABLE governance_blocked_facts(user_id TEXT, generation INTEGER, fact_id TEXT, reason TEXT);
        ''')


def store_query(path, query, *, text_suffix='', vector=1.0, status='ready', value='PRIVATE VALUE', generation=1):
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO governance_users VALUES (?, ?, ?)', (query['user'], generation, 1))
        for source in query['history']:
            sid, message = source['id'], source['message']
            db.execute('INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (sid, query['user'], message['content'], message['role'], message['timestamp'],
                 message['content'] + text_suffix, struct.pack('<f', vector), 1))
            state = dict(source_id=sid, status=status, processed_ranges=[[0, len(message['content'])]], error_type=None)
            db.execute('INSERT INTO governance_source_status VALUES (?, ?, ?, ?, ?)',
                (query['user'], generation, sid, status, json.dumps(state)))
            fact = dict(id='fact-' + sid, subject_id='entity-' + query['user'], subject_name='Aster',
                predicate='residence', scope='home', value=value, kind='state', operation='assert',
                source_time=message['timestamp'], recorded_at=query['user'], commit_sequence=source['line'],
                evidence=[dict(source_id=sid, start=0, end=len(message['content']), quote=message['content'],
                    content_hash=sha(message['content']), purpose='assertion')])
            db.execute('INSERT INTO governance_facts VALUES (?, ?, ?, ?, ?, ?)',
                (query['user'], generation, fact['id'], fact['subject_id'], sid, json.dumps(fact)))


def paired(tmp_path):
    a = log(tmp_path, 'before.jsonl', [add(user('a'), 'old', 'PRIVATE SOURCE'), get(user('a'))])
    b = log(tmp_path, 'after.jsonl', [add(user('b'), 'new', 'PRIVATE SOURCE'), get(user('b'))])
    return a, b


def audit(path, a, b):
    with read_only_database(path) as db:
        return compare(a, b, IndexReader(db), IndexReader(db))


def test_reordered_duplicate_questions_match_their_own_history(tmp_path):
    before = log(tmp_path, 'before.jsonl', [add(user('a'), 'a', 'Alpha'), add(user('b'), 'b', 'Beta'),
        get(user('a')), get(user('b'))])
    after = log(tmp_path, 'after.jsonl', [add(user('c'), 'c', 'Beta'), add(user('d'), 'd', 'Alpha'),
        get(user('c')), get(user('d'))])
    assert [(a['user'], b['user']) for a, b in pair_queries(before, after)] == [
        (user('a'), user('d')), (user('b'), user('c'))]


def test_log_snapshot_excludes_failed_foreign_future_and_duplicate_writes(tmp_path):
    first = add(user('a'), 'first', 'first')
    data = log(tmp_path, 'input.jsonl', [first, first, add(user('b'), 'foreign', 'foreign'),
        add(user('a'), 'failed', 'failed', 500), get(user('a')), add(user('a'), 'later', 'later'), get(user('a'))])
    assert [[s['message']['content'] for s in q['history']] for q in data['queries']] == [['first'], ['first', 'later']]
    conflict = add(user('a'), 'first', 'different')
    with pytest.raises(ValueError, match='Conflicting successful request'):
        log(tmp_path, 'conflict.jsonl', [first, conflict])


def test_empty_database_is_unverified_not_identical(tmp_path):
    a, b = paired(tmp_path)
    path = tmp_path / 'store.db'
    schema(path)
    result = audit(path, a, b)
    assert result['summary']['memory_comparison_available'] == 0
    assert result['cases'][0]['same_observed_indices'] is None
    assert result['summary']['same_observed_indices'] == 0


def test_generated_ids_and_persistence_clocks_do_not_fake_fact_changes(tmp_path):
    a, b = paired(tmp_path)
    path = tmp_path / 'store.db'
    schema(path)
    for data in (a, b):
        store_query(path, data['queries'][0])
    before = path.read_bytes()
    result = audit(path, a, b)
    assert result['cases'][0]['same_observed_indices'] is True
    assert path.read_bytes() == before
    exported = json.dumps(result)
    assert 'PRIVATE SOURCE' not in exported and 'PRIVATE VALUE' not in exported
    assert 'What is the current location?' not in exported


def test_same_raw_messages_can_have_different_model_derived_indices(tmp_path):
    a, b = paired(tmp_path)
    path = tmp_path / 'store.db'
    schema(path)
    store_query(path, a['queries'][0])
    store_query(path, b['queries'][0], text_suffix=' expanded', vector=0.5, status='partial', value='different')
    result = audit(path, a, b)
    row = result['cases'][0]
    assert row['same_logged_history'] and row['same_observed_indices'] is False
    assert set(row['changed_messages'][0]['changed']) >= {
        'search_text_sha256', 'embedding_sha256', 'fact_content_sha256', 'status'}


def test_reader_uses_active_generation_and_logged_sources_only(tmp_path):
    a, b = paired(tmp_path)
    path = tmp_path / 'store.db'
    schema(path)
    for data in (a, b):
        store_query(path, data['queries'][0], generation=2)
    with sqlite3.connect(path) as db:
        row = db.execute('SELECT * FROM governance_facts WHERE user_id=?', (a['queries'][0]['user'],)).fetchone()
        db.execute('INSERT INTO governance_facts VALUES (?, 1, ?, ?, ?, ?)', (row[0], 'retired', row[3], row[4], row[5]))
        db.execute('INSERT INTO governance_facts VALUES (?, 2, ?, ?, ?, ?)', (row[0], 'future', row[3], 'future-source', row[5]))
        db.execute('INSERT INTO governance_relations VALUES (?, 2, ?, ?, ?)', (row[0], 'future', row[2], 'changes'))
    assert audit(path, a, b)['cases'][0]['same_observed_indices'] is True


def test_reader_rejects_writes_and_does_not_create_missing_database(tmp_path):
    path = tmp_path / 'store.db'
    schema(path)
    with read_only_database(path) as db:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            db.execute('DELETE FROM memories')
    missing = tmp_path / 'missing.db'
    with pytest.raises(sqlite3.OperationalError):
        with read_only_database(missing):
            pass
    assert not missing.exists()


def test_cross_user_source_id_cannot_satisfy_expected_memory(tmp_path):
    a, b = paired(tmp_path)
    path = tmp_path / 'store.db'
    schema(path)
    store_query(path, b['queries'][0])
    with sqlite3.connect(path) as db:
        db.execute('UPDATE memories SET id=? WHERE user_id=?',
            (memory_id('old', 0), b['queries'][0]['user']))
    result = audit(path, a, b)
    assert result['cases'][0]['before']['present_messages'] == 0
    assert result['cases'][0]['same_observed_indices'] is None


def test_cli_saves_incomplete_report_and_refuses_existing_output(tmp_path, monkeypatch):
    from scripts.audit_smoke_indices import main
    import sys
    paired(tmp_path)
    path, output = tmp_path / 'store.db', tmp_path / 'audit.json'
    schema(path)
    monkeypatch.setattr(sys, 'argv', ['audit', '--before-log', str(tmp_path / 'before.jsonl'),
        '--after-log', str(tmp_path / 'after.jsonl'), '--database', str(path), '--output', str(output)])
    assert main() == 2
    assert json.loads(output.read_text(encoding='utf-8'))['summary']['same_observed_indices'] == 0
    with pytest.raises(ValueError, match='new report file'):
        main()


def test_empty_log_and_corrupt_vectors_are_not_verified(tmp_path):
    with pytest.raises(ValueError, match='No successful official'):
        log(tmp_path, 'empty.jsonl', [])
    a, b = paired(tmp_path)
    path = tmp_path / 'store.db'
    schema(path)
    for data in (a, b):
        store_query(path, data['queries'][0])
    with sqlite3.connect(path) as db:
        db.execute('UPDATE memories SET embedding_dim=20')
    result = audit(path, a, b)
    assert result['cases'][0]['same_observed_indices'] is None


def test_current_application_schema_is_supported_without_model_calls_or_mutations(tmp_path):
    from tests.test_governance import make_service, draft
    from app.schemas import MemoryMessage
    text = 'I work at Aster.'
    before = log(tmp_path, 'before.jsonl', [add(user('a'), 'old', text), get(user('a'))])
    after = log(tmp_path, 'after.jsonl', [add(user('b'), 'new', text), get(user('b'))])
    service, sdk = make_service(tmp_path, {text: [draft(text, 'Aster')]})
    for who, request in ((user('a'), 'old'), (user('b'), 'new')):
        service.add(request, [MemoryMessage(role='user', content=text, timestamp=1704067200000)], who, 'session-' + who)
    path = service.settings.database_path
    stored, calls = path.read_bytes(), len(sdk.calls)
    report = audit(path, before, after)
    assert report['summary']['governance_comparison_available'] == 1
    assert report['cases'][0]['same_observed_indices'] is True
    assert report['cases'][0]['before']['fact_count'] == 1
    assert path.read_bytes() == stored and len(sdk.calls) == calls
