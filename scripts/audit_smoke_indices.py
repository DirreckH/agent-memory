"""Read-only comparison of persisted indices for two smoke request logs.

Uses only the standard library. Never initializes a service, replays Add, calls a
model, or reconstructs original search responses. Reports hashes/counts, not text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sqlite3


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def memory_id(request, ordinal):
    return 'mem_' + sha(f'{request}:{ordinal}')[:24]


def load_log(path):
    histories, sessions = defaultdict(list), defaultdict(dict)
    seen, queries, counts = {}, [], Counter()
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        record = json.loads(line)
        body = record.get('body', {})
        body = json.loads(body) if isinstance(body, str) else body
        user = body.get('user_id', '')
        if not re.fullmatch(r'u_[0-9a-f]{64}', user):
            continue
        counts[f"{record['path']}:{record['status']}"] += 1
        if record['status'] != 200:
            continue
        if record['path'] == '/set':
            key = user, body['request_id']
            fingerprint = sha(canonical(body))
            if key in seen:
                if seen[key] != fingerprint:
                    raise ValueError(f'Conflicting successful request at {path.name}:{line_no}')
                continue
            seen[key] = fingerprint
            sm = sessions[user]
            sm.setdefault(body['session_id'], len(sm) + 1)
            for ordinal, message in enumerate(body['messages']):
                histories[user].append({'id': memory_id(body['request_id'], ordinal),
                    'line': line_no, 'message': message, 'session': sm[body['session_id']], 'ordinal': ordinal})
        elif record['path'] == '/get':
            queries.append({'line': line_no, 'time': record.get('ts'), 'query': body['query'],
                'user': user, 'history': list(histories[user]),
                'options': {k: v for k, v in body.items() if k not in ('query', 'user_id')}})
    if not queries:
        raise ValueError(f'No successful official /get requests in {path.name}.')
    return {'file': path.name, 'sha256': sha(path.read_bytes()), 'counts': dict(counts), 'queries': queries}


def history(query):
    return [canonical({k: value[k] for k in ('message', 'session', 'ordinal')}) for value in query['history']]


def pair_queries(before, after):
    """Repeated identical questions may belong to different users/corpora."""
    left, right = defaultdict(list), defaultdict(list)
    for q in before['queries']:
        left[q['query']].append(q)
    for q in after['queries']:
        right[q['query']].append(q)
    if Counter({k: len(v) for k, v in left.items()}) != Counter({k: len(v) for k, v in right.items()}):
        raise ValueError('Query multisets differ; align unmatched questions before comparing indices.')
    pairs = []
    for query, group in left.items():
        candidates = right[query]
        if len(group) > 8:
            raise ValueError('More than eight identical queries require explicit disambiguation.')
        counters = {id(q): Counter(history(q)) for q in group + candidates}
        def distance(a, b):
            x, y = counters[id(a)], counters[id(b)]
            return sum((x - y).values()) + sum((y - x).values())
        chosen = min(itertools.permutations(candidates), key=lambda permutation: (
            sum(distance(a, b) for a, b in zip(group, permutation)),
            sum(a['options'] != b['options'] for a, b in zip(group, permutation))))
        pairs.extend(zip(group, chosen))
    pairs.sort(key=lambda pair: pair[0]['line'])
    # A query-level match must also preserve a consistent user partition.
    forward, reverse = defaultdict(set), defaultdict(set)
    for a, b in pairs:
        forward[a['user']].add(b['user'])
        reverse[b['user']].add(a['user'])
    if any(len(v) != 1 for v in (*forward.values(), *reverse.values())):
        raise ValueError('Ambiguous user alignment; inspect repeated query histories manually.')
    return pairs


@contextmanager
def read_only_database(path):
    # mode=ro cannot create a missing DB. query_only adds a second write guard.
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        yield connection
    finally:
        connection.close()


class IndexReader:
    def __init__(self, connection):
        self.connection, self.cache = connection, {}
        self.tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {r['name'] for r in connection.execute('PRAGMA table_info(memories)')}
        if not {'id', 'user_id', 'raw_content', 'role', 'source_timestamp', 'search_text', 'embedding', 'embedding_dim'} <= columns:
            raise ValueError('Unsupported memory schema; this audit does not migrate databases.')

    def user(self, user):
        if user not in self.cache:
            db = self.connection
            memories = {r['id']: dict(r) for r in db.execute('SELECT * FROM memories WHERE user_id=?', (user,))}
            state = (db.execute('SELECT active_generation, revision FROM governance_users WHERE user_id=?', (user,)).fetchone()
                     if 'governance_users' in self.tables else None)
            generation = state['active_generation'] if state else None
            def rows(table):
                if generation is None or table not in self.tables:
                    return []
                return [dict(r) for r in db.execute(
                    f'SELECT * FROM {table} WHERE user_id=? AND generation=?', (user, generation))]
            self.cache[user] = {'memories': memories, 'generation': generation,
                'revision': state['revision'] if state else None,
                'statuses': {r['source_id']: r for r in rows('governance_source_status')},
                'facts': rows('governance_facts'), 'relations': rows('governance_relations'),
                'blocked': rows('governance_blocked_facts')}
        return self.cache[user]

    def inspect(self, query):
        stored = self.user(query['user'])
        refs = {s['id']: f'm{i}' for i, s in enumerate(query['history'])}
        facts_by_source, fact_keys, entities = defaultdict(list), {}, defaultdict(list)
        external_evidence = 0
        for row in stored['facts']:
            if row['source_id'] not in refs:
                continue  # Never include a message written after the logged query.
            fact = json.loads(row['data'])
            if any(ev['source_id'] not in refs for ev in fact.get('evidence', [])):
                external_evidence += 1
                continue
            # Generated IDs / persistence clocks differ between runs. Retain all
            # semantic fields, source times, evidence offsets, and identity context.
            normalized = {k: v for k, v in fact.items()
                          if k not in ('id', 'subject_id', 'recorded_at', 'commit_sequence', 'evidence')}
            normalized['evidence'] = [{**{k: v for k, v in ev.items() if k not in ('source_id', 'quote')},
                'source': refs[ev['source_id']], 'quote_sha256': sha(ev['quote'])} for ev in fact.get('evidence', [])]
            key = sha(canonical(normalized))
            fact_keys[row['id']] = key
            facts_by_source[row['source_id']].append(key)
            entities[row['subject_id']].append(key)
        messages = []
        for source in query['history']:
            sid, expected = source['id'], source['message']
            record, status = stored['memories'].get(sid), stored['statuses'].get(sid)
            item = {'source_line': source['line'], 'source_id': sid, 'present': record is not None}
            if record is not None:
                source_matches = (record['raw_content'] == expected['content'] and record['role'] == expected['role']
                                  and record['source_timestamp'] == expected.get('timestamp'))
                status_data = json.loads(status['data']) if status else None
                item.update(source_matches_log=source_matches,
                    search_text_sha256=sha(record['search_text']), embedding_sha256=sha(record['embedding']),
                    embedding_dim=record['embedding_dim'], embedding_bytes_valid=len(record['embedding']) == 4 * record['embedding_dim'],
                    status=status['status'] if status else 'missing',
                    status_sha256=sha(canonical({k: v for k, v in status_data.items() if k != 'source_id'})) if status_data else None,
                    fact_count=len(facts_by_source[sid]), fact_content_sha256=sha(canonical(sorted(facts_by_source[sid]))))
            messages.append(item)
        relations = sorted((fact_keys[r['newer']], fact_keys[r['older']], r['relation'])
            for r in stored['relations'] if r['newer'] in fact_keys and r['older'] in fact_keys)
        blocked = sorted((fact_keys[r['fact_id']], r['reason']) for r in stored['blocked'] if r['fact_id'] in fact_keys)
        return {'generation': stored['generation'], 'revision_at_audit': stored['revision'],
            'expected_messages': len(messages), 'present_messages': sum(m['present'] for m in messages),
            'status_counts': dict(Counter(m.get('status', 'missing_memory') for m in messages)),
            'fact_count': len(fact_keys), 'external_or_later_evidence_facts_excluded': external_evidence,
            'entity_partition_sha256': sha(canonical(sorted(sorted(v) for v in entities.values()))),
            'relations_sha256': sha(canonical(relations)), 'blocked_sha256': sha(canonical(blocked)), 'messages': messages}


def compare(before, after, old_reader, new_reader):
    cases = []
    for old, new in pair_queries(before, after):
        a, b = old_reader.inspect(old), new_reader.inspect(new)
        same_history = history(old) == history(new)
        diffs = []
        if same_history:
            for x, y in zip(a['messages'], b['messages']):
                keys = ('source_matches_log', 'search_text_sha256', 'embedding_sha256', 'embedding_dim',
                        'embedding_bytes_valid', 'status', 'status_sha256', 'fact_content_sha256')
                changed = [k for k in keys if x.get(k) != y.get(k)] if x['present'] and y['present'] else ['missing_memory']
                if changed:
                    diffs.append({'old_line': x['source_line'], 'new_line': y['source_line'],
                        'old_source_id': x['source_id'], 'new_source_id': y['source_id'], 'changed': changed})
        records_available = bool(a['expected_messages'] and b['expected_messages']) and all(
            all(m['present'] and m['source_matches_log'] and m['embedding_bytes_valid']
                for m in result['messages']) for result in (a, b))
        governance_available = records_available and all(result['generation'] is not None and
            'missing' not in result['status_counts'] and not result['external_or_later_evidence_facts_excluded'] for result in (a, b))
        cases.append({'old_query_line': old['line'], 'new_query_line': new['line'], 'query_sha256': sha(old['query']),
            'same_logged_history': same_history, 'same_query_options': old['options'] == new['options'],
            'memory_comparison_available': records_available, 'governance_comparison_available': governance_available,
            'before': a, 'after': b, 'changed_messages': diffs,
            'same_observed_indices': not diffs and all(a[k] == b[k] for k in
                ('entity_partition_sha256', 'relations_sha256', 'blocked_sha256'))
                if same_history and governance_available else None})
    return {'captured_at': datetime.now(timezone.utc).isoformat(),
        'before_log': {k: v for k, v in before.items() if k != 'queries'},
        'after_log': {k: v for k, v in after.items() if k != 'queries'},
        'boundaries': ['Persisted state at audit time; not original server responses or historical index revisions.',
            'No model calls, query expansion, query vectors, final answers, or official category labels.',
            'Sources are limited to successful writes preceding each query; later repairs/deletions cannot be undone.',
            'Identity is compared by fact grouping; these fingerprints do not establish answer quality.',
            'Repeated exact queries are matched one-to-one by prior-history distance and checked for user consistency.'],
        'summary': {'paired_queries': len(cases), 'same_logged_history': sum(r['same_logged_history'] for r in cases),
            'memory_comparison_available': sum(r['memory_comparison_available'] for r in cases),
            'governance_comparison_available': sum(r['governance_comparison_available'] for r in cases),
            'same_observed_indices': sum(r['same_observed_indices'] is True for r in cases),
            'different_observed_indices': sum(r['same_observed_indices'] is False for r in cases)}, 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-log', type=Path, required=True)
    parser.add_argument('--after-log', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--after-database', type=Path, help='Optional separate after-run backup; otherwise use the same DB.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protected = {p.resolve() for p in (args.before_log, args.after_log, args.database, args.after_database or args.database)}
    if args.output.resolve() in protected or args.output.exists():
        raise ValueError('Output must be a new report file, separate from logs and databases.')
    before, after = load_log(args.before_log), load_log(args.after_log)
    with ExitStack() as stack:
        db = stack.enter_context(read_only_database(args.database))
        other_path = args.after_database or args.database
        other = db if other_path.resolve() == args.database.resolve() else stack.enter_context(read_only_database(other_path))
        report = compare(before, after, IndexReader(db), IndexReader(other))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(canonical(report['summary']))
    return 0 if report['summary']['governance_comparison_available'] == report['summary']['paired_queries'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
