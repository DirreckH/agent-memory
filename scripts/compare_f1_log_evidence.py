"""Compare frozen/current F1 on request-log source snapshots, without model calls.

This is a raw-source packing probe, NOT a reconstruction of server retrieval or
final-answer scores: logs contain neither embeddings, extracted facts nor answers.
Only successful official requests and history available before each query are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASELINE = '0ec14657f7aaf80b5ab873bb156d3b49b507d768'


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def encode(value):
    if hasattr(value, '__dataclass_fields__'):
        return encode(asdict(value))
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [encode(v) for v in (sorted(value, key=str) if isinstance(value, (set, frozenset)) else value)]
    return value


def intervals(spans):
    groups = {}
    for span in spans:
        groups.setdefault(span['source_id'], []).append((span['start'], span['end']))
    for sid, ranges in groups.items():
        merged = []
        for a, b in sorted(ranges):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        groups[sid] = merged
    return groups


def removed_ranges(left, right):
    missing = []
    a, b = intervals(left), intervals(right)
    for sid, ranges in a.items():
        for start, end in ranges:
            remaining = [(start, end)]
            for x, y in b.get(sid, ()):
                remaining = [(p, q) for l, r in remaining
                             for p, q in ((l, min(x, r)), (max(y, l), r)) if p < q]
            missing += [{'source_id': sid, 'start': l, 'end': r} for l, r in remaining]
    return missing


def worker(args):
    sys.path.insert(0, str(args.root))
    from app.governance.models import Source, UserSnapshot
    from app.governance.summary_planning import build_summary_plan
    from app.governance.summary_retrieval import collect_summary
    from app.governance.summary_assembly import assemble_summary
    from app.governance.evidence import valid_evidence
    from app.governance.extraction import token_count
    # Historical roots use their own implementation; current roots exercise the
    # same guard as production, with the deliberately absent baseline/facts.
    assembler = assemble_summary
    if (args.root / 'app/governance/summary_guard.py').exists():
        from app.governance.summary_guard import guarded_summary
        assembler = guarded_summary

    output = []
    for path in args.logs:
        histories, requests = {}, {}
        for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            record = json.loads(line)
            if record.get('status') != 200 or record.get('path') not in ('/set', '/get'):
                continue
            body = record.get('body', {})
            body = json.loads(body) if isinstance(body, str) else body
            user = body.get('user_id', record.get('user_id', ''))
            if not re.fullmatch(r'u_[0-9a-f]{64}', user):
                continue
            sources = histories.setdefault(user, {})
            if record['path'] == '/set':
                request = body['request_id']
                identity = (user, request)
                fingerprint = sha(json.dumps(body, sort_keys=True))
                if identity in requests:
                    if requests[identity] != fingerprint:
                        raise ValueError('Conflicting successful request IDs cannot be reconstructed.')
                    continue
                requests[identity] = fingerprint
                for ordinal, msg in enumerate(body['messages']):
                    sid = f'L{line_no}M{ordinal + 1}'
                    stamp = msg.get('timestamp')
                    if stamp is not None:
                        stamp = int(stamp)
                    sources[sid] = Source(sid, msg['content'], msg['role'], body['session_id'], ordinal,
                                          stamp, record.get('ts', ''), line_no)
                continue
            snapshot = UserSnapshot(user, 1, len(requests), sources=dict(sources))
            reference = max((s.timestamp for s in sources.values() if s.timestamp is not None), default=None)
            query = body['query']
            plan = build_summary_plan(query, snapshot, reference=reference)
            row = {'file': path.name, 'line': line_no, 'query_sha256': sha(query), 'query': query,
                   'source_count': len(sources), 'raw_tokens': token_count('\n'.join(s.content for s in sources.values())),
                   'plan': encode(plan), 'emitted_spans': [], 'packet_tokens': 0, 'hits': []}
            if plan.applicable:
                collection = collect_summary(snapshot, plan)
                result = assembler(collection, plan, [], snapshot,
                                          limit=body.get('top_k', 10), token_budget=args.visible_tokens)
                assert all(valid_evidence(ev, snapshot) for e in result.evidence for ev in e.spans)
                assert result.packet_tokens <= args.visible_tokens
                spans = [ev for e in result.evidence for ev in e.spans]
                row.update(candidate_units=len(collection.evidence), emitted_units=len(result.evidence),
                           coverage=encode(result.coverage), protection=encode(getattr(result, 'protection', None)),
                           packet_tokens=result.packet_tokens,
                           quote_tokens=token_count('\n'.join(ev.quote for ev in spans)),
                           short_primary_units=sum(len(e.spans[0].quote.strip()) <= 10 for e in result.evidence),
                           emitted_spans=[{'source_id': s.source_id, 'start': s.start, 'end': s.end,
                                           'content_hash': s.content_hash} for s in spans], hits=encode(result.hits))
            output.append(row)
    write(args.output / f'{args.worker}.json', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--visible-tokens', type=int, default=8192)
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--worker', choices=('baseline', 'current'), help=argparse.SUPPRESS)
    parser.add_argument('--root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    commit = subprocess.run(['git', 'rev-parse', '--verify', args.baseline + '^{commit}'], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    manifest = {'baseline_commit': commit, 'visible_tokens': args.visible_tokens,
                'modes': {'GOVERNANCE_MODE': 'active', 'F1_MODE': 'active'},
                'probe': 'raw_sources_only_no_facts_no_vectors_no_baseline_hits',
                'source_ids': 'stable short log line/message IDs in both probes',
                'reference': 'maximum prior source timestamp; V3 resolved window unavailable',
                'logs': {p.name: sha(p.read_text(encoding='utf-8-sig')) for p in args.logs},
                'code_sha256': sha(json.dumps({str(p.relative_to(ROOT)): sha(p.read_text(encoding='utf-8-sig'))
                    for p in sorted((ROOT / 'app').rglob('*.py'))}, sort_keys=True)),
                'official_labels_available': False, 'final_answer_correctness': None}
    existing = args.output / 'manifest.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Use a new output directory for changed code/data/budgets.')
    write(existing, manifest)
    with tempfile.TemporaryDirectory(prefix='f1-log-baseline-') as temp:
        archive = subprocess.run(['git', 'archive', '--format=zip', commit], cwd=ROOT, capture_output=True, check=True).stdout
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            z.extractall(temp)
        for mode, root in (('baseline', Path(temp)), ('current', ROOT)):
            subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker', mode,
                '--root', str(root), '--logs', *(str(p.resolve()) for p in args.logs),
                '--output', str(args.output.resolve()), '--visible-tokens', str(args.visible_tokens)],
                cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED='0'), check=True)
    left = json.loads((args.output / 'baseline.json').read_text(encoding='utf-8'))
    right = json.loads((args.output / 'current.json').read_text(encoding='utf-8'))
    assert [(r['file'], r['line'], r['query_sha256']) for r in left] == [
        (r['file'], r['line'], r['query_sha256']) for r in right]
    rows = []
    for a, b in zip(left, right):
        lost, gained = removed_ranges(a['emitted_spans'], b['emitted_spans']), removed_ranges(b['emitted_spans'], a['emitted_spans'])
        rows.append({'file': a['file'], 'line': a['line'], 'baseline_applicable': a['plan']['applicable'],
            'current_applicable': b['plan']['applicable'], 'lost_ranges': lost, 'gained_ranges': gained,
            'lost_characters': sum(s['end'] - s['start'] for s in lost),
            'gained_characters': sum(s['end'] - s['start'] for s in gained),
            'baseline_packet_tokens': a['packet_tokens'], 'current_packet_tokens': b['packet_tokens'],
            'protection': b.get('protection'),
            'packing_probe_only': True, 'answer_quality': 'unverified'})
    write(args.output / 'comparison.json', {'manifest': manifest, 'cases': rows})
    print(json.dumps({'queries': len(rows), 'baseline_applicable': sum(r['baseline_applicable'] for r in rows),
        'current_applicable': sum(r['current_applicable'] for r in rows),
        'cases_with_lost_spans': sum(bool(r['lost_ranges']) for r in rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
