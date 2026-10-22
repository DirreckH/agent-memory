"""G evidence regression: reuse isolated Add/Search replay; grade declared support.

This is a synthetic development diagnostic, not platform or answer accuracy.
Every support obligation refers to complete authored source clauses, including
their conditions, examples, attribution and update dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_memory_joint_regression as replay
from scripts.governance_e2e_support import validate_dataset, visible_context

PROTECTION_BASELINE = 'f97629b9a9e3b4712ba6a5850906a796f286c8a7'
HISTORICAL_BASELINE = 'f930ac62b730b4cd6cd558cd412b0136d3c4437b'
DATASET = ROOT / 'tests/fixtures/g_execution_cases.json'


def source_ids(scene):
    groups, mapped = {}, {}
    for source in scene['sources']:
        key = (source.get('request_id', source['id']), source['session'])
        ordinal = groups.get(key, 0)
        groups[key] = ordinal + 1
        mapped[source['id']] = 'mem_' + hashlib.sha256(f'{key[0]}:{ordinal}'.encode()).hexdigest()[:24]
    return mapped


def support_visible(span, hits, mapped):
    """Require the full clause beside its source reference, or in that raw hit."""
    ident = mapped[span['source_id']]
    header = re.compile(r'\[[^\]\n]*来源:([^:;\]\s]+)(?::\d+-\d+)?[^\]\n]*\]')
    for hit in hits:
        if hit['id'] == ident and span['quote'] in hit['content']:
            return True
        matches = list(header.finditer(hit['content']))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(hit['content'])
            if match[1] == ident and span['quote'] in hit['content'][match.end():end]:
                return True
    return False


def evaluate(scene, question, row, budget):
    hits = row['visible_hits']
    mapped = source_ids(scene)
    serialized, omitted, count = visible_context(hits, budget)
    checks = {'caller_visible_budget': not omitted and serialized == hits and count <= budget}
    groups = []
    for atom in question['gold']['atoms']:
        alternatives = [atom['support'], *atom.get('support_alternatives', [])]
        covered = any(all(support_visible(span, hits, mapped) for span in alternative)
                      for alternative in alternatives)
        groups.append({'claim': atom['claim'], 'complete_source_bound_support': covered})
        if not question.get('expect_base_response'):
            checks['complete_support:' + atom['claim']] = covered
    text = '\n'.join(hit['content'] for hit in hits)
    for source_id in question.get('forbidden_sources', []):
        source = next(source for source in scene['sources'] if source['id'] == source_id)
        checks['exclude_source:' + source_id] = not any(
            hit['id'] == mapped[source_id] or mapped[source_id] in hit['content']
            or source['content'] in hit['content'] for hit in hits)
    for dependency in question.get('dependencies', []):
        triggered = any(fragment in text for fragment in dependency['if_any'])
        checks['dependency:' + dependency['id']] = not triggered or all(
            support_visible(span, hits, mapped) for span in dependency['support'])
    return {'checks': checks, 'support_groups': groups, 'passed': all(checks.values()),
            'answer_correct': None, 'qualifier_semantics': None,
            'boundary': 'Complete authored clauses and their provenance are checked; arbitrary entailment and final answers need human review.'}


def audit_report(dataset, report, budget):
    lookup = {(scene['id'], question['id']): (scene, question)
              for scene in dataset['scenarios'] for question in scene['questions']}
    rows = []
    for row in report['rows']:
        if row['mode'] == 'full':
            scene, question = lookup[(row['scenario'], row['question'])]
            rows.append({'scenario': scene['id'], 'question': question['id'],
                         **evaluate(scene, question, row, budget)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default=PROTECTION_BASELINE)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/g_execution_regression')
    parser.add_argument('--budget', type=int, default=1500)
    parser.add_argument('--scenario', action='append', default=[])
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    dataset = replay.read(args.dataset)
    validate_dataset(dataset)
    if args.validate_only:
        print(replay.canonical({'status': 'valid', 'scenarios': len(dataset['scenarios'])}))
        return 0
    replay_code = replay.compare(args)
    old = audit_report(dataset, replay.read(args.output / 'baseline.json'), args.budget)
    new = audit_report(dataset, replay.read(args.output / 'current.json'), args.budget)
    indexed = {(r['scenario'], r['question']): r for r in old}
    for row in new:
        prior = indexed[(row['scenario'], row['question'])]
        row['change'] = ('improved' if row['passed'] and not prior['passed'] else
                         'regressed' if prior['passed'] and not row['passed'] else 'unchanged')
    result = {'evaluation': 'Synthetic frozen-draft/vector Add/Search; not hidden-platform questions.',
              'dataset_sha256': replay.digest(dataset), 'visible_budget': args.budget,
              'baseline_commit': replay.read(args.output / 'comparison.json')['baseline_commit'],
              'historical_G_reference': HISTORICAL_BASELINE, 'protection_reference': PROTECTION_BASELINE,
              'answer_status': 'not_run', 'answer_correct': None, 'baseline': old, 'current': new}
    replay.write(args.output / 'g_support_comparison.json', result)
    failed = [r for r in new if not r['passed']]
    print(replay.canonical({'g_support_cases': len(new), 'g_support_failures': len(failed)}))
    return int(bool(replay_code or failed))


if __name__ == '__main__':
    raise SystemExit(main())
