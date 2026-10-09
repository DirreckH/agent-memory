"""治理索引管理：备份、迁移、回填、修复、摘要重建、显式代次激活。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import Settings
from app.llm import build_memory_llm
from app.service import MemoryService
from app.storage import SQLiteMemoryStore


class NoEmbedding:
    def embed(self, texts):
        raise RuntimeError("维护任务不允许生成向量或重放 Add")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['backup','migrate','status','backfill','repair','rebuild','activate'])
    parser.add_argument('--database', type=Path)
    parser.add_argument('--user-id')
    parser.add_argument('--generation', type=int)
    parser.add_argument('--expected-revision', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    settings = Settings(governance_mode='shadow', warmup_embedding_on_startup=False)
    if args.database:
        settings = settings.model_copy(update={'database_path': args.database})
    store = SQLiteMemoryStore(settings.database_path)
    if args.action == 'backup':
        if not args.output or not store.path.exists():
            parser.error('backup 需要现存数据库和 --output')
        store.backup_to(args.output)
        print(json.dumps({'backed_up': True}))
        return 0
    store.initialize()
    if args.action == 'migrate':
        print(json.dumps({'initialized': True}))
        return 0
    if not args.user_id:
        parser.error('此操作需要 --user-id')
    llm = build_memory_llm(settings) if args.action in ('backfill','repair') else None
    service = MemoryService(settings, store, NoEmbedding(), llm)
    if args.action == 'backfill':
        generation = args.generation or store.create_index_generation(args.user_id)
        result = service.index_sources(args.user_id, generation=generation)
    elif args.action == 'repair':
        result = service.index_sources(args.user_id, generation=args.generation)
    elif args.action == 'rebuild':
        result = {'published': service.rebuild_summaries(args.user_id, args.generation)}
    elif args.action == 'activate':
        if args.generation is None or args.expected_revision is None:
            parser.error('activate 需要 --generation 和 --expected-revision')
        result = {'activated': store.activate_index_generation(args.user_id, args.generation, args.expected_revision)}
    else:
        snapshot = store.fetch_by_user(args.user_id, governance=True, generation=args.generation)
        result = {'generation': snapshot.generation, 'revision': snapshot.revision,
                  'sources': len(snapshot.sources), 'facts': len(snapshot.facts),
                  'incomplete': len(snapshot.incomplete), 'ready_summaries': len(snapshot.summaries)}
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result.get('committed') is False or result.get('activated') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
