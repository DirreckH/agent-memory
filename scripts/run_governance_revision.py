"""冻结 Git 版本的离线证据对照；不读取官网日志，不修改运行数据库。

语料、人工金标、向量器、扩展和可见预算固定。它不是最终回答评测；
报告保留完整来源及人工核验队列，不能用答案词命中冒充回答正确率。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'tests/fixtures/governance_revision_cases.json'


def worker(args):
    sys.path.insert(0, str(args.root))
    import numpy as np
    import tiktoken
    from app.config import Settings
    from app.llm import LLMConnection, OpenAICompatibleMemoryLLM
    from app.schemas import MemoryMessage
    from app.service import MemoryService
    from app.storage import SQLiteMemoryStore
    corpus = json.loads(DATASET.read_text(encoding='utf-8'))
    encoding = tiktoken.get_encoding('cl100k_base')

    class Embedder:
        # 不使用哈希映射，避免碰撞；所有版本的词表和数值完全相同。
        def tokens(self, text):
            return re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', text.casefold())

        def __init__(self):
            texts = [s['content'] for s in corpus['sources']] + [c['query'] for c in corpus['cases']]
            self.vocab = {t:i for i,t in enumerate(sorted({t for x in texts for t in self.tokens(x)}))}

        def embed(self, texts):
            out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
            for row,text in enumerate(texts):
                for token in self.tokens(text):
                    if token in self.vocab:
                        out[row,self.vocab[token]] += 1
            return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    class SDK:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
            self.queries = 0
            self.facts = {s['content']:s['facts'] for s in corpus['sources']}

        def with_options(self, **kwargs):
            return self

        def create(self, **kwargs):
            payload = json.loads(kwargs['messages'][1]['content'])
            if 'chunks' in payload:
                result = {'items':[{'chunk_id':c['chunk_id'], 'complete':True,
                                    'facts':self.facts.get(c['text'], [])} for c in payload['chunks']]}
            else:
                self.queries += 1
                result = {'expanded_query':''}
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content=json.dumps(result, ensure_ascii=False)))])

    with tempfile.TemporaryDirectory(prefix='gov-revision-db-') as td:
        settings = Settings(_env_file=None, database_path=Path(td)/'memory.db',
            llm_provider='none', llm_add_enrichment=False, llm_search_expansion=True,
            temporal_extraction_mode='rules', temporal_mode='soft',
            min_relevance_score=0.2, multihop_enabled=False,
            governance_mode='off' if args.mode in ('v3','off') else 'shadow')
        llm = OpenAICompatibleMemoryLLM(LLMConnection('fixture','https://example.invalid','fixture'),1,0,False,True)
        sdk = SDK()
        llm._client = sdk
        kwargs = {}
        if args.mode=='preserve':
            from app.governance.retrieval import GovernanceRetriever
            kwargs['governance_retriever'] = GovernanceRetriever(preserve_only=True)
        service = MemoryService(settings, SQLiteMemoryStore(settings.database_path), Embedder(), llm, **kwargs)
        service.initialize()
        ids = {}
        for s in corpus['sources']:
            service.add(s['id'], [MemoryMessage(role='user', content=s['content'], timestamp=s['timestamp'])], 'fixture', s['id'])
            ids[s['id']] = service._memory_id(s['id'], 0)
        if args.mode not in ('v3','off','shadow'):
            service.settings = settings.model_copy(update={'governance_mode':'active'})
        outputs=[]
        for case in corpus['cases']:
            before=sdk.queries
            hits=service.search(case['query'], 'fixture', case['top_k'], query_time_ms=1742083199999)
            # 统一回答上下文预算，只按完整结果收纳，所有组适用同一规则。
            visible=[]; used=0
            for hit in hits:
                cost=len(encoding.encode(hit.content, disallowed_special=()))
                if used+cost <= 8192:
                    visible.append(hit); used+=cost
            text='\n'.join(h.content for h in visible)
            covered=[]
            for sid in case['required_sources']:
                source=next(s for s in corpus['sources'] if s['id']==sid)
                if source['content'] in text or any(f['quote'] in text for f in source['facts']):
                    covered.append(sid)
            outputs.append({'case':case['id'], 'query':case['query'], 'gold':case['answer'], 'reason':case['reason'],
                'required_sources':case['required_sources'], 'covered_sources':covered,
                'required_source_coverage':len(covered)/len(case['required_sources']),
                'query_model_calls':sdk.queries-before, 'visible_tokens':used,
                'answer_correctness':None, 'semantic_review':'pending',
                'hits':[dict(id=h.id,content=h.content,score=h.score,created_at=h.created_at) for h in visible]})
        print(json.dumps({'mode':args.mode, 'source_ids':ids, 'cases':outputs},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--mode', choices=['v3','v4','preserve','full','off','shadow'], default='full')
    p.add_argument('--freeze-only', action='store_true')
    p.add_argument('--report', type=Path, default=ROOT/'data/governance_revision/comparison.json')
    args=p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    if args.worker:
        worker(args); return
    runs={}
    env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    groups=[('v3','9f096cf'),('v4','e90b153')]
    if not args.freeze_only:
        groups += [('off',None),('shadow',None),('preserve',None),('full',None)]
    for mode,revision in groups:
        with tempfile.TemporaryDirectory(prefix='gov-frozen-') as td:
            root=ROOT
            if revision:
                data=subprocess.run(['git','archive','--format=zip',revision],cwd=ROOT,check=True,capture_output=True).stdout
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    archive.extractall(td)
                root=Path(td)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--root',str(root),'--mode',mode]
            result=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'{mode}: {result.stderr[-3000:]}')
            runs[mode]=json.loads(result.stdout)
    report={'dataset_sha256':hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        'baselines':{'v3':'9f096cf','v4':'e90b153'}, 'embedding':'frozen collision-free lexical vectors',
        'expansion':'frozen empty expansion', 'visible_token_budget':8192,
        'final_answer_evaluation':'not performed; source coverage is not answer accuracy', 'runs':runs}
    if not args.freeze_only:
        report['compatibility']={mode:all(
            a['hits']==b['hits'] and a['query_model_calls']==b['query_model_calls']
            for a,b in zip(runs['v3']['cases'],runs[mode]['cases'],strict=True)) for mode in ('off','shadow')}
        if not all(report['compatibility'].values()):
            raise AssertionError('V3 off/shadow compatibility gate failed')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Report: {args.report}')


if __name__=='__main__':
    main()
