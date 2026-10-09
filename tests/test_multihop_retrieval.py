"""通过公开服务接口验证证据驱动的多跳检索；模型/向量器为外部替身。"""

from __future__ import annotations

from types import SimpleNamespace
import json
import pytest

import numpy as np

from app.config import Settings
from app.embeddings import normalize_rows
from app.llm import LLMConnection, NoOpMemoryLLM, OpenAICompatibleMemoryLLM
from app.schemas import MemoryMessage
from app.service import MemoryService
from app.storage import SQLiteMemoryStore
from app.prompts import QUERY_EXPANSION_PROMPT_V3


class WordEmbedder:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), 2048), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = ''.join(c.casefold() if c.isalnum() else ' ' for c in text).split()
            for token in tokens:
                index = self.vocabulary.setdefault(token, len(self.vocabulary))
                matrix[row, index] += 1
        return normalize_rows(matrix)


class PlannedLLM(NoOpMemoryLLM):
    enabled = True

    def expand_query(self, query, options):
        return SimpleNamespace(
            text='', temporal=None,
            retrieval_steps=(
                {'query': 'museum behind my internship', 'evidence': 'museum behind my internship'},
                {'query': 'city hosts the museum', 'evidence': 'city hosts the museum'},
            ),
        )

def test_search_follows_a_bridge_from_original_evidence(tmp_path):
    settings = Settings(
        _env_file=None, database_path=tmp_path / 'memory.db',
        temporal_mode='off', min_relevance_score=0.20,
    )
    service = MemoryService(settings, SQLiteMemoryStore(settings.database_path), WordEmbedder(), PlannedLLM())
    service.initialize()
    messages = [
        MemoryMessage(role='user', content='My internship was arranged by Cedar Museum.'),
        MemoryMessage(role='user', content='Cedar Museum has its headquarters in Birchford.'),
    ] + [
        MemoryMessage(role='user', content=f'Museum exhibit schedule near local city zone {i}.')
        for i in range(20)
    ]
    service.add('bridge-test', messages, 'owner', 'session')
    hits = service.search('Which city hosts the museum behind my internship?', 'owner', 100)
    contents = [h.content for h in hits]
    assert any('arranged by Cedar Museum' in c for c in contents)
    assert any('Birchford' in c for c in contents)


def test_provider_json_plan_drives_grounded_bridge_search(tmp_path):
    class SDKClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            # JSON 模式要求消息中明确出现 JSON；在 SDK 边界复现服务器校验。
            assert any('json' in message['content'].lower() for message in kwargs['messages'])
            payload = json.loads(kwargs['messages'][1]['content'])
            if 'messages' in payload:
                output = {'items': []}
            else:
                assert set(payload) == {'query', 'options'}
                plan = PlannedLLM().expand_query(payload['query'], None)
                output = {'expanded_query': '', 'retrieval_steps': plan.retrieval_steps}
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(output)))])

    llm = OpenAICompatibleMemoryLLM(
        LLMConnection('test-only', 'https://example.invalid/v1', 'test-model'),
        timeout_seconds=1, max_retries=0, add_enrichment=True, search_expansion=True,
    )
    llm.multihop_retrieval = True
    llm._client = SDKClient()
    settings = Settings(_env_file=None, database_path=tmp_path / 'sdk.db',
                        temporal_mode='off', min_relevance_score=0.20)
    service = MemoryService(settings, SQLiteMemoryStore(settings.database_path), WordEmbedder(), llm)
    service.initialize()
    service.add('sdk-plan', [
        MemoryMessage(role='user', content='My internship was arranged by Cedar Museum.'),
        MemoryMessage(role='user', content='Cedar Museum has its headquarters in Birchford.'),
    ], 'owner', 'session')
    hits = service.search('Which city hosts the museum behind my internship?', 'owner', 10)
    assert any('Birchford' in hit.content for hit in hits)


@pytest.mark.parametrize('fault', ['invented_evidence', 'wrong_type', 'duplicate_step', 'too_many_steps'])
def test_invalid_query_plan_preserves_baseline_results(tmp_path, fault):
    class UnverifiedLLM(PlannedLLM):
        def expand_query(self, query, options):
            plan = super().expand_query(query, options)
            steps = list(plan.retrieval_steps)
            if fault == 'invented_evidence':
                steps[0] = {'query': 'Hidden Museum', 'evidence': 'Hidden Museum'}
            elif fault == 'wrong_type':
                steps[0] = {'query': True, 'evidence': query}
            elif fault == 'duplicate_step':
                steps[1] = steps[0]
            else:
                steps = steps * 3
            return SimpleNamespace(text=plan.text, temporal=None, retrieval_steps=steps)

    settings = Settings(_env_file=None, database_path=tmp_path / 'invalid.db',
                        temporal_mode='off', min_relevance_score=0.20)
    store = SQLiteMemoryStore(settings.database_path)
    service = MemoryService(settings, store, WordEmbedder(), UnverifiedLLM())
    service.initialize()
    service.add('invalid', [
        MemoryMessage(role='user', content='My internship was arranged by Cedar Museum.'),
        MemoryMessage(role='user', content='Hidden Museum has its headquarters in Secretford.'),
    ], 'owner', 'session')
    hits = service.search('Which city hosts the museum behind my internship?', 'owner', 10)
    baseline = MemoryService(settings.model_copy(update={'multihop_enabled': False}), store, service.embedder, UnverifiedLLM())
    assert hits == baseline.search('Which city hosts the museum behind my internship?', 'owner', 10)
    assert service.search('Which city hosts the museum behind my internship?', 'other-user', 10) == []


def test_supplement_embedding_failure_preserves_baseline_results(tmp_path):
    class BrokenBatchEmbedder(WordEmbedder):
        def embed(self, texts):
            if len(texts) > 2:
                from app.embeddings import EmbeddingError
                raise EmbeddingError('supplement batch unavailable')
            return super().embed(texts)

    settings = Settings(_env_file=None, database_path=tmp_path / 'failure.db', temporal_mode='off',
                        multihop_embed_goals=True)
    store, embedder = SQLiteMemoryStore(settings.database_path), BrokenBatchEmbedder()
    enhanced = MemoryService(settings, store, embedder, PlannedLLM())
    baseline = MemoryService(settings.model_copy(update={'multihop_enabled': False}), store, embedder, PlannedLLM())
    enhanced.initialize()
    enhanced.add('failure', [MemoryMessage(role='user', content='My internship was arranged by Cedar Museum.')], 'owner', 's')
    query = 'Which city hosts the museum behind my internship?'
    assert enhanced.search(query, 'owner', 10) == baseline.search(query, 'owner', 10)


def test_tiny_top_k_does_not_spend_calls_on_unreturnable_supplements(tmp_path):
    class RecordingEmbedder(WordEmbedder):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def embed(self, texts):
            self.inputs.append(texts)
            return super().embed(texts)

    settings = Settings(_env_file=None, database_path=tmp_path / 'tiny.db', temporal_mode='off', multihop_embed_goals=True)
    embedder = RecordingEmbedder()
    service = MemoryService(settings, SQLiteMemoryStore(settings.database_path), embedder, PlannedLLM())
    service.initialize()
    service.add('tiny', [MemoryMessage(role='user', content='My internship was arranged by Cedar Museum.')], 'owner', 's')
    embedder.inputs.clear()
    query = 'Which city hosts the museum behind my internship?'
    assert len(service.search(query, 'owner', 1)) == 1
    assert embedder.inputs == [[query]]


def test_three_hops_share_one_planning_call_without_sending_raw_sources(tmp_path):
    calls = []

    def completion(**kwargs):
        payload = json.loads(kwargs['messages'][1]['content'])
        calls.append(payload)
        assert set(payload) == {'query', 'options'}
        assert 'Cedar' not in kwargs['messages'][1]['content']
        output = {
            'expanded_query': '',
            'retrieval_steps': [
                {'query': 'museum behind my internship', 'evidence': 'museum behind my internship'},
                {'query': 'sculpture commissioned by the museum', 'evidence': 'sculpture commissioned by the museum'},
                {'query': 'color of sculpture', 'evidence': 'color'},
            ],
        }
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(output)))])

    llm = OpenAICompatibleMemoryLLM(
        LLMConnection('test-only', 'https://example.invalid/v1', 'test-model'),
        timeout_seconds=1, max_retries=0, add_enrichment=False, search_expansion=True,
        multihop_retrieval=True,
    )
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    settings = Settings(_env_file=None, database_path=tmp_path / 'three-hop.db', temporal_mode='off')
    store, embedder = SQLiteMemoryStore(settings.database_path), WordEmbedder()
    service = MemoryService(settings, store, embedder, llm)
    writer = MemoryService(settings, store, embedder, NoOpMemoryLLM())
    writer.initialize()
    writer.add('root', [MemoryMessage(role='user', content='My internship was arranged by Cedar Museum.')], 'owner', 's1')
    writer.add('middle', [MemoryMessage(role='user', content='Cedar Museum commissioned the Azurite artwork.')], 'owner', 's2')
    writer.add('leaf', [MemoryMessage(role='user', content='Azurite artwork glows scarlet.')], 'owner', 's3')
    writer.add('private', [MemoryMessage(role='user', content='Outsider Museum commissioned the Azurite artwork.')], 'other', 's4')
    query = 'What color is the sculpture commissioned by the museum behind my internship?'
    baseline = MemoryService(settings.model_copy(update={'multihop_enabled': False}), store, embedder, NoOpMemoryLLM())
    assert all('scarlet' not in h.content for h in baseline.search(query, 'owner', 10))
    hits = service.search(query, 'owner', 10)
    assert any('scarlet' in h.content for h in hits)
    assert len(calls) == 1


def test_local_bridges_respect_word_boundaries_and_user_isolation(tmp_path):
    settings = Settings(_env_file=None, database_path=tmp_path / 'isolated.db', temporal_mode='off')
    service = MemoryService(settings, SQLiteMemoryStore(settings.database_path), WordEmbedder(), PlannedLLM())
    service.initialize()
    service.add('local-root', [MemoryMessage(role='user', content='My internship was arranged by Cedar Museum.')], 'owner', 's1')
    service.add('substring', [MemoryMessage(role='user', content='Cedarwood estate overlooks Secretford.')], 'owner', 's2')
    service.add('private-leaf', [MemoryMessage(role='user', content='Cedar Museum has its headquarters in Privateford.')], 'other', 's3')
    hits = service.search('Which city hosts the museum behind my internship?', 'owner', 10)
    assert all('Secretford' not in h.content and 'Privateford' not in h.content for h in hits)


@pytest.mark.parametrize('top_k', [5, 10, 100])
def test_original_returned_evidence_keeps_its_order_and_score(tmp_path, top_k):
    settings = Settings(_env_file=None, database_path=tmp_path / 'stable.db', temporal_mode='off')
    store, embedder = SQLiteMemoryStore(settings.database_path), WordEmbedder()
    enhanced = MemoryService(settings, store, embedder, PlannedLLM())
    baseline = MemoryService(settings.model_copy(update={'multihop_enabled': False}), store, embedder, PlannedLLM())
    enhanced.initialize()
    enhanced.add('stable', [
        MemoryMessage(role='user', content='My internship was arranged by Cedar Museum.'),
        MemoryMessage(role='user', content='Cedar Museum has its headquarters in Birchford.'),
    ] + [MemoryMessage(role='user', content=f'Museum exhibit schedule near local city zone {i}.')
         for i in range(20)], 'owner', 'session')
    query = 'Which city hosts the museum behind my internship?'
    original = baseline.search(query, 'owner', top_k)
    result = enhanced.search(query, 'owner', top_k)
    assert result[:len(original)] == original
    assert all(result[i].score >= result[i + 1].score for i in range(len(result) - 1))


def test_pronoun_context_only_expands_within_the_original_write_batch(tmp_path):
    query = 'Where will I meet the person who assigned my grant application?'

    class ContextPlanner(NoOpMemoryLLM):
        enabled = True

        def expand_query(self, query, options):
            return SimpleNamespace(text='', temporal=None, retrieval_steps=[
                {'query': 'person who assigned my grant application', 'evidence': 'person who assigned my grant application'},
                {'query': 'meeting location', 'evidence': 'Where will I meet'},
            ])

    settings = Settings(_env_file=None, database_path=tmp_path / 'context.db', temporal_mode='off')
    store, embedder = SQLiteMemoryStore(settings.database_path), WordEmbedder()
    service = MemoryService(settings, store, embedder, ContextPlanner())
    service.initialize()
    service.add('context-batch', [
        MemoryMessage(role='user', content='Rayna asked me to prepare the grant application.'),
        MemoryMessage(role='user', content='She moved the discussion to her office.'),
        MemoryMessage(role='user', content='We agreed to meet in the blue alcove.'),
    ], 'owner', 'conversation')
    service.add('unrelated-batch', [MemoryMessage(role='user', content='Hidden ceremony at PrivateHouse.')], 'owner', 'conversation')
    baseline = MemoryService(settings.model_copy(update={'multihop_enabled': False}), store, embedder, ContextPlanner())
    original = baseline.search(query, 'owner', 10)
    assert all('blue alcove' not in hit.content for hit in original)
    result = service.search(query, 'owner', 10)
    assert result[:len(original)] == original
    assert any('blue alcove' in hit.content for hit in result)
    assert all('PrivateHouse' not in hit.content for hit in result)


def test_exhausted_local_budget_returns_original_evidence(tmp_path):
    settings = Settings(_env_file=None, database_path=tmp_path / 'budget.db', temporal_mode='off',
                        min_relevance_score=0.30, multihop_embed_goals=True)
    store, embedder = SQLiteMemoryStore(settings.database_path), WordEmbedder()
    normal = MemoryService(settings, store, embedder, PlannedLLM())
    normal.initialize()
    normal.add('budget', [
        MemoryMessage(role='user', content='My internship was arranged by Cedar Museum.'),
        MemoryMessage(role='user', content='Cedar Museum identifies Birchford as its principal city.'),
    ], 'owner', 's')
    query = 'Which city hosts the museum behind my internship?'
    baseline = MemoryService(settings.model_copy(update={'multihop_enabled': False}), store, embedder, PlannedLLM())
    original = baseline.search(query, 'owner', 10)
    assert all('Birchford' not in hit.content for hit in original)
    assert any('Birchford' in hit.content for hit in normal.search(query, 'owner', 10))
    starved = MemoryService(settings.model_copy(update={'multihop_budget_seconds': 1e-12}), store, embedder, PlannedLLM())
    assert starved.search(query, 'owner', 10) == original


@pytest.mark.parametrize('query,expansion,expected_steps', [
    ('Which city hosts the museum behind my internship?',
     'the museum behind my internship; the city where that museum is based', 2),
    ('What drink do I prefer?', 'favorite drink; preferred beverage', 0),
])
def test_default_planning_reuses_v3_request_and_splits_existing_goals(query, expansion, expected_steps):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({'expanded_query': expansion})))])

    llm = OpenAICompatibleMemoryLLM(
        LLMConnection('test-only', 'https://example.invalid/v1', 'test-model'),
        timeout_seconds=1, max_retries=0, add_enrichment=False, search_expansion=True,
        multihop_retrieval=True,
    )
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    result = llm.expand_query(query, None)
    assert result.text == expansion
    assert len(result.retrieval_steps) == expected_steps
    assert calls[0]['messages'][0]['content'] == QUERY_EXPANSION_PROMPT_V3
    assert calls[0]['max_tokens'] == 512
    assert len(calls) == 1
