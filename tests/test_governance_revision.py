"""通过服务接口测试人工声明的证据需求；不调用外部模型。"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from app.service import MemoryService
from app.governance.extraction import token_count
from app.governance.retrieval import GovernanceRetriever, GovernanceSearchContext
from app.governance.planning import plan
from tests.test_governance import make_service, add, draft

CORPUS = json.loads((Path(__file__).parent / 'fixtures/governance_revision_cases.json').read_text(encoding='utf-8'))


def test_dated_unindexed_source_preserves_quote_and_date(tmp_path):
    raw = "Riley: I've had these dogs for 3 years."
    service, _ = make_service(tmp_path, {}, temporal_mode='off')
    add(service, 'dogs', raw, 1679922600000)
    hits = service.search('Which year did Riley adopt these dogs?', 'u', 3)
    assert raw in '\n'.join(h.content for h in hits)
    assert '2023-03-27' in '\n'.join(h.content for h in hits)


@pytest.mark.parametrize('case', CORPUS['cases'][:4], ids=lambda c:c['id'])
def test_state_then_relation_chain_uses_sources_not_future_plan(tmp_path, case):
    facts = {s['content']:s['facts'] for s in CORPUS['sources']}
    service, _ = make_service(tmp_path, facts, temporal_mode='off')
    for source in CORPUS['sources']:
        add(service, source['id'], source['content'], source['timestamp'])
    hits = service.search(case['query'], 'u', case['top_k'])
    text = '\n'.join(h.content for h in hits)
    assert '【当前状态】' in text
    assert '【关系证据】' in text
    assert '青禾公司目前的负责人是陈宁。' in text
    assert service._memory_id('M2',0) in text and service._memory_id('M3',0) in text
    assert '林峰' not in text.split('【待确认原文】')[0]
    assert len(hits) <= case['top_k']


@pytest.mark.parametrize('query', ['Please recall the knowledge in my notes.', 'My acknowledgement of the project.',
                                  'Tell me about Acme.', 'I prefer tea.'])
def test_ordinary_queries_are_exactly_baseline(tmp_path, query):
    text='I work at Acme.'
    service,_=make_service(tmp_path,{text:[draft(text,'Acme')]})
    add(service,'ordinary',text,1704067200000)
    base=MemoryService(service.settings.model_copy(update={'governance_mode':'off'}),service.store,service.embedder,service.llm)
    assert service.search(query,'u',3)==base.search(query,'u',3)


def test_later_explicit_correction_and_historical_knowledge_are_different(tmp_path):
    old='I worked at Acme on 2025-03-01.'
    correction='Correction: on 2025-03-01 I actually worked at Beta, not Acme.'
    service,_=make_service(tmp_path,{
        old:[draft(old,'Acme',valid_from='2025-03-01',time_quote=old)],
        correction:[draft(correction,'Beta',operation='correct',target_value='Acme',update_quote=correction,
                          polarity='positive',valid_from='2025-03-01',time_quote=correction)]})
    # 非否定式的完整更正，避免把 not Acme 的极性错误赋给 Beta。
    correction='Correction: the Acme record was mistaken. I worked at Beta on 2025-03-01.'
    service.llm._client.facts[correction]=[draft('I worked at Beta on 2025-03-01.','Beta',
        operation='correct',target_value='Acme',update_quote=correction,
        valid_from='2025-03-01',time_quote='I worked at Beta on 2025-03-01.')]
    add(service,'old',old,1740787200000)
    add(service,'fix',correction,1743465600000)
    truth=service.search('Where did I work as of March 15, 2025?','u',3)[0].content
    known=service.search('Where did I work as of March 15, 2025, according to records available then?','u',3)[0].content
    assert 'Beta' in truth.split('【待确认原文】')[0]
    assert 'Acme' in known and 'Beta' not in known


def test_future_change_does_not_rewrite_earlier_state(tmp_path):
    old,new='I work at Acme.','I now work at Beta.'
    service,_=make_service(tmp_path,{old:[draft(old,'Acme')],new:[draft(new,'Beta')]})
    add(service,'old',old,1740787200000)
    add(service,'future',new,1743465600000)
    text=service.search('Where did I work as of March 15, 2025?','u',2)[0].content
    assert old in text and new not in text


def test_summary_keeps_no_fact_rule_and_numeric_source(tmp_path):
    rule='Rule: approve only if pH is below 7, except when the control sample is missing.'
    data='Experiment: at pH 6, removal was 42%; at pH 8, removal was 13%.'
    state='I work at Acme.'
    service,_=make_service(tmp_path,{state:[draft(state,'Acme')]})
    for i,text in enumerate((rule,data,state)):
        add(service,str(i),text,1704067200000+i*86400000)
    text=service.search('Summarize my entire history, experiments and rules.','u',1)[0].content
    assert rule in text and data in text and state in text


def test_mixed_message_preserves_unindexed_property_without_restoring_employer(tmp_path):
    mixed='I work at Acme. My notebook code is ZX-42.'
    new='I now work at Beta.'
    service,_=make_service(tmp_path,{mixed:[draft('I work at Acme.','Acme')],new:[draft(new,'Beta')]})
    add(service,'mixed',mixed,1704067200000)
    add(service,'new',new,1706745600000)
    text=service.search('What is my current employer and notebook code?','u',3)[0].content
    assert 'ZX-42' in text and 'Beta' in text
    assert 'Acme' not in text.split('【待确认原文】')[0]
    assert '已失效' in text


def test_relation_cooccurrence_cannot_complete_an_edge(tmp_path):
    employer='I work at Beta.'
    unrelated='Network founder is Ada and Network recommends Beta.'
    service,_=make_service(tmp_path,{employer:[draft(employer,'Beta')],unrelated:[
        dict(subject='Network',predicate='founder',value='Ada',quote=unrelated,kind='other',object_entities=['Beta'])]})
    add(service,'employer',employer,1704067200000)
    add(service,'unrelated',unrelated,1706745600000)
    text=service.search('Who founded my current employer?','u',3)[0].content
    assert '【关系证据】' not in text and '覆盖待确认：关系证据' in text


def test_service_temporal_window_is_consumed_without_mutation(tmp_path):
    from app.temporal import TemporalWindow
    event1='I decided to deploy project Aurora.'
    event2='I decided to cancel project Aurora.'
    facts={t:[dict(subject='self',predicate='event',value='project Aurora',quote=t,kind='decision')] for t in (event1,event2)}
    service,_=make_service(tmp_path,facts)
    add(service,'a',event1,1704067200000)
    add(service,'b',event2,1706745600000)
    snapshot=service.store.fetch_by_user('u',governance=True)
    window=TemporalWindow(1704067200000,1706745600000)
    p=plan('Which events happened?',snapshot,window=window)
    context=GovernanceSearchContext('Which events happened?',[],1706745600000,service.settings,3,window=window,plan=p)
    hits=GovernanceRetriever().retrieve(snapshot,context)
    text='\n'.join(h.content for h in hits)
    assert event1 in text and event2 not in text
    assert context.window is window and p.window is window


def test_summary_as_of_keeps_earlier_stage_and_excludes_future_event(tmp_path):
    early='I decided to preserve the audit record.'
    future='I decided to delete the audit record.'
    facts={t:[dict(subject='self',predicate='event',value='audit record',quote=t,kind='decision')] for t in (early,future)}
    service,_=make_service(tmp_path,facts)
    add(service,'early',early,1704067200000)
    add(service,'future',future,1743465600000)
    text='\n'.join(h.content for h in service.search('Summarize my entire history as of March 15, 2025.','u',3))
    assert early in text and future not in text


def test_summary_retains_complete_correction_explanation(tmp_path):
    old='I work at Acme.'
    fix='Correction: I never worked at Acme. I work at Beta.'
    service,_=make_service(tmp_path,{old:[draft(old,'Acme')],fix:[draft('I work at Beta.','Beta',
        operation='correct',target_value='Acme',update_quote=fix)]})
    add(service,'old',old,1704067200000)
    add(service,'fix',fix,1706745600000)
    text=service.search('Summarize my entire employment history.','u',3)[0].content
    assert fix in text
    assert old not in text.split('【历史变化】')[-1].split('【冲突与纠正】')[0]


def test_knowledge_cutoff_with_no_visible_records_never_falls_back_to_future(tmp_path):
    future='I work at Beta.'
    service,_=make_service(tmp_path,{future:[draft(future,'Beta')]})
    add(service,'future',future,1743465600000)
    hits=service.search('Where did I work as of March 15, 2025, according to records available then?','u',3)
    assert all(future not in hit.content for hit in hits)


def test_summary_scope_does_not_include_another_project(tmp_path):
    a='My project Atlas status is delayed.'
    b='My project Boreal status is complete.'
    service,_=make_service(tmp_path,{
        a:[dict(subject='self',predicate='project_status',scope='Atlas',value='delayed',quote=a)],
        b:[dict(subject='self',predicate='project_status',scope='Boreal',value='complete',quote=b)]})
    add(service,'a',a,1704067200000)
    add(service,'b',b,1706745600000)
    text='\n'.join(h.content for h in service.search('Summarize project Atlas.','u',3))
    assert a in text and b not in text


def test_budget_exhaustion_preserves_a_complete_dated_baseline(tmp_path):
    raw='Riley said: I have had the dogs for three years.'
    service,_=make_service(tmp_path,{},governance_query_budget_ms=0)
    add(service,'a',raw,1679922600000)
    hits=service.search('When did Riley adopt the dogs?','u',3)
    assert raw in hits[0].content and '2023-03-27' in hits[0].content
    assert '预算' in hits[0].content


def test_generation_and_source_integrity_are_enforced(tmp_path):
    text='I work at Acme.'
    service,_=make_service(tmp_path,{text:[draft(text,'Acme')]})
    add(service,'a',text,1704067200000)
    add(service,'private','I work at SecretCompany.',1704067200000,user='other')
    snapshot=service.store.fetch_by_user('u',governance=True)
    corrupt=replace(snapshot.facts[0],evidence=(replace(snapshot.facts[0].evidence[0],quote='I work at SecretCompany.'),))
    snapshot=replace(snapshot,facts=(corrupt,))
    query='Where do I currently work?'
    context=GovernanceSearchContext(query,[],1704067200000,service.settings,3)
    assert all('SecretCompany' not in h.content for h in GovernanceRetriever().retrieve(snapshot,context))


def test_relation_alias_validates_chinese_quote_without_assuming_single_value(tmp_path):
    text='白鹭公司的负责人是周宁。'
    service,_=make_service(tmp_path,{text:[dict(subject='白鹭公司',predicate='leader',value='周宁',quote=text,kind='other')]})
    add(service,'leader',text,1740787200000)
    snapshot=service.store.fetch_by_user('u',governance=True)
    assert not snapshot.incomplete and len(snapshot.facts)==1
    hits=service.search('截至 2025-03-15，白鹭公司的负责人是谁？','u',1)
    assert '【当前状态】' in hits[0].content and text in hits[0].content


def test_ambiguous_values_are_preserved_as_conflicts_not_confirmed_state(tmp_path):
    a,b='I work at Alpha.','I work at Beta.'
    service,_=make_service(tmp_path,{a:[draft(a,'Alpha')],b:[draft(b,'Beta')]})
    add(service,'a',a)
    add(service,'b',b)
    text=service.search('Where do I currently work?','u',1)[0].content
    assert '【当前状态】' not in text
    assert '【冲突与纠正】' in text and a in text and b in text
    assert '不能确定唯一当前值' in text
