from __future__ import annotations

from dataclasses import dataclass, replace
import time
import logging

from app.governance.assembly import assemble
from app.governance.evidence import EvidenceItem, GovernanceResult, fact_item, raw_item, valid_evidence
from app.governance.planning import QueryPlan, plan, needs_governance, relation_key, contains
from app.governance.resolution import StateResolver
from app.governance.summaries import SummaryBuilder, BUILD_VERSION

logger=logging.getLogger(__name__)


@dataclass(frozen=True)
class GovernanceSearchContext:
    query: str
    baseline: list
    reference: int | None
    settings: object
    limit: int
    steps: tuple = ()
    query_texts: tuple[str, ...] = ()
    query_vectors: object = None
    score_record: object = None
    window: object = None
    plan: QueryPlan | None = None
    governance_evidence: GovernanceResult | None = None


class GovernanceRetriever:
    def __init__(self, *, versions=True, summaries=True, preserve_only=False):
        self.versions=versions
        self.summaries=summaries
        self.preserve_only=preserve_only

    def retrieve(self, snapshot, search_context):
        ctx=search_context
        p=ctx.plan or plan(ctx.query,snapshot,reference=ctx.reference,window=ctx.window)
        result=self.collect(snapshot,ctx,p)
        logger.info('治理覆盖 applicable=%s covered=%s missing=%s conflicts=%s',result.applicable,
                    len(result.covered_requirements),len(result.missing_requirements),len(result.conflicts))
        baseline=ctx.baseline
        if result.applicable and p.knowledge_cutoff is not None:
            baseline=[h for h in baseline if h.id in snapshot.sources
                      and snapshot.sources[h.id].timestamp is not None
                      and snapshot.sources[h.id].timestamp<=p.knowledge_cutoff]
        legacy = assemble(result,baseline,snapshot,limit=ctx.limit,
                          token_budget=ctx.settings.governance_content_tokens)
        if (getattr(ctx.settings, 'f1_mode', 'legacy') != 'legacy'
                and self.summaries and self.versions and not self.preserve_only):
            summary_plan, outcome = self.summarize(snapshot, replace(ctx, governance_evidence=result))
            if summary_plan.applicable:
                logger.info('F1 coverage emitted=%s omitted=%s missing=%s',
                            len(outcome.emitted), len(outcome.omitted), len(outcome.missing))
                if outcome.protection is not None:
                    audit = outcome.protection
                    logger.info('F1 protection passed=%s required=%s retained=%s missing=%s',
                                audit.passed, audit.required_units, audit.retained_units, len(audit.missing_requirements))
                if ctx.settings.f1_mode == 'active':
                    # 已识别的时间/身份范围不能因缺证或预算不足而回退到范围外原文。
                    # 无完整可用证据时返回空结果；不生成只有标题的伪摘要。
                    return list(outcome.hits)
        return legacy

    def summarize(self, snapshot, ctx):
        """独立 F1 执行入口；返回不可变审计结果，不在共享实例保存请求状态。"""
        from app.governance.summary_planning import build_summary_plan
        from app.governance.summary_retrieval import collect_summary
        from app.governance.summary_assembly import SummaryOutcome
        from app.governance.summary_guard import guarded_summary
        summary_plan = build_summary_plan(ctx.query, snapshot, reference=ctx.reference, window=ctx.window)
        if not summary_plan.applicable:
            return summary_plan, SummaryOutcome()
        outcome = guarded_summary(collect_summary(snapshot, summary_plan), summary_plan,
            ctx.baseline, snapshot, limit=ctx.limit, token_budget=ctx.settings.governance_content_tokens,
            governance_result=ctx.governance_evidence)
        return summary_plan, outcome

    def collect(self, snapshot, ctx, p):
        if p.intent=='ordinary' or not self.versions and p.intent!='summary':
            return GovernanceResult(False)
        resolver=StateResolver()
        view=resolver.knowledge_view(snapshot,p.knowledge_cutoff)
        deadline=time.perf_counter()+ctx.settings.governance_query_budget_ms/1000
        if ctx.settings.governance_query_budget_ms<=0:
            return GovernanceResult(True,tuple(raw_item(view.sources[h.id]) for h in ctx.baseline if h.id in view.sources),
                                    missing_requirements=p.requirements,warnings=('处理预算不足；保留完整来源待确认',))
        warnings=[]
        if snapshot.incomplete:
            warnings.append(f'覆盖不完整：{len(snapshot.incomplete)} 条来源待确认')
        if p.knowledge_cutoff is not None and any(s.timestamp is None for s in snapshot.sources.values()):
            warnings.append('部分来源消息时间未知；未用于当时已知状态')
        if p.ambiguous_identity:
            warnings.append('身份待确认：同名主体缺少跨会话身份依据')
        items=[]; covered=[]; missing=[]; conflicts=(); excluded={}; spans=[]
        if self.preserve_only:
            items=[raw_item(view.sources[h.id]) for h in ctx.baseline if h.id in view.sources]
            return GovernanceResult(True,tuple(items),missing_requirements=('治理语义未执行',))
        if p.intent=='summary' and self.summaries:
            use_cache=(self.versions and p.window is None and p.knowledge_cutoff is None
                       and p.slots==frozenset(f.slot for f in view.facts))
            units=view.summaries if use_cache and view.summaries and all(
                u.build_version==BUILD_VERSION and u.reference_ms==p.reference for u in view.summaries) else ()
            if not units:
                built=SummaryBuilder(versions=self.versions).build(view,p.slots,
                    budget_ms=min(ctx.settings.governance_summary_budget_ms,max(0,(deadline-time.perf_counter())*1000)),
                    reference=p.reference,time_window=p.window)
                units=built.units
                if not built.complete:
                    warnings.append('摘要覆盖待确认')
            by_id={f.id:f for f in view.facts}
            seen=set()
            for unit in units:
                if unit.level!='session' or unit.section=='directory':
                    continue
                if unit.section=='conflicts':
                    for fid in unit.fact_ids:
                        if fid in by_id:
                            for ev in by_id[fid].evidence:
                                if ev.purpose=='update' and valid_evidence(ev,view):
                                    items.append(replace(raw_item(view.sources[ev.source_id],
                                        '纠正/撤回记录；错误旧说法不作为历史事实',text=ev.quote),section='conflicts'))
                    continue
                for fid in unit.display_fact_ids or unit.fact_ids:
                    if fid in by_id and (unit.section,fid) not in seen:
                        item=fact_item(by_id[fid],view,unit.section,'summarize')
                        if item:
                            items.append(item); seen.add((unit.section,fid))
            (covered if items else missing).append('summarize')
            state=resolver.state_at(view,reference=p.reference)
            excluded=state.excluded; conflicts=state.conflicts
        else:
            if p.intent=='summary':
                state=resolver.state_at(view,p.slots,p.reference)
            elif 'events_between' in p.operations:
                state=resolver.events_between(view,p.slots,p.window)
            elif 'history_until' in p.operations:
                state=resolver.history_until(view,p.slots,p.reference)
            else:
                state=resolver.state_at(view,p.slots,p.reference)
            excluded=state.excluded; conflicts=state.conflicts
            facts=state.selected if self.versions else tuple(f for f in view.facts if f.slot in p.slots)
            section='current' if p.intent in ('current','summary') else 'history'
            requirement=p.operations[0]
            for fact in facts:
                item=fact_item(fact,view,section,requirement)
                if item:
                    items.append(item)
            (covered if items else missing).append(requirement)
            if state.unknown_slots:
                missing.append('当前状态待确认')
            if conflicts:
                if requirement in covered:
                    covered.remove(requirement)
                if requirement not in missing:
                    missing.append(requirement)
                ids={fid for group in conflicts for fid in group}
                for fact in view.facts:
                    if fact.id in ids:
                        item=fact_item(fact,view,'conflicts')
                        if item:
                            items.append(item)
                items=[e for e in items if e.section!='current' or not ids.intersection(e.fact_ids)]
            if p.intent!='current':
                for ev in state.update_evidence:
                    if valid_evidence(ev,view) and ev.purpose=='update':
                        item=raw_item(view.sources[ev.source_id],
                                      '纠正/撤回记录；错误旧说法不作为历史事实',text=ev.quote)
                        items.append(replace(item,section='conflicts'))
            if p.relation:
                extra,reason=self._relations(view,p,state,ctx.settings.multihop_max_rounds)
                for fact in extra:
                    item=fact_item(fact,view,'relations','relation')
                    if item:
                        items.append(item)
                (covered if extra else missing).append('relation')
                if reason:
                    warnings.append(reason)
        by_source={}
        for f in view.facts:
            by_source.setdefault(f.source_id,[]).append(f)
            if f.id in excluded:
                for e in f.evidence[:1]:
                    spans.append((e.source_id,e.start,e.end,excluded[f.id]))
        represented={sid for item in items for sid in item.source_ids}
        source_ids=[h.id for h in ctx.baseline if h.id in view.sources]
        if p.intent=='summary' and p.global_summary:
            source_ids += sorted(view.sources)
        elif p.intent=='summary':
            source_ids += [f.source_id for f in view.facts if f.slot in p.slots]
        for sid in dict.fromkeys(source_ids):
            source=view.sources[sid]
            if p.scope_keys and not any(contains(source.content,k) for k in p.scope_keys):
                continue
            facts=by_source.get(sid,[])
            if p.window is not None and 'events_between' in p.operations and source.timestamp is not None and not p.window.contains(source.timestamp):
                continue
            if p.intent in ('current','summary') and p.reference is not None and source.timestamp is not None and source.timestamp>p.reference and sid not in represented:
                continue
            if sid in view.incomplete or not facts:
                items.append(raw_item(source))
                continue
            ranges=sorted((e.start,e.end) for f in facts for e in f.evidence if valid_evidence(e,view))
            pieces=[]; cursor=0
            for start,end in ranges:
                if start>cursor:
                    pieces.append(source.content[cursor:start])
                cursor=max(cursor,end)
            pieces.append(source.content[cursor:])
            if any(c.isalnum() for c in ''.join(pieces)):
                annotations=[(f.quote,excluded[f.id]) for f in facts if f.id in excluded]
                label='混合原文；部分属性适用性待确认'
                if annotations:
                    label+='；已失效/非当前片段：'+'；'.join(q+' ('+r+')' for q,r in annotations)
                items.append(raw_item(source,label))
            if p.intent=='summary':
                for fact in facts:
                    if fact.modality!='asserted' and fact.slot in p.slots:
                        item=fact_item(fact,view,'plans','summarize')
                        if item:
                            items.append(item)
        if not items:
            blocked_ids={fid for fid,_ in view.blocked_facts}
            for hit in ctx.baseline:
                if hit.id not in view.sources:
                    continue
                source=view.sources[hit.id]
                if p.scope_keys and not any(contains(source.content,k) for k in p.scope_keys):
                    continue
                if p.reference is not None and source.timestamp is not None and source.timestamp>p.reference:
                    continue
                if any(f.id in blocked_ids for f in by_source.get(hit.id,())):
                    items.append(EvidenceItem('pending',f'[来源:{hit.id};系统删除依赖记录] 更新来源已删除；当前状态待确认，不恢复已失效观察。',(hit.id,)))
                else:
                    items.append(raw_item(view.sources[hit.id]))
            if items:
                missing.append('当前状态待确认：仅有原文')
        if time.perf_counter()>deadline:
            warnings.append('处理预算不足；覆盖待确认')
        return GovernanceResult(True,tuple(items),tuple(dict.fromkeys(covered)),tuple(dict.fromkeys(missing)),
                                conflicts,tuple(spans),tuple(warnings))

    @staticmethod
    def _relations(snapshot,p,primary,max_rounds):
        if max_rounds<1 or primary.conflicts or p.ambiguous_identity:
            return (), '关系链待确认：状态或身份不唯一'
        anchors={f.value.casefold() for f in primary.selected if f.predicate in ('employer','residence')}
        if not anchors:
            return (), '关系链缺少适用主体'
        matches=[f for f in snapshot.facts if f.subject_name.casefold() in anchors
                 or any(a.casefold() in anchors for a in f.aliases)]
        for anchor in anchors:
            identities={f.subject_id for f in matches if f.subject_name.casefold()==anchor}
            if len(identities)>1:
                return (), '身份待确认：关系对象存在同名实体'
        slots={f.slot for f in matches if relation_key(f.predicate)==p.relation}
        state=StateResolver().state_at(snapshot,slots,p.reference)
        if state.conflicts:
            return (), '关系链存在冲突'
        return tuple(f for f in state.selected if relation_key(f.predicate)==p.relation), ''
