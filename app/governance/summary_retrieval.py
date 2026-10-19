"""F1 来源视图：在同一用户快照中读取事实和原文，保持 D1 为唯一版本解释器。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re

from app.governance.evidence import date_text, valid_evidence
from app.governance.extraction import digest, token_count
from app.governance.models import Evidence
from app.governance.planning import contains
from app.governance.resolution import StateResolver
from app.governance.summary_planning import FACETS, SummaryRequirement, mentioned_projects


@dataclass(frozen=True)
class SummaryEvidence:
    spans: tuple[Evidence, ...]
    facets: frozenset[str]
    labels: tuple[str, ...] = ()
    fact_ids: tuple[str, ...] = ()
    support_key: tuple = ()
    moments: tuple[int, ...] = ()


@dataclass(frozen=True)
class SummaryCollection:
    evidence: tuple[SummaryEvidence, ...] = ()
    warnings: tuple[str, ...] = ()
    requirements: tuple[SummaryRequirement, ...] = ()


_CONTINUATION = re.compile(r'^(?:except\b|unless\b|however\b|otherwise\b|but\b|only\b|'
                           r'this\b|that\b|it\b|they\b|because\b|therefore\b|'
                           r'例外|除非|否则|但|然而|其中|该|这|因为|因此)', re.I)


def source_spans(source, facts=()):
    """保留准确字符范围。只沿段落/句子边界切分；限定与相邻主句共同收纳。"""
    paragraphs = [(m.start(), m.end()) for m in re.finditer(r'[^\n]+', source.content)
                  if m[0].strip()]
    spans = []
    for start, end in paragraphs:
        text = source.content[start:end]
        if token_count(text) <= 240:
            pieces = [(start, end)]
        else:
            # 不在小数、英文缩写内部或分号处断开；不可再分的长句保留完整。
            cuts = [start]
            for m in re.finditer(r'[。！？!?][”’\"\']?\s*|[.]\s+(?=[A-Z\u3400-\u9fff])', text):
                prefix = text[:m.start()]
                if re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|vs|e\.g|i\.e)$', prefix):
                    continue
                cuts.append(start + m.end())
            cuts.append(end)
            pieces = [(a, b) for a, b in zip(cuts, cuts[1:]) if source.content[a:b].strip()]
        for a, b in pieces:
            if spans and (_CONTINUATION.search(source.content[a:b].lstrip())
                          or source.content[spans[-1][0]:spans[-1][1]].rstrip().endswith((':', '：'))
                          or re.fullmatch(r'\s*(?:Project\s+[\w-]+|项目[\w-]+)\s*',
                                          source.content[spans[-1][0]:spans[-1][1]], re.I)):
                spans[-1] = (spans[-1][0], b)
            else:
                spans.append((a, b))
    # 已验证的断言/限定跨越段落时，以完整证据范围为不可分割边界。
    for fact in facts:
        for ev in fact.evidence:
            if ev.source_id != source.id:
                continue
            overlap = [(a, b) for a, b in spans if a < ev.end and b > ev.start]
            if len(overlap) > 1:
                spans = [s for s in spans if s not in overlap]
                spans.append((min(a for a, _ in overlap), max(b for _, b in overlap)))
                spans.sort()
    return tuple(Evidence(source.id, a, b, source.content[a:b], digest(source.content[a:b])) for a, b in spans)


def continuation_parents(view, summary_plan):
    """只连接同批、同会话、相邻且同说话角色的明确续句；不猜跨会话指代。"""
    sessions = {}
    topics = {}
    for s in view.sources.values():
        sessions.setdefault(s.session_id, []).append(s)
        topics.setdefault(s.session_id, set()).update(n.casefold() for n in mentioned_projects(s.content))
    parents = {}
    for session, sources in sessions.items():
        if summary_plan.scopes and topics[session] != {s.casefold() for s in summary_plan.scopes}:
            continue
        sources.sort(key=lambda s: (s.commit_sequence if s.commit_sequence is not None else -1, s.ordinal, s.id))
        for a, b in zip(sources, sources[1:]):
            if (a.commit_sequence is not None and a.commit_sequence == b.commit_sequence and a.ordinal + 1 == b.ordinal
                    and a.role == b.role and _CONTINUATION.search(b.content.lstrip()) and not mentioned_projects(b.content)):
                parents[b.id] = a.id
    return parents


def collect_summary(snapshot, summary_plan):
    resolver = StateResolver()
    view = resolver.knowledge_view(snapshot, summary_plan.knowledge_cutoff)
    current = resolver.state_at(view, summary_plan.slots, summary_plan.reference)
    history = resolver.history_until(view, summary_plan.slots, summary_plan.reference)
    current_ids = {f.id for f in current.selected}
    history_ids = {f.id for f in history.selected}
    conflict_ids = {fid for group in current.conflicts for fid in group}
    by_source = {}
    values_by_slot = {}
    for fact in view.facts:
        if fact.evidence and all(valid_evidence(e, view) for e in fact.evidence):
            by_source.setdefault(fact.source_id, []).append(fact)
            values_by_slot.setdefault(fact.slot, set()).add(fact.value.casefold())
    updates = tuple(e for e in history.update_evidence if e.purpose == 'update' and valid_evidence(e, view))
    parents = continuation_parents(view, summary_plan)
    items = []
    for source in view.sources.values():
        facts = by_source.get(source.id, ())
        if summary_plan.scopes and not (any(contains(source.content, s) for s in summary_plan.scopes)
                or any(f.slot in summary_plan.slots for f in facts) or source.id in parents):
            continue
        projects_in_source = mentioned_projects(source.content)
        in_scope = source.id in parents or bool(summary_plan.scopes and len(projects_in_source) <= 1
                                                and any(contains(source.content, s) for s in summary_plan.scopes))
        for ev in source_spans(source, facts):
            linked = [f for f in facts if any(e.source_id == source.id and e.start < ev.end and e.end > ev.start
                                             for e in f.evidence)]
            if summary_plan.scopes:
                projects = mentioned_projects(ev.quote)
                if projects:
                    in_scope = bool({p.casefold() for p in projects} & {s.casefold() for s in summary_plan.scopes})
                if any(contains(ev.quote, s) for s in summary_plan.scopes):
                    in_scope = True
                if not in_scope and not any(f.slot in summary_plan.slots for f in linked):
                    continue
            elif not summary_plan.global_scope and not (
                    summary_plan.predicates and any(f.slot in summary_plan.slots for f in linked)
                    or any(contains(ev.quote, t) for t in summary_plan.focus_terms)):
                continue
            is_update = any(e.source_id == source.id and e.start < ev.end and e.end > ev.start for e in updates)
            moments = [f.observation_time for f in linked if f.observation_time is not None]
            if not moments and source.timestamp is not None:
                moments = [source.timestamp]
            if not is_update and moments:
                if summary_plan.reference is not None and all(t > summary_plan.reference for t in moments):
                    continue
                if summary_plan.window is not None and not any(summary_plan.window.contains(t) for t in moments):
                    continue
            facets = {k for k, (_, pattern) in FACETS.items() if re.search(pattern, ev.quote, re.I)}
            facets.add('overview')
            if linked:
                if any(f.id in current_ids and f.id not in conflict_ids and f.kind in ('state', 'preference') for f in linked):
                    facets.add('current')
                else:
                    facets.discard('current')
                if any(f.modality != 'asserted' for f in linked):
                    facets.add('plans')
                if any(f.kind == 'rule' for f in linked):
                    facets.add('rules')
                if any(f.kind == 'decision' and f.modality == 'asserted' for f in linked):
                    facets.add('decisions')
            labels = []
            dependencies = []
            for f in linked:
                location = f'{f.evidence[0].start}-{f.evidence[0].end}'
                if f.id in conflict_ids:
                    state = '冲突待确认'
                elif f.id in current_ids:
                    state = '适用观察'
                elif f.id in history_ids:
                    state = '历史观察'
                elif f.id in history.excluded and history.excluded[f.id] != 'not_confirmed_state':
                    state = f'非有效事实({history.excluded[f.id]})'
                    for rel in view.relations:
                        if rel.older == f.id and rel.relation in ('corrects', 'retracts'):
                            dependencies.extend(e for e in rel.evidence if e.purpose == 'update' and valid_evidence(e, view))
                else:
                    state = f'非确认状态({f.modality}/{f.polarity})'
                labels.append(f'{location}:{state};归因:{f.subject_name};主体:{f.subject_id}')
                if f.valid_from is not None:
                    labels.append('明确有效日:' + date_text(f.valid_from))
                if f.valid_to is not None:
                    labels.append('有效期结束:' + date_text(f.valid_to))
            if not linked:
                labels.append('未结构化原文；适用性待核验')
            support_key = ()
            if len(linked) == 1:
                f = linked[0]
                if (f.id in history_ids and f.kind in ('state', 'preference') and f.operation == 'assert'
                        and len(values_by_slot[f.slot]) == 1 and f.quote.strip() == ev.quote.strip()):
                    support_key = (f.slot, f.value, f.kind, f.polarity, f.modality, source.role,
                                   f.identity_context, f.valid_from, f.valid_to, ev.quote.strip())
            # 混合原文中的错误说法必须与更正依据共同收纳，不能单独挤入返回结果。
            spans = tuple(dict.fromkeys((ev, *dependencies)))
            items.append(SummaryEvidence(spans, frozenset(facets), tuple(dict.fromkeys(labels)),
                                         tuple(f.id for f in linked), support_key, tuple(sorted(set(moments)))))
    # 续句与主句共同竞争预算。相邻输入顺序只表示文档关系，不作为事件时间。
    removed = set()
    for child, parent in parents.items():
        child_i = next((i for i, e in enumerate(items) if e.spans[0].source_id == child
                        and not view.sources[child].content[:e.spans[0].start].strip()), None)
        parent_i = next((i for i in range(len(items)-1, -1, -1) if i not in removed and
                         any(e.source_id == parent for e in items[i].spans)), None)
        if child_i is not None and parent_i is not None:
            a, b = items[parent_i], items[child_i]
            items[parent_i] = replace(a, spans=a.spans + b.spans, facets=a.facets | b.facets,
                labels=tuple(dict.fromkeys((*a.labels, *b.labels, '同批相邻续句；不作为事件先后依据'))),
                fact_ids=a.fact_ids + b.fact_ids, support_key=(), moments=tuple(sorted(set(a.moments + b.moments))))
            removed.add(child_i)
    items = [e for i, e in enumerate(items) if i not in removed]
    groups = {}
    compressed = []
    for item in items:
        if item.support_key and item.support_key in groups:
            i = groups[item.support_key]
            old = compressed[i]
            compressed[i] = replace(old, spans=old.spans + item.spans, fact_ids=old.fact_ids + item.fact_ids,
                labels=('重复支持；同值观察，不推断连续有效期', '归因:' + str(item.support_key[0][0]),
                        *(label for label in old.labels if label.startswith(('明确有效日:', '有效期结束:')))),
                moments=tuple(sorted(set(old.moments + item.moments))))
        else:
            if item.support_key:
                groups[item.support_key] = len(compressed)
            compressed.append(item)
    requirements = list(summary_plan.requirements)
    times = sorted({t for e in compressed for t in e.moments})
    if summary_plan.order == 'chronological' and len(times) > 1:
        requirements += [SummaryRequirement('stage_first', '最早有日期的观察', summary_plan.task),
                         SummaryRequirement('stage_last', '最晚有日期的观察', summary_plan.task)]
        compressed = [replace(e, facets=e.facets | frozenset(
            (['stage_first'] if times[0] in e.moments else []) +
            (['stage_last'] if times[-1] in e.moments else []) +
            (['changes'] if e.moments and any(r.key == 'changes' for r in requirements) else []))) for e in compressed]
    warnings = list(summary_plan.warnings)
    if view.incomplete:
        warnings.append('部分来源抽取不完整；保留原文证据')
    if any(not e.moments for e in compressed):
        warnings.append('无可靠日期的片段仅按文档位置展示；不推断事件先后')
    if summary_plan.knowledge_cutoff is not None and any(s.timestamp is None for s in snapshot.sources.values()):
        warnings.append('未知消息日期未用于当时已知信息')
    return SummaryCollection(tuple(compressed), tuple(warnings), tuple(requirements))
