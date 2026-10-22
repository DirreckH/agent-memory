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
from app.governance.procedural import (
    has_procedure_body, procedural_groups, procedure_role_allowed, procedure_scope_allowed,
    requirement_matches, rule_fact, source_scope_context,
)


@dataclass(frozen=True)
class SummaryEvidence:
    spans: tuple[Evidence, ...]
    facets: frozenset[str]
    labels: tuple[str, ...] = ()
    fact_ids: tuple[str, ...] = ()
    support_key: tuple = ()
    moments: tuple[int, ...] = ()
    # Required exact source ranges, not inferred semantic truth. They travel with
    # the group through compression and are rechecked on the serialized packet.
    required_spans: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class SummaryCollection:
    evidence: tuple[SummaryEvidence, ...] = ()
    warnings: tuple[str, ...] = ()
    requirements: tuple[SummaryRequirement, ...] = ()


_CONTINUATION = re.compile(r'^(?:except\b|exceptions?\b|unless\b|however\b|otherwise\b|but\b|only\b|'
                           r'this\b|that\b|it\b|they\b|he\b|she\b|his\b|her\b|their\b|'
                           r'these\b|those\b|because\b|therefore\b|'
                           r'例外|除非|否则|但|然而|其中|该|这|因为|因此|他|她|其)', re.I)
_FENCE = re.compile(r'^\s{0,3}(`{3,}|~{3,})')
_TABLE_DIVIDER = re.compile(r'^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$')
_PROJECT_TITLE = re.compile(
    r'^\s*Project\s+[\w-]+(?:\s+(?:review|rules?|goals?|decisions?|status|history|'
    r'implementation|rollout|plans?|meeting|notes?|report|format|configuration)){0,5}\s*:?\s*$', re.I)


def _structural_blocks(content):
    """Keep fenced code and a Markdown table intact, including internal newlines."""
    lines = list(re.finditer(r'[^\n]+', content))
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line[0].strip():
            i += 1
            continue
        fence = _FENCE.match(line[0])
        if fence:
            marker = fence[1]
            j = i + 1
            close = re.compile(r'^\s{0,3}' + re.escape(marker[0]) + '{' + str(len(marker)) + r',}\s*$')
            while j < len(lines) and not close.fullmatch(lines[j][0]):
                j += 1
            # An unclosed fence is one whole source tail, never a set of snippets.
            end = lines[j].end() if j < len(lines) else len(content)
            yield line.start(), end, 'structure'
            i = j + 1
            continue
        if ('|' in line[0] and i + 1 < len(lines)
                and _TABLE_DIVIDER.fullmatch(lines[i + 1][0])
                and not content[line.end():lines[i + 1].start()].strip('\r\n')):
            j = i + 2
            while (j < len(lines) and '|' in lines[j][0] and lines[j][0].strip()
                   and content[lines[j - 1].end():lines[j].start()].count('\n') == 1):
                j += 1
            yield line.start(), lines[j - 1].end(), 'structure'
            i = j
            continue
        yield line.start(), line.end(), 'text'
        i += 1


def _heading_level(text):
    markdown = re.match(r'^\s{0,3}(#{1,6})\s+\S', text)
    if markdown:
        return len(markdown[1])
    if _PROJECT_TITLE.fullmatch(text):
        return 1
    if len(text) <= 160 and text.rstrip().endswith((':', '：')):
        return 7
    return None


def source_groups(source, facts=()):
    """Local source units plus explicit heading context; no cross-source inference.

    Only immediately adjacent continuations in the same heading/project context
    are joined. Unrelated preceding paragraphs remain independent candidates.
    """
    ranges = []
    headings = []
    for start, end, kind in _structural_blocks(source.content):
        text = source.content[start:end]
        level = _heading_level(text) if kind == 'text' else None
        if level is not None:
            new_projects = {p.casefold() for p in mentioned_projects(text)}
            old_projects = {p.casefold() for _, a, b in headings
                            for p in mentioned_projects(source.content[a:b])}
            if new_projects and old_projects and new_projects.isdisjoint(old_projects):
                # Markdown nesting is a document relation, not evidence that a
                # different named project belongs to the earlier query scope.
                headings = []
            headings = [(lev, a, b) for lev, a, b in headings if lev < level]
            headings.append((level, start, end))
            continue
        context = tuple((a, b) for _, a, b in headings)
        if kind == 'structure' or token_count(text) <= 240:
            pieces = [(start, end)]
        else:
            cuts = [start]
            for match in re.finditer(r'[。！？!?][”’\"\']?\s*|[.]\s+(?=[A-Z\u3400-\u9fff])', text):
                if re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|vs|e\.g|i\.e)$', text[:match.start()]):
                    continue
                cuts.append(start + match.end())
            cuts.append(end)
            pieces = [(a, b) for a, b in zip(cuts, cuts[1:]) if source.content[a:b].strip()]
        for a, b in pieces:
            previous_projects = mentioned_projects(source.content[ranges[-1][0]:ranges[-1][1]]) if ranges else set()
            projects = mentioned_projects(source.content[a:b])
            same_topic = not previous_projects or not projects or {
                p.casefold() for p in previous_projects} == {p.casefold() for p in projects}
            if (ranges and ranges[-1][2] == context and same_topic and kind == 'text'
                    and _CONTINUATION.search(source.content[a:b].lstrip())
                    and not source.content[ranges[-1][1]:a].strip()):
                old_a, _, old_context = ranges[-1]
                ranges[-1] = (old_a, b, old_context)
            else:
                ranges.append((a, b, context))
    # A non-Markdown title-like message may itself be the complete raw observation.
    if not ranges and headings and not re.match(r'^\s*#', source.content):
        ranges = [(headings[0][1], headings[-1][2], ())]
    for fact in facts:
        for ev in fact.evidence:
            if ev.source_id != source.id:
                continue
            overlap = [entry for entry in ranges if entry[0] < ev.end and entry[1] > ev.start]
            if len(overlap) > 1:
                ranges = [entry for entry in ranges if entry not in overlap]
                a, b = min(r[0] for r in overlap), max(r[1] for r in overlap)
                contexts = tuple(dict.fromkeys(c for r in overlap for c in r[2] if not (a <= c[0] and c[1] <= b)))
                ranges.append((a, b, contexts))
                ranges.sort()
    def evidence(a, b):
        quote = source.content[a:b]
        return Evidence(source.id, a, b, quote, digest(quote))
    return tuple((evidence(a, b), tuple(evidence(c, d) for c, d in context)) for a, b, context in ranges)


def source_spans(source, facts=()):
    """Compatibility view of the primary, structure-preserving source ranges."""
    return tuple(ev for ev, _ in source_groups(source, facts))


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
    by_source = {}
    values_by_slot = {}
    for fact in view.facts:
        if fact.evidence and all(valid_evidence(e, view) for e in fact.evidence):
            by_source.setdefault(fact.source_id, []).append(fact)
            values_by_slot.setdefault(fact.slot, set()).add(fact.value.casefold())
    needs = tuple(getattr(summary_plan, 'procedure_requirements', ()))
    procedures = tuple(r for r in needs if r.kind == 'procedure')
    session_projects, known_projects = source_scope_context(view) if procedures else ({}, set())
    prepared_groups = {}
    procedure_slots = {r.key: set() for r in procedures}
    procedure_fact_ids = {r.key: set() for r in procedures}
    if needs:
        for source in view.sources.values():
            facts = by_source.get(source.id, ())
            groups = procedural_groups(source, source_groups(source, facts), needs)
            prepared_groups[source.id] = groups
            for primary, context in groups:
                spans = (*context, primary)
                text = '\n'.join(ev.quote for ev in spans)
                linked = [f for f in facts if any(e.source_id == source.id and e.start < span.end and e.end > span.start
                                                 for e in f.evidence for span in spans)]
                if not procedure_scope_allowed(text, source, summary_plan, linked,
                        session_projects=session_projects.get(source.session_id, ()), known_projects=known_projects):
                    continue
                for req in procedures:
                    if procedure_role_allowed(source, req) and has_procedure_body(primary, context, req):
                        procedure_slots[req.key].update(f.slot for f in linked if rule_fact(f))
                        procedure_fact_ids[req.key].update(f.id for f in linked if rule_fact(f))
        # A multi-valued rules slot may contain independent procedures. Resolve
        # its complete state, but follow only encoded version/support relations
        # to nominate unnamed successor versions of the requested procedure.
        for req in procedures:
            changed = True
            while changed:
                changed = False
                for rel in view.relations:
                    if rel.relation not in ('corrects', 'retracts', 'changes', 'supports', 'conflicts'):
                        continue
                    if {rel.older, rel.newer} & procedure_fact_ids[req.key]:
                        previous = len(procedure_fact_ids[req.key])
                        procedure_fact_ids[req.key].update((rel.older, rel.newer))
                        changed |= len(procedure_fact_ids[req.key]) != previous
    # Select complete rule slots, including versions whose correction no longer
    # repeats the template name. StateResolver remains the only update policy.
    slots = summary_plan.slots | frozenset(slot for values in procedure_slots.values() for slot in values)
    current = resolver.state_at(view, slots, summary_plan.reference)
    history = resolver.history_until(view, slots, summary_plan.reference)
    current_ids = {f.id for f in current.selected}
    history_ids = {f.id for f in history.selected}
    conflict_ids = {fid for group in current.conflicts for fid in group}
    updates = tuple(e for e in history.update_evidence if e.purpose == 'update' and valid_evidence(e, view))
    parents = continuation_parents(view, summary_plan)
    order_basis = getattr(summary_plan, 'order_basis', 'observation')
    items = []
    for source in view.sources.values():
        facts = by_source.get(source.id, ())
        if not procedures and summary_plan.scopes and not (any(contains(source.content, s) for s in summary_plan.scopes)
                or any(f.slot in summary_plan.slots for f in facts) or source.id in parents):
            continue
        projects_in_source = mentioned_projects(source.content)
        in_scope = source.id in parents or bool(summary_plan.scopes and len(projects_in_source) <= 1
                                                and any(contains(source.content, s) for s in summary_plan.scopes))
        for ev, context in prepared_groups.get(source.id, ()) if needs else source_groups(source, facts):
            group_spans = (*context, ev)
            scope_text = '\n'.join(e.quote for e in group_spans)
            linked = [f for f in facts if any(e.source_id == source.id and e.start < span.end and e.end > span.start
                                             for e in f.evidence for span in group_spans)]
            procedural_matches = tuple(r for r in procedures if procedure_role_allowed(source, r) and (
                has_procedure_body(ev, context, r)
                or any(f.id in procedure_fact_ids[r.key] for f in linked if rule_fact(f))))
            if procedural_matches and not procedure_scope_allowed(scope_text, source, summary_plan, linked,
                    session_projects=session_projects.get(source.session_id, ()), known_projects=known_projects):
                continue
            normal_scope = True
            if summary_plan.scopes:
                explicit_projects = mentioned_projects(ev.quote)
                projects = explicit_projects or mentioned_projects(scope_text)
                if projects:
                    in_scope = bool({p.casefold() for p in projects} & {s.casefold() for s in summary_plan.scopes})
                if (any(contains(ev.quote, s) for s in summary_plan.scopes)
                        or not explicit_projects and any(contains(scope_text, s) for s in summary_plan.scopes)):
                    in_scope = True
                if not in_scope and not any(f.slot in summary_plan.slots for f in linked):
                    normal_scope = False
            elif not summary_plan.global_scope and not (
                    summary_plan.predicates and any(f.slot in summary_plan.slots for f in linked)
                    or any(contains(scope_text, t) for t in summary_plan.focus_terms)):
                normal_scope = False
            if not normal_scope and not procedural_matches:
                continue
            is_update = any(e.source_id == source.id and e.start < ev.end and e.end > ev.start for e in updates)
            if order_basis == 'event':
                supported_dates = [f for f in linked if f.valid_from is not None
                                   and any(e.purpose == 'time' for e in f.evidence)]
                # Effective/event dates are usable only with validated time
                # evidence and an asserted, applicable fact. Message timestamps
                # cannot fill a missing event date or turn a plan into an event.
                dated = [f for f in supported_dates if f.modality == 'asserted' and f.polarity == 'positive'
                         and f.id in current_ids | history_ids and f.id not in conflict_ids]
                moments = [f.valid_from for f in dated]
                # Filtering and display eligibility have different purposes. A
                # future/retracted/planned dated record does not become undated
                # merely because StateResolver did not select it for display.
                filter_moments = [f.valid_from for f in supported_dates]
            elif order_basis == 'message':
                moments = [source.timestamp] if source.timestamp is not None else []
                filter_moments = moments
            else:
                moments = [f.observation_time for f in linked if f.observation_time is not None]
                if not moments and source.timestamp is not None:
                    moments = [source.timestamp]
                filter_moments = moments
            if not is_update and filter_moments:
                if summary_plan.reference is not None and all(t > summary_plan.reference for t in filter_moments):
                    continue
                if summary_plan.window is not None and not any(summary_plan.window.contains(t) for t in filter_moments):
                    continue
            facets = {k for k, (_, pattern) in FACETS.items() if re.search(pattern, scope_text, re.I)}
            facets.add('overview')
            applicable_procedure = False
            matched_rule_facts = [f for f in linked if rule_fact(f)]
            for req in procedural_matches:
                # Raw definitions remain usable, explicitly unverified evidence.
                # A known invalid rule never returns through that raw fallback.
                relevant_rules = [f for f in matched_rule_facts if f.id in procedure_fact_ids[req.key]]
                if not matched_rule_facts or any(f.id in current_ids and f.id not in conflict_ids for f in relevant_rules):
                    facets.add(req.key)
                    applicable_procedure = True
            if normal_scope:
                facets.update(r.key for r in needs if r.kind == 'explicit' and requirement_matches(scope_text, r))
            if moments and order_basis in ('event', 'message'):
                facets.add('timeline')
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
            if procedural_matches and matched_rule_facts and not applicable_procedure:
                labels.append('程序性材料不可作为当前执行规则；保留版本诊断依据')
            elif procedural_matches and not matched_rule_facts:
                labels.append('程序性原文；适用性待核验，不自动视为当前有效指令')
            if order_basis == 'event' and not moments:
                labels.append('已知日期的更新/非确认记录；不推断事件先后' if filter_moments else
                              '事件日期未知；仅保留来源位置，不推断事件先后')
            elif order_basis == 'message' and not moments:
                labels.append('消息日期未知；不推断提及先后')
            for f in linked:
                location = f'{f.evidence[0].start}-{f.evidence[0].end}'
                if procedural_matches and rule_fact(f):
                    # Separate assertion, update, condition and time spans are
                    # one procedure dependency, even when only one matched the
                    # query name. Do not expose a replacement without its basis.
                    dependencies.extend(f.evidence)
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
                if procedural_matches and f.id in current.excluded and current.excluded[f.id] != 'duplicate_support':
                    state = f'非当前执行规则({current.excluded[f.id]})'
                    for rel in view.relations:
                        if rel.older == f.id and rel.relation in ('corrects', 'retracts', 'changes'):
                            dependencies.extend(e for e in rel.evidence if e.purpose == 'update' and valid_evidence(e, view))
                labels.append(f'{location}:{state};归因:{f.subject_name};主体:{f.subject_id}')
                if f.valid_from is not None:
                    labels.append('明确有效日:' + date_text(f.valid_from))
                if f.valid_to is not None:
                    labels.append('有效期结束:' + date_text(f.valid_to))
                if f.valid_from is not None or f.valid_to is not None:
                    dependencies.extend(e for e in f.evidence if e.purpose == 'time')
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
            # A date/update subrange already inside an emitted assertion needs
            # no second quote, but remains an independently encoded dependency.
            required_spans = tuple(dict.fromkeys((ev, *context, *dependencies)))
            spans = tuple(dict.fromkeys((ev, *context)))
            spans += tuple(dependency for dependency in dependencies if not any(
                existing.source_id == dependency.source_id and existing.start <= dependency.start
                and existing.end >= dependency.end for existing in spans))
            spans = tuple(dict.fromkeys(spans))
            items.append(SummaryEvidence(spans, frozenset(facets), tuple(dict.fromkeys(labels)),
                                         tuple(f.id for f in linked), support_key, tuple(sorted(set(moments))), required_spans))
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
                fact_ids=a.fact_ids + b.fact_ids, support_key=(), moments=tuple(sorted(set(a.moments + b.moments))),
                required_spans=tuple(dict.fromkeys((*a.required_spans, *b.required_spans))))
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
                moments=tuple(sorted(set(old.moments + item.moments))),
                required_spans=tuple(dict.fromkeys((*old.required_spans, *item.required_spans))))
        else:
            if item.support_key:
                groups[item.support_key] = len(compressed)
            compressed.append(item)
    requirements = list(summary_plan.requirements)
    times = sorted({t for e in compressed for t in e.moments})
    if summary_plan.order == 'chronological' and len(times) > 1:
        dated_kind = {'event': '明确事件/生效记录', 'message': '消息'}.get(order_basis, '观察')
        requirements += [SummaryRequirement('stage_first', '最早有日期的' + dated_kind, summary_plan.task),
                         SummaryRequirement('stage_last', '最晚有日期的' + dated_kind, summary_plan.task)]
        compressed = [replace(e, facets=e.facets | frozenset(
            (['stage_first'] if times[0] in e.moments else []) +
            (['stage_last'] if times[-1] in e.moments else []) +
            (['changes'] if e.moments and any(r.key == 'changes' for r in requirements) else []))) for e in compressed]
    warnings = list(summary_plan.warnings)
    if summary_plan.order == 'chronological' and order_basis == 'event':
        warnings.append('排序依据为有来源的事件/生效日期；消息时间仅表示来源记录时间')
    elif summary_plan.order == 'chronological' and order_basis == 'message':
        warnings.append('按消息时间展示提及顺序；不推断事件发生先后')
    if view.incomplete:
        warnings.append('部分来源抽取不完整；保留原文证据')
    if any(not e.moments for e in compressed):
        warnings.append('部分片段无适用排序时点；按文档位置展示，不推断事件先后' if order_basis == 'event' else
                        '无可靠日期的片段仅按文档位置展示；不推断事件先后')
    if summary_plan.knowledge_cutoff is not None and any(s.timestamp is None for s in snapshot.sources.values()):
        warnings.append('未知消息日期未用于当时已知信息')
    return SummaryCollection(tuple(compressed), tuple(warnings), tuple(requirements))
