"""Source-bound procedural dependencies; no independent version or update policy."""
from __future__ import annotations

import re

from app.governance.extraction import digest
from app.governance.models import Evidence
from app.governance.planning import contains
from app.governance.summary_planning import mentioned_projects
from app.governance.task_requirements import matches_requirement_terms


_DEPENDENCY = re.compile(
    r'^\s*(?:#{1,6}\s*)?(?:variables?|definitions?|units?|fields?|columns?|parameters?|'
    r'inputs?|outputs?|examples?|counterexamples?|steps?|conditions?|preconditions?|'
    r'exceptions?|unless|if|when|otherwise|never|do not|must|only|where)\b|'
    r'^\s*(?:变量|定义|单位|字段|参数|输入|输出|示例|反例|步骤|条件|前提|例外|除非|如果|当|否则|禁止)|'
    r'^\s*(?:[-*]\s+|\d+[.)]\s+|[A-Za-z][\w]*\s*=)', re.I)
_FOREIGN_OWNER = re.compile(r"\b([A-Z][A-Za-z-]*(?:\s+[A-Z][A-Za-z-]*)?)['’]s\b")
_SHARED = re.compile(r'\b(?:reusable|general|generic|standard|all projects|every project|'
                     r'all reports|every report)\b|通用|所有项目|全部项目|每个项目|每份报告', re.I)
_EMPTY_DEFINITION = re.compile(r'^\s*(?:not available|unknown|not defined|no (?:template|rule|formula)'
                               r'|未定义|未知|暂无|没有(?:模板|规则|公式))\b', re.I)


def rule_fact(fact):
    return fact.kind == 'rule' or fact.predicate == 'rules'


def requirement_matches(text, requirement):
    return matches_requirement_terms(text, getattr(requirement, 'terms', ()))


def procedure_role_allowed(source, requirement):
    """Respect an explicit teaching speaker; generic 'agreed' stays unconfirmed."""
    trigger = requirement.trigger
    if re.search(r'\bI\s+(?:(?:previously|earlier|already)\s+)?taught\b|我(?:之前|以前|此前)?教', trigger, re.I):
        return source.role == 'user'
    if re.search(r'\byou\s+(?:(?:previously|earlier|already)\s+)?taught\b|你(?:之前|以前|此前)?教', trigger, re.I):
        return source.role == 'assistant'
    return True


def source_scope_context(view):
    sessions = {}
    projects = set()
    for source in view.sources.values():
        names = {name.casefold() for name in mentioned_projects(source.content)}
        sessions.setdefault(source.session_id, set()).update(names)
        projects.update(names)
    return sessions, projects


def procedure_scope_allowed(text, source, plan, linked=(), *, session_projects=(), known_projects=()):
    """Explicit invocation permits a matching unqualified definition as raw evidence.

    A different named project, owner or a known project-specific fact scope is not
    made global by its matching title. Unknown applicability stays labeled by the
    caller; this is candidate eligibility, not permission to execute instructions.
    """
    requested = {scope.casefold() for scope in plan.scopes}
    projects = {name.casefold() for name in mentioned_projects(text)}
    if projects and (not requested or not projects <= requested):
        return False
    for match in _FOREIGN_OWNER.finditer(text):
        if match[1].casefold() not in requested and not contains(plan.task, match[1]):
            return False
    speaker = re.match(r'^\s*([A-Z][a-z]+):\s+(?:my|our|the)\b', text)
    if speaker and speaker[1].casefold() not in requested and not contains(plan.task, speaker[1]):
        return False
    for fact in linked:
        if fact.subject_id != 'self' and fact.subject_name.casefold() not in ('self', '我', '本人'):
            if fact.subject_name.casefold() not in requested and not contains(plan.task, fact.subject_name):
                return False
        fact_projects = {p for p in known_projects if contains(fact.scope, p)}
        if fact_projects and not fact_projects <= requested:
            return False
    if not projects and session_projects and not set(session_projects) <= requested and not _SHARED.search(text):
        return False
    return True


def _merge(source, first, second):
    a, b = first[0], second[0]
    start, end = min(a.start, b.start), max(a.end, b.end)
    quote = source.content[start:end]
    # Keep headings outside the merged body, including common target headings.
    context = tuple(dict.fromkeys(ev for ev in (*first[1], *second[1])
                                  if not (start <= ev.start and ev.end <= end)))
    return Evidence(source.id, start, end, quote, digest(quote)), context


def procedural_groups(source, groups, requirements):
    """Bind only explicit heading sections and locally marked dependencies.

    Unmarked neighboring prose does not become a dependency. The caller still
    applies scope, source validation, temporal and version checks to the group.
    """
    procedures = tuple(r for r in requirements if r.kind == 'procedure')
    if not procedures:
        return groups
    output = []
    active = None
    active_roots = set()
    active_projects = set()
    for primary, context in groups:
        text = '\n'.join(ev.quote for ev in (*context, primary))
        roots = {(ev.start, ev.end) for ev in context if any(requirement_matches(ev.quote, r) for r in procedures)}
        target = any(requirement_matches(text, r) for r in procedures)
        projects = {p.casefold() for p in mentioned_projects(text)}
        common = bool(active_roots & roots)
        dependency = bool(_DEPENDENCY.search(text))
        if active is not None:
            previous = output[active]
            new_starts = [ev.start for ev in (*context, primary) if ev.start >= previous[0].end]
            adjacent = bool(new_starts) and not source.content[previous[0].end:min(new_starts)].strip()
            same_project = not projects or not active_projects or projects == active_projects
            if same_project and (common or dependency and adjacent and not target):
                output[active] = _merge(source, previous, (primary, context))
                active_roots |= roots
                active_projects |= projects
                continue
        output.append((primary, context))
        active = len(output) - 1 if target else None
        active_roots = roots
        active_projects = projects
    return tuple(output)


def has_procedure_body(primary, context, requirement):
    if not primary.quote.strip() or _EMPTY_DEFINITION.search(primary.quote):
        return False
    # A heading alone, or a mention without its definition, cannot satisfy a task.
    if primary.quote.strip().endswith((':', '：')) and '\n' not in primary.quote:
        return False
    if any(requirement_matches(ev.quote, requirement) for ev in context):
        return True
    if not requirement_matches(primary.quote, requirement):
        return False
    # The definition follows its target name. "Project X uses our template" is
    # merely a reference, even though it contains both the name and a verb.
    for term in requirement.terms:
        if term.isascii():
            pattern = r'[\s-]+'.join(re.escape(w) for w in re.split(r'[\s-]+', term.strip()))
            if term[-1:].isalpha() and not term.endswith('s'):
                pattern += 's?'
            matches = re.finditer(r'(?<!\w)' + pattern + r'(?!\w)', primary.quote, re.I)
        else:
            matches = re.finditer(re.escape(term), primary.quote, re.I)
        for match in matches:
            tail = re.split(r'[.!?。！？]', primary.quote[match.end():], maxsplit=1)[0]
            if re.search(r'[:=]|\b(?:is|are|uses?|requires?|must|fields?|keys?|columns?|'
                         r'consists?|return|format|calculate|means|times|steps?|defined)\b|'
                         r'为|是|要求|必须|字段|键|计算|步骤|定义', tail, re.I):
                return True
    return False
