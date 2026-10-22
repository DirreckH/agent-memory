"""Deterministic task dependencies; term matches nominate evidence, not applicability."""
from __future__ import annotations

from dataclasses import dataclass
import re


_PROCEDURE = r'(?:templates?|formulas?|rules?|methods?|procedures?|rubrics?|checklists?|workflows?|schemas?|formats?|algorithms?|conventions?|protocols?)'
_ZH_PROCEDURE = r'(?:模板|公式|规则|方法|步骤|流程|规范|格式|算法|量表|清单)'
_HYPOTHETICAL = re.compile(r'\b(?:what if|suppose|hypothetically|would I|would we)\b|假如|假设|如果我|如果我们', re.I)
_NON_RECALL = re.compile(
    r'\b(?:should I|should we)\b|'
    r'\b(?:how (?:do|can|should) (?:I|we)|how to)\b[^.?!\n]*'
    r'\b(?:change|set|configure|enable|disable|implement|adjust)\b|'
    r'^\s*(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:recommend|suggest|advise)\b|'
    r'\b(?:give|provide|offer)\s+(?:me|us)\s+(?:some\s+)?(?:advice|recommendations?|suggestions?)\b|'
    r'(?:我|我们)(?:应该|该)|^(?:请)?(?:推荐|建议)|(?:给我|给我们)(?:一些|提供)?建议|如何(?:设置|修改|配置)', re.I)


@dataclass(frozen=True)
class TaskRequirement:
    key: str
    label: str
    trigger: str
    terms: tuple[str, ...]
    mandatory: bool = True
    kind: str = 'procedure'


def matches_requirement_terms(text: str, terms: tuple[str, ...]) -> bool:
    """OR across complete anchors; never OR across words of a target phrase."""
    for term in terms:
        if not term:
            continue
        if term.isascii():
            words = re.split(r'[\s-]+', term.strip())
            pattern = r'[\s-]+'.join(re.escape(w) for w in words)
            if words[-1].isalpha() and not words[-1].endswith('s'):
                pattern += 's?'
            if re.search(r'(?<!\w)' + pattern + r'(?!\w)', text, re.I):
                return True
        elif term.casefold() in text.casefold():
            return True
    return False


def procedure_request(task: str) -> bool:
    """A historical noun phrase alone is not a retrieval/application request."""
    return bool(re.search(r'(?:^|[.!?;,\n])\s*(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?'
                          r'(?:use|using|apply|follow|recall|retrieve|remember|what|which|how|'
                          r'calculate|compute|answer|respond|write|draft|format|according to|based on|'
                          r'summari[sz]e|list|compare|recap)\b|'
                          r'(?:^|[。！？；，\n])\s*(?:现在)?(?:请)?'
                          r'(?:使用|应用|按照|根据|用|回忆|回想|回答|计算|什么|如何|怎样|总结|梳理|列出)', task, re.I))


def quoted_procedure_term(body: str) -> bool:
    """A short noun phrase can be a quoted argument; imperative sentences cannot."""
    return bool(re.fullmatch(rf'(?:[A-Za-z][\w-]*\s+){{0,3}}{_PROCEDURE}', body, re.I)
                and not re.search(r'\b(?:use|apply|follow|ignore|delete|summarize|recap|instructions?)\b', body, re.I))


def _term(raw: str) -> str:
    value = raw.strip(' \t\r\n,.;:!?，。；：！？\"“”「」').casefold()
    if value.isascii():
        words = re.findall(r'[a-z][a-z0-9_-]*', value)
        # Strip request scaffolding while keeping meaningful modifiers together.
        boundaries = {'a', 'an', 'the', 'my', 'our', 'your', 'their', 'its', 'his', 'her', 'this', 'that',
                      'these', 'those', 'use', 'using', 'apply', 'follow', 'configure',
                      'retain', 'include', 'preserve', 'what', 'which', 'was', 'is',
                      'for', 'of', 'with', 'in', 'and', 'or', 'to', 'by'}
        last = max((i for i, word in enumerate(words[:-1]) if word in boundaries), default=-1)
        words = words[last + 1:]
        if not words or len(words) > 5:
            return ''
        if re.fullmatch(_PROCEDURE, words[-1]) and words[-1].endswith('s'):
            words[-1] = words[-1][:-1]
        return ' '.join(words)
    return re.sub(r'^(?:该|这个|那个|我们的|我的|它的|的)', '', value)


def _requirement(term: str, trigger: str, kind: str) -> TaskRequirement:
    slug = re.sub(r'[^\w]+', '_', term).strip('_')
    return TaskRequirement(f'need:{kind}:{slug}', term, trigger, (term,), kind=kind)


def procedure_requirements(task: str) -> tuple[TaskRequirement, ...]:
    """Only explicit historical procedure references create execution dependencies."""
    found = []
    word = r'[A-Za-z][\w-]*'
    patterns = (
        rf'\b(?:agreed|established|saved|learned|taught|previous|prior)\s+'
        rf'(?P<term>(?:{word}\s+){{0,3}}?{_PROCEDURE})\b',
        rf'\b(?P<term>(?:{word}\s+){{0,3}}?{_PROCEDURE})\s+'
        r'(?:(?:that|which)\s+)?(?:I|we|you)\s+(?:(?:previously|earlier|already)\s+)?'
        r'(?:agreed|taught|established|defined|saved|used|specified)\b',
        rf'(?:我们|我|你)?(?:之前|以前|此前|已经|曾经)?(?:约定|教给你|教你|教过你|商定|确定|定义|规定|用过)(?:过)?的?'
        rf'(?P<term>[\u3400-\u9fffA-Za-z0-9_-]{{0,12}}?{_ZH_PROCEDURE})',
        rf'(?:之前|以前|此前|先前|原先)的(?P<term>[\u3400-\u9fffA-Za-z0-9_-]{{0,12}}?{_ZH_PROCEDURE})',
    )
    for pattern in patterns:
        for match in re.finditer(pattern, task, re.I):
            prefix = re.split(r'\bbut\b|\binstead\b|[。；\n]|但是|而是', task[:match.end('term')], flags=re.I)[-1]
            if re.search(r"\b(?:do not|don't|not|never|without|ignore|discard)\b[^.!?;\n]{0,55}$|"
                         r'(?:不要|不用|无需|不按|忽略|不使用)[^。；\n]{0,25}$', prefix, re.I):
                continue
            term = _term(match['term'])
            if term:
                found.append(_requirement(term, match[0], 'procedure'))
    return tuple({r.key: r for r in found}.values())


def memory_task_text(task: str) -> str:
    """Retain recall/application clauses, without admitting pure advice or hypotheticals."""
    clauses = re.split(
        r'[!?;。？！；]\s*|\.\s+(?=[A-Za-z\u3400-\u9fff])|\n+|'
        r',?\s+(?:and\s+(?:then\s+)?|then\s+)(?=(?:recommend|suggest|advise)\b)|'
        r'[，,]?(?:然后|并且|并)(?=(?:建议|推荐))', task, flags=re.I)
    kept = []
    removed = False
    for clause in clauses:
        if not clause.strip():
            continue
        excluded = bool(_HYPOTHETICAL.search(clause) or (
            _NON_RECALL.search(clause) and not procedure_requirements(clause)))
        if excluded:
            removed = True
        else:
            kept.append(clause.strip())
    return '\n'.join(kept) if removed else task


def explicit_requirements(task: str) -> tuple[TaskRequirement, ...]:
    """Encode concrete positive inclusion targets; negative instructions are not evidence."""
    found = []
    for match in re.finditer(r'\b(?:include|retain|preserve)\s+([^.;!?\n]+)|'
                             r'(?:保留|包括|包含)([^。；！？\n]+)', task, re.I):
        if re.search(r"\b(?:do not|don't|never|not)\s*$|(?:不|不要|无需)$", task[:match.start()], re.I):
            continue
        body = match[1] or match[2]
        for item in re.split(r'\s+and\s+|,|、|以及|和|及', body, flags=re.I):
            item = re.split(r'\s+(?:for|from|in|under|using|with)\s+', item, maxsplit=1, flags=re.I)[0]
            term = _term(item)
            # Clausal instructions need a richer parser; do not turn a full sentence
            # or a quantifier into a broad evidence obligation.
            if (not term or term in {'it', 'them', 'everything', 'all', 'details', '全部', '所有', '细节'}
                    or re.search(r'\b(?:should|must|will|can|that|which)\b', term)
                    or not term.isascii() and len(term) > 20):
                continue
            found.append(_requirement(term, match[0], 'explicit'))
    return tuple({r.key: r for r in found}.values())


def task_requirements(task: str) -> tuple[TaskRequirement, ...]:
    return procedure_requirements(task) + explicit_requirements(task)
