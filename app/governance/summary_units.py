"""抽取式结构单元：所有范围直接指向原文，不重写正文或猜测代码语义。"""
from __future__ import annotations

import re


CONTINUATION = re.compile(
    r'^(?:except\b|unless\b|however\b|otherwise\b|but\b|only\b|this\b|that\b|it\b|they\b|'
    r'because\b|therefore\b|notes?\s*:|footnotes?\s*:|例外|除非|否则|但|然而|其中|该|这|因为|因此|注[:：])', re.I)
_FENCE = re.compile(r'^\s*(`{3,}|~{3,})')
_HEADING = re.compile(r'^\s*(?:#{1,6}\s+.+|(?:Project\s+[\w-]+|项目[\w-]+)\s*[:：]?)\s*$', re.I)


def structural_ranges(text):
    """保留换行段落、围栏代码、表格及其注释；不按 token 数截断语义。

    空行和显式标题才分段。代码围栏内空行不分段；未闭合围栏保留到来源结束。
    过长的不可分单元由打包器整体舍弃并报告缺口，不拆条件/表头/代码语法。
    """
    blocks = []
    start = end = None
    fence = None
    for line in re.finditer(r'[^\n]+', text):
        raw = line[0].rstrip('\r')
        if not raw.strip() and fence is None:
            continue
        marker = _FENCE.match(raw)
        # 两行之间有空白行时，中间至少含两个换行。包含空格的空白行也适用。
        gap = end is not None and text[end:line.start()].count('\n') > 1
        new_heading = bool(_HEADING.fullmatch(raw))
        if start is not None and fence is None and (gap or new_heading):
            blocks.append((start, end))
            start = None
        if start is None:
            start = line.start()
        end = line.start() + len(raw)
        if marker:
            run = marker[1]
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not raw.strip()[len(run):].strip():
                fence = None
    if start is not None:
        blocks.append((start, end))
    joined = []
    for a, b in blocks:
        previous = text[joined[-1][0]:joined[-1][1]].rstrip() if joined else ''
        following = text[a:b].lstrip()
        # 显式新标题是主题边界，不能与前一主题的末尾冒号合并。
        if joined and not _HEADING.match(following.splitlines()[0]) and (
                CONTINUATION.search(following) or previous.endswith((':', '：'))
                or _HEADING.fullmatch(previous)):
            joined[-1] = (joined[-1][0], b)
        else:
            joined.append((a, b))
    return joined


def unit_kind(text):
    if re.search(r'^\s*(`{3,}|~{3,})', text, re.M):
        return 'code'
    if re.search(r'^\s*\|?.*\|.*\n\s*\|?\s*:?-{3,}', text, re.M):
        return 'table'
    if re.search(r'^\s*(?:[-*+]\s|\d+[.)]\s)', text, re.M):
        return 'list'
    return 'paragraph'
