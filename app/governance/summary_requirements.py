"""需求与来源的保守绑定：候选命中不等同于语义完成，更不评价最终答案。"""
import re

from app.governance.planning import contains


def source_support(item, requirement):
    """返回直接来源支持的需求；仅有关键词/标题/问题/无效状态保持待核验。

    source_supported 表示原文包含对应陈述/条件/数据，不保证该陈述为真，
    也不保证跨片段因果或最终综合结论成立。角色与 D1 版本标签必须随文保留。
    """
    key = requirement.key
    # 同批续句也属于该证据单元；后来的更正说明不能借此证明被更正的原说法。
    text = '\n'.join(ev.quote for ev in item.spans if ev.purpose != 'update').strip()
    if key not in item.facets:
        return False
    if key in ('stage_first', 'stage_last'):
        return bool(item.moments)
    if key == 'overview':
        return True
    if text.endswith(('?', '？')):
        return False
    if key == 'current':
        return any('适用观察' in label for label in item.labels) and not any(
            '冲突' in label or '非有效' in label for label in item.labels)
    if key.startswith('aspect:'):
        return all(contains(text, t) for t in requirement.terms) and bool(re.search(
            r'\b(?:was|were|is|are|equals?)\s+[\d.]|[=:]\s*[\d.]|\d\s*(?:%|MPa|kPa|mg|kg|ms|cycles|秒|次)|\|.*\d.*\|', text, re.I))
    patterns = {
        'reasons': r'\b(?:because|due to|as a result of)\b\s+\S.+|因为.+|由于.+|原因[是为：:].+',
        'decisions': r'\b(?:decided|approved|cancelled|canceled|agreed|rejected)\b\s+\S.+|决定.+|取消.+|批准.+',
        'rules': r'\b(?:must|shall|only if|only after|unless|except)\b\s+\S.+|必须.+|除非.+|只有.+|例外[:：].+',
        'exceptions': r'\b(?:unless|except)\b\s+\S.+|\bexception\s*[:：].+|除非.+|例外[:：].+',
        'structure': r'\b(?:use\b|include\b|begin with\b|sections?\s*[:：]|format\s*[:：]|outline\s*[:：]).+|格式[:：].+|结构[:：].+',
        'goals': r'\b(?:goal|objective|aim)\b.{0,30}\b(?:is|was|to)\b\s+\S.+|目标[是为：:].+',
        'plans': r'\b(?:plan|planned|intend)\b.{0,20}\bto\b\s+\S.+|计划.+|打算.+',
        'issues': r'\b(?:pending|blocked|unresolved|risk)\b\s*[:：]?\s+\S.+|未解决.+|待确认.+|风险[:：].+',
        'limitations': r'\b(?:only|except|not tested|uncertainty|cannot|do not infer)\b\s*\S.*|仅限.+|未测试.+|不确定.+|不得.+',
        'results': r'\d\s*(?:%|MPa|kPa|mg|kg|ms|cycles|秒|次)|\b(?:result|finding|outcome)s?\s*[:：].+|结果[:：].+',
        'changes': r'\b(?:changed|shifted|initially|no longer|previously)\b\s+\S.+|从.+变为.+|不再.+',
    }
    return bool(re.search(patterns.get(key, r'(?!)'), text, re.I))
