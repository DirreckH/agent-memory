"""治理证据打包：预算、来源和基础结果保留由同一入口处理。"""
from collections import defaultdict, deque
from app.governance.extraction import digest, token_count

LABELS = {'current':'当前状态', 'relations':'关系证据', 'history':'历史变化',
          'events':'事件与决定', 'plans':'计划', 'conflicts':'冲突与纠正', 'pending':'待确认原文'}
REQUIREMENTS = {'state_at':'指定时点的状态证据', 'history_until':'历史证据',
                'events_between':'指定时期的事件证据', 'relation':'关系证据', 'summarize':'综合覆盖'}


def assemble(result, baseline, snapshot, *, limit, token_budget):
    if not result.applicable:
        return baseline
    if not result.evidence:
        return []
    from app.service import SearchHit
    header = '[抽取式证据；提及时间不等于生效时间，不推断未知历史区间]'
    if result.missing_requirements:
        header += '\n[覆盖待确认：' + '、'.join(REQUIREMENTS.get(x,x) for x in result.missing_requirements) + ']'
    if result.conflicts:
        header += '\n[存在未解决冲突；不能确定唯一当前值]'
    for warning in result.warnings:
        header += '\n[' + warning + ']'
    buckets=defaultdict(deque); seen=set()
    for item in result.evidence:
        if (item.section,item.text) not in seen:
            buckets[(item.section,item.topic)].append(item)
            seen.add((item.section,item.text))
    ordered=[]
    for requirement in result.covered_requirements:
        for queue in buckets.values():
            match=next((e for e in queue if e.requirement==requirement),None)
            if match:
                ordered.append(match); queue.remove(match); break
    queues={section:deque(key for key,queue in buckets.items() if key[0]==section and queue) for section in LABELS}
    while any(queues.values()):
        for section in LABELS:
            if queues[section]:
                key=queues[section].popleft()
                ordered.append(buckets[key].popleft())
                if buckets[key]:
                    queues[section].append(key)
    chosen=[]; omitted=0
    reserve='\n[内容预算不足；存在完整证据未覆盖]'
    def render(items, suffix=''):
        groups=defaultdict(list)
        for item in items:
            groups[item.section].append(item.text)
        return header+suffix+''.join(f'\n【{label}】\n'+'\n'.join(groups[s])
                                     for s,label in LABELS.items() if groups[s])
    for item in ordered:
        if token_count(render(chosen+[item],reserve)) <= token_budget:
            chosen.append(item)
        else:
            omitted+=1
    if not chosen:
        output=[]; used=0
        for hit in baseline:
            cost=token_count(hit.content)
            if used+cost <= token_budget:
                output.append(hit); used+=cost
        return output[:limit]
    text=render(chosen,reserve if omitted else '')
    ids={sid for item in chosen for sid in item.source_ids}
    anchors=[hit for hit in baseline if hit.id in ids]
    score=max((hit.score for hit in anchors),default=baseline[0].score if baseline else 0.0)
    stamp=anchors[0].created_at if anchors else max((snapshot.sources[s].recorded_at for s in ids),default='')
    uid='gov_'+digest(str((snapshot.user_id,snapshot.generation,snapshot.revision,tuple(sorted(ids)),text)))[:24]
    return [SearchHit(uid,text,score,stamp)][:limit]
