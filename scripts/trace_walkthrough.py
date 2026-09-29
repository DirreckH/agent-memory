"""文档走查复现脚本：docs/方案说明.html 第 3 节"用例走查"的全部真实数值。

运行：python -m scripts.trace_walkthrough
使用真实 FastEmbed 向量与固定测量纪元（threads=1、NOW=1.8e12），
每次运行产出与文档完全一致的中间值。
"""

import json
import tempfile
from pathlib import Path

from app.config import Settings
from app.embeddings import FastEmbedder
from app.llm import QueryExpansion
from app.schemas import MemoryMessage
from app.service import MemoryService
from app.storage import SQLiteMemoryStore
from app.temporal import EventAnchorSpec, parse_temporal, time_match

NOW = 1_800_000_000_000
DAY = 86_400_000


class StubLLM:
    enabled = True

    def __init__(self, expansion):
        self.expansion = expansion

    def enrich_messages(self, messages):
        return {}

    def expand_query(self, query, options):
        return self.expansion


settings = Settings(
    _env_file=None, database_path=Path(tempfile.mkdtemp()) / "trace.db"
)
embedder = FastEmbedder(
    model_name=settings.embedding_model,
    cache_dir=settings.embedding_cache_dir,
    threads=1,
)
expansion = QueryExpansion(
    text="搬到滨江路之后养的宠物；新家养的动物",
    temporal=parse_temporal(
        {"event_anchor": {"event": "搬到滨江路", "direction": "after"}}
    ),
)
service = MemoryService(
    settings, SQLiteMemoryStore(settings.database_path), embedder, StubLLM(expansion)
)
service.initialize()

mems = [
    ("我上周搬进了滨江路的新公寓。", 7),
    ("我在新家养了一只虎皮鹦鹉，取名叫团团。", 3),
    ("住在老房子的时候我养过一只仓鼠。", 45),
]
service.add(
    request_id="walkthrough",
    messages=[
        MemoryMessage(role="user", content=c, timestamp=NOW - d * DAY)
        for c, d in mems
    ],
    user_id="user-1",
    session_id="session-1",
)

query = "搬到滨江路之后我养了什么宠物？"
records = service.store.fetch_by_user("user-1")
context = service._resolve_temporal_context(query, expansion.temporal, records)
print("WINDOW:", context.window, "ordering:", context.ordering)
anchor = EventAnchorSpec(event="搬到滨江路", direction="after")
print("ANCHOR WINDOW:", service._resolve_event_anchor_window(anchor, records))

# 事件锚点的定位分数（防线 4：低于 0.30 则放弃约束）
event_vector = embedder.embed([anchor.event])[0]
anchor_weights = settings.normalized_score_weights + (0.0,)
anchor_scores = []
for rec in records:
    score = service._score_record(
        anchor.event, event_vector, rec, weights=anchor_weights, time_match=None
    )
    if score is not None:
        anchor_scores.append((round(score, 4), rec.content))
print("ANCHOR SCORES:", json.dumps(sorted(anchor_scores, reverse=True), ensure_ascii=False))

base_text = query
expanded_text = query + "\n查询扩展：" + expansion.text
vectors = embedder.embed([base_text, expanded_text])
weights = (
    settings.normalized_temporal_weights
    if context.window
    else settings.normalized_score_weights + (0.0,)
)
print("WEIGHTS:", [round(w, 4) for w in weights])

rows = []
for rec in records:
    t = service._record_time_ms(rec)
    tm = (
        time_match(
            t,
            context.window,
            half_life_days=settings.temporal_decay_half_life_days,
        )
        if context.window
        else None
    )
    sA = service._score_record(
        base_text, vectors[0], rec, weights=weights, time_match=tm
    )
    sB = service._score_record(
        expanded_text, vectors[1], rec, weights=weights, time_match=tm
    )
    rows.append(
        {
            "content": rec.content,
            "days_ago": (NOW - t) // DAY if t else None,
            "time_match": round(tm, 4) if tm is not None else None,
            "scoreA": round(sA, 4),
            "scoreB": round(sB, 4),
            "final": round(max(sA, sB), 4),
        }
    )
for row in sorted(rows, key=lambda x: -x["final"]):
    print(json.dumps(row, ensure_ascii=False))

print("SEARCH RESULT:")
for hit in service.search(query=query, user_id="user-1", top_k=5):
    print(
        json.dumps(
            {
                "id": hit.id,
                "content": hit.content,
                "score": hit.score,
                "created_at": hit.created_at,
            },
            ensure_ascii=False,
        )
    )
