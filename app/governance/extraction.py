from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import tiktoken

from app.governance.models import (Chunk, Evidence, Fact, IndexBatch, IndexBatchDraft,
                                   Source, SourceStatus, ValidatedIndexBatch)
from app.governance.registry import property_spec


class FactIndexError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def encoding():
    # 固定工程计数编码，并非所有提供方的实际计费 tokenizer。
    return tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(encoding().encode(text, disallowed_special=()))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def batch_payload(chunks):
    return {"chunks": [{"chunk_id": c.chunk_id, "source_id": c.source.id,
                        "role": c.source.role, "session_id": c.source.session_id,
                        "ordinal": c.source.ordinal, "source_timestamp": c.source.timestamp,
                        "start": c.start, "end": c.end, "text": c.text,
                        **({"source_format":"legacy stored text; a leading role/date display header is metadata, not effective-date evidence"}
                           if not c.source.raw_available else {})} for c in chunks]}


def input_size(chunks):
    from app.prompts import FACT_INDEX_PROMPT
    return token_count(FACT_INDEX_PROMPT) + token_count(json.dumps(batch_payload(chunks), ensure_ascii=False))


def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


class FactIndexLLM:
    """独立写入调用；输入只能是当前写入的来源块。"""
    def __init__(self, memory_llm):
        self.provider = memory_llm
        self.call_count = 0

    def index_messages(self, batch: IndexBatch) -> IndexBatchDraft:
        from app.prompts import FACT_INDEX_PROMPT
        if not self.provider.enabled or not hasattr(self.provider, "_get_client"):
            raise FactIndexError("Unavailable")
        payload = batch_payload(batch.chunks)
        try:
            client = self.provider._get_client().with_options(
                timeout=batch.timeout_seconds, max_retries=0)
            self.call_count += 1
            response = client.chat.completions.create(
                model=self.provider.connection.model,
                messages=[{"role": "system", "content": FACT_INDEX_PROMPT},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                response_format={"type": "json_object"}, temperature=0,
                max_tokens=batch.output_tokens, stream=False)
            choice = response.choices[0]
            parsed = json.loads(choice.message.content)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
                raise ValueError("InvalidShape")
            return IndexBatchDraft(tuple(parsed["items"]), choice.finish_reason == "stop")
        except Exception as exc:
            # 不透传模型原文或提供方请求内容。
            raise FactIndexError(type(exc).__name__) from None


class FactIndexer:
    def validate(self, draft: IndexBatchDraft, chunks: tuple[Chunk, ...]) -> ValidatedIndexBatch:
        facts: dict[str, Fact] = {}
        statuses = []
        for chunk in chunks:
            items = [i for i in draft.items if isinstance(i, dict) and i.get("chunk_id") == chunk.chunk_id]
            ranges = ()
            state, error = "partial", "MissingOrInvalidOutput"
            if len(items) == 1 and isinstance(items[0].get("facts"), list):
                invalid = False
                for item in items[0]["facts"]:
                    try:
                        fact = self._fact(item, chunk)
                        facts[fact.id] = fact
                    except (ValueError, TypeError, KeyError):
                        invalid = True
                if items[0].get("complete") is True and draft.complete and not invalid:
                    ranges = ((chunk.start, chunk.end),)
                    state = "ready" if items[0]["facts"] else "no_fact"
                    error = None
            statuses.append(SourceStatus(chunk.source.id, state, ranges, error))
        return ValidatedIndexBatch(tuple(facts.values()), tuple(statuses))

    def _fact(self, item: Any, chunk: Chunk) -> Fact:
        if not isinstance(item, dict):
            raise ValueError("InvalidFact")
        required = ("subject", "predicate", "value", "quote")
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required):
            raise ValueError("InvalidField")
        quote = item["quote"]
        legacy_header = (re.match(r"^\[" + re.escape(chunk.source.role) + r"(?:\s+\|\s+[^\]\n]+)?\]\r?\n", chunk.source.content)
                         if not chunk.source.raw_available else None)
        header_end = legacy_header.end() if legacy_header else 0
        start = item.get("start")
        if start is None:
            if chunk.text.count(quote) != 1:
                raise ValueError("AmbiguousQuote")
            start = chunk.start + chunk.text.index(quote)
        end = item.get("end", start + len(quote))
        if (type(start) is not int or type(end) is not int or start < chunk.start or end > chunk.end
                or chunk.source.content[start:end] != quote):
            raise ValueError("InvalidSpan")
        value = item["value"].strip()
        if value.casefold() not in quote.casefold():
            raise ValueError("UngroundedValue")
        enums = {"kind": ("state", "event", "decision", "rule", "preference", "other"),
                 "polarity": ("positive", "negative"),
                 "modality": ("asserted", "planned", "hypothetical", "uncertain"),
                 "operation": ("assert", "change", "correct", "add", "retract")}
        for key, allowed in enums.items():
            if key in item and item[key] not in allowed:
                raise ValueError("InvalidEnum")
        kind = item.get("kind", "state")
        modality = item.get("modality", "asserted")
        polarity = item.get("polarity", "positive")
        before = chunk.source.content[:start]
        after = chunk.source.content[end:]
        # 检查同一语句的限定条件，防止只摘录一个无条件片段。
        sentence = re.split(r"[.!?。！？\n]", before)[-1] + quote + re.split(r"[.!?。！？\n]", after)[0]
        plans = r"\b(plan(?:ning)? to|intend to|will|going to|want to|hope to|might|may)\b|计划|打算|将要|准备|想要|希望"
        conditions = r"\b(if|unless|provided that|hypothetically)\b|如果|假设|除非|前提|仅当"
        uncertain = r"\b(maybe|perhaps|possibly|not sure)\b|可能|或许|不确定|据说"
        negative = r"\b(not|never|no longer|isn't|wasn't|don't|didn't)\b|不再|并非|不是|没有|没在"
        normative_rule = (kind == "rule" and property_spec(item["predicate"]).key == "rules"
                          and re.search(r"\b(rule|policy|must|shall|required|allow|prohibit|exception|do not|never)\b|规则|规定|必须|不得|允许|政策|制度|例外|不要|禁止", sentence, re.I))
        if not normative_rule:
            if re.search(plans, sentence, re.I) and modality == "asserted":
                raise ValueError("PlanMisclassified")
            if re.search(conditions, sentence, re.I) and modality != "hypothetical":
                raise ValueError("ConditionMisclassified")
            if re.search(uncertain, sentence, re.I) and modality == "asserted":
                raise ValueError("UncertaintyMisclassified")
        if polarity == "positive" and re.search(negative, sentence, re.I):
            raise ValueError("NegationMisclassified")
        for pattern in (plans, conditions, uncertain, negative):
            if re.search(pattern, sentence, re.I) and not re.search(pattern, quote, re.I):
                raise ValueError("OmittedQualification")
        subject = item["subject"].strip()
        identity = item.get("identity_context", "")
        if not isinstance(identity, str) or (identity and identity.casefold() not in quote.casefold()):
            raise ValueError("UngroundedIdentity")
        if subject == "self":
            if chunk.source.role != "user" or not re.search(r"\b(I|my|me|we|our)\b|我", quote, re.I):
                raise ValueError("UngroundedSubject")
            attribution = re.split(r"[.!?。！？\n]", before)[-1]
            if (re.search(r"\b(said|says|wrote|told)\b|说|表示|写道", attribution, re.I)
                    and not re.search(r"\bI\s+(said|say|wrote|told)\b|我(?:说|表示|写道)", attribution, re.I)):
                raise ValueError("ReportedSpeakerIsNotSelf")
            subject_id = "self"
        else:
            if subject.casefold() not in quote.casefold():
                raise ValueError("UngroundedSubject")
            subject_id = "entity_" + digest(str((subject.casefold(), identity.casefold() or chunk.source.session_id)))[:20]
        spec = property_spec(item["predicate"])
        scope = item.get("scope") or spec.default_scope
        if not isinstance(scope, str) or not scope.strip() or len(scope) > 256:
            raise ValueError("InvalidScope")
        if spec.cardinality == "unknown" and spec.key not in quote.casefold():
            raise ValueError("UngroundedPredicate")
        if scope not in ("general", "work", "home", "temporary") and scope.casefold() not in quote.casefold():
            raise ValueError("UngroundedScope")
        if scope == "temporary" and not re.search(r"temporary|hotel|business trip|临时|旅馆|酒店|出差", quote, re.I):
            raise ValueError("UngroundedScope")
        objects = item.get("object_entities", [])
        if not isinstance(objects, list) or any(not isinstance(o, str) or not o or o.casefold() not in quote.casefold() for o in objects):
            raise ValueError("UngroundedObject")
        aliases = item.get("aliases", [])
        if (not isinstance(aliases, list) or any(not isinstance(a, str) or not a or a.casefold() not in quote.casefold() for a in aliases)
                or (aliases and not re.search(r"known as|alias|also called|又名|别名|也叫", quote, re.I))):
            raise ValueError("UngroundedAlias")
        ev = Evidence(chunk.source.id, start, end, quote, digest(quote))
        evidence = [ev]
        operation = item.get("operation", "assert")
        target = item.get("target_value")
        target_date = None
        if operation in ("correct", "change", "retract"):
            update = item.get("update_quote")
            cues = {"correct": r"correction|correct|mistake|actually|更正|纠正|说错|不是|口误",
                    "change": r"now|moved|changed|switched|left|joined|现在|换|搬|离开|加入|改为",
                    "retract": r"retract|withdraw|no longer|撤回|不再|作废|取消"}
            if (not isinstance(update, str) or chunk.text.count(update) != 1
                    or not re.search(cues[operation], update, re.I)
                    or (target is not None and (not isinstance(target, str) or target.casefold() not in update.casefold()))):
                raise ValueError("UngroundedUpdate")
            offset = chunk.start + chunk.text.index(update)
            evidence.append(Evidence(chunk.source.id, offset, offset + len(update), update, digest(update), "update"))
            if item.get("target_date") is not None:
                date_text = item["target_date"]
                if not isinstance(date_text, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text) or date_text not in update:
                    raise ValueError("UngroundedTargetDate")
                if offset + update.index(date_text) < header_end:
                    raise ValueError("DisplayTimestampIsNotTargetDate")
                target_date = int(datetime.fromisoformat(date_text).replace(tzinfo=timezone.utc).timestamp() * 1000)
        dates = []
        time_quote = item.get("time_quote")
        for key in ("valid_from", "valid_to"):
            raw_date = item.get(key)
            if raw_date is None:
                dates.append(None)
                continue
            if (not isinstance(raw_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date)
                    or not isinstance(time_quote, str) or chunk.text.count(time_quote) != 1
                    or raw_date not in time_quote):
                raise ValueError("UngroundedDate")
            if chunk.start + chunk.text.index(time_quote) < header_end:
                raise ValueError("DisplayTimestampIsNotEffectiveDate")
            when = datetime.strptime(raw_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            dates.append(int(when.timestamp() * 1000))
        if dates[0] is not None and dates[1] is not None and dates[1] <= dates[0]:
            raise ValueError("InvalidInterval")
        if any(d is not None for d in dates):
            offset = chunk.start + chunk.text.index(time_quote)
            evidence.append(Evidence(chunk.source.id, offset, offset + len(time_quote), time_quote, digest(time_quote), "time"))
        fid = "fact_" + digest(json.dumps([chunk.source.id, start, end, subject_id, spec.key, scope,
                                           value, polarity, modality, operation, target, dates, target_date], ensure_ascii=False))[:24]
        return Fact(fid, subject_id, subject, spec.key, scope,
                    value, kind, polarity, modality, operation, tuple(evidence),
                    chunk.source.timestamp, chunk.source.recorded_at, chunk.source.commit_sequence,
                    chunk.source.ordinal, dates[0], dates[1], "day" if any(d is not None for d in dates) else "unknown",
                    target_value=item.get("target_value"),
                    object_entities=tuple(objects), identity_context=identity,
                    current_observation=bool(re.search(r"\b(now|currently)\b|现在|目前", quote, re.I)), aliases=tuple(aliases),
                    target_date=target_date)

    def extract(self, sources: list[Source], model: FactIndexLLM, settings) -> ValidatedIndexBatch:
        started = time.perf_counter()
        deadline = started + settings.governance_add_budget_seconds
        facts, states = {}, {s.id: [] for s in sources}
        batch = []

        def submit():
            if not batch:
                return
            chunks = tuple(batch)
            batch.clear()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                for c in chunks:
                    states[c.source.id].append(SourceStatus(c.source.id, "pending", error_type="BudgetExhausted"))
                return
            try:
                response = model.index_messages(IndexBatch(chunks, min(remaining, settings.llm_timeout_seconds),
                                                           settings.governance_output_tokens))
                valid = self.validate(response, chunks)
                for f in valid.facts:
                    facts[f.id] = f
                for status in valid.statuses:
                    states[status.source_id].append(status)
            except FactIndexError as exc:
                for c in chunks:
                    states[c.source.id].append(SourceStatus(c.source.id, "failed", error_type=str(exc)))

        for source in sources:
            offset, serial = 0, 0
            while offset < len(source.content):
                if time.perf_counter() >= deadline:
                    states[source.id].append(SourceStatus(source.id, "pending", error_type="BudgetExhausted"))
                    break
                # 先用字符二分找满足工程 token 上限的块，再回退到最近句/段边界。
                lo, hi = offset + 1, len(source.content)
                end = offset
                while lo <= hi:
                    mid = (lo + hi) // 2
                    chunk = Chunk(f"{source.id}:{serial}", source, offset, mid)
                    if input_size((chunk,)) <= settings.governance_input_tokens:
                        end, lo = mid, mid + 1
                    else:
                        hi = mid - 1
                if end == offset:
                    states[source.id].append(SourceStatus(source.id, "failed", error_type="InputBudgetTooSmall"))
                    break
                if end < len(source.content):
                    boundaries = list(re.finditer(r"[.!?。！？](?:\s+|$)|\n", source.content[offset:end]))
                    if boundaries:
                        end = offset + boundaries[-1].end()
                chunk = Chunk(f"{source.id}:{serial}", source, offset, end)
                if batch and input_size((*batch, chunk)) > settings.governance_input_tokens:
                    submit()
                batch.append(chunk)
                # 重叠最多一个完整句子且不超过输入预算的 10%，保留绝对偏移。
                previous = list(re.finditer(r"[.!?。！？](?:\s+|$)|\n", source.content[offset:end]))
                next_offset = end
                if end < len(source.content) and len(previous) >= 2:
                    overlap = offset + previous[-2].end()
                    if overlap > offset and token_count(source.content[overlap:end]) <= settings.governance_input_tokens // 10:
                        next_offset = overlap
                offset, serial = next_offset, serial + 1
        submit()
        statuses = []
        for source in sources:
            results = states[source.id]
            ranges = merge_ranges(r for s in results for r in s.processed_ranges)
            covered = ranges == ((0, len(source.content)),) or not source.content
            errors = [s.error_type for s in results if s.error_type]
            full = covered and not errors
            has_facts = any(f.source_id == source.id for f in facts.values())
            state = ("ready" if has_facts else "no_fact") if full else (
                "partial" if ranges or has_facts else (results[0].status if results else "pending"))
            statuses.append(SourceStatus(source.id, state, ranges, errors[0] if errors else None))
        return ValidatedIndexBatch(tuple(facts.values()), tuple(statuses), model.call_count, (time.perf_counter()-started)*1000)
