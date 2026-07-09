"""Repositories for evaluation observability detail tables."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from evaluation.db import connect_db, init_db
from evaluation.trace_schemas import (
    EvaluationClaim,
    EvaluationContextPack,
    EvaluationHumanRating,
    EvaluationLlmCall,
    EvaluationRetrievalChunk,
    EvaluationRetrievalEvent,
    EvaluationRoutingDecision,
    EvaluationToolCall,
    EvaluationTraceEvent,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Decimal, UUID)):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=_json_default)


def _json_loads(payload: str | None, fallback: Any) -> Any:
    if not payload:
        return fallback
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return fallback


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class EvaluationObservabilityRepository:
    """Persistence operations for normalized evaluation observability rows."""

    async def record_trace_event(self, event: EvaluationTraceEvent) -> None:
        await self.record_trace_events([event])

    async def record_trace_events(self, events: list[EvaluationTraceEvent]) -> None:
        if not events:
            return
        await init_db()
        async with connect_db() as connection:
            for event in events:
                await connection.execute(
                    """
                    INSERT OR REPLACE INTO evaluation_trace_events (
                        event_id, run_id, campaign_id, span_id, parent_event_id, parent_span_id,
                        event_type, event_schema_version, sequence, stage_type, stage_name,
                        started_at, ended_at, duration_ms, status, retry_count,
                        payload_json, error_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.run_id,
                        event.campaign_id,
                        event.span_id,
                        event.parent_event_id,
                        event.parent_span_id,
                        event.event_type,
                        event.event_schema_version,
                        event.sequence,
                        event.stage_type,
                        event.stage_name,
                        event.started_at.isoformat(),
                        _dt(event.ended_at),
                        event.duration_ms,
                        event.status,
                        event.retry_count,
                        _json_dumps(event.payload),
                        _json_dumps(event.error),
                        event.created_at.isoformat(),
                    ),
                )
            await connection.commit()

    async def list_trace_events_for_run(self, run_id: str) -> list[EvaluationTraceEvent]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_trace_events WHERE run_id = ? ORDER BY sequence ASC, started_at ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationTraceEvent(
                event_id=row["event_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                span_id=row["span_id"],
                parent_event_id=row["parent_event_id"],
                parent_span_id=row["parent_span_id"],
                event_type=row["event_type"],
                event_schema_version=row["event_schema_version"],
                sequence=row["sequence"],
                stage_type=row["stage_type"],
                stage_name=row["stage_name"],
                started_at=datetime.fromisoformat(row["started_at"]),
                ended_at=_parse_dt(row["ended_at"]),
                duration_ms=row["duration_ms"],
                status=row["status"],
                retry_count=row["retry_count"],
                payload=_json_loads(row["payload_json"], {}),
                error=_json_loads(row["error_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def list_trace_events_for_campaign(self, campaign_id: str) -> dict[str, list[EvaluationTraceEvent]]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_trace_events
                WHERE campaign_id = ?
                ORDER BY run_id ASC, sequence ASC, started_at ASC
                """,
                (campaign_id,),
            )
            rows = await cursor.fetchall()
        grouped: dict[str, list[EvaluationTraceEvent]] = defaultdict(list)
        for row in rows:
            grouped[str(row["run_id"])].append(
                EvaluationTraceEvent(
                    event_id=row["event_id"],
                    run_id=row["run_id"],
                    campaign_id=row["campaign_id"],
                    span_id=row["span_id"],
                    parent_event_id=row["parent_event_id"],
                    parent_span_id=row["parent_span_id"],
                    event_type=row["event_type"],
                    event_schema_version=row["event_schema_version"],
                    sequence=row["sequence"],
                    stage_type=row["stage_type"],
                    stage_name=row["stage_name"],
                    started_at=datetime.fromisoformat(row["started_at"]),
                    ended_at=_parse_dt(row["ended_at"]),
                    duration_ms=row["duration_ms"],
                    status=row["status"],
                    retry_count=row["retry_count"],
                    payload=_json_loads(row["payload_json"], {}),
                    error=_json_loads(row["error_json"], {}),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return dict(grouped)

    async def record_llm_call(self, call: EvaluationLlmCall) -> None:
        await init_db()
        async with connect_db() as connection:
            await connection.execute(
                """
                INSERT OR REPLACE INTO evaluation_llm_calls (
                    llm_call_id, run_id, campaign_id, span_id, provider, model_name,
                    purpose, prompt_tokens, completion_tokens, total_tokens,
                    estimated_cost_usd, estimated_cost_twd, prompt_hash, prompt_preview,
                    response_hash, latency_ms, status, error_json, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call.llm_call_id,
                    call.run_id,
                    call.campaign_id,
                    call.span_id,
                    call.provider,
                    call.model_name,
                    call.purpose,
                    call.prompt_tokens,
                    call.completion_tokens,
                    call.total_tokens,
                    call.estimated_cost_usd,
                    call.estimated_cost_twd,
                    call.prompt_hash,
                    call.prompt_preview,
                    call.response_hash,
                    call.latency_ms,
                    call.status,
                    _json_dumps(call.error),
                    _json_dumps(call.payload),
                    call.created_at.isoformat(),
                ),
            )
            await connection.commit()

    async def list_llm_calls_for_run(self, run_id: str) -> list[EvaluationLlmCall]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_llm_calls WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationLlmCall(
                llm_call_id=row["llm_call_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                span_id=row["span_id"],
                provider=row["provider"],
                model_name=row["model_name"],
                purpose=row["purpose"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
                estimated_cost_usd=row["estimated_cost_usd"],
                estimated_cost_twd=row["estimated_cost_twd"],
                prompt_hash=row["prompt_hash"],
                prompt_preview=row["prompt_preview"],
                response_hash=row["response_hash"],
                latency_ms=row["latency_ms"],
                status=row["status"],
                error=_json_loads(row["error_json"], {}),
                payload=_json_loads(row["payload_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def list_llm_calls_for_campaign(self, campaign_id: str) -> dict[str, list[EvaluationLlmCall]]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_llm_calls
                WHERE campaign_id = ?
                ORDER BY run_id ASC, created_at ASC
                """,
                (campaign_id,),
            )
            rows = await cursor.fetchall()
        grouped: dict[str, list[EvaluationLlmCall]] = defaultdict(list)
        for row in rows:
            grouped[str(row["run_id"])].append(
                EvaluationLlmCall(
                    llm_call_id=row["llm_call_id"],
                    run_id=row["run_id"],
                    campaign_id=row["campaign_id"],
                    span_id=row["span_id"],
                    provider=row["provider"],
                    model_name=row["model_name"],
                    purpose=row["purpose"],
                    prompt_tokens=row["prompt_tokens"],
                    completion_tokens=row["completion_tokens"],
                    total_tokens=row["total_tokens"],
                    estimated_cost_usd=row["estimated_cost_usd"],
                    estimated_cost_twd=row["estimated_cost_twd"],
                    prompt_hash=row["prompt_hash"],
                    prompt_preview=row["prompt_preview"],
                    response_hash=row["response_hash"],
                    latency_ms=row["latency_ms"],
                    status=row["status"],
                    error=_json_loads(row["error_json"], {}),
                    payload=_json_loads(row["payload_json"], {}),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return dict(grouped)

    async def record_retrieval_event(self, event: EvaluationRetrievalEvent) -> None:
        await init_db()
        async with connect_db() as connection:
            await connection.execute(
                """
                INSERT OR REPLACE INTO evaluation_retrieval_events (
                    retrieval_event_id, run_id, campaign_id, span_id, query, query_hash,
                    retriever_name, top_k, result_count, latency_ms, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.retrieval_event_id,
                    event.run_id,
                    event.campaign_id,
                    event.span_id,
                    event.query,
                    event.query_hash,
                    event.retriever_name,
                    event.top_k,
                    event.result_count,
                    event.latency_ms,
                    _json_dumps(event.payload),
                    event.created_at.isoformat(),
                ),
            )
            await connection.commit()

    async def list_retrieval_events_for_run(self, run_id: str) -> list[EvaluationRetrievalEvent]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_retrieval_events WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationRetrievalEvent(
                retrieval_event_id=row["retrieval_event_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                span_id=row["span_id"],
                query=row["query"],
                query_hash=row["query_hash"],
                retriever_name=row["retriever_name"],
                top_k=row["top_k"],
                result_count=row["result_count"],
                latency_ms=row["latency_ms"],
                payload=_json_loads(row["payload_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def record_retrieval_chunk(self, chunk: EvaluationRetrievalChunk) -> None:
        await init_db()
        async with connect_db() as connection:
            await connection.execute(
                """
                INSERT OR REPLACE INTO evaluation_retrieval_chunks (
                    retrieval_chunk_id, run_id, campaign_id, span_id, retrieval_event_id,
                    chunk_id, doc_id, page_start, page_end, modality, rank_before_rerank,
                    rank_after_rerank, dense_score, bm25_score, rerank_score,
                    used_in_context, used_in_answer, expected_evidence_match,
                    excerpt, content_hash, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.retrieval_chunk_id,
                    chunk.run_id,
                    chunk.campaign_id,
                    chunk.span_id,
                    chunk.retrieval_event_id,
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.modality,
                    chunk.rank_before_rerank,
                    chunk.rank_after_rerank,
                    chunk.dense_score,
                    chunk.bm25_score,
                    chunk.rerank_score,
                    1 if chunk.used_in_context else 0,
                    1 if chunk.used_in_answer else 0,
                    1 if chunk.expected_evidence_match else 0,
                    chunk.excerpt,
                    chunk.content_hash,
                    _json_dumps(chunk.payload),
                    chunk.created_at.isoformat(),
                ),
            )
            await connection.commit()

    async def list_retrieval_chunks_for_run(self, run_id: str) -> list[EvaluationRetrievalChunk]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_retrieval_chunks WHERE run_id = ? ORDER BY retrieval_event_id ASC, rank_after_rerank ASC, created_at ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationRetrievalChunk(
                retrieval_chunk_id=row["retrieval_chunk_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                span_id=row["span_id"],
                retrieval_event_id=row["retrieval_event_id"],
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                modality=row["modality"],
                rank_before_rerank=row["rank_before_rerank"],
                rank_after_rerank=row["rank_after_rerank"],
                dense_score=row["dense_score"],
                bm25_score=row["bm25_score"],
                rerank_score=row["rerank_score"],
                used_in_context=bool(row["used_in_context"]),
                used_in_answer=bool(row["used_in_answer"]),
                expected_evidence_match=bool(row["expected_evidence_match"]),
                excerpt=row["excerpt"],
                content_hash=row["content_hash"],
                payload=_json_loads(row["payload_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def list_retrieval_chunks_for_campaign(self, campaign_id: str) -> dict[str, list[EvaluationRetrievalChunk]]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_retrieval_chunks
                WHERE campaign_id = ?
                ORDER BY run_id ASC, retrieval_event_id ASC, rank_after_rerank ASC, created_at ASC
                """,
                (campaign_id,),
            )
            rows = await cursor.fetchall()
        grouped: dict[str, list[EvaluationRetrievalChunk]] = defaultdict(list)
        for row in rows:
            grouped[str(row["run_id"])].append(
                EvaluationRetrievalChunk(
                    retrieval_chunk_id=row["retrieval_chunk_id"],
                    run_id=row["run_id"],
                    campaign_id=row["campaign_id"],
                    span_id=row["span_id"],
                    retrieval_event_id=row["retrieval_event_id"],
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    modality=row["modality"],
                    rank_before_rerank=row["rank_before_rerank"],
                    rank_after_rerank=row["rank_after_rerank"],
                    dense_score=row["dense_score"],
                    bm25_score=row["bm25_score"],
                    rerank_score=row["rerank_score"],
                    used_in_context=bool(row["used_in_context"]),
                    used_in_answer=bool(row["used_in_answer"]),
                    expected_evidence_match=bool(row["expected_evidence_match"]),
                    excerpt=row["excerpt"],
                    content_hash=row["content_hash"],
                    payload=_json_loads(row["payload_json"], {}),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return dict(grouped)

    async def record_context_pack(self, pack: EvaluationContextPack) -> None:
        await init_db()
        async with connect_db() as connection:
            await connection.execute(
                """
                INSERT OR REPLACE INTO evaluation_context_packs (
                    context_pack_id, run_id, campaign_id, span_id, input_chunk_count,
                    packed_chunk_count, token_count, retrieved_but_not_packed_evidence_json,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack.context_pack_id,
                    pack.run_id,
                    pack.campaign_id,
                    pack.span_id,
                    pack.input_chunk_count,
                    pack.packed_chunk_count,
                    pack.token_count,
                    _json_dumps(pack.retrieved_but_not_packed_evidence),
                    _json_dumps(pack.payload),
                    pack.created_at.isoformat(),
                ),
            )
            await connection.commit()

    async def list_context_packs_for_run(self, run_id: str) -> list[EvaluationContextPack]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_context_packs WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationContextPack(
                context_pack_id=row["context_pack_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                span_id=row["span_id"],
                input_chunk_count=row["input_chunk_count"],
                packed_chunk_count=row["packed_chunk_count"],
                token_count=row["token_count"],
                retrieved_but_not_packed_evidence=_json_loads(
                    row["retrieved_but_not_packed_evidence_json"], []
                ),
                payload=_json_loads(row["payload_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def record_tool_call(self, call: EvaluationToolCall) -> None:
        await self._record_simple(
            "evaluation_tool_calls",
            (
                "tool_call_id",
                "run_id",
                "campaign_id",
                "span_id",
                "tool_name",
                "action",
                "latency_ms",
                "status",
                "payload_json",
                "created_at",
            ),
            (
                call.tool_call_id,
                call.run_id,
                call.campaign_id,
                call.span_id,
                call.tool_name,
                call.action,
                call.latency_ms,
                call.status,
                _json_dumps(call.payload),
                call.created_at.isoformat(),
            ),
        )

    async def list_tool_calls_for_run(self, run_id: str) -> list[EvaluationToolCall]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_tool_calls WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationToolCall(
                tool_call_id=row["tool_call_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                span_id=row["span_id"],
                tool_name=row["tool_name"],
                action=row["action"],
                latency_ms=row["latency_ms"],
                status=row["status"],
                payload=_json_loads(row["payload_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def record_routing_decision(self, decision: EvaluationRoutingDecision) -> None:
        await self._record_simple(
            "evaluation_routing_decisions",
            (
                "routing_decision_id",
                "run_id",
                "campaign_id",
                "span_id",
                "selected_mode",
                "analysis_type",
                "confidence",
                "reason",
                "payload_json",
                "created_at",
            ),
            (
                decision.routing_decision_id,
                decision.run_id,
                decision.campaign_id,
                decision.span_id,
                decision.selected_mode,
                decision.analysis_type,
                decision.confidence,
                decision.reason,
                _json_dumps(decision.payload),
                decision.created_at.isoformat(),
            ),
        )

    async def list_routing_decisions_for_run(self, run_id: str) -> list[EvaluationRoutingDecision]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_routing_decisions WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationRoutingDecision(
                routing_decision_id=row["routing_decision_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                span_id=row["span_id"],
                selected_mode=row["selected_mode"],
                analysis_type=row["analysis_type"],
                confidence=row["confidence"],
                reason=row["reason"],
                payload=_json_loads(row["payload_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def record_claim(self, claim: EvaluationClaim) -> None:
        await self._record_simple(
            "evaluation_claims",
            (
                "claim_id",
                "run_id",
                "campaign_id",
                "span_id",
                "claim_text",
                "claim_type",
                "support_status",
                "evidence_json",
                "unsupported_reason",
                "payload_json",
                "created_at",
            ),
            (
                claim.claim_id,
                claim.run_id,
                claim.campaign_id,
                claim.span_id,
                claim.claim_text,
                claim.claim_type,
                claim.support_status,
                _json_dumps(claim.evidence),
                claim.unsupported_reason,
                _json_dumps(claim.payload),
                claim.created_at.isoformat(),
            ),
        )

    async def list_claims_for_run(self, run_id: str) -> list[EvaluationClaim]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_claims WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationClaim(
                claim_id=row["claim_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                span_id=row["span_id"],
                claim_text=row["claim_text"],
                claim_type=row["claim_type"],
                support_status=row["support_status"],
                evidence=_json_loads(row["evidence_json"], []),
                unsupported_reason=row["unsupported_reason"],
                payload=_json_loads(row["payload_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def list_claims_for_campaign(self, campaign_id: str) -> dict[str, list[EvaluationClaim]]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_claims
                WHERE campaign_id = ?
                ORDER BY run_id ASC, created_at ASC
                """,
                (campaign_id,),
            )
            rows = await cursor.fetchall()
        grouped: dict[str, list[EvaluationClaim]] = defaultdict(list)
        for row in rows:
            grouped[str(row["run_id"])].append(
                EvaluationClaim(
                    claim_id=row["claim_id"],
                    run_id=row["run_id"],
                    campaign_id=row["campaign_id"],
                    span_id=row["span_id"],
                    claim_text=row["claim_text"],
                    claim_type=row["claim_type"],
                    support_status=row["support_status"],
                    evidence=_json_loads(row["evidence_json"], []),
                    unsupported_reason=row["unsupported_reason"],
                    payload=_json_loads(row["payload_json"], {}),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return dict(grouped)

    async def record_human_rating(self, rating: EvaluationHumanRating) -> None:
        await self._record_simple(
            "evaluation_human_ratings",
            (
                "human_rating_id",
                "run_id",
                "campaign_id",
                "span_id",
                "rater_id_hash",
                "rubric_version",
                "correctness_score",
                "faithfulness_score",
                "completeness_score",
                "citation_quality_score",
                "usefulness_score",
                "comments",
                "is_blinded",
                "shown_mode_label",
                "payload_json",
                "created_at",
            ),
            (
                rating.human_rating_id,
                rating.run_id,
                rating.campaign_id,
                rating.span_id,
                rating.rater_id_hash,
                rating.rubric_version,
                rating.correctness_score,
                rating.faithfulness_score,
                rating.completeness_score,
                rating.citation_quality_score,
                rating.usefulness_score,
                rating.comments,
                1 if rating.is_blinded else 0,
                1 if rating.shown_mode_label else 0,
                _json_dumps(rating.payload),
                rating.created_at.isoformat(),
            ),
        )

    async def list_human_ratings_for_run(self, run_id: str) -> list[EvaluationHumanRating]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_human_ratings WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationHumanRating(
                human_rating_id=row["human_rating_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                span_id=row["span_id"],
                rater_id_hash=row["rater_id_hash"],
                rubric_version=row["rubric_version"],
                correctness_score=row["correctness_score"],
                faithfulness_score=row["faithfulness_score"],
                completeness_score=row["completeness_score"],
                citation_quality_score=row["citation_quality_score"],
                usefulness_score=row["usefulness_score"],
                comments=row["comments"],
                is_blinded=bool(row["is_blinded"]),
                shown_mode_label=bool(row["shown_mode_label"]),
                payload=_json_loads(row["payload_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def list_human_ratings_for_campaign(self, campaign_id: str) -> dict[str, list[EvaluationHumanRating]]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_human_ratings
                WHERE campaign_id = ?
                ORDER BY run_id ASC, created_at ASC
                """,
                (campaign_id,),
            )
            rows = await cursor.fetchall()
        grouped: dict[str, list[EvaluationHumanRating]] = defaultdict(list)
        for row in rows:
            grouped[str(row["run_id"])].append(
                EvaluationHumanRating(
                    human_rating_id=row["human_rating_id"],
                    run_id=row["run_id"],
                    campaign_id=row["campaign_id"],
                    span_id=row["span_id"],
                    rater_id_hash=row["rater_id_hash"],
                    rubric_version=row["rubric_version"],
                    correctness_score=row["correctness_score"],
                    faithfulness_score=row["faithfulness_score"],
                    completeness_score=row["completeness_score"],
                    citation_quality_score=row["citation_quality_score"],
                    usefulness_score=row["usefulness_score"],
                    comments=row["comments"],
                    is_blinded=bool(row["is_blinded"]),
                    shown_mode_label=bool(row["shown_mode_label"]),
                    payload=_json_loads(row["payload_json"], {}),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return dict(grouped)

    async def _record_simple(
        self,
        table_name: str,
        columns: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> None:
        await init_db()
        placeholders = ", ".join("?" for _ in columns)
        column_list = ", ".join(columns)
        async with connect_db() as connection:
            await connection.execute(
                f"INSERT OR REPLACE INTO {table_name} ({column_list}) VALUES ({placeholders})",
                values,
            )
            await connection.commit()
