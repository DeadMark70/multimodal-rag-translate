"""Repositories for evaluation observability detail tables."""

from __future__ import annotations

import json
import html
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from evaluation.db import connect_db, init_db
from evaluation.schemas import EvaluationGraphEvent, EvaluationGraphEvidenceItem
from evaluation.trace_schemas import (
    EvaluationClaim,
    EvaluationContextPack,
    EvaluationEvidencePacket,
    EvaluationHumanRating,
    EvaluationLlmCall,
    EvaluationRetrievalChunk,
    EvaluationRetrievalEvent,
    EvaluationRoutingDecision,
    EvaluationSlotResolution,
    EvaluationToolCall,
    EvaluationTraceEvent,
    EvaluationV9AttemptMaterialization,
)


MAX_V9_OBSERVABILITY_PAYLOAD_BYTES = 256 * 1024
DEFAULT_EVIDENCE_EXCERPT_CHARS = 500
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]+"),
    re.compile(r"(?:api[_-]?key|authorization|bearer)\s*[:=]?\s*\S+", re.IGNORECASE),
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


def _v9_json_dumps(payload: Any) -> str:
    serialized = _json_dumps(payload)
    if len(serialized.encode("utf-8")) > MAX_V9_OBSERVABILITY_PAYLOAD_BYTES:
        raise ValueError("v9 observability payload exceeds 262144 bytes")
    return serialized


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


def safe_plain_text_excerpt(value: Any, *, limit: int = DEFAULT_EVIDENCE_EXCERPT_CHARS) -> str:
    """Render untrusted evidence as a bounded, secret-redacted text excerpt."""
    text = html.unescape(redact_sensitive_text(value))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_#>~]", "", text)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    text = " ".join(text.split())
    if len(text) > limit:
        return f"{text[: max(0, limit - 3)]}..."
    return text


def redact_sensitive_text(value: Any) -> str:
    """Redact credentials without altering otherwise authorized export content."""
    text = str(value or "")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


def redact_sensitive_value(value: Any) -> Any:
    """Apply credential redaction recursively to JSON-shaped export payloads."""
    if isinstance(value, dict):
        return {str(key): redact_sensitive_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_sensitive_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_value(item) for item in value]
    return redact_sensitive_text(value) if isinstance(value, str) else value


def _validate_materialization_rows(
    *,
    attempt_id: str,
    run_id: str,
    campaign_id: str,
    condition_id: str,
    schema_version: str,
    evidence_packets: list[EvaluationEvidencePacket],
    slot_resolutions: list[EvaluationSlotResolution],
    claims: list[EvaluationClaim],
) -> None:
    """Reject mixed-attempt writes before opening the transaction."""
    for row in [*evidence_packets, *slot_resolutions, *claims]:
        if (
            row.attempt_id != attempt_id
            or row.run_id != run_id
            or row.campaign_id != campaign_id
            or row.condition_id != condition_id
            or row.schema_version != schema_version
        ):
            raise ValueError("v9 materialization row does not match its attempt identity")


def _sanitize_v9_trace_payload(
    payload: dict[str, Any], *, cancelled: bool
) -> dict[str, Any]:
    """Keep an auditable trace while never retaining rejected source names."""
    sanitized = _json_loads(_v9_json_dumps(payload), {})
    contract = sanitized.get("query_contract")
    if isinstance(contract, dict):
        scope = contract.get("resolved_source_scope")
        if isinstance(scope, dict):
            # Source display names can reveal documents the caller was not
            # authorized to access.  IDs remain only when authorization was
            # actually granted, and name fields are never persisted in traces.
            scope["requested_source_names"] = []
            scope["rejected_source_names"] = []
            if not scope.get("authorized_doc_ids"):
                scope["resolved_doc_ids"] = []
    if cancelled:
        sanitized.pop("completion", None)
    return sanitized


def _graph_event_from_row(row: Any) -> EvaluationGraphEvent:
    return EvaluationGraphEvent(
        graph_event_id=row["graph_event_id"],
        run_id=row["run_id"],
        campaign_id=row["campaign_id"],
        span_id=row["span_id"],
        graph_query=row["graph_query"],
        graph_search_mode=row["graph_search_mode"],
        graph_evidence_mode=row["graph_evidence_mode"],
        graph_route=row["graph_route"],
        router_reason=row["router_reason"],
        graph_feature_flags=_json_loads(row["graph_feature_flags_json"], {}),
        graph_snapshot_version=row["graph_snapshot_version"],
        graph_schema_version=row["graph_schema_version"],
        graph_extraction_prompt_version=row["graph_extraction_prompt_version"],
        matched_entity_ids=_json_loads(row["matched_entity_ids_json"], []),
        community_ids=_json_loads(row["community_ids_json"], []),
        node_count=row["node_count"],
        edge_count=row["edge_count"],
        path_count=row["path_count"],
        graph_latency_ms=row["graph_latency_ms"],
        graph_context_tokens=row["graph_context_tokens"],
        graph_to_chunk_success_rate=row["graph_to_chunk_success_rate"],
        graph_noise_ratio=row["graph_noise_ratio"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _graph_evidence_item_from_row(row: Any) -> EvaluationGraphEvidenceItem:
    return EvaluationGraphEvidenceItem(
        graph_evidence_item_id=row["graph_evidence_item_id"],
        graph_event_id=row["graph_event_id"],
        node_ids=_json_loads(row["node_ids_json"], []),
        edge_ids=_json_loads(row["edge_ids_json"], []),
        relation_path=_json_loads(row["relation_path_json"], []),
        source_doc_ids=_json_loads(row["source_doc_ids_json"], []),
        source_chunk_ids=_json_loads(row["source_chunk_ids_json"], []),
        pages=_json_loads(row["pages_json"], []),
        asset_ids=_json_loads(row["asset_ids_json"], []),
        confidence=row["confidence"],
        provenance_status=row["provenance_status"],
        used_as_locator=bool(row["used_as_locator"]),
        packed_in_context=bool(row["packed_in_context"]),
        used_in_answer=bool(row["used_in_answer"]),
        supported_claim_ids=_json_loads(row["supported_claim_ids_json"], []),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class EvaluationGraphEventRepository:
    """Persistence operations for GraphRAG event rows."""

    async def record_graph_event(self, event: EvaluationGraphEvent) -> None:
        await init_db()
        async with connect_db() as connection:
            await connection.execute(
                """
                INSERT OR REPLACE INTO evaluation_graph_events (
                    graph_event_id, run_id, campaign_id, span_id, graph_query, graph_search_mode,
                    graph_evidence_mode, graph_route, router_reason, graph_feature_flags_json,
                    graph_snapshot_version, graph_schema_version, graph_extraction_prompt_version,
                    matched_entity_ids_json, community_ids_json, node_count, edge_count, path_count,
                    graph_latency_ms, graph_context_tokens, graph_to_chunk_success_rate,
                    graph_noise_ratio, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.graph_event_id,
                    event.run_id,
                    event.campaign_id,
                    event.span_id,
                    event.graph_query,
                    event.graph_search_mode,
                    event.graph_evidence_mode,
                    event.graph_route,
                    event.router_reason,
                    _json_dumps(event.graph_feature_flags),
                    event.graph_snapshot_version,
                    event.graph_schema_version,
                    event.graph_extraction_prompt_version,
                    _json_dumps(event.matched_entity_ids),
                    _json_dumps(event.community_ids),
                    event.node_count,
                    event.edge_count,
                    event.path_count,
                    event.graph_latency_ms,
                    event.graph_context_tokens,
                    event.graph_to_chunk_success_rate,
                    event.graph_noise_ratio,
                    event.created_at.isoformat(),
                ),
            )
            await connection.commit()

    async def list_graph_events_for_run(self, run_id: str) -> list[EvaluationGraphEvent]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_graph_events WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [_graph_event_from_row(row) for row in rows]

    async def list_graph_events_for_campaign(
        self, campaign_id: str
    ) -> dict[str, list[EvaluationGraphEvent]]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_graph_events
                WHERE campaign_id = ?
                ORDER BY run_id ASC, created_at ASC
                """,
                (campaign_id,),
            )
            rows = await cursor.fetchall()
        grouped: dict[str, list[EvaluationGraphEvent]] = defaultdict(list)
        for row in rows:
            grouped[str(row["run_id"])].append(_graph_event_from_row(row))
        return dict(grouped)


class EvaluationGraphEvidenceItemRepository:
    """Persistence operations for GraphRAG evidence rows."""

    async def record_graph_evidence_item(self, item: EvaluationGraphEvidenceItem) -> None:
        await self.record_graph_evidence_items([item])

    async def record_graph_evidence_items(self, items: list[EvaluationGraphEvidenceItem]) -> None:
        if not items:
            return
        await init_db()
        async with connect_db() as connection:
            for item in items:
                await connection.execute(
                    """
                    INSERT OR REPLACE INTO evaluation_graph_evidence_items (
                        graph_evidence_item_id, graph_event_id, node_ids_json, edge_ids_json,
                        relation_path_json, source_doc_ids_json, source_chunk_ids_json, pages_json,
                        asset_ids_json, confidence, provenance_status, used_as_locator,
                        packed_in_context, used_in_answer, supported_claim_ids_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.graph_evidence_item_id,
                        item.graph_event_id,
                        _json_dumps(item.node_ids),
                        _json_dumps(item.edge_ids),
                        _json_dumps(item.relation_path),
                        _json_dumps(item.source_doc_ids),
                        _json_dumps(item.source_chunk_ids),
                        _json_dumps(item.pages),
                        _json_dumps(item.asset_ids),
                        item.confidence,
                        item.provenance_status,
                        1 if item.used_as_locator else 0,
                        1 if item.packed_in_context else 0,
                        1 if item.used_in_answer else 0,
                        _json_dumps(item.supported_claim_ids),
                        item.created_at.isoformat(),
                    ),
                )
            await connection.commit()

    async def list_graph_evidence_items_for_graph_event(
        self, graph_event_id: str
    ) -> list[EvaluationGraphEvidenceItem]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_graph_evidence_items
                WHERE graph_event_id = ?
                ORDER BY created_at ASC, graph_evidence_item_id ASC
                """,
                (graph_event_id,),
            )
            rows = await cursor.fetchall()
        return [_graph_evidence_item_from_row(row) for row in rows]

    async def list_graph_evidence_items_for_run(
        self, run_id: str
    ) -> list[EvaluationGraphEvidenceItem]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT items.*
                FROM evaluation_graph_evidence_items AS items
                JOIN evaluation_graph_events AS events
                  ON events.graph_event_id = items.graph_event_id
                WHERE events.run_id = ?
                ORDER BY events.created_at ASC, items.created_at ASC, items.graph_evidence_item_id ASC
                """,
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [_graph_evidence_item_from_row(row) for row in rows]

    async def list_graph_evidence_items_for_campaign(
        self, campaign_id: str
    ) -> dict[str, list[EvaluationGraphEvidenceItem]]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT events.run_id, items.*
                FROM evaluation_graph_evidence_items AS items
                JOIN evaluation_graph_events AS events
                  ON events.graph_event_id = items.graph_event_id
                WHERE events.campaign_id = ?
                ORDER BY events.run_id ASC, events.created_at ASC, items.created_at ASC, items.graph_evidence_item_id ASC
                """,
                (campaign_id,),
            )
            rows = await cursor.fetchall()
        grouped: dict[str, list[EvaluationGraphEvidenceItem]] = defaultdict(list)
        for row in rows:
            grouped[str(row["run_id"])].append(_graph_evidence_item_from_row(row))
        return dict(grouped)


class EvaluationObservabilityRepository(
    EvaluationGraphEventRepository,
    EvaluationGraphEvidenceItemRepository,
):
    """Persistence operations for normalized evaluation observability rows."""

    async def materialize_v9_attempt(
        self,
        *,
        attempt_id: str,
        run_id: str,
        campaign_id: str,
        condition_id: str,
        schema_version: str,
        trace_payload: dict[str, Any],
        evidence_packets: list[EvaluationEvidencePacket],
        slot_resolutions: list[EvaluationSlotResolution],
        claims: list[EvaluationClaim] | None = None,
    ) -> EvaluationV9AttemptMaterialization:
        """Atomically materialize one v9 attempt without promoting cancelled work.

        The authoritative execution result and provider usage remain owned by the
        durable job/accounting stores.  This method only commits attempt-scoped
        evidence state after proving the attempt belongs to the supplied campaign.
        """
        _validate_materialization_rows(
            attempt_id=attempt_id,
            run_id=run_id,
            campaign_id=campaign_id,
            condition_id=condition_id,
            schema_version=schema_version,
            evidence_packets=evidence_packets,
            slot_resolutions=slot_resolutions,
            claims=claims or [],
        )
        await init_db()
        async with connect_db() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    """
                    SELECT attempt.status
                    FROM evaluation_attempts AS attempt
                    JOIN evaluation_jobs AS job ON job.id = attempt.job_id
                    WHERE attempt.id = ? AND job.campaign_id = ?
                    """,
                    (attempt_id, campaign_id),
                )
                attempt = await cursor.fetchone()
                if attempt is None:
                    raise ValueError("attempt does not belong to the supplied campaign")
                cancelled = str(attempt["status"]) == "cancelled"
                sanitized_trace = _sanitize_v9_trace_payload(
                    trace_payload, cancelled=cancelled
                )
                serialized_trace = _v9_json_dumps(sanitized_trace)
                now = datetime.now(timezone.utc)
                materialization_status = "cancelled" if cancelled else "completed"
                completed_at = None if cancelled else now
                await connection.execute(
                    """
                    INSERT INTO evaluation_v9_attempt_materializations (
                        attempt_id, run_id, campaign_id, condition_id, schema_version,
                        trace_json, materialization_status, completed_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(attempt_id) DO UPDATE SET
                        run_id = excluded.run_id,
                        campaign_id = excluded.campaign_id,
                        condition_id = excluded.condition_id,
                        schema_version = excluded.schema_version,
                        trace_json = excluded.trace_json,
                        materialization_status = excluded.materialization_status,
                        completed_at = excluded.completed_at
                    """,
                    (
                        attempt_id,
                        run_id,
                        campaign_id,
                        condition_id,
                        schema_version,
                        serialized_trace,
                        materialization_status,
                        _dt(completed_at),
                        _dt(now),
                    ),
                )
                if not cancelled:
                    for packet in evidence_packets:
                        await self._insert_evidence_packet(connection, packet)
                    for resolution in slot_resolutions:
                        await self._insert_slot_resolution(connection, resolution)
                    for claim in claims or []:
                        await self._insert_claim(connection, claim)
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return EvaluationV9AttemptMaterialization(
            attempt_id=attempt_id,
            run_id=run_id,
            campaign_id=campaign_id,
            condition_id=condition_id,
            schema_version=schema_version,
            trace_payload=sanitized_trace,
            materialization_status=materialization_status,
            completed_at=completed_at,
            created_at=now,
        )

    async def get_v9_attempt_materialization(
        self, attempt_id: str
    ) -> EvaluationV9AttemptMaterialization | None:
        """Return the retained, sanitized state for a single durable attempt."""
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_v9_attempt_materializations WHERE attempt_id = ?",
                (attempt_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return EvaluationV9AttemptMaterialization(
            attempt_id=row["attempt_id"],
            run_id=row["run_id"],
            campaign_id=row["campaign_id"],
            condition_id=row["condition_id"],
            schema_version=row["schema_version"],
            trace_payload=_json_loads(row["trace_json"], {}),
            materialization_status=row["materialization_status"],
            completed_at=_parse_dt(row["completed_at"]),
            created_at=_parse_dt(row["created_at"]) or datetime.now(timezone.utc),
        )

    @staticmethod
    async def _insert_evidence_packet(connection: Any, packet: EvaluationEvidencePacket) -> None:
        await connection.execute(
            """
            INSERT INTO evaluation_evidence_packets (
                evidence_packet_row_id, attempt_id, run_id, campaign_id, condition_id,
                schema_version, evidence_id, packet_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(attempt_id, evidence_id) DO UPDATE SET
                run_id = excluded.run_id,
                campaign_id = excluded.campaign_id,
                condition_id = excluded.condition_id,
                schema_version = excluded.schema_version,
                packet_json = excluded.packet_json,
                created_at = excluded.created_at
            """,
            (
                f"{packet.attempt_id}:{packet.evidence_id}",
                packet.attempt_id,
                packet.run_id,
                packet.campaign_id,
                packet.condition_id,
                packet.schema_version,
                packet.evidence_id,
                _v9_json_dumps(packet.packet),
                packet.created_at.isoformat(),
            ),
        )

    @staticmethod
    async def _assert_attempt_campaign(
        connection: Any, *, attempt_id: str, campaign_id: str
    ) -> None:
        cursor = await connection.execute(
            """
            SELECT 1
            FROM evaluation_attempts AS attempt
            JOIN evaluation_jobs AS job ON job.id = attempt.job_id
            WHERE attempt.id = ? AND job.campaign_id = ?
            """,
            (attempt_id, campaign_id),
        )
        if await cursor.fetchone() is None:
            raise ValueError("attempt does not belong to the supplied campaign")

    @staticmethod
    async def _insert_slot_resolution(connection: Any, resolution: EvaluationSlotResolution) -> None:
        await connection.execute(
            """
            INSERT INTO evaluation_slot_resolutions (
                slot_resolution_row_id, attempt_id, run_id, campaign_id, condition_id,
                schema_version, slot_id, resolution_stage, resolution_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(attempt_id, slot_id, resolution_stage) DO UPDATE SET
                run_id = excluded.run_id,
                campaign_id = excluded.campaign_id,
                condition_id = excluded.condition_id,
                schema_version = excluded.schema_version,
                resolution_json = excluded.resolution_json,
                created_at = excluded.created_at
            """,
            (
                f"{resolution.attempt_id}:{resolution.slot_id}:{resolution.resolution_stage}",
                resolution.attempt_id,
                resolution.run_id,
                resolution.campaign_id,
                resolution.condition_id,
                resolution.schema_version,
                resolution.slot_id,
                resolution.resolution_stage,
                _v9_json_dumps(resolution.resolution),
                resolution.created_at.isoformat(),
            ),
        )

    @staticmethod
    async def _insert_claim(connection: Any, claim: EvaluationClaim) -> None:
        await connection.execute(
            """
            INSERT INTO evaluation_claims (
                claim_id, run_id, campaign_id, attempt_id, condition_id, schema_version,
                span_id, claim_text, claim_type, support_status, evidence_json,
                unsupported_reason, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                run_id = excluded.run_id,
                campaign_id = excluded.campaign_id,
                attempt_id = excluded.attempt_id,
                condition_id = excluded.condition_id,
                schema_version = excluded.schema_version,
                span_id = excluded.span_id,
                claim_text = excluded.claim_text,
                claim_type = excluded.claim_type,
                support_status = excluded.support_status,
                evidence_json = excluded.evidence_json,
                unsupported_reason = excluded.unsupported_reason,
                payload_json = excluded.payload_json,
                created_at = excluded.created_at
            """,
            (
                claim.claim_id,
                claim.run_id,
                claim.campaign_id,
                claim.attempt_id,
                claim.condition_id,
                claim.schema_version,
                claim.span_id,
                claim.claim_text,
                claim.claim_type,
                claim.support_status,
                _v9_json_dumps(claim.evidence),
                claim.unsupported_reason,
                _v9_json_dumps(claim.payload),
                claim.created_at.isoformat(),
            ),
        )

    async def record_evidence_packet(self, packet: EvaluationEvidencePacket) -> None:
        """Persist one v9 evidence packet idempotently for its attempt."""
        await init_db()
        async with connect_db() as connection:
            await self._assert_attempt_campaign(
                connection, attempt_id=packet.attempt_id, campaign_id=packet.campaign_id
            )
            await connection.execute(
                """
                INSERT INTO evaluation_evidence_packets (
                    evidence_packet_row_id, attempt_id, run_id, campaign_id, condition_id,
                    schema_version, evidence_id, packet_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attempt_id, evidence_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    campaign_id = excluded.campaign_id,
                    condition_id = excluded.condition_id,
                    schema_version = excluded.schema_version,
                    packet_json = excluded.packet_json,
                    created_at = excluded.created_at
                """,
                (
                    f"{packet.attempt_id}:{packet.evidence_id}",
                    packet.attempt_id,
                    packet.run_id,
                    packet.campaign_id,
                    packet.condition_id,
                    packet.schema_version,
                    packet.evidence_id,
                    _v9_json_dumps(packet.packet),
                    packet.created_at.isoformat(),
                ),
            )
            await connection.commit()

    async def list_evidence_packets_for_attempt(
        self, attempt_id: str
    ) -> list[EvaluationEvidencePacket]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_evidence_packets
                WHERE attempt_id = ? ORDER BY created_at ASC, evidence_id ASC
                """,
                (attempt_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationEvidencePacket(
                attempt_id=row["attempt_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                condition_id=row["condition_id"],
                schema_version=row["schema_version"],
                evidence_id=row["evidence_id"],
                packet=_json_loads(row["packet_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def record_slot_resolution(self, resolution: EvaluationSlotResolution) -> None:
        """Persist one v9 slot resolution idempotently for its attempt and stage."""
        await init_db()
        async with connect_db() as connection:
            await self._assert_attempt_campaign(
                connection,
                attempt_id=resolution.attempt_id,
                campaign_id=resolution.campaign_id,
            )
            await connection.execute(
                """
                INSERT INTO evaluation_slot_resolutions (
                    slot_resolution_row_id, attempt_id, run_id, campaign_id, condition_id,
                    schema_version, slot_id, resolution_stage, resolution_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attempt_id, slot_id, resolution_stage) DO UPDATE SET
                    run_id = excluded.run_id,
                    campaign_id = excluded.campaign_id,
                    condition_id = excluded.condition_id,
                    schema_version = excluded.schema_version,
                    resolution_json = excluded.resolution_json,
                    created_at = excluded.created_at
                """,
                (
                    f"{resolution.attempt_id}:{resolution.slot_id}:{resolution.resolution_stage}",
                    resolution.attempt_id,
                    resolution.run_id,
                    resolution.campaign_id,
                    resolution.condition_id,
                    resolution.schema_version,
                    resolution.slot_id,
                    resolution.resolution_stage,
                    _v9_json_dumps(resolution.resolution),
                    resolution.created_at.isoformat(),
                ),
            )
            await connection.commit()

    async def list_slot_resolutions_for_attempt(
        self, attempt_id: str
    ) -> list[EvaluationSlotResolution]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_slot_resolutions
                WHERE attempt_id = ? ORDER BY created_at ASC, slot_id ASC, resolution_stage ASC
                """,
                (attempt_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvaluationSlotResolution(
                attempt_id=row["attempt_id"],
                run_id=row["run_id"],
                campaign_id=row["campaign_id"],
                condition_id=row["condition_id"],
                schema_version=row["schema_version"],
                slot_id=row["slot_id"],
                resolution_stage=row["resolution_stage"],
                resolution=_json_loads(row["resolution_json"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

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
                    context_pack_id, run_id, campaign_id, attempt_id, condition_id, schema_version,
                    span_id, input_chunk_count,
                    packed_chunk_count, token_count, retrieved_but_not_packed_evidence_json,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack.context_pack_id,
                    pack.run_id,
                    pack.campaign_id,
                    pack.attempt_id,
                    pack.condition_id,
                    pack.schema_version,
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
                attempt_id=row["attempt_id"],
                condition_id=row["condition_id"],
                schema_version=row["schema_version"],
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

    async def list_routing_decisions_for_campaign(
        self, campaign_id: str
    ) -> dict[str, list[EvaluationRoutingDecision]]:
        """Load all routing decisions for a campaign in one query.

        Campaign analytics must not issue one routing query per result/run. The
        response is grouped by run_id so callers can preserve their existing
        run-oriented response shapes without reopening the database.
        """
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_routing_decisions
                WHERE campaign_id = ?
                ORDER BY run_id ASC, created_at ASC, routing_decision_id ASC
                """,
                (campaign_id,),
            )
            rows = await cursor.fetchall()
        grouped: dict[str, list[EvaluationRoutingDecision]] = defaultdict(list)
        for row in rows:
            grouped[str(row["run_id"])].append(
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
            )
        return dict(grouped)

    async def record_claim(self, claim: EvaluationClaim) -> None:
        if claim.attempt_id:
            await init_db()
            async with connect_db() as connection:
                await self._assert_attempt_campaign(
                    connection,
                    attempt_id=claim.attempt_id,
                    campaign_id=claim.campaign_id,
                )
                await self._insert_claim(connection, claim)
                await connection.commit()
            return
        await self._record_simple(
            "evaluation_claims",
            (
                "claim_id",
                "run_id",
                "campaign_id",
                "attempt_id",
                "condition_id",
                "schema_version",
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
                claim.attempt_id,
                claim.condition_id,
                claim.schema_version,
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
                attempt_id=row["attempt_id"],
                condition_id=row["condition_id"],
                schema_version=row["schema_version"],
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
                    attempt_id=row["attempt_id"],
                    condition_id=row["condition_id"],
                    schema_version=row["schema_version"],
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
