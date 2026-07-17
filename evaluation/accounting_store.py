"""Durable persistence for evaluation usage-accounting scopes and events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Mapping

from pydantic import BaseModel

from evaluation.accounting_schemas import (
    AccountingScope,
    AccountingScopeStart,
    AccountingScopeTarget,
    ScopeStatus,
    UsageEvent,
    UsageEventCreate,
)
from evaluation.db import connect_db, init_db


_TERMINAL_SCOPE_STATUSES = frozenset({"completed", "failed", "interrupted", "cancelled"})


class ScopeTokenSummary(BaseModel):
    """Token totals and completeness derived from events in one scope."""

    scope_id: str
    input_tokens: int = 0
    output_text_tokens: int = 0
    reasoning_tokens: int = 0
    other_tokens: int = 0
    total_tokens: int = 0
    observed_call_count: int = 0
    measured_call_count: int = 0
    missing_usage_call_count: int = 0
    failed_call_count: int = 0
    reconciliation_status: str = "unavailable"

    def as_legacy_usage(self, *, accounting_schema_version: str) -> dict[str, int | str]:
        """Return the backward-compatible token usage projection for a result."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_text_tokens,
            "output_text_tokens": self.output_text_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "other_tokens": self.other_tokens,
            "total_tokens": self.total_tokens,
            "accounting_schema_version": accounting_schema_version,
        }


class EvaluationAccountingStore:
    """Transactional SQLite store for accounting scopes and callback events."""

    async def start_scope(self, request: AccountingScopeStart) -> AccountingScope:
        """Create a running accounting scope and all of its targets atomically."""
        now = _utc_now_iso()
        await init_db()
        async with connect_db() as connection:
            await connection.execute(
                """INSERT INTO evaluation_accounting_scopes (
                       scope_id, campaign_id, scope_type, scope_key, run_id, metric_name,
                       accounting_schema_version, status, started_at, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)""",
                (
                    request.scope_id,
                    request.campaign_id,
                    request.scope_type,
                    request.scope_key,
                    request.run_id,
                    request.metric_name,
                    request.accounting_schema_version,
                    now,
                    now,
                    now,
                ),
            )
            for target in request.targets:
                await connection.execute(
                    """INSERT INTO evaluation_accounting_scope_targets (
                           scope_id, campaign_result_id, job_id, work_item_id, attempt_id,
                           metric_name, is_official, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        request.scope_id,
                        target.campaign_result_id,
                        target.job_id,
                        target.work_item_id,
                        target.attempt_id,
                        target.metric_name,
                        int(target.is_official),
                        now,
                    ),
                )
            await connection.commit()
        return await self.get_scope(request.scope_id)

    async def record_event(self, event: UsageEventCreate) -> None:
        """Persist one callback event and its counter effects in one transaction.

        The usage event ID is the idempotency key. Retried callbacks therefore do
        not increment the owning scope's counters a second time.
        """
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """INSERT OR IGNORE INTO evaluation_usage_events (
                       usage_event_id, scope_id, campaign_id, scope_type, scope_key, run_id,
                       provider_run_id, phase, purpose, metric_name, provider, model_name,
                       input_tokens, output_text_tokens, reasoning_tokens, other_tokens,
                       reported_total_tokens, raw_usage_json, usage_status,
                       reconciliation_status, estimated_cost_usd, estimated_cost_twd,
                       pricing_status, price_snapshot_id, latency_ms, status, error_json,
                       created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                             ?, ?, ?, ?, ?, ?, ?, ?)""",
                _usage_event_values(event),
            )
            if cursor.rowcount == 1:
                await connection.execute(
                    """UPDATE evaluation_accounting_scopes SET
                           observed_call_count = observed_call_count + 1,
                           measured_call_count = measured_call_count + ?,
                           missing_usage_call_count = missing_usage_call_count + ?,
                           unclassified_phase_call_count = unclassified_phase_call_count + ?,
                           updated_at = ?
                       WHERE scope_id = ?""",
                    (
                        int(event.usage_status == "measured"),
                        int(event.usage_status == "missing"),
                        int(event.phase == "unclassified"),
                        event.created_at.isoformat(),
                        event.scope_id,
                    ),
                )
            await connection.commit()

    async def finalize_scope(self, scope_id: str, status: ScopeStatus) -> AccountingScope:
        """Set a scope to a terminal status."""
        if status not in _TERMINAL_SCOPE_STATUSES:
            raise ValueError("Accounting scopes can only be finalized with a terminal status")
        now = _utc_now_iso()
        await init_db()
        async with connect_db() as connection:
            await connection.execute(
                """UPDATE evaluation_accounting_scopes
                   SET status = ?, completed_at = ?, updated_at = ?
                   WHERE scope_id = ?""",
                (status, now, now, scope_id),
            )
            await connection.commit()
        return await self.get_scope(scope_id)

    async def mark_targets_official(
        self, scope_id: str, campaign_result_ids_by_attempt_id: Mapping[str, str]
    ) -> None:
        """Promote selected scope targets and attach their durable result IDs."""
        if not campaign_result_ids_by_attempt_id:
            return
        await init_db()
        async with connect_db() as connection:
            for attempt_id, campaign_result_id in campaign_result_ids_by_attempt_id.items():
                await connection.execute(
                    """UPDATE evaluation_accounting_scope_targets
                       SET is_official = 1, campaign_result_id = ?
                       WHERE scope_id = ? AND attempt_id = ?""",
                    (campaign_result_id, scope_id, attempt_id),
                )
            await connection.commit()

    async def interrupt_running_scopes(self) -> int:
        """Mark unfinished scopes as interrupted during process recovery."""
        now = _utc_now_iso()
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """UPDATE evaluation_accounting_scopes
                   SET status = 'interrupted', completed_at = ?, updated_at = ?
                   WHERE status = 'running'""",
                (now, now),
            )
            await connection.commit()
        return cursor.rowcount

    async def get_scope(self, scope_id: str) -> AccountingScope:
        """Return one scope with ordered targets."""
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_accounting_scopes WHERE scope_id = ?", (scope_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                raise KeyError(f"Accounting scope not found: {scope_id}")
            targets = await _load_scope_targets(connection, scope_id)
        return _scope_from_row(row, targets)

    async def list_campaign_scopes(self, campaign_id: str) -> list[AccountingScope]:
        """List every campaign scope with its targets in creation order."""
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """SELECT * FROM evaluation_accounting_scopes
                   WHERE campaign_id = ? ORDER BY created_at ASC, scope_id ASC""",
                (campaign_id,),
            )
            rows = await cursor.fetchall()
            scopes = [
                _scope_from_row(row, await _load_scope_targets(connection, row["scope_id"]))
                for row in rows
            ]
        return scopes

    async def list_campaign_events(self, campaign_id: str) -> list[UsageEvent]:
        """List every campaign usage event in deterministic creation order."""
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """SELECT * FROM evaluation_usage_events
                   WHERE campaign_id = ? ORDER BY created_at ASC, usage_event_id ASC""",
                (campaign_id,),
            )
            rows = await cursor.fetchall()
        return [_event_from_row(row) for row in rows]

    async def summarize_scope_tokens(self, scope_id: str) -> ScopeTokenSummary:
        """Return non-overlapping event token sums plus completeness metadata."""
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """SELECT
                       COUNT(*) AS observed_call_count,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_text_tokens), 0) AS output_text_tokens,
                       COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                       COALESCE(SUM(other_tokens), 0) AS other_tokens,
                       COALESCE(SUM(input_tokens + output_text_tokens + reasoning_tokens + other_tokens), 0)
                           AS total_tokens,
                       COALESCE(SUM(usage_status = 'measured'), 0) AS measured_call_count,
                       COALESCE(SUM(usage_status = 'missing'), 0) AS missing_usage_call_count,
                       COALESCE(SUM(usage_status = 'failed'), 0) AS failed_call_count,
                       COALESCE(SUM(reconciliation_status != 'balanced'), 0) AS unbalanced_call_count
                   FROM evaluation_usage_events WHERE scope_id = ?""",
                (scope_id,),
            )
            row = await cursor.fetchone()
        observed = int(row["observed_call_count"])
        if observed == 0:
            reconciliation_status = "unavailable"
        elif row["missing_usage_call_count"] or row["failed_call_count"] or row["unbalanced_call_count"]:
            reconciliation_status = "partial"
        else:
            reconciliation_status = "balanced"
        return ScopeTokenSummary(
            scope_id=scope_id,
            input_tokens=row["input_tokens"],
            output_text_tokens=row["output_text_tokens"],
            reasoning_tokens=row["reasoning_tokens"],
            other_tokens=row["other_tokens"],
            total_tokens=row["total_tokens"],
            observed_call_count=observed,
            measured_call_count=row["measured_call_count"],
            missing_usage_call_count=row["missing_usage_call_count"],
            failed_call_count=row["failed_call_count"],
            reconciliation_status=reconciliation_status,
        )


async def _load_scope_targets(connection, scope_id: str) -> list[AccountingScopeTarget]:
    cursor = await connection.execute(
        """SELECT campaign_result_id, job_id, work_item_id, attempt_id, metric_name, is_official
           FROM evaluation_accounting_scope_targets
           WHERE scope_id = ? ORDER BY created_at ASC, attempt_id ASC""",
        (scope_id,),
    )
    rows = await cursor.fetchall()
    return [
        AccountingScopeTarget(
            campaign_result_id=row["campaign_result_id"], job_id=row["job_id"],
            work_item_id=row["work_item_id"], attempt_id=row["attempt_id"],
            metric_name=row["metric_name"], is_official=bool(row["is_official"]),
        )
        for row in rows
    ]


def _scope_from_row(row, targets: list[AccountingScopeTarget]) -> AccountingScope:
    return AccountingScope(
        scope_id=row["scope_id"], campaign_id=row["campaign_id"], scope_type=row["scope_type"],
        scope_key=row["scope_key"], run_id=row["run_id"], metric_name=row["metric_name"],
        accounting_schema_version=row["accounting_schema_version"], status=row["status"],
        observed_call_count=row["observed_call_count"], measured_call_count=row["measured_call_count"],
        missing_usage_call_count=row["missing_usage_call_count"],
        unclassified_phase_call_count=row["unclassified_phase_call_count"],
        started_at=row["started_at"], completed_at=row["completed_at"], created_at=row["created_at"],
        updated_at=row["updated_at"], targets=targets,
    )


def _event_from_row(row) -> UsageEvent:
    return UsageEvent.model_validate({
        **dict(row),
        "raw_usage": json.loads(row["raw_usage_json"]),
        "error": json.loads(row["error_json"]),
    })


def _usage_event_values(event: UsageEventCreate) -> tuple:
    return (
        event.usage_event_id, event.scope_id, event.campaign_id, event.scope_type, event.scope_key,
        event.run_id, event.provider_run_id, event.phase, event.purpose, event.metric_name,
        event.provider, event.model_name, event.input_tokens, event.output_text_tokens,
        event.reasoning_tokens, event.other_tokens, event.reported_total_tokens,
        json.dumps(event.raw_usage, separators=(",", ":")), event.usage_status,
        event.reconciliation_status, event.estimated_cost_usd, event.estimated_cost_twd,
        event.pricing_status, event.price_snapshot_id, event.latency_ms, event.status,
        json.dumps(event.error, separators=(",", ":")), event.created_at.isoformat(),
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
