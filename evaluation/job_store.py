"""SQLite-backed ledger for durable evaluation work."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
import json
from typing import Any
from uuid import uuid4

import aiosqlite

from core.errors import AppError, ErrorCode
from evaluation.db import connect_db, init_db
from evaluation.error_policy import ErrorDecision, retry_delay_seconds
from evaluation.job_schemas import (
    ClaimedEvaluationWork,
    EvaluationAttempt,
    EvaluationJob,
    EvaluationJobType,
    WorkItemSpec,
)

class EvaluationJobStore:
    """Persist jobs, immutable work inputs, and append-only attempts."""

    async def create_job_with_items(
        self,
        *,
        user_id: str,
        campaign_id: str,
        job_type: EvaluationJobType,
        selection: dict[str, Any],
        config_snapshot: dict[str, Any],
        items: Sequence[WorkItemSpec],
    ) -> EvaluationJob:
        """Create a job and fresh job-item retry budgets for stable work keys."""
        await init_db()
        now = _as_iso(datetime.now(timezone.utc))
        job_id = str(uuid4())
        async with connect_db() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await connection.execute(
                    """
                    INSERT INTO evaluation_jobs (
                        id, user_id, campaign_id, job_type, selection_json,
                        config_snapshot_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        user_id,
                        campaign_id,
                        _enum_value(job_type),
                        _json_dumps(selection),
                        _json_dumps(config_snapshot),
                        now,
                    ),
                )
                for spec in items:
                    work_item_id = await self._upsert_work_item(
                        connection, campaign_id=campaign_id, spec=spec, created_at=now
                    )
                    await connection.execute(
                        """
                        INSERT INTO evaluation_job_items (
                            id, job_id, work_item_id, status, max_attempts,
                            next_retry_at, active_attempt_id, created_at, updated_at
                        ) VALUES (?, ?, ?, 'pending', ?, NULL, NULL, ?, ?)
                        ON CONFLICT(job_id, work_item_id) DO NOTHING
                        """,
                        (str(uuid4()), job_id, work_item_id, spec.max_attempts, now, now),
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return EvaluationJob(
            job_id=job_id,
            job_type=job_type,
            user_id=user_id,
            campaign_id=campaign_id,
            selection=selection,
            config_snapshot=config_snapshot,
            created_at=_from_iso(now),
        )

    async def claim_ready_items(
        self, *, limit: int, now: datetime
    ) -> list[ClaimedEvaluationWork]:
        """Atomically claim ready work, recording one append-only running attempt."""
        if limit <= 0:
            return []
        await init_db()
        now_iso = _as_iso(now)
        claimed: list[ClaimedEvaluationWork] = []
        async with connect_db() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    """
                    SELECT ji.id AS job_item_id, ji.job_id, ji.work_item_id,
                           wi.input_snapshot_json, ji.created_at
                    FROM evaluation_job_items AS ji
                    JOIN evaluation_work_items AS wi ON wi.id = ji.work_item_id
                    WHERE (
                        ji.status = 'pending'
                        OR (ji.status = 'retry_wait' AND ji.next_retry_at <= ?)
                    )
                    AND NOT EXISTS (
                        SELECT 1
                        FROM evaluation_job_items AS active
                        WHERE active.work_item_id = ji.work_item_id
                          AND active.status IN ('pending', 'running', 'retry_wait')
                          AND (
                              active.status = 'running'
                              OR active.created_at < ji.created_at
                              OR (
                                  active.created_at = ji.created_at
                                  AND active.id < ji.id
                              )
                          )
                    )
                    ORDER BY ji.created_at ASC, ji.id ASC
                    LIMIT ?
                    """,
                    (now_iso, limit),
                )
                rows = await cursor.fetchall()
                for row in rows:
                    if len(claimed) >= limit:
                        break
                    attempt_id = str(uuid4())
                    attempt_cursor = await connection.execute(
                        """
                        SELECT COALESCE(MAX(attempt_number), 0) + 1 AS attempt_number
                        FROM evaluation_attempts
                        WHERE work_item_id = ?
                        """,
                        (row["work_item_id"],),
                    )
                    attempt_number = (await attempt_cursor.fetchone())["attempt_number"]
                    await connection.execute(
                        """
                        INSERT INTO evaluation_attempts (
                            id, job_id, job_item_id, work_item_id, attempt_number,
                            status, started_at, last_heartbeat_at, finished_at,
                            error_type, safe_error_message
                        ) VALUES (?, ?, ?, ?, ?, 'running', ?, NULL, NULL, NULL, NULL)
                        """,
                        (
                            attempt_id,
                            row["job_id"],
                            row["job_item_id"],
                            row["work_item_id"],
                            attempt_number,
                            now_iso,
                        ),
                    )
                    await connection.execute(
                        """
                        UPDATE evaluation_job_items
                        SET status = 'running', active_attempt_id = ?, next_retry_at = NULL,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (attempt_id, now_iso, row["job_item_id"]),
                    )
                    claimed.append(
                        ClaimedEvaluationWork(
                            job_id=row["job_id"],
                            job_item_id=row["job_item_id"],
                            work_item_id=row["work_item_id"],
                            attempt_id=attempt_id,
                            attempt_number=attempt_number,
                            input_snapshot=json.loads(row["input_snapshot_json"]),
                        )
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return claimed

    async def fail_attempt(
        self,
        claim: ClaimedEvaluationWork,
        decision: ErrorDecision,
        *,
        next_retry_at: datetime | None,
    ) -> EvaluationAttempt:
        """Finish a failed attempt and either requeue it or exhaust its budget."""
        now = datetime.now(timezone.utc)
        now_iso = _as_iso(now)
        await init_db()
        async with connect_db() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                row = await self._active_claim_row(connection, claim)
                retry = decision.retryable and row["attempt_count"] < row["max_attempts"]
                item_status = "retry_wait" if retry else "failed"
                retry_at = _as_iso(next_retry_at) if retry and next_retry_at else None
                if retry and retry_at is None:
                    retry_at = _as_iso(
                        now
                        + timedelta(
                            seconds=retry_delay_seconds(
                                row["attempt_count"], decision.retry_after_seconds
                            )
                        )
                    )
                await connection.execute(
                    """
                    UPDATE evaluation_attempts
                    SET status = 'failed', finished_at = ?, error_type = ?,
                        safe_error_message = ?
                    WHERE id = ?
                    """,
                    (now_iso, decision.error_type, decision.safe_message, claim.attempt_id),
                )
                await connection.execute(
                    """
                    UPDATE evaluation_job_items
                    SET status = ?, next_retry_at = ?, active_attempt_id = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (item_status, retry_at, now_iso, claim.job_item_id),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return EvaluationAttempt(
            attempt_id=claim.attempt_id,
            job_id=claim.job_id,
            job_item_id=claim.job_item_id,
            work_item_id=claim.work_item_id,
            attempt_number=claim.attempt_number,
            status="failed",
            started_at=_from_iso(row["started_at"]),
            last_heartbeat_at=(
                _from_iso(row["last_heartbeat_at"])
                if row["last_heartbeat_at"] is not None
                else None
            ),
            finished_at=now,
            error_type=decision.error_type,
            safe_error_message=decision.safe_message,
        )

    async def cancel_attempt(
        self, claim: ClaimedEvaluationWork, *, safe_message: str
    ) -> EvaluationAttempt:
        """Cancel the active claim without deleting its attempt history."""
        now = datetime.now(timezone.utc)
        now_iso = _as_iso(now)
        await init_db()
        async with connect_db() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                row = await self._active_claim_row(connection, claim)
                await connection.execute(
                    """
                    UPDATE evaluation_attempts
                    SET status = 'cancelled', finished_at = ?, error_type = 'cancelled',
                        safe_error_message = ?
                    WHERE id = ?
                    """,
                    (now_iso, safe_message, claim.attempt_id),
                )
                await connection.execute(
                    """
                    UPDATE evaluation_job_items
                    SET status = 'cancelled', active_attempt_id = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (now_iso, claim.job_item_id),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return EvaluationAttempt(
            attempt_id=claim.attempt_id,
            job_id=claim.job_id,
            job_item_id=claim.job_item_id,
            work_item_id=claim.work_item_id,
            attempt_number=claim.attempt_number,
            status="cancelled",
            started_at=_from_iso(row["started_at"]),
            last_heartbeat_at=(
                _from_iso(row["last_heartbeat_at"])
                if row["last_heartbeat_at"] is not None
                else None
            ),
            finished_at=now,
            error_type="cancelled",
            safe_error_message=safe_message,
        )

    async def heartbeat_attempt(self, attempt_id: str, *, at: datetime) -> None:
        """Record liveness only while an attempt remains running."""
        await init_db()
        async with connect_db() as connection:
            await connection.execute(
                """
                UPDATE evaluation_attempts
                SET last_heartbeat_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (_as_iso(at), attempt_id),
            )
            await connection.commit()

    async def recover_interrupted_attempts(self, *, at: datetime) -> int:
        """Mark in-flight attempts interrupted and requeue items with budget left."""
        await init_db()
        at_iso = _as_iso(at)
        async with connect_db() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    """
                    SELECT attempt.id AS attempt_id, attempt.job_item_id,
                           item.max_attempts,
                           (SELECT COUNT(*) FROM evaluation_attempts AS counted
                            WHERE counted.job_item_id = item.id) AS attempt_count
                    FROM evaluation_attempts AS attempt
                    JOIN evaluation_job_items AS item ON item.id = attempt.job_item_id
                    WHERE attempt.status = 'running'
                    """
                )
                rows = await cursor.fetchall()
                for row in rows:
                    await connection.execute(
                        """
                        UPDATE evaluation_attempts
                        SET status = 'interrupted', finished_at = ?, error_type = 'interrupted',
                            safe_error_message = 'Evaluation processing was interrupted.'
                        WHERE id = ?
                        """,
                        (at_iso, row["attempt_id"]),
                    )
                    item_status = "pending" if row["attempt_count"] < row["max_attempts"] else "failed"
                    await connection.execute(
                        """
                        UPDATE evaluation_job_items
                        SET status = ?, next_retry_at = NULL, active_attempt_id = NULL,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (item_status, at_iso, row["job_item_id"]),
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return len(rows)

    async def list_jobs(self, *, user_id: str, campaign_id: str) -> list[EvaluationJob]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM evaluation_jobs
                WHERE user_id = ? AND campaign_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (user_id, campaign_id),
            )
            return [_row_to_job(row) for row in await cursor.fetchall()]

    async def get_job(self, *, user_id: str, job_id: str) -> EvaluationJob:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM evaluation_jobs WHERE id = ? AND user_id = ?", (job_id, user_id)
            )
            row = await cursor.fetchone()
        if row is None:
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Evaluation job not found",
                status_code=404,
            )
        return _row_to_job(row)

    async def list_attempts(
        self, *, user_id: str, work_item_id: str
    ) -> list[EvaluationAttempt]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT attempt.*
                FROM evaluation_attempts AS attempt
                JOIN evaluation_jobs AS job ON job.id = attempt.job_id
                WHERE job.user_id = ? AND attempt.work_item_id = ?
                ORDER BY attempt.attempt_number ASC
                """,
                (user_id, work_item_id),
            )
            return [_row_to_attempt(row) for row in await cursor.fetchall()]

    async def count_work_items(self, *, campaign_id: str) -> int:
        return await self._count("evaluation_work_items", campaign_id)

    async def count_job_items(self, *, campaign_id: str) -> int:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM evaluation_job_items AS item
                JOIN evaluation_jobs AS job ON job.id = item.job_id
                WHERE job.campaign_id = ?
                """,
                (campaign_id,),
            )
            return (await cursor.fetchone())["count"]

    async def get_job_item_status(self, job_item_id: str) -> str | None:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT status FROM evaluation_job_items WHERE id = ?", (job_item_id,)
            )
            row = await cursor.fetchone()
        return row["status"] if row is not None else None

    async def _upsert_work_item(
        self,
        connection: aiosqlite.Connection,
        *,
        campaign_id: str,
        spec: WorkItemSpec,
        created_at: str,
    ) -> str:
        cursor = await connection.execute(
            """
            SELECT id FROM evaluation_work_items
            WHERE campaign_id = ? AND logical_key = ?
            """,
            (campaign_id, spec.logical_key),
        )
        row = await cursor.fetchone()
        if row is not None:
            return row["id"]
        work_item_id = str(uuid4())
        await connection.execute(
            """
            INSERT INTO evaluation_work_items (
                id, campaign_id, logical_key, work_type, input_snapshot_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                work_item_id,
                campaign_id,
                spec.logical_key,
                _enum_value(spec.work_type),
                _json_dumps(spec.input_snapshot),
                created_at,
            ),
        )
        return work_item_id

    async def _active_claim_row(
        self, connection: aiosqlite.Connection, claim: ClaimedEvaluationWork
    ) -> aiosqlite.Row:
        cursor = await connection.execute(
            """
            SELECT item.max_attempts, attempt.started_at, attempt.last_heartbeat_at,
                   (SELECT COUNT(*) FROM evaluation_attempts AS counted
                    WHERE counted.job_item_id = item.id) AS attempt_count
            FROM evaluation_job_items AS item
            JOIN evaluation_attempts AS attempt ON attempt.id = item.active_attempt_id
            WHERE item.id = ? AND item.job_id = ? AND item.work_item_id = ?
              AND item.active_attempt_id = ? AND item.status = 'running'
              AND attempt.status = 'running'
            """,
            (claim.job_item_id, claim.job_id, claim.work_item_id, claim.attempt_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise ValueError("The claim is no longer active")
        return row

    async def _count(self, table: str, campaign_id: str) -> int:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                f"SELECT COUNT(*) AS count FROM {table} WHERE campaign_id = ?", (campaign_id,)
            )
            return (await cursor.fetchone())["count"]


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_job(row: aiosqlite.Row) -> EvaluationJob:
    return EvaluationJob(
        job_id=row["id"],
        job_type=row["job_type"],
        user_id=row["user_id"],
        campaign_id=row["campaign_id"],
        selection=json.loads(row["selection_json"]),
        config_snapshot=json.loads(row["config_snapshot_json"]),
        created_at=_from_iso(row["created_at"]),
    )


def _row_to_attempt(row: aiosqlite.Row) -> EvaluationAttempt:
    return EvaluationAttempt(
        attempt_id=row["id"],
        job_id=row["job_id"],
        job_item_id=row["job_item_id"],
        work_item_id=row["work_item_id"],
        attempt_number=row["attempt_number"],
        status=row["status"],
        started_at=_from_iso(row["started_at"]),
        last_heartbeat_at=(
            _from_iso(row["last_heartbeat_at"])
            if row["last_heartbeat_at"] is not None
            else None
        ),
        finished_at=_from_iso(row["finished_at"]) if row["finished_at"] is not None else None,
        error_type=row["error_type"],
        safe_error_message=row["safe_error_message"],
    )
