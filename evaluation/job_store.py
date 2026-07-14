"""SQLite-backed ledger for durable evaluation work."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import aiosqlite

from core.errors import AppError, ErrorCode
from evaluation.campaign_schemas import CampaignResult, CampaignResultStatus
from evaluation.db import CampaignResultRepository, connect_db, init_db
from evaluation.error_policy import ErrorDecision, retry_delay_seconds
from evaluation.job_schemas import (
    ClaimedEvaluationWork,
    EvaluationAttempt,
    EvaluationJob,
    EvaluationJobType,
    EvaluationWorkType,
    ExecutionAttemptOutput,
    RagasAttemptOutput,
    WorkItemSpec,
)


JobCreatedNotifier = Callable[[], None]


def build_evaluation_signature(
    *,
    result: CampaignResult,
    evaluator_model: str,
    evaluator_config: dict[str, Any],
    metric_name: str,
    metric_version: str,
    ground_truth_hash: str,
    context_metrics_enabled: bool,
) -> str:
    """Return a stable identity for a metric evaluated against one result."""
    payload = {
        "campaign_result_id": result.id,
        "final_answer_hash": result.final_answer_hash,
        "evaluator_model": evaluator_model,
        "evaluator_config": evaluator_config,
        "metric_name": metric_name,
        "metric_version": metric_version,
        "context_policy_version": result.context_policy_version,
        "ground_truth_hash": ground_truth_hash,
        "context_metrics_enabled": context_metrics_enabled,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(canonical.encode("utf-8")).hexdigest()

class EvaluationJobStore:
    """Persist jobs, immutable work inputs, and append-only attempts."""

    def __init__(self, *, on_job_created: JobCreatedNotifier | None = None) -> None:
        self._on_job_created = on_job_created

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
        created_job = EvaluationJob(
            job_id=job_id,
            job_type=job_type,
            user_id=user_id,
            campaign_id=campaign_id,
            selection=selection,
            config_snapshot=config_snapshot,
            created_at=_from_iso(now),
        )
        if self._on_job_created is not None:
            self._on_job_created()
        return created_job

    async def claim_ready_items(
        self,
        *,
        limit: int,
        now: datetime,
        work_type: EvaluationWorkType | None = None,
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
                           wi.input_snapshot_json, wi.work_type, wi.logical_key, ji.created_at
                    FROM evaluation_job_items AS ji
                    JOIN evaluation_work_items AS wi ON wi.id = ji.work_item_id
                    WHERE (
                        ji.status = 'pending'
                        OR (ji.status = 'retry_wait' AND ji.next_retry_at <= ?)
                    )
                    AND (? IS NULL OR wi.work_type = ?)
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
                    (
                        now_iso,
                        _enum_value(work_type) if work_type is not None else None,
                        _enum_value(work_type) if work_type is not None else None,
                        limit,
                    ),
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
                            work_type=row["work_type"],
                            logical_key=row["logical_key"],
                        )
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return claimed

    async def next_ready_at(self) -> datetime | None:
        """Return the next retry deadline, if retry-wait work remains."""
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT MIN(next_retry_at) AS next_retry_at
                FROM evaluation_job_items
                WHERE status = 'retry_wait' AND next_retry_at IS NOT NULL
                """
            )
            row = await cursor.fetchone()
        if row is None or row["next_retry_at"] is None:
            return None
        return _from_iso(row["next_retry_at"])

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

    async def complete_execution_attempt(
        self,
        claim: ClaimedEvaluationWork,
        output: ExecutionAttemptOutput,
    ) -> CampaignResult:
        """Atomically retain successful output and promote its official result."""
        now_iso = _as_iso(datetime.now(timezone.utc))
        result = output.result
        if result.status != CampaignResultStatus.COMPLETED:
            raise ValueError("execution output result must be completed")
        await init_db()
        async with connect_db() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await self._active_claim_row(connection, claim)
                job_cursor = await connection.execute(
                    "SELECT user_id, campaign_id FROM evaluation_jobs WHERE id = ?", (claim.job_id,)
                )
                job = await job_cursor.fetchone()
                if job is None or result.campaign_id != job["campaign_id"]:
                    raise ValueError("execution output does not belong to the claimed campaign")
                existing_cursor = await connection.execute(
                    """
                    SELECT id FROM campaign_results
                    WHERE campaign_id = ? AND user_id = ? AND question_id = ?
                      AND mode = ? AND run_number = ?
                    """,
                    (job["campaign_id"], job["user_id"], result.question_id, result.mode, result.run_number),
                )
                existing = await existing_cursor.fetchone()
                result_id = existing["id"] if existing is not None else result.id
                await connection.execute(
                    """
                    INSERT INTO campaign_results (
                        id, campaign_id, user_id, question_id, question, ground_truth,
                        ground_truth_short, key_points_json, ragas_focus_json, mode,
                        execution_profile, context_policy_version, run_number, answer,
                        contexts_json, source_doc_ids_json, expected_sources_json, latency_ms,
                        token_usage_json, category, difficulty, status, error_message,
                        question_version, request_id, started_at, completed_at, total_latency_ms,
                        total_tokens, question_snapshot_json, model_config_snapshot_json,
                        system_version_snapshot_json, derived_metrics_json, final_answer_hash,
                        source_attempt_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(campaign_id, question_id, mode, run_number) DO UPDATE SET
                        question = excluded.question,
                        ground_truth = excluded.ground_truth,
                        ground_truth_short = excluded.ground_truth_short,
                        key_points_json = excluded.key_points_json,
                        ragas_focus_json = excluded.ragas_focus_json,
                        execution_profile = excluded.execution_profile,
                        context_policy_version = excluded.context_policy_version,
                        answer = excluded.answer,
                        contexts_json = excluded.contexts_json,
                        source_doc_ids_json = excluded.source_doc_ids_json,
                        expected_sources_json = excluded.expected_sources_json,
                        latency_ms = excluded.latency_ms,
                        token_usage_json = excluded.token_usage_json,
                        category = excluded.category,
                        difficulty = excluded.difficulty,
                        status = excluded.status,
                        error_message = excluded.error_message,
                        question_version = excluded.question_version,
                        request_id = excluded.request_id,
                        started_at = excluded.started_at,
                        completed_at = excluded.completed_at,
                        total_latency_ms = excluded.total_latency_ms,
                        total_tokens = excluded.total_tokens,
                        question_snapshot_json = excluded.question_snapshot_json,
                        model_config_snapshot_json = excluded.model_config_snapshot_json,
                        system_version_snapshot_json = excluded.system_version_snapshot_json,
                        derived_metrics_json = excluded.derived_metrics_json,
                        final_answer_hash = excluded.final_answer_hash,
                        source_attempt_id = excluded.source_attempt_id
                    """,
                    _campaign_result_values(
                        result_id=result_id,
                        user_id=job["user_id"],
                        result=result,
                        source_attempt_id=claim.attempt_id,
                        created_at=now_iso,
                    ),
                )
                await connection.execute(
                    "UPDATE evaluation_attempts SET status = 'succeeded', finished_at = ?, output_json = ? WHERE id = ?",
                    (now_iso, _json_dumps(output.model_dump(mode="json")), claim.attempt_id),
                )
                await connection.execute(
                    "UPDATE evaluation_work_items SET latest_success_attempt_id = ? WHERE id = ?",
                    (claim.attempt_id, claim.work_item_id),
                )
                await connection.execute(
                    """
                    UPDATE evaluation_job_items
                    SET status = 'succeeded', active_attempt_id = NULL, next_retry_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now_iso, claim.job_item_id),
                )
                await self._recompute_campaign_counts(connection, campaign_id=job["campaign_id"])
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return await CampaignResultRepository().get(
            user_id=job["user_id"], campaign_id=job["campaign_id"], result_id=result_id
        )

    async def complete_ragas_attempt(
        self,
        claim: ClaimedEvaluationWork,
        output: RagasAttemptOutput,
    ) -> None:
        """Atomically retain a successful metric attempt and safely promote scores."""
        now_iso = _as_iso(datetime.now(timezone.utc))
        await init_db()
        async with connect_db() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await self._active_claim_row(connection, claim)
                job_cursor = await connection.execute(
                    "SELECT user_id, campaign_id FROM evaluation_jobs WHERE id = ?", (claim.job_id,)
                )
                job = await job_cursor.fetchone()
                if job is None:
                    raise ValueError("evaluation job not found")
                for score in output.scores:
                    result_id = str(score["campaign_result_id"])
                    metric_name = str(score["metric_name"])
                    signature = score.get("evaluation_signature")
                    result_cursor = await connection.execute(
                        "SELECT 1 FROM campaign_results WHERE id = ? AND campaign_id = ? AND user_id = ?",
                        (result_id, job["campaign_id"], job["user_id"]),
                    )
                    if await result_cursor.fetchone() is None:
                        raise ValueError("RAGAS score does not belong to the claimed campaign")
                    existing_cursor = await connection.execute(
                        "SELECT evaluation_signature FROM ragas_scores WHERE campaign_result_id = ? AND metric_name = ?",
                        (result_id, metric_name),
                    )
                    existing = await existing_cursor.fetchone()
                    if (
                        existing is not None
                        and (
                            existing["evaluation_signature"] is None
                            or signature is None
                            or existing["evaluation_signature"] != signature
                        )
                    ):
                        continue
                    await connection.execute(
                        """
                        INSERT INTO ragas_scores (
                            id, campaign_id, campaign_result_id, user_id, metric_name,
                            metric_value, details_json, source_attempt_id, evaluation_signature, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(campaign_result_id, metric_name) DO UPDATE SET
                            metric_value = excluded.metric_value,
                            details_json = excluded.details_json,
                            source_attempt_id = excluded.source_attempt_id,
                            evaluation_signature = excluded.evaluation_signature,
                            created_at = excluded.created_at
                        """,
                        (
                            str(uuid4()), job["campaign_id"], result_id, job["user_id"], metric_name,
                            float(score["metric_value"]), _json_dumps(score.get("details", {})),
                            claim.attempt_id, signature, now_iso,
                        ),
                    )
                await connection.execute(
                    "UPDATE evaluation_attempts SET status = 'succeeded', finished_at = ?, output_json = ? WHERE id = ?",
                    (now_iso, _json_dumps(output.model_dump(mode="json")), claim.attempt_id),
                )
                await connection.execute(
                    "UPDATE evaluation_work_items SET latest_success_attempt_id = ? WHERE id = ?",
                    (claim.attempt_id, claim.work_item_id),
                )
                await connection.execute(
                    "UPDATE evaluation_job_items SET status = 'succeeded', active_attempt_id = NULL, next_retry_at = NULL, updated_at = ? WHERE id = ?",
                    (now_iso, claim.job_item_id),
                )
                await self._recompute_campaign_counts(connection, campaign_id=job["campaign_id"])
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def backfill_legacy_attempts(self) -> None:
        """Link pre-ledger official projections to deterministic synthetic attempts."""
        await init_db()
        now_iso = _as_iso(datetime.now(timezone.utc))
        async with connect_db() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                campaigns = await (
                    await connection.execute(
                        """
                        SELECT DISTINCT campaign_id, user_id FROM campaign_results WHERE source_attempt_id IS NULL
                        UNION
                        SELECT DISTINCT campaign_id, user_id FROM ragas_scores WHERE source_attempt_id IS NULL
                        """
                    )
                ).fetchall()
                for campaign in campaigns:
                    job_id = str(uuid5(NAMESPACE_URL, f"evaluation-legacy:{campaign['campaign_id']}"))
                    await connection.execute(
                        """
                        INSERT INTO evaluation_jobs (id, user_id, campaign_id, job_type, selection_json, config_snapshot_json, created_at)
                        VALUES (?, ?, ?, 'initial', '{\"legacy_backfill\":true}', '{}', ?)
                        ON CONFLICT(id) DO NOTHING
                        """,
                        (job_id, campaign["user_id"], campaign["campaign_id"], now_iso),
                    )
                    rows = await (
                        await connection.execute(
                            "SELECT * FROM campaign_results WHERE campaign_id = ? AND user_id = ? AND source_attempt_id IS NULL",
                            (campaign["campaign_id"], campaign["user_id"]),
                        )
                    ).fetchall()
                    for row in rows:
                        succeeded = row["status"] == CampaignResultStatus.COMPLETED.value
                        attempt_id = await self._insert_legacy_attempt(
                            connection, job_id=job_id, campaign_id=campaign["campaign_id"],
                            kind="execution", key=row["id"], succeeded=succeeded, now_iso=now_iso,
                        )
                        if succeeded:
                            await connection.execute(
                                "UPDATE campaign_results SET source_attempt_id = ? WHERE id = ? AND source_attempt_id IS NULL",
                                (attempt_id, row["id"]),
                            )
                    scores = await (
                        await connection.execute(
                            "SELECT id, campaign_result_id, metric_name FROM ragas_scores WHERE campaign_id = ? AND user_id = ? AND source_attempt_id IS NULL",
                            (campaign["campaign_id"], campaign["user_id"]),
                        )
                    ).fetchall()
                    for score in scores:
                        attempt_id = await self._insert_legacy_attempt(
                            connection, job_id=job_id, campaign_id=campaign["campaign_id"],
                            kind="ragas", key=f"{score['campaign_result_id']}:{score['metric_name']}",
                            succeeded=True, now_iso=now_iso,
                        )
                        await connection.execute(
                            "UPDATE ragas_scores SET source_attempt_id = ? WHERE id = ? AND source_attempt_id IS NULL",
                            (attempt_id, score["id"]),
                        )
                    await self._recompute_campaign_counts(connection, campaign_id=campaign["campaign_id"])
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

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

    async def cancel_campaign_jobs(self, *, user_id: str, campaign_id: str) -> None:
        """Close active durable work for a cancelled campaign without deleting attempts."""
        now_iso = _as_iso(datetime.now(timezone.utc))
        await init_db()
        async with connect_db() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await connection.execute(
                    """
                    UPDATE evaluation_attempts
                    SET status = 'cancelled', finished_at = ?, error_type = 'cancelled',
                        safe_error_message = 'Campaign cancellation was requested.'
                    WHERE job_id IN (
                        SELECT id FROM evaluation_jobs WHERE user_id = ? AND campaign_id = ?
                    ) AND status = 'running'
                    """,
                    (now_iso, user_id, campaign_id),
                )
                await connection.execute(
                    """
                    UPDATE evaluation_job_items
                    SET status = 'cancelled', active_attempt_id = NULL, next_retry_at = NULL,
                        updated_at = ?
                    WHERE job_id IN (
                        SELECT id FROM evaluation_jobs WHERE user_id = ? AND campaign_id = ?
                    ) AND status IN ('pending', 'running', 'retry_wait')
                    """,
                    (now_iso, user_id, campaign_id),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def _recompute_campaign_counts(
        self, connection: aiosqlite.Connection, *, campaign_id: str
    ) -> None:
        """Keep campaign evaluation counters aligned with terminal job items."""
        await connection.execute(
            """
            UPDATE campaigns
            SET evaluation_total_units = (
                    SELECT COUNT(*)
                    FROM evaluation_job_items AS item
                    JOIN evaluation_jobs AS job ON job.id = item.job_id
                    WHERE job.campaign_id = campaigns.id
                ),
                evaluation_completed_units = (
                    SELECT COUNT(*)
                    FROM evaluation_job_items AS item
                    JOIN evaluation_jobs AS job ON job.id = item.job_id
                    WHERE job.campaign_id = campaigns.id AND item.status = 'succeeded'
                ),
                updated_at = ?
            WHERE id = ?
            """,
            (_as_iso(datetime.now(timezone.utc)), campaign_id),
        )

    async def _insert_legacy_attempt(
        self,
        connection: aiosqlite.Connection,
        *,
        job_id: str,
        campaign_id: str,
        kind: str,
        key: str,
        succeeded: bool,
        now_iso: str,
    ) -> str:
        work_item_id = str(uuid5(NAMESPACE_URL, f"evaluation-legacy-work:{campaign_id}:{kind}:{key}"))
        job_item_id = str(uuid5(NAMESPACE_URL, f"evaluation-legacy-job-item:{job_id}:{work_item_id}"))
        attempt_id = str(uuid5(NAMESPACE_URL, f"evaluation-legacy-attempt:{job_item_id}"))
        status = "succeeded" if succeeded else "failed"
        work_type = "dataset_execution" if kind == "execution" else "ragas_metric"
        await connection.execute(
            """
            INSERT INTO evaluation_work_items (
                id, campaign_id, logical_key, work_type, input_snapshot_json,
                latest_success_attempt_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id, logical_key) DO NOTHING
            """,
            (
                work_item_id, campaign_id, f"legacy:{kind}:{key}", work_type,
                _json_dumps({"legacy": True, "key": key}),
                attempt_id if succeeded else None, now_iso,
            ),
        )
        await connection.execute(
            """
            INSERT INTO evaluation_job_items (
                id, job_id, work_item_id, status, max_attempts, next_retry_at,
                active_attempt_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 1, NULL, NULL, ?, ?)
            ON CONFLICT(job_id, work_item_id) DO NOTHING
            """,
            (job_item_id, job_id, work_item_id, status, now_iso, now_iso),
        )
        await connection.execute(
            """
            INSERT INTO evaluation_attempts (
                id, job_id, job_item_id, work_item_id, attempt_number, status,
                started_at, last_heartbeat_at, finished_at, error_type,
                safe_error_message, output_json
            ) VALUES (?, ?, ?, ?, 1, ?, ?, NULL, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                attempt_id, job_id, job_item_id, work_item_id, status, now_iso, now_iso,
                None if succeeded else "legacy_failed",
                None if succeeded else "Legacy evaluation result was unsuccessful.",
                _json_dumps({"legacy": True, "key": key}),
            ),
        )
        return attempt_id

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


def _campaign_result_values(
    *,
    result_id: str,
    user_id: str,
    result: CampaignResult,
    source_attempt_id: str,
    created_at: str,
) -> tuple[Any, ...]:
    return (
        result_id, result.campaign_id, user_id, result.question_id, result.question,
        result.ground_truth, result.ground_truth_short, _json_dumps(result.key_points),
        _json_dumps(result.ragas_focus), result.mode, result.execution_profile,
        result.context_policy_version, result.run_number, result.answer,
        _json_dumps(result.contexts), _json_dumps(result.source_doc_ids),
        _json_dumps(result.expected_sources), result.latency_ms, _json_dumps(result.token_usage),
        result.category, result.difficulty, result.status.value, result.error_message,
        result.question_version, result.request_id,
        _as_iso(result.started_at) if result.started_at else None,
        _as_iso(result.completed_at) if result.completed_at else None,
        result.total_latency_ms, 0 if result.total_tokens is None else result.total_tokens,
        _json_dumps(result.question_snapshot), _json_dumps(result.model_config_snapshot),
        _json_dumps(result.system_version_snapshot), _json_dumps(result.derived_metrics),
        result.final_answer_hash, source_attempt_id, created_at,
    )


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
