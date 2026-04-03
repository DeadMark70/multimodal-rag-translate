"""SQLite persistence for evaluation campaigns."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import aiosqlite

from core.errors import AppError, ErrorCode
from evaluation.agentic_evaluation_service import LEGACY_SHARED_PROFILE
from evaluation.campaign_schemas import (
    CampaignConfig,
    CampaignLifecycleStatus,
    CampaignResult,
    CampaignResultStatus,
    CampaignStatus,
)
from evaluation.trace_schemas import AgentTraceDetail, AgentTraceSummary, summarize_agent_trace

EVALUATION_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation.db"
_UNSET = object()
logger = logging.getLogger(__name__)
ROUTE_PROFILE_ALIASES = {
    "hybrid_graph": "generic_graph",
}
_INIT_SQL = """
CREATE TABLE IF NOT EXISTS campaigns (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT,
    status TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'execution',
    config_json TEXT NOT NULL,
    completed_units INTEGER NOT NULL DEFAULT 0,
    total_units INTEGER NOT NULL DEFAULT 0,
    evaluation_completed_units INTEGER NOT NULL DEFAULT 0,
    evaluation_total_units INTEGER NOT NULL DEFAULT 0,
    current_question_id TEXT,
    current_mode TEXT,
    error_message TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_campaigns_user_created
ON campaigns(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS campaign_results (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    question TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    ground_truth_short TEXT,
    key_points_json TEXT NOT NULL DEFAULT '[]',
    ragas_focus_json TEXT NOT NULL DEFAULT '[]',
    mode TEXT NOT NULL,
    execution_profile TEXT,
    context_policy_version TEXT,
    run_number INTEGER NOT NULL,
    answer TEXT NOT NULL,
    contexts_json TEXT NOT NULL,
    source_doc_ids_json TEXT NOT NULL,
    expected_sources_json TEXT NOT NULL,
    latency_ms REAL NOT NULL DEFAULT 0,
    token_usage_json TEXT NOT NULL,
    category TEXT,
    difficulty TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_campaign_results_campaign_created
ON campaign_results(campaign_id, created_at ASC);

CREATE TABLE IF NOT EXISTS agent_traces (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    campaign_result_id TEXT,
    user_id TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ragas_scores (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    campaign_result_id TEXT,
    user_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ragas_scores_result_metric
ON ragas_scores(campaign_result_id, metric_name);

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_traces_result
ON agent_traces(campaign_result_id);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def connect_db():
    """Open SQLite connection with WAL-friendly settings."""
    db_path = Path(EVALUATION_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(db_path)
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA journal_mode=WAL;")
    await connection.execute("PRAGMA synchronous=NORMAL;")
    await connection.execute("PRAGMA foreign_keys=ON;")
    try:
        yield connection
    finally:
        await connection.close()


async def init_db() -> None:
    """Initialize evaluation database and future-proof tables."""
    async with connect_db() as connection:
        await connection.executescript(_INIT_SQL)
        await _apply_migrations(connection)
        await connection.commit()


async def _apply_migrations(connection: aiosqlite.Connection) -> None:
    """Apply additive migrations for existing Phase 2 databases."""
    campaign_columns = await _table_columns(connection, "campaigns")
    if "phase" not in campaign_columns:
        await connection.execute(
            "ALTER TABLE campaigns ADD COLUMN phase TEXT NOT NULL DEFAULT 'execution'"
        )
    if "evaluation_completed_units" not in campaign_columns:
        await connection.execute(
            "ALTER TABLE campaigns ADD COLUMN evaluation_completed_units INTEGER NOT NULL DEFAULT 0"
        )
    if "evaluation_total_units" not in campaign_columns:
        await connection.execute(
            "ALTER TABLE campaigns ADD COLUMN evaluation_total_units INTEGER NOT NULL DEFAULT 0"
        )

    campaign_result_columns = await _table_columns(connection, "campaign_results")
    if "execution_profile" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN execution_profile TEXT"
        )
    if "context_policy_version" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN context_policy_version TEXT"
        )
    if "ground_truth_short" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN ground_truth_short TEXT"
        )
    if "key_points_json" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN key_points_json TEXT NOT NULL DEFAULT '[]'"
        )
    if "ragas_focus_json" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN ragas_focus_json TEXT NOT NULL DEFAULT '[]'"
        )

    await connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ragas_scores_result_metric
        ON ragas_scores(campaign_result_id, metric_name)
        """
    )
    await connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_traces_result
        ON agent_traces(campaign_result_id)
        """
    )


async def _table_columns(connection: aiosqlite.Connection, table_name: str) -> set[str]:
    cursor = await connection.execute(f"PRAGMA table_info({table_name})")
    rows = await cursor.fetchall()
    return {str(row[1]) for row in rows}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _json_loads(payload: str | None, fallback: Any) -> Any:
    if not payload:
        return fallback
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return fallback


def _row_to_campaign_status(row: aiosqlite.Row) -> CampaignStatus:
    config_payload = _json_loads(row["config_json"], {})
    return CampaignStatus(
        id=row["id"],
        name=row["name"],
        status=CampaignLifecycleStatus(row["status"]),
        phase=row["phase"],
        config=CampaignConfig.model_validate(config_payload),
        completed_units=row["completed_units"],
        total_units=row["total_units"],
        evaluation_completed_units=row["evaluation_completed_units"],
        evaluation_total_units=row["evaluation_total_units"],
        current_question_id=row["current_question_id"],
        current_mode=row["current_mode"],
        error_message=row["error_message"],
        cancel_requested=bool(row["cancel_requested"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_campaign_result(row: aiosqlite.Row) -> CampaignResult:
    execution_profile = row["execution_profile"] if "execution_profile" in row.keys() else None
    context_policy_version = (
        row["context_policy_version"] if "context_policy_version" in row.keys() else None
    )
    if not execution_profile and row["mode"] == "agentic":
        execution_profile = LEGACY_SHARED_PROFILE

    return CampaignResult(
        id=row["id"],
        campaign_id=row["campaign_id"],
        question_id=row["question_id"],
        question=row["question"],
        ground_truth=row["ground_truth"],
        ground_truth_short=row["ground_truth_short"] if "ground_truth_short" in row.keys() else None,
        key_points=_json_loads(row["key_points_json"], []) if "key_points_json" in row.keys() else [],
        ragas_focus=_json_loads(row["ragas_focus_json"], []) if "ragas_focus_json" in row.keys() else [],
        mode=row["mode"],
        execution_profile=execution_profile,
        context_policy_version=context_policy_version,
        run_number=row["run_number"],
        answer=row["answer"],
        contexts=_json_loads(row["contexts_json"], []),
        source_doc_ids=_json_loads(row["source_doc_ids_json"], []),
        expected_sources=_json_loads(row["expected_sources_json"], []),
        latency_ms=row["latency_ms"],
        token_usage=_json_loads(row["token_usage_json"], {}),
        category=row["category"],
        difficulty=row["difficulty"],
        status=CampaignResultStatus(row["status"]),
        error_message=row["error_message"],
        has_trace=bool(row["has_trace"]) if "has_trace" in row.keys() else False,
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _normalize_route_profile(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return ROUTE_PROFILE_ALIASES.get(value, value)


def _normalize_trace_route_profiles(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["route_profile"] = _normalize_route_profile(normalized.get("route_profile"))
    steps = normalized.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            metadata = step.get("metadata")
            if isinstance(metadata, dict):
                metadata["route_profile"] = _normalize_route_profile(metadata.get("route_profile"))
    return normalized


def _row_to_agent_trace_detail(row: aiosqlite.Row) -> AgentTraceDetail:
    payload = _normalize_trace_route_profiles(_json_loads(row["trace_json"], {}))
    if not payload:
        raise AppError(
            code=ErrorCode.INTERNAL_ERROR,
            message="Stored agent trace is invalid",
            status_code=500,
        )
    try:
        if not payload.get("execution_profile") and payload.get("mode") == "agentic":
            payload["execution_profile"] = LEGACY_SHARED_PROFILE
        return AgentTraceDetail.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse agent trace row %s: %s", row["id"], exc)
        raise AppError(
            code=ErrorCode.INTERNAL_ERROR,
            message="Stored agent trace is invalid",
            status_code=500,
        ) from exc


class CampaignRepository:
    """CRUD operations for campaign lifecycle rows."""

    async def create(
        self,
        *,
        user_id: str,
        name: Optional[str],
        config: CampaignConfig,
    ) -> CampaignStatus:
        await init_db()
        campaign_id = str(uuid4())
        now = _utc_now_iso()
        total_units = len(config.test_case_ids) * len(config.modes) * config.repeat_count
        async with connect_db() as connection:
            await connection.execute(
                """
                INSERT INTO campaigns (
                    id, user_id, name, status, phase, config_json, completed_units, total_units,
                    evaluation_completed_units, evaluation_total_units, current_question_id,
                    current_mode, error_message, cancel_requested, created_at, started_at,
                    completed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, NULL, NULL, NULL, 0, ?, NULL, NULL, ?)
                """,
                (
                    campaign_id,
                    user_id,
                    name,
                    CampaignLifecycleStatus.PENDING.value,
                    "execution",
                    _json_dumps(config.model_dump(mode="json", by_alias=True)),
                    0,
                    total_units,
                    now,
                    now,
                ),
            )
            await connection.commit()
        return await self.get(user_id=user_id, campaign_id=campaign_id)

    async def get(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM campaigns WHERE id = ? AND user_id = ?",
                (campaign_id, user_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Campaign not found",
                status_code=404,
            )
        return _row_to_campaign_status(row)

    async def list_by_user(self, *, user_id: str) -> list[CampaignStatus]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM campaigns WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
            rows = await cursor.fetchall()
        return [_row_to_campaign_status(row) for row in rows]

    async def mark_running(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            status=CampaignLifecycleStatus.RUNNING,
            phase="execution",
            started_at=_utc_now_iso(),
        )

    async def update_progress(
        self,
        *,
        user_id: str,
        campaign_id: str,
        completed_units: int,
        evaluation_completed_units: Optional[int] = None,
        evaluation_total_units: Optional[int] = None,
        current_question_id: Optional[str],
        current_mode: Optional[str],
    ) -> CampaignStatus:
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            completed_units=completed_units,
            evaluation_completed_units=evaluation_completed_units,
            evaluation_total_units=evaluation_total_units,
            current_question_id=current_question_id,
            current_mode=current_mode,
        )

    async def mark_evaluating(
        self,
        *,
        user_id: str,
        campaign_id: str,
        evaluation_total_units: int,
    ) -> CampaignStatus:
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            status=CampaignLifecycleStatus.EVALUATING,
            phase="evaluation",
            evaluation_completed_units=0,
            evaluation_total_units=evaluation_total_units,
            error_message=None,
        )

    async def mark_completed(
        self,
        *,
        user_id: str,
        campaign_id: str,
        phase: Optional[str] = None,
    ) -> CampaignStatus:
        now = _utc_now_iso()
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            status=CampaignLifecycleStatus.COMPLETED,
            phase=phase,
            completed_at=now,
            current_question_id=None,
            current_mode=None,
        )

    async def mark_failed(
        self,
        *,
        user_id: str,
        campaign_id: str,
        error_message: str,
        phase: Optional[str] = None,
    ) -> CampaignStatus:
        now = _utc_now_iso()
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            status=CampaignLifecycleStatus.FAILED,
            error_message=error_message,
            phase=phase,
            completed_at=now,
            current_question_id=None,
            current_mode=None,
        )

    async def mark_cancelled(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        now = _utc_now_iso()
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            status=CampaignLifecycleStatus.CANCELLED,
            completed_at=now,
            current_question_id=None,
            current_mode=None,
        )

    async def request_cancel(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            cancel_requested=True,
        )

    async def is_cancel_requested(self, *, user_id: str, campaign_id: str) -> bool:
        campaign = await self.get(user_id=user_id, campaign_id=campaign_id)
        return campaign.cancel_requested

    async def _update_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        status: Optional[CampaignLifecycleStatus] = None,
        phase: Optional[str] = None,
        completed_units: Optional[int] = None,
        evaluation_completed_units: Optional[int] = None,
        evaluation_total_units: Optional[int] = None,
        current_question_id: Optional[str] | object = _UNSET,
        current_mode: Optional[str] | object = _UNSET,
        error_message: Optional[str] | object = _UNSET,
        cancel_requested: Optional[bool] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ) -> CampaignStatus:
        await init_db()
        updates: list[str] = ["updated_at = ?"]
        values: list[Any] = [_utc_now_iso()]

        if status is not None:
            updates.append("status = ?")
            values.append(status.value)
        if phase is not None:
            updates.append("phase = ?")
            values.append(phase)
        if completed_units is not None:
            updates.append("completed_units = ?")
            values.append(completed_units)
        if evaluation_completed_units is not None:
            updates.append("evaluation_completed_units = ?")
            values.append(evaluation_completed_units)
        if evaluation_total_units is not None:
            updates.append("evaluation_total_units = ?")
            values.append(evaluation_total_units)
        if current_question_id is not _UNSET:
            updates.append("current_question_id = ?")
            values.append(current_question_id)
        if current_mode is not _UNSET:
            updates.append("current_mode = ?")
            values.append(current_mode)
        if error_message is not _UNSET:
            updates.append("error_message = ?")
            values.append(error_message)
        if cancel_requested is not None:
            updates.append("cancel_requested = ?")
            values.append(1 if cancel_requested else 0)
        if started_at is not None:
            updates.append("started_at = ?")
            values.append(started_at)
        if completed_at is not None:
            updates.append("completed_at = ?")
            values.append(completed_at)

        values.extend([campaign_id, user_id])

        async with connect_db() as connection:
            cursor = await connection.execute(
                f"UPDATE campaigns SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
                values,
            )
            await connection.commit()
            if cursor.rowcount == 0:
                raise AppError(
                    code=ErrorCode.NOT_FOUND,
                    message="Campaign not found",
                    status_code=404,
                )
        return await self.get(user_id=user_id, campaign_id=campaign_id)


class CampaignResultRepository:
    """Persistence for per-unit campaign outputs."""

    async def create(
        self,
        *,
        user_id: str,
        campaign_id: str,
        question_id: str,
        question: str,
        ground_truth: str,
        ground_truth_short: Optional[str],
        key_points: list[str],
        ragas_focus: list[str],
        mode: str,
        execution_profile: Optional[str],
        context_policy_version: Optional[str],
        run_number: int,
        answer: str,
        contexts: list[str],
        source_doc_ids: list[str],
        expected_sources: list[str],
        latency_ms: float,
        token_usage: dict[str, Any],
        category: Optional[str],
        difficulty: Optional[str],
        status: CampaignResultStatus,
        error_message: Optional[str] = None,
    ) -> CampaignResult:
        await init_db()
        result_id = str(uuid4())
        created_at = _utc_now_iso()
        async with connect_db() as connection:
            await connection.execute(
                """
                INSERT INTO campaign_results (
                    id, campaign_id, user_id, question_id, question, ground_truth,
                    ground_truth_short, key_points_json, ragas_focus_json, mode, execution_profile,
                    context_policy_version, run_number, answer, contexts_json, source_doc_ids_json,
                    expected_sources_json, latency_ms, token_usage_json, category,
                    difficulty, status, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    campaign_id,
                    user_id,
                    question_id,
                    question,
                    ground_truth,
                    ground_truth_short,
                    _json_dumps(key_points),
                    _json_dumps(ragas_focus),
                    mode,
                    execution_profile,
                    context_policy_version,
                    run_number,
                    answer,
                    _json_dumps(contexts),
                    _json_dumps(source_doc_ids),
                    _json_dumps(expected_sources),
                    latency_ms,
                    _json_dumps(token_usage),
                    category,
                    difficulty,
                    status.value,
                    error_message,
                    created_at,
                ),
            )
            await connection.commit()
        return await self.get(user_id=user_id, campaign_id=campaign_id, result_id=result_id)

    async def get(
        self,
        *,
        user_id: str,
        campaign_id: str,
        result_id: str,
    ) -> CampaignResult:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT campaign_results.*,
                       EXISTS(
                           SELECT 1
                           FROM agent_traces
                           WHERE agent_traces.campaign_result_id = campaign_results.id
                       ) AS has_trace
                FROM campaign_results
                WHERE id = ? AND campaign_id = ? AND user_id = ?
                """,
                (result_id, campaign_id, user_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Campaign result not found",
                status_code=404,
            )
        return _row_to_campaign_result(row)

    async def list_for_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> list[CampaignResult]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT campaign_results.*,
                       EXISTS(
                           SELECT 1
                           FROM agent_traces
                           WHERE agent_traces.campaign_result_id = campaign_results.id
                       ) AS has_trace
                FROM campaign_results
                WHERE campaign_id = ? AND user_id = ?
                ORDER BY created_at ASC, question_id ASC, mode ASC, run_number ASC
                """,
                (campaign_id, user_id),
            )
            rows = await cursor.fetchall()
        return [_row_to_campaign_result(row) for row in rows]


class AgentTraceRepository:
    """Persistence helpers for campaign-linked agent traces."""

    async def replace_for_result(
        self,
        *,
        user_id: str,
        campaign_id: str,
        campaign_result_id: str,
        trace_payload: dict[str, Any],
    ) -> AgentTraceDetail:
        await init_db()
        normalized_payload = _normalize_trace_route_profiles(trace_payload)
        steps = normalized_payload.get("steps", [])
        tool_call_count = sum(len(step.get("tool_calls", [])) for step in steps)
        total_tokens = sum(int(step.get("token_usage", {}).get("total_tokens", 0) or 0) for step in steps)
        detail = AgentTraceDetail.model_validate(
            {
                "trace_id": normalized_payload.get("trace_id") or str(uuid4()),
                "campaign_id": campaign_id,
                "campaign_result_id": campaign_result_id,
                "question_id": normalized_payload.get("question_id", ""),
                "question": normalized_payload.get("question", ""),
                "mode": normalized_payload.get("mode"),
                "execution_profile": (
                    normalized_payload.get("execution_profile")
                    or (LEGACY_SHARED_PROFILE if normalized_payload.get("mode") == "agentic" else None)
                ),
                "question_intent": normalized_payload.get("question_intent"),
                "strategy_tier": normalized_payload.get("strategy_tier"),
                "route_profile": normalized_payload.get("route_profile"),
                "required_coverage": normalized_payload.get("required_coverage", []),
                "coverage_gaps": normalized_payload.get("coverage_gaps", []),
                "subtask_coverage_status": normalized_payload.get("subtask_coverage_status", {}),
                "supported_claim_count": normalized_payload.get("supported_claim_count", 0),
                "unsupported_claim_count": normalized_payload.get("unsupported_claim_count", 0),
                "claims": normalized_payload.get("claims", []),
                "visual_verification_attempted": normalized_payload.get("visual_verification_attempted", False),
                "visual_tool_call_count": normalized_payload.get("visual_tool_call_count", 0),
                "visual_force_fallback_used": normalized_payload.get("visual_force_fallback_used", False),
                "run_number": normalized_payload.get("run_number", 1),
                "trace_status": normalized_payload.get("trace_status", "completed"),
                "summary": normalized_payload.get("summary", ""),
                "step_count": len(steps),
                "tool_call_count": tool_call_count,
                "total_tokens": total_tokens,
                "created_at": normalized_payload.get("created_at") or _utc_now_iso(),
                "steps": steps,
            }
        )
        async with connect_db() as connection:
            await connection.execute(
                """
                DELETE FROM agent_traces
                WHERE campaign_result_id = ? AND user_id = ?
                """,
                (campaign_result_id, user_id),
            )
            await connection.execute(
                """
                INSERT INTO agent_traces (
                    id, campaign_id, campaign_result_id, user_id, trace_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    detail.trace_id,
                    campaign_id,
                    campaign_result_id,
                    user_id,
                    _json_dumps(detail.model_dump(mode="json")),
                    detail.created_at.isoformat(),
                ),
            )
            await connection.commit()
        return detail

    async def list_for_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> list[AgentTraceSummary]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT *
                FROM agent_traces
                WHERE campaign_id = ? AND user_id = ?
                ORDER BY created_at DESC
                """,
                (campaign_id, user_id),
            )
            rows = await cursor.fetchall()
        return [summarize_agent_trace(_row_to_agent_trace_detail(row)) for row in rows]

    async def get_for_result(
        self,
        *,
        user_id: str,
        campaign_id: str,
        campaign_result_id: str,
    ) -> AgentTraceDetail:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT *
                FROM agent_traces
                WHERE campaign_id = ? AND campaign_result_id = ? AND user_id = ?
                """,
                (campaign_id, campaign_result_id, user_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Agent trace not found",
                status_code=404,
            )
        return _row_to_agent_trace_detail(row)


class RagasScoreRepository:
    """Persistence for per-result RAGAS metrics."""

    async def _insert_score_rows(
        self,
        *,
        connection: Any,
        user_id: str,
        campaign_id: str,
        score_rows: list[dict[str, Any]],
    ) -> None:
        for row in score_rows:
            await connection.execute(
                """
                INSERT INTO ragas_scores (
                    id, campaign_id, campaign_result_id, user_id, metric_name,
                    metric_value, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    campaign_id,
                    row["campaign_result_id"],
                    user_id,
                    row["metric_name"],
                    row["metric_value"],
                    _json_dumps(row.get("details", {})),
                    _utc_now_iso(),
                ),
            )

    async def replace_for_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        score_rows: list[dict[str, Any]],
    ) -> None:
        await init_db()
        async with connect_db() as connection:
            await connection.execute(
                "DELETE FROM ragas_scores WHERE campaign_id = ? AND user_id = ?",
                (campaign_id, user_id),
            )
            await self._insert_score_rows(
                connection=connection,
                user_id=user_id,
                campaign_id=campaign_id,
                score_rows=score_rows,
            )
            await connection.commit()

    async def replace_for_campaign_subset(
        self,
        *,
        user_id: str,
        campaign_id: str,
        selected_result_ids: list[str],
        score_rows: list[dict[str, Any]],
    ) -> None:
        if not selected_result_ids:
            return
        await init_db()
        placeholders = ",".join("?" for _ in selected_result_ids)
        params: tuple[Any, ...] = (
            campaign_id,
            user_id,
            *selected_result_ids,
        )
        async with connect_db() as connection:
            await connection.execute(
                f"""
                DELETE FROM ragas_scores
                WHERE campaign_id = ?
                  AND user_id = ?
                  AND campaign_result_id IN ({placeholders})
                """,
                params,
            )
            await self._insert_score_rows(
                connection=connection,
                user_id=user_id,
                campaign_id=campaign_id,
                score_rows=score_rows,
            )
            await connection.commit()

    async def list_for_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> list[dict[str, Any]]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT campaign_result_id, metric_name, metric_value, details_json
                FROM ragas_scores
                WHERE campaign_id = ? AND user_id = ?
                ORDER BY created_at ASC
                """,
                (campaign_id, user_id),
            )
            rows = await cursor.fetchall()

        return [
            {
                "campaign_result_id": row["campaign_result_id"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "details": _json_loads(row["details_json"], {}),
            }
            for row in rows
        ]



