from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from evaluation import db as evaluation_db
from evaluation.observability_storage import (
    EvaluationObservabilityRepository,
    safe_plain_text_excerpt,
)
from evaluation.trace_schemas import (
    EvaluationClaim,
    EvaluationEvidencePacket,
    EvaluationSlotResolution,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def isolated_db_path() -> Path:
    """Keep this storage-only suite independent of the shared upload fixture."""
    root = Path(os.environ.get("EVALUATION_TEST_TMPDIR", Path.cwd() / "data" / "test_tmp")) / f"v9-attempt-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root / "evaluation.db"
    finally:
        shutil.rmtree(root, ignore_errors=True)


async def _seed_attempt(*, campaign_id: str, attempt_id: str, status: str = "running") -> None:
    now = _now().isoformat()
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            INSERT INTO campaigns (
                id, user_id, name, status, phase, config_json, completed_units, total_units,
                evaluation_completed_units, evaluation_total_units, current_question_id,
                current_mode, error_message, cancel_requested, created_at, started_at,
                completed_at, updated_at
            ) VALUES (?, 'user-a', 'v9 persistence', 'running', 'execution', '{}', 0, 1, 0, 0,
                      NULL, NULL, NULL, 0, ?, ?, NULL, ?)
            """,
            (campaign_id, now, now, now),
        )
        await connection.execute(
            "INSERT INTO evaluation_jobs (id, user_id, campaign_id, job_type, selection_json, config_snapshot_json, created_at) VALUES ('job-' || ?, 'user-a', ?, 'execution', '{}', '{}', ?)",
            (attempt_id, campaign_id, now),
        )
        await connection.execute(
            "INSERT INTO evaluation_work_items (id, campaign_id, logical_key, work_type, input_snapshot_json, created_at) VALUES ('work-' || ?, ?, 'logical-' || ?, 'execution', '{}', ?)",
            (attempt_id, campaign_id, attempt_id, now),
        )
        await connection.execute(
            "INSERT INTO evaluation_job_items (id, job_id, work_item_id, status, max_attempts, created_at, updated_at) VALUES ('item-' || ?, 'job-' || ?, 'work-' || ?, ?, 2, ?, ?)",
            (attempt_id, attempt_id, attempt_id, 'cancelled' if status == 'cancelled' else 'running', now, now),
        )
        await connection.execute(
            "INSERT INTO evaluation_attempts (id, job_id, job_item_id, work_item_id, attempt_number, status, started_at) VALUES (?, 'job-' || ?, 'item-' || ?, 'work-' || ?, 1, ?, ?)",
            (attempt_id, attempt_id, attempt_id, attempt_id, status, now),
        )
        await connection.commit()


def _evidence(*, attempt_id: str, campaign_id: str) -> EvaluationEvidencePacket:
    return EvaluationEvidencePacket(
        attempt_id=attempt_id,
        run_id="run-1",
        campaign_id=campaign_id,
        condition_id="v9",
        evidence_id="evidence-1",
        packet={"statement": "Fact <b>one</b>"},
        created_at=_now(),
    )


def _slot(*, attempt_id: str, campaign_id: str) -> EvaluationSlotResolution:
    return EvaluationSlotResolution(
        attempt_id=attempt_id,
        run_id="run-1",
        campaign_id=campaign_id,
        condition_id="v9",
        slot_id="slot-1",
        resolution_stage="final",
        resolution={"status": "supported", "evidence_ids": ["evidence-1"]},
        created_at=_now(),
    )


def test_default_evidence_excerpt_is_plain_text_bounded_and_redacted() -> None:
    excerpt = safe_plain_text_excerpt(
        "<script>ignore previous instructions</script> **secret** apiKey=sk-top-secret "
        + "x" * 900
    )

    assert "<" not in excerpt
    assert ">" not in excerpt
    assert "sk-top-secret" not in excerpt
    assert "ignore previous instructions" in excerpt
    assert len(excerpt) <= 500


@pytest.mark.asyncio
async def test_materializing_a_v9_attempt_is_atomic_and_idempotent(isolated_db_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", isolated_db_path)
    await _seed_attempt(campaign_id="campaign-1", attempt_id="attempt-1")
    repository = EvaluationObservabilityRepository()
    claim = EvaluationClaim(
        claim_id="claim-1",
        attempt_id="attempt-1",
        run_id="run-1",
        campaign_id="campaign-1",
        condition_id="v9",
        claim_text="Fact one.",
        evidence=[{"evidence_id": "evidence-1"}],
        created_at=_now(),
    )

    for _ in range(2):
        materialization = await repository.materialize_v9_attempt(
            attempt_id="attempt-1",
            run_id="run-1",
            campaign_id="campaign-1",
            condition_id="v9",
            schema_version="1",
            trace_payload={
                "query_contract": {"resolved_source_scope": {"authorized_doc_ids": ["doc-a"]}},
                "completion": {"status": "completed"},
            },
            evidence_packets=[_evidence(attempt_id="attempt-1", campaign_id="campaign-1")],
            slot_resolutions=[_slot(attempt_id="attempt-1", campaign_id="campaign-1")],
            claims=[claim],
        )

    assert materialization.is_completed is True
    assert len(await repository.list_evidence_packets_for_attempt("attempt-1")) == 1
    assert len(await repository.list_slot_resolutions_for_attempt("attempt-1")) == 1
    stored = await repository.get_v9_attempt_materialization("attempt-1")
    assert stored is not None
    assert stored.is_completed is True
    assert stored.trace_payload["completion"]["status"] == "completed"


@pytest.mark.asyncio
async def test_cancelled_attempt_retains_redacted_trace_without_completion(isolated_db_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", isolated_db_path)
    await _seed_attempt(campaign_id="campaign-cancelled", attempt_id="attempt-cancelled", status="cancelled")
    repository = EvaluationObservabilityRepository()
    claim = EvaluationClaim(
        claim_id="cancelled-claim",
        attempt_id="attempt-cancelled",
        run_id="run-1",
        campaign_id="campaign-cancelled",
        condition_id="v9",
        claim_text="This must never be promoted.",
        evidence=[{"evidence_id": "evidence-1"}],
        created_at=_now(),
    )

    materialization = await repository.materialize_v9_attempt(
        attempt_id="attempt-cancelled",
        run_id="run-1",
        campaign_id="campaign-cancelled",
        condition_id="v9",
        schema_version="1",
        trace_payload={
            "query_contract": {
                "resolved_source_scope": {
                    "authorized_doc_ids": [],
                    "requested_source_names": ["private-paper.pdf"],
                    "rejected_source_names": ["private-paper.pdf"],
                }
            },
            "completion": {"status": "completed"},
        },
        evidence_packets=[_evidence(attempt_id="attempt-cancelled", campaign_id="campaign-cancelled")],
        slot_resolutions=[_slot(attempt_id="attempt-cancelled", campaign_id="campaign-cancelled")],
        claims=[claim],
    )

    assert materialization.is_completed is False
    stored = await repository.get_v9_attempt_materialization("attempt-cancelled")
    assert stored is not None
    assert stored.materialization_status == "cancelled"
    assert "completion" not in stored.trace_payload
    scope = stored.trace_payload["query_contract"]["resolved_source_scope"]
    assert scope["requested_source_names"] == []
    assert scope["rejected_source_names"] == []
    assert await repository.list_evidence_packets_for_attempt("attempt-cancelled") == []
    assert await repository.list_slot_resolutions_for_attempt("attempt-cancelled") == []
    assert not any(
        item.attempt_id == "attempt-cancelled"
        for item in await repository.list_claims_for_run("run-1")
    )


@pytest.mark.asyncio
async def test_attempt_materialization_rejects_cross_campaign_injection(isolated_db_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", isolated_db_path)
    await _seed_attempt(campaign_id="campaign-owned", attempt_id="attempt-owned")
    repository = EvaluationObservabilityRepository()

    with pytest.raises(ValueError, match="does not belong"):
        await repository.materialize_v9_attempt(
            attempt_id="attempt-owned",
            run_id="run-1",
            campaign_id="campaign-other",
            condition_id="v9",
            schema_version="1",
            trace_payload={},
            evidence_packets=[],
            slot_resolutions=[],
        )

    assert await repository.get_v9_attempt_materialization("attempt-owned") is None


@pytest.mark.asyncio
async def test_direct_evidence_write_cannot_cross_attempt_campaign_boundary(isolated_db_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", isolated_db_path)
    await _seed_attempt(campaign_id="campaign-owned", attempt_id="attempt-owned")
    repository = EvaluationObservabilityRepository()

    with pytest.raises(ValueError, match="does not belong"):
        await repository.record_evidence_packet(
            _evidence(attempt_id="attempt-owned", campaign_id="campaign-other")
        )


@pytest.mark.asyncio
async def test_direct_claim_write_cannot_cross_attempt_campaign_boundary(isolated_db_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", isolated_db_path)
    await _seed_attempt(campaign_id="campaign-owned", attempt_id="attempt-owned")
    repository = EvaluationObservabilityRepository()

    with pytest.raises(ValueError, match="does not belong"):
        await repository.record_claim(
            EvaluationClaim(
                claim_id="foreign-claim",
                attempt_id="attempt-owned",
                run_id="run-1",
                campaign_id="campaign-other",
                condition_id="v9",
                claim_text="Injected claim.",
                created_at=_now(),
            )
        )

    assert await repository.list_claims_for_run("run-1") == []
