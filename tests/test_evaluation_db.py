"""Regression coverage for compact campaign agent-trace summaries."""

from __future__ import annotations

import json

import pytest

from evaluation import db as evaluation_db
from evaluation.campaign_schemas import CampaignResultStatus
from evaluation.db import (
    MAX_EVALUATION_ANSWER_BYTES,
    AgentTraceRepository,
    CampaignResultRepository,
)


def _trace_payload() -> dict[str, object]:
    return {
        "trace_id": "trace-1",
        "question_id": "question-1",
        "question": "What changed?",
        "mode": "agentic",
        "run_number": 1,
        "trace_status": "completed",
        "summary": "Completed trace",
        "created_at": "2026-07-24T00:00:00+00:00",
        "steps": [],
    }


@pytest.mark.asyncio
async def test_campaign_result_repository_accepts_the_exact_answer_byte_limit_and_rejects_more(
    tmp_path, monkeypatch
) -> None:
    await evaluation_db.force_init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            INSERT INTO campaigns (
                id, user_id, status, phase, config_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "campaign-1",
                "user-1",
                "completed",
                "execution",
                "{}",
                "2026-07-24T00:00:00+00:00",
                "2026-07-24T00:00:00+00:00",
            ),
        )
        await connection.commit()
    repository = CampaignResultRepository()

    result_fields = dict(
        user_id="user-1",
        campaign_id="campaign-1",
        question_id="question-1",
        question="What changed?",
        ground_truth="The answer",
        ground_truth_short=None,
        key_points=[],
        ragas_focus=[],
        mode="naive",
        execution_profile=None,
        context_policy_version=None,
        run_number=1,
        answer="😀" * (MAX_EVALUATION_ANSWER_BYTES // 4),
        contexts=[],
        source_doc_ids=[],
        expected_sources=[],
        latency_ms=0,
        token_usage={},
        category=None,
        difficulty=None,
        status=CampaignResultStatus.COMPLETED,
    )
    result = await repository.create(**result_fields)
    duplicate = await repository.create(**result_fields)
    assert duplicate.id == result.id

    assert result.status == CampaignResultStatus.COMPLETED
    assert len(result.answer.encode("utf-8")) == MAX_EVALUATION_ANSWER_BYTES
    with pytest.raises(ValueError, match="EVALUATION_ANSWER_TOO_LARGE"):
        await repository.create(
            user_id="user-1",
            campaign_id="campaign-1",
            question_id="question-2",
            question="What changed?",
            ground_truth="The answer",
            ground_truth_short=None,
            key_points=[],
            ragas_focus=[],
            mode="naive",
            execution_profile=None,
            context_policy_version=None,
            run_number=1,
            answer="😀" * ((MAX_EVALUATION_ANSWER_BYTES // 4) + 1),
            contexts=[],
            source_doc_ids=[],
            expected_sources=[],
            latency_ms=0,
            token_usage={},
            category=None,
            difficulty=None,
            status=CampaignResultStatus.COMPLETED,
        )


@pytest.mark.asyncio
async def test_agent_trace_summary_persists_and_indexes_campaign_lists(
    tmp_path, monkeypatch
) -> None:
    await evaluation_db.force_init_db()
    repository = AgentTraceRepository()
    await repository.replace_for_result(
        user_id="user-1",
        campaign_id="campaign-1",
        campaign_result_id="result-1",
        trace_payload=_trace_payload(),
    )

    async with evaluation_db.connect_db() as connection:
        columns = await evaluation_db._table_columns(connection, "agent_traces")
        assert "summary_json" in columns

        await connection.execute("SET LOCAL enable_seqscan=off")
        cursor = await connection.execute(
            "SELECT summary_json FROM agent_traces WHERE id = ?", ("trace-1",)
        )
        stored_summary = json.loads((await cursor.fetchone())["summary_json"])
        assert stored_summary["trace_status"] == "completed"
        assert stored_summary["question"] == "What changed?"

        cursor = await connection.execute(
            """
            EXPLAIN
            SELECT id, campaign_id, campaign_result_id, user_id, summary_json, created_at
            FROM agent_traces
            WHERE campaign_id = ? AND user_id = ?
            ORDER BY created_at DESC
            """,
            ("campaign-1", "user-1"),
        )
        plan = "\n".join(row["QUERY PLAN"] for row in await cursor.fetchall())
    assert "idx_agent_traces_campaign_user_created" in plan


@pytest.mark.asyncio
async def test_agent_trace_campaign_list_reads_stored_summary_without_trace_json(
    tmp_path, monkeypatch
) -> None:
    await evaluation_db.force_init_db()
    repository = AgentTraceRepository()
    await repository.replace_for_result(
        user_id="user-1",
        campaign_id="campaign-1",
        campaign_result_id="result-1",
        trace_payload=_trace_payload(),
    )
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            "UPDATE agent_traces SET trace_json = ? WHERE id = ?",
            ("not valid JSON", "trace-1"),
        )
        await connection.commit()

    summaries = await repository.list_for_campaign(
        user_id="user-1", campaign_id="campaign-1"
    )

    assert len(summaries) == 1
    assert summaries[0].trace_id == "trace-1"
    assert summaries[0].summary == "Completed trace"
    assert summaries[0].trace_status == "completed"


@pytest.mark.asyncio
async def test_agent_trace_campaign_list_returns_not_instrumented_summary_for_legacy_blank(
    tmp_path, monkeypatch
) -> None:
    await evaluation_db.force_init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            INSERT INTO agent_traces (
                id, campaign_id, campaign_result_id, user_id, trace_json, summary_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-trace",
                "campaign-1",
                "legacy-result",
                "user-1",
                "not valid JSON",
                "",
                "2026-07-24T00:00:00+00:00",
            ),
        )
        await connection.commit()

    summaries = await AgentTraceRepository().list_for_campaign(
        user_id="user-1", campaign_id="campaign-1"
    )

    assert len(summaries) == 1
    assert summaries[0].trace_id == "legacy-trace"
    assert summaries[0].campaign_result_id == "legacy-result"
    assert summaries[0].trace_status == "not_instrumented"


@pytest.mark.asyncio
async def test_missing_postgres_configuration_never_falls_back_to_local_file(
    tmp_path, monkeypatch
) -> None:
    from evaluation.postgres import close_db

    await close_db()
    monkeypatch.delenv("EVALUATION_DATABASE_URL")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="EVALUATION_DATABASE_URL"):
        await evaluation_db.init_db()
    assert list(tmp_path.iterdir()) == []
