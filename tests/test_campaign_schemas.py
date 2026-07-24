from __future__ import annotations

from datetime import datetime, UTC

import pytest
from pydantic import ValidationError

from evaluation.campaign_schemas import (
    AgentBehaviorResponse,
    AgentBehaviorRow,
    CampaignResult,
    CampaignResultStatus,
)


def test_campaign_result_allows_nested_token_usage_payloads() -> None:
    result = CampaignResult(
        id="result-1",
        campaign_id="cmp-1",
        question_id="Q1",
        question="question",
        ground_truth="ground truth",
        mode="naive",
        run_number=1,
        answer="answer",
        token_usage={
            "total_tokens": 42,
            "input_tokens": 21,
            "input_token_details": {"cache_read": 0},
        },
        status=CampaignResultStatus.COMPLETED,
        created_at=datetime.now(UTC),
    )

    assert result.token_usage["total_tokens"] == 42
    assert result.token_usage["input_token_details"] == {"cache_read": 0}


def test_campaign_result_snapshot_fields_default_to_none_or_empty() -> None:
    result = CampaignResult(
        id="result-1",
        campaign_id="cmp-1",
        question_id="Q1",
        question="question",
        ground_truth="ground truth",
        mode="naive",
        run_number=1,
        answer="answer",
        status=CampaignResultStatus.COMPLETED,
        created_at=datetime.now(UTC),
    )

    assert result.question_version is None
    assert result.request_id is None
    assert result.started_at is None
    assert result.completed_at is None
    assert result.total_latency_ms is None
    assert result.total_tokens is None
    assert result.question_snapshot == {}
    assert result.model_config_snapshot == {}
    assert result.system_version_snapshot == {}
    assert result.derived_metrics == {}
    assert result.final_answer_hash is None


def test_campaign_config_ragas_fields_default_and_bounds() -> None:
    from evaluation.campaign_schemas import CampaignConfig
    from evaluation.schemas import ModelConfig

    config = CampaignConfig(
        test_case_ids=["Q1"],
        modes=["naive"],
        model_config=ModelConfig(
            id="cfg-1",
            name="Balanced",
            model_name="gemini-2.5-flash",
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_input_tokens=8192,
            max_output_tokens=2048,
            thinking_mode=False,
            thinking_budget=8192,
        ),
    )

    assert config.ragas_batch_size == 8
    assert config.ragas_parallel_batches == 8
    assert config.ragas_rpm_limit == 1000


def test_campaign_config_ragas_fields_reject_invalid_values() -> None:
    from evaluation.campaign_schemas import CampaignConfig
    from evaluation.schemas import ModelConfig

    with pytest.raises(ValidationError):
        CampaignConfig(
            test_case_ids=["Q1"],
            modes=["naive"],
            model_config=ModelConfig(
                id="cfg-1",
                name="Balanced",
                model_name="gemini-2.5-flash",
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_input_tokens=8192,
                max_output_tokens=2048,
                thinking_mode=False,
                thinking_budget=8192,
            ),
            ragas_batch_size=9,
        )


def test_agent_behavior_v9_contract_preserves_evidence_metrics_without_legacy_zeros() -> None:
    response = AgentBehaviorResponse(
        campaign_id="cmp-v9",
        analysis_unit="execution",
        sample_count=1,
        independent_question_count=1,
        repeat_count=1,
        behavior_schema_version="2",
        rows=[
            AgentBehaviorRow(
                run_id="run-v9",
                campaign_id="cmp-v9",
                question_id="Q1",
                mode="agentic",
                repeat_number=1,
                behavior_schema="v9",
                trace_status="completed",
                accounting_status="complete",
                total_tokens=42,
                legacy=None,
                v9={
                    "route": "multi_hop",
                    "graph_policy": "required_locator",
                    "visual_required": False,
                    "evidence_extraction_required": True,
                    "retrieval_query_count": 2,
                    "provider_attempt_count": 1,
                    "final_generation_count": 1,
                    "evidence_packet_count": 13,
                    "packed_evidence_count": 6,
                    "slot_resolution_count": 2,
                    "required_slot_count": 2,
                    "supported_slot_count": 2,
                    "repair_count": 0,
                    "final_claim_count": 1,
                    "reserved_tokens": 64,
                    "reconciled_tokens": 42,
                    "graph_execution": "required_but_not_satisfied",
                    "visual_execution": "not_requested",
                },
            )
        ],
    )

    row = response.rows[0]
    assert response.behavior_schema_version == "2"
    assert row.behavior_schema == "v9"
    assert row.legacy is None
    assert row.v9 is not None
    assert row.v9.evidence_packet_count == 13
    assert row.v9.graph_execution == "required_but_not_satisfied"
