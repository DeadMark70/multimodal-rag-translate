from __future__ import annotations

from datetime import datetime, UTC

import pytest
from pydantic import ValidationError

from evaluation.campaign_schemas import CampaignResult, CampaignResultStatus


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
