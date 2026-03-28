from __future__ import annotations

from datetime import datetime, UTC

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
