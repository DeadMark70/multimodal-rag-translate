from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from evaluation.observability import EvaluationRunRecorder
from evaluation.token_cost import normalize_llm_usage, price_llm_usage


class FakeLlmCallRepository:
    def __init__(self) -> None:
        self.calls = []

    async def record_llm_call(self, call):
        self.calls.append(call)


def test_normalize_llm_usage_accepts_multiple_provider_shapes() -> None:
    normalized = normalize_llm_usage(
        {
            "input_tokens": "100",
            "output_tokens": 40,
            "total_tokens": 0,
            "output_token_details": {"reasoning": 12},
        }
    )

    assert normalized == {
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
        "reasoning_tokens": 12,
    }

    legacy = normalize_llm_usage(
        Mock(
            usage_metadata={
                "prompt_token_count": 7,
                "candidates_token_count": 5,
                "total_token_count": 12,
                "thoughts_token_count": 3,
            }
        )
    )
    assert legacy == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
        "reasoning_tokens": 3,
    }


def test_price_llm_usage_uses_snapshot_rates_and_returns_none_for_unknown_model() -> None:
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "reasoning_tokens": 0,
    }

    priced = price_llm_usage(
        model_name="gemini-test",
        usage=usage,
        price_snapshot={
            "snapshot_id": "local-test",
            "currency": "USD",
            "usd_to_twd": 32.0,
            "models": {
                "gemini-test": {
                    "input_per_1m_usd": 0.10,
                    "output_per_1m_usd": 0.40,
                }
            },
        },
    )

    assert priced == {
        "estimated_cost_usd": pytest.approx(0.0003),
        "estimated_cost_twd": pytest.approx(0.0096),
        "price_snapshot_id": "local-test",
    }

    assert price_llm_usage(model_name="unknown", usage=usage)["estimated_cost_usd"] is None


@pytest.mark.asyncio
async def test_recorder_records_llm_usage_as_normalized_llm_call() -> None:
    repository = FakeLlmCallRepository()
    recorder = EvaluationRunRecorder(
        run_id="run-1",
        campaign_id="campaign-1",
        user_id="user-a",
        llm_call_repository=repository,
    )

    await recorder.record_llm_usage(
        purpose="generation",
        provider="google",
        model_name="gemini-test",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        latency_ms=25.5,
        created_at=datetime(2026, 7, 8, tzinfo=timezone.utc),
        price_snapshot={
            "snapshot_id": "local-test",
            "usd_to_twd": 32.0,
            "models": {
                "gemini-test": {
                    "input_per_1m_usd": 0.10,
                    "output_per_1m_usd": 0.40,
                }
            },
        },
    )

    assert len(repository.calls) == 1
    call = repository.calls[0]
    assert call.purpose == "generation"
    assert call.provider == "google"
    assert call.model_name == "gemini-test"
    assert call.prompt_tokens == 10
    assert call.completion_tokens == 5
    assert call.total_tokens == 15
    assert call.estimated_cost_usd == pytest.approx(0.000003)
    assert call.payload["price_snapshot_id"] == "local-test"
