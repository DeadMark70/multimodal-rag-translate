from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from evaluation.observability import EvaluationRunRecorder
from evaluation.token_cost import (
    load_price_snapshot,
    normalize_llm_usage,
    price_llm_usage,
    price_normalized_usage,
)
from evaluation.token_normalizers import normalize_provider_usage


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


def test_price_llm_usage_uses_snapshot_rates_and_returns_none_for_unknown_model() -> (
    None
):
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

    assert (
        price_llm_usage(model_name="unknown", usage=usage)["estimated_cost_usd"] is None
    )


def test_price_normalized_usage_charges_non_overlapping_categories() -> None:
    usage = normalize_provider_usage(
        "openai",
        {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 3_000_000,
            "total_tokens": 4_000_000,
            "output_token_details": {"reasoning": 1_000_000},
        },
    )

    priced = price_normalized_usage(
        "openai-test",
        usage,
        {
            "snapshot_id": "audited-test",
            "currency": "USD",
            "usd_to_twd": 32.0,
            "models": {
                "openai-test": {
                    "input_per_1m_usd": 1.0,
                    "output_per_1m_usd": 2.0,
                    "reasoning_per_1m_usd": 3.0,
                }
            },
        },
    )

    assert priced == {
        "estimated_cost_usd": pytest.approx(8.0),
        "estimated_cost_twd": pytest.approx(256.0),
        "pricing_status": "priced",
        "price_snapshot_id": "audited-test",
    }


def test_price_normalized_usage_keeps_unknown_model_cost_missing() -> None:
    usage = normalize_provider_usage("google", {"total_token_count": 0})

    priced = price_normalized_usage("unknown", usage, load_price_snapshot())

    assert priced["estimated_cost_usd"] is None
    assert priced["pricing_status"] == "unknown_model"


def test_load_price_snapshot_rejects_invalid_configured_json(
    tmp_path, monkeypatch
) -> None:
    snapshot_path = tmp_path / "invalid-prices.json"
    snapshot_path.write_text(
        '{"snapshot_id": "bad", "currency": "TWD"}', encoding="utf-8"
    )
    monkeypatch.setenv("EVALUATION_PRICE_SNAPSHOT_PATH", str(snapshot_path))

    with pytest.raises(ValueError, match="currency"):
        load_price_snapshot()


def test_load_price_snapshot_rejects_non_finite_rates(tmp_path) -> None:
    snapshot_path = tmp_path / "non-finite-prices.json"
    snapshot_path.write_text(
        """{
            "snapshot_id": "bad-rate",
            "currency": "USD",
            "models": {
                "test": {
                    "input_per_1m_usd": 1,
                    "output_per_1m_usd": 2,
                    "reasoning_per_1m_usd": "NaN"
                }
            }
        }""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reasoning_per_1m_usd"):
        load_price_snapshot(snapshot_path)


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
