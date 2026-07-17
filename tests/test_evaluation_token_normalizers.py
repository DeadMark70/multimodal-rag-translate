from evaluation.token_normalizers import normalize_provider_usage


def test_google_usage_is_non_overlapping_and_balanced() -> None:
    usage = normalize_provider_usage(
        "google",
        {
            "prompt_token_count": 10,
            "candidates_token_count": 4,
            "thoughts_token_count": 3,
            "total_token_count": 17,
        },
    )

    assert usage.model_dump() == {
        "input_tokens": 10,
        "output_text_tokens": 4,
        "reasoning_tokens": 3,
        "other_tokens": 0,
        "reported_total_tokens": 17,
        "usage_status": "measured",
        "reconciliation_status": "balanced",
    }


def test_completion_reasoning_subset_is_not_double_counted() -> None:
    usage = normalize_provider_usage(
        "openai",
        {
            "prompt_tokens": 10,
            "completion_tokens": 9,
            "total_tokens": 19,
            "output_token_details": {"reasoning": 4},
        },
    )

    assert usage.output_text_tokens == 5
    assert usage.reasoning_tokens == 4
    assert usage.reconciliation_status == "balanced"


def test_missing_usage_is_not_zero() -> None:
    usage = normalize_provider_usage("google", {})

    assert usage.reported_total_tokens is None
    assert usage.usage_status == "missing"
    assert usage.reconciliation_status == "unavailable"


def test_usage_with_overlapping_categories_is_partial() -> None:
    usage = normalize_provider_usage(
        "google",
        {
            "prompt_token_count": 10,
            "candidates_token_count": 4,
            "thoughts_token_count": 3,
            "total_token_count": 12,
        },
    )

    assert usage.other_tokens == 0
    assert usage.reconciliation_status == "partial"
