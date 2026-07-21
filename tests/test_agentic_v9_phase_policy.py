"""Regression coverage for setup-authoritative Agentic v9 phase policy."""

import pytest

from core.llm_factory import current_llm_runtime_overrides, llm_runtime_override
from data_base.agentic_v9.phase_policy import (
    PHASE_POLICIES,
    agentic_phase_policy_scope,
    resolve_phase_policy,
)
from evaluation.model_capabilities import normalize_model_config_for_runtime


@pytest.mark.parametrize(
    ("phase", "temperature", "top_p", "top_k", "output_cap"),
    [
        ("route_plan", 0.10, 0.80, 20, 384),
        ("query_rewrite", 0.10, 0.80, 20, 192),
        ("retrieval_judge", 0.10, 0.70, 10, 96),
        ("graph_route", 0.10, 0.70, 10, 128),
        ("visual_extract", 0.10, 0.80, 20, 768),
        ("evidence_extract", 0.10, 0.80, 20, 768),
        ("conflict_arbitration", 0.10, 0.80, 20, 256),
        ("claim_verifier", 0.10, 0.80, 20, 384),
        ("final_answer", 0.25, 0.90, 40, 1536),
    ],
)
def test_phase_matrix_has_frozen_sampling_and_output_caps(
    phase: str,
    temperature: float,
    top_p: float,
    top_k: int,
    output_cap: int,
) -> None:
    policy = resolve_phase_policy(
        phase,
        setup_output_ceiling=8192,
        setup_input_ceiling=16384,
        remaining_input_budget=12000,
    )

    assert PHASE_POLICIES[phase].max_output_tokens == output_cap
    assert (policy.temperature, policy.top_p, policy.top_k) == (
        temperature,
        top_p,
        top_k,
    )
    assert policy.max_output_tokens == output_cap
    assert policy.max_input_tokens == 12000


def test_phase_policy_applies_setup_and_remaining_budget_ceilings() -> None:
    policy = resolve_phase_policy(
        "final_answer",
        setup_output_ceiling=800,
        setup_input_ceiling=8192,
        remaining_input_budget=5000,
    )

    assert policy.max_output_tokens == 800
    assert policy.max_input_tokens == 5000


def test_phase_scope_preserves_setup_model_thinking_and_ceilings() -> None:
    with llm_runtime_override(
        model_name="gemini-2.5-flash-lite",
        thinking_enabled=False,
        max_output_tokens=8192,
        max_input_tokens=16384,
    ):
        policy = resolve_phase_policy(
            "route_plan",
            setup_output_ceiling=8192,
            setup_input_ceiling=16384,
            remaining_input_budget=5000,
        )
        with agentic_phase_policy_scope(policy):
            config = current_llm_runtime_overrides()

            assert config["model_name"] == "gemini-2.5-flash-lite"
            assert config["thinking_enabled"] is False
            assert config["setup_max_output_tokens"] == 8192
            assert config["setup_max_input_tokens"] == 16384
            assert config["max_output_tokens"] == 384
            assert config["max_input_tokens"] == 5000
            assert (config["temperature"], config["top_p"], config["top_k"]) == (
                0.10,
                0.80,
                20,
            )


def test_runtime_normalization_carries_setup_token_ceilings() -> None:
    runtime = normalize_model_config_for_runtime(
        {
            "model_name": "gemini-2.5-flash-lite",
            "max_input_tokens": 16384,
            "max_output_tokens": 8192,
            "thinking_mode": False,
        }
    )

    assert runtime["setup_max_input_tokens"] == 16384
    assert runtime["setup_max_output_tokens"] == 8192
