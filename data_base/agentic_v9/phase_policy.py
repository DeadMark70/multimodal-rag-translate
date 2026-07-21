"""Setup-constrained sampling policy for Agentic v9 provider phases."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from core.llm_factory import llm_runtime_override


@dataclass(frozen=True, slots=True)
class PhasePolicy:
    """Fixed sampling and output cap for one provider phase."""

    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class EffectivePhasePolicy:
    """One phase policy after applying Setup and remaining-input ceilings."""

    phase: str
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int
    max_input_tokens: int


PHASE_POLICIES: dict[str, PhasePolicy] = {
    "route_plan": PhasePolicy(0.10, 0.80, 20, 384),
    "query_rewrite": PhasePolicy(0.10, 0.80, 20, 192),
    "retrieval_judge": PhasePolicy(0.10, 0.70, 10, 96),
    "graph_route": PhasePolicy(0.10, 0.70, 10, 128),
    "visual_extract": PhasePolicy(0.10, 0.80, 20, 768),
    "evidence_extract": PhasePolicy(0.10, 0.80, 20, 768),
    "conflict_arbitration": PhasePolicy(0.10, 0.80, 20, 256),
    "claim_verifier": PhasePolicy(0.10, 0.80, 20, 384),
    "final_answer": PhasePolicy(0.25, 0.90, 40, 1536),
}


def resolve_phase_policy(
    phase: str,
    *,
    setup_output_ceiling: int,
    setup_input_ceiling: int,
    remaining_input_budget: int,
) -> EffectivePhasePolicy:
    """Resolve v9 sampling while keeping Setup as the ceiling authority."""
    try:
        phase_policy = PHASE_POLICIES[phase]
    except KeyError as error:
        raise ValueError(f"Unsupported Agentic v9 phase: {phase}") from error

    if setup_output_ceiling < 1:
        raise ValueError("setup_output_ceiling must be at least one token")
    if setup_input_ceiling < 1:
        raise ValueError("setup_input_ceiling must be at least one token")
    if remaining_input_budget < 1:
        raise ValueError("remaining_input_budget must be at least one token")

    return EffectivePhasePolicy(
        phase=phase,
        temperature=phase_policy.temperature,
        top_p=phase_policy.top_p,
        top_k=phase_policy.top_k,
        max_output_tokens=min(setup_output_ceiling, phase_policy.max_output_tokens),
        max_input_tokens=min(setup_input_ceiling, remaining_input_budget),
    )


@contextmanager
def agentic_phase_policy_scope(policy: EffectivePhasePolicy) -> Iterator[None]:
    """Apply only a resolved v9 phase's sampling and effective ceilings."""
    with llm_runtime_override(
        temperature=policy.temperature,
        top_p=policy.top_p,
        top_k=policy.top_k,
        max_output_tokens=policy.max_output_tokens,
        max_input_tokens=policy.max_input_tokens,
    ):
        yield
