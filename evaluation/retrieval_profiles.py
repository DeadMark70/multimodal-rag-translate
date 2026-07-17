"""Versioned retrieval policies shared by Evaluation Center execution paths."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from data_base.indexing_service import DEFAULT_PRODUCTION_INDEXING_PROFILE

EVALUATION_RETRIEVAL_POLICY_VERSION = "v2_multiquery_locator"

ADVANCED_EVAL_PROFILE = (
    f"advanced_eval_v2_multiquery_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
)
GRAPH_EVAL_PROFILE = (
    f"graph_eval_v2_multiquery_locator_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
)
AGENTIC_EVAL_PROFILE = (
    f"agentic_eval_v8_multiquery_locator_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
)
AGENTIC_LEGACY_CHAT_PROFILE = (
    f"agentic_eval_v7_semantic_router_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
)

GRAPH_ABLATION_MODES = frozenset(
    {
        "graph_raw_current",
        "graph_provenance_gated",
        "graph_locator_to_chunk",
        "graph_locator_claim_gate",
        "always_no_graph",
        "always_graph_locator",
        "router_auto_graph",
        "oracle_graph_router",
        "graph_local_first",
        "graph_global_first",
        "graph_blended",
        "graph_path_pruned",
        "graph_planning_only",
    }
)


def multi_query_settings() -> dict[str, bool]:
    return {"enable_hyde": False, "enable_multi_query": True}


def no_query_expansion_settings() -> dict[str, bool]:
    return {"enable_hyde": False, "enable_multi_query": False}


def locator_to_chunk_graph_hints(
    *,
    stage_hint: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    hints: dict[str, Any] = {
        "graph_evidence_mode": "locator_to_chunk",
        "graph_feature_flags": {
            "graph_raw_current_enabled": False,
            "graph_evidence_locator_enabled": True,
            "graph_provenance_gate_enabled": True,
            "graph_to_chunk_enabled": True,
            "graph_auto_gate_enabled": False,
        },
    }
    if stage_hint is not None:
        hints["stage_hint"] = stage_hint
        hints["prefer_global"] = stage_hint == "exploration"
        hints["prefer_local"] = (
            stage_hint == "verification" and task_type != "graph_analysis"
        )
    if task_type is not None:
        hints["task_type_hint"] = task_type
        if stage_hint is None:
            hints["prefer_global"] = task_type == "graph_analysis"
            hints["prefer_local"] = False
    return hints


def apply_no_hyde_policy(
    modes: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized = deepcopy({name: dict(config) for name, config in modes.items()})
    changed_modes = {"advanced", "graph", "agentic", *GRAPH_ABLATION_MODES}
    for name, config in normalized.items():
        if name in changed_modes:
            config["enable_hyde"] = False
    return normalized


def evaluation_execution_profile(mode: str) -> str | None:
    if mode == "advanced":
        return ADVANCED_EVAL_PROFILE
    if mode == "graph":
        return GRAPH_EVAL_PROFILE
    if mode == "agentic":
        return AGENTIC_EVAL_PROFILE
    if mode in GRAPH_ABLATION_MODES:
        return f"{mode}_eval_v2_multiquery_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
    return None


def evaluation_failure_execution_profile(
    mode: str,
    payload: object,
) -> str | None:
    """Resolve a failed run's captured profile before using the mode baseline."""
    trace = getattr(payload, "agent_trace", None)
    if isinstance(trace, Mapping):
        trace_profile = trace.get("execution_profile")
        if isinstance(trace_profile, str) and trace_profile:
            return trace_profile
    payload_profile = getattr(payload, "execution_profile", None)
    if isinstance(payload_profile, str) and payload_profile:
        return payload_profile
    return evaluation_execution_profile(mode)
