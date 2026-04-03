from evaluation.db import _normalize_trace_route_profiles


def test_normalize_trace_route_profile_alias_for_top_level() -> None:
    payload = {
        "route_profile": "hybrid_graph",
        "steps": [],
    }

    normalized = _normalize_trace_route_profiles(payload)
    assert normalized["route_profile"] == "generic_graph"


def test_normalize_trace_route_profile_alias_for_step_metadata() -> None:
    payload = {
        "route_profile": "hybrid_compare",
        "steps": [
            {"metadata": {"route_profile": "hybrid_graph"}},
            {"metadata": {"route_profile": "visual_verify"}},
        ],
    }

    normalized = _normalize_trace_route_profiles(payload)
    assert normalized["route_profile"] == "hybrid_compare"
    assert normalized["steps"][0]["metadata"]["route_profile"] == "generic_graph"
    assert normalized["steps"][1]["metadata"]["route_profile"] == "visual_verify"
