from evaluation.retrieval_profiles import (
    ADVANCED_EVAL_PROFILE,
    AGENTIC_EVAL_PROFILE,
    GRAPH_EVAL_PROFILE,
    apply_no_hyde_policy,
    evaluation_execution_profile,
    locator_to_chunk_graph_hints,
    multi_query_settings,
    no_query_expansion_settings,
)


def test_query_policy_factories_return_fresh_explicit_settings() -> None:
    first = multi_query_settings()
    second = multi_query_settings()
    assert first == {"enable_hyde": False, "enable_multi_query": True}
    assert no_query_expansion_settings() == {
        "enable_hyde": False,
        "enable_multi_query": False,
    }
    assert first is not second


def test_locator_to_chunk_hints_are_source_backed_and_not_auto_gated() -> None:
    hints = locator_to_chunk_graph_hints(
        stage_hint="exploration",
        task_type="graph_analysis",
    )
    assert hints["graph_evidence_mode"] == "locator_to_chunk"
    assert hints["stage_hint"] == "exploration"
    assert hints["task_type_hint"] == "graph_analysis"
    assert hints["prefer_global"] is True
    assert hints["prefer_local"] is False
    assert hints["graph_feature_flags"] == {
        "graph_raw_current_enabled": False,
        "graph_evidence_locator_enabled": True,
        "graph_provenance_gate_enabled": True,
        "graph_to_chunk_enabled": True,
        "graph_auto_gate_enabled": False,
    }


def test_no_hyde_policy_does_not_mutate_input() -> None:
    source = {
        "naive": {"enable_hyde": False, "enable_multi_query": False},
        "advanced": {"enable_hyde": True, "enable_multi_query": True},
        "graph_raw_current": {"enable_hyde": True, "enable_multi_query": True},
        "future_unrelated": {"enable_hyde": True, "enable_multi_query": False},
    }
    normalized = apply_no_hyde_policy(source)
    assert source["advanced"]["enable_hyde"] is True
    assert normalized["naive"]["enable_hyde"] is False
    assert normalized["advanced"]["enable_hyde"] is False
    assert normalized["graph_raw_current"]["enable_hyde"] is False
    assert normalized["future_unrelated"]["enable_hyde"] is True


def test_execution_profiles_version_changed_modes() -> None:
    assert evaluation_execution_profile("naive") is None
    assert evaluation_execution_profile("advanced") == ADVANCED_EVAL_PROFILE
    assert evaluation_execution_profile("graph") == GRAPH_EVAL_PROFILE
    assert evaluation_execution_profile("agentic") == AGENTIC_EVAL_PROFILE
    assert evaluation_execution_profile("graph_raw_current").startswith(
        "graph_raw_current_eval_v2_multiquery_"
    )
