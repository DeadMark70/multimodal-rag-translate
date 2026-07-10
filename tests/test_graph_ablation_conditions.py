from evaluation.rag_modes import RAG_MODES


def test_graph_evidence_ablation_conditions_exist() -> None:
    for mode in [
        "graph_raw_current",
        "graph_provenance_gated",
        "graph_locator_to_chunk",
        "graph_locator_claim_gate",
    ]:
        assert mode in RAG_MODES
        assert RAG_MODES[mode]["enable_graph_rag"] is True
        assert RAG_MODES[mode]["ablation_family"] == "graph_evidence"


def test_router_policy_and_query_strategy_are_separate_families() -> None:
    assert RAG_MODES["router_auto_graph"]["ablation_family"] == "graph_usage_policy"
    assert RAG_MODES["oracle_graph_router"]["ablation_family"] == "graph_usage_policy"
    assert RAG_MODES["graph_local_first"]["ablation_family"] == "graph_query_strategy"
    assert RAG_MODES["graph_path_pruned"]["ablation_family"] == "graph_query_strategy"
    assert RAG_MODES["graph_planning_only"]["ablation_family"] == "graph_query_strategy"
    assert RAG_MODES["graph_locator_to_chunk"]["graph_evidence_mode"] == "locator_to_chunk"
