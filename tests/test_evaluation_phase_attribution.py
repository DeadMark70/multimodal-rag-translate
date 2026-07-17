from pathlib import Path

import pytest


PHASE_CASES = [
    ("data_base/query_transformer.py", "query_expansion"),
    ("data_base/query_transformer.py", "retrieval_rewrite"),
    ("graph_rag/local_search.py", "graph_reasoning"),
    ("graph_rag/global_search.py", "graph_reasoning"),
    ("graph_rag/generic_mode.py", "graph_reasoning"),
    ("evaluation/agentic_evaluation_service.py", "agent_planning"),
    ("data_base/RAG_QA_service.py", "answer_generation"),
    ("multimodal_rag/image_summarizer.py", "visual_verification"),
    ("agents/synthesizer.py", "agent_synthesis"),
]


@pytest.mark.parametrize(("relative_path", "expected_phase"), PHASE_CASES)
def test_evaluation_call_sites_declare_controlled_phase(
    relative_path: str, expected_phase: str
) -> None:
    source = Path(relative_path).read_text(encoding="utf-8")

    assert f'llm_accounting_phase("{expected_phase}")' in source
