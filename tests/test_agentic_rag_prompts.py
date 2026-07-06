from pathlib import Path

from core.prompt_loader import format_agentic_rag_prompt, get_agentic_rag_prompt_registry

EXPECTED_KEYS = {
    "planner",
    "graph_planner",
    "followup",
    "refine_query",
    "intent_classifier",
    "conflict_arbitration",
    "synthesizer",
    "academic_report",
    "retrieval_eval",
    "faithfulness_eval",
    "detailed_eval",
    "pure_llm_eval",
    "fact_state",
}

EXPECTED_REQUIRED_VARIABLES = {
    "planner": ["question"],
    "graph_planner": ["question"],
    "followup": ["original_question", "current_findings", "existing_questions"],
    "refine_query": ["original_question", "evaluation_reason", "failed_answer"],
    "intent_classifier": ["question"],
    "conflict_arbitration": ["sub_results"],
    "synthesizer": ["original_question", "sub_results"],
    "academic_report": ["original_question", "sub_results"],
    "retrieval_eval": ["question", "documents"],
    "faithfulness_eval": ["question", "documents", "answer"],
    "detailed_eval": ["question", "documents", "answer"],
    "pure_llm_eval": ["question", "answer", "ground_truth"],
    "fact_state": ["question", "source_doc_ids", "answer"],
}

SOURCE_FILES = [
    Path(__file__).resolve().parents[1] / "agents" / "planner.py",
    Path(__file__).resolve().parents[1] / "agents" / "synthesizer.py",
    Path(__file__).resolve().parents[1] / "agents" / "evaluator.py",
    Path(__file__).resolve().parents[1] / "data_base" / "research_execution_core.py",
]

OLD_MARKERS = [
    "_PLANNER_PROMPT =",
    "_GRAPH_PLANNER_PROMPT =",
    "_FOLLOWUP_PROMPT =",
    "_REFINE_QUERY_PROMPT =",
    "_INTENT_CLASSIFIER_PROMPT =",
    "_CONFLICT_ARBITRATION_PROMPT =",
    "_SYNTHESIZER_PROMPT =",
    "_ACADEMIC_REPORT_PROMPT =",
    "_RETRIEVAL_EVAL_PROMPT =",
    "_FAITHFULNESS_EVAL_PROMPT =",
    "_DETAILED_EVAL_PROMPT =",
    "_PURE_LLM_EVAL_PROMPT =",
    "_FACT_STATE_PROMPT =",
]


def test_agentic_prompt_registry_has_expected_keys():
    registry = get_agentic_rag_prompt_registry()
    keys = {key for key in EXPECTED_KEYS if registry.get(key)}
    assert keys == EXPECTED_KEYS


def test_agentic_prompt_registry_required_variables_match_contract():
    registry = get_agentic_rag_prompt_registry()
    for key, expected in EXPECTED_REQUIRED_VARIABLES.items():
        assert list(registry.get(key).required_variables) == expected


def test_agentic_prompt_sources_no_longer_define_constants():
    for path in SOURCE_FILES:
        content = path.read_text(encoding="utf-8")
        for marker in OLD_MARKERS:
            assert marker not in content, f"unexpected legacy prompt marker in {path.name}: {marker}"


def test_agentic_prompt_format_smoke():
    planner = format_agentic_rag_prompt("planner", question="What is X?")
    followup = format_agentic_rag_prompt(
        "followup",
        original_question="What is X?",
        current_findings="- finding",
        existing_questions="- prior question",
    )
    conflict = format_agentic_rag_prompt("conflict_arbitration", sub_results="[Task 1] A")
    retrieval = format_agentic_rag_prompt("retrieval_eval", question="What is X?", documents="[1] doc text")
    fact_state = format_agentic_rag_prompt(
        "fact_state",
        question="What is X?",
        source_doc_ids="doc-1, doc-2",
        answer="Answer text",
    )

    assert "What is X?" in planner
    assert "What is X?" in followup
    assert "[Task 1] A" in conflict
    assert "[1] doc text" in retrieval
    assert "doc-1, doc-2" in fact_state
