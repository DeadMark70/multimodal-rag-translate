from pathlib import Path


PRODUCTION_DIRS = (
    Path("agents"),
    Path("graph_rag"),
    Path("data_base"),
)

FORBIDDEN_MARKERS = (
    "_PLANNER_PROMPT = ",
    "_GRAPH_PLANNER_PROMPT = ",
    "_FOLLOWUP_PROMPT = ",
    "_REFINE_QUERY_PROMPT = ",
    "_INTENT_CLASSIFIER_PROMPT = ",
    "_CONFLICT_ARBITRATION_PROMPT = ",
    "_SYNTHESIZER_PROMPT = ",
    "_ACADEMIC_REPORT_PROMPT = ",
    "_RETRIEVAL_EVAL_PROMPT = ",
    "_FAITHFULNESS_EVAL_PROMPT = ",
    "_DETAILED_EVAL_PROMPT = ",
    "_PURE_LLM_EVAL_PROMPT = ",
    "_FACT_STATE_PROMPT = ",
    "_ENTITY_EXTRACTION_PROMPT = ",
    "_RELATION_EXTRACTION_PROMPT = ",
    "_ONE_PASS_EXTRACTION_PROMPT = ",
    "_RELEVANCE_CHECK_PROMPT = ",
    "_COMMUNITY_ANSWER_PROMPT = ",
    "_SYNTHESIS_PROMPT = ",
    "_ENTITY_IDENTIFICATION_PROMPT = ",
    "_COMMUNITY_SUMMARY_PROMPT = ",
    "_PARENT_COMMUNITY_PROMPT = ",
    "_ROUTER_PROMPT = ",
    "_HYDE_PROMPT = ",
    "_MULTI_QUERY_PROMPT = ",
    "_PROPOSITION_PROMPT = ",
    "_CONTEXT_PROMPT_TEMPLATE = ",
)


def test_rag_prompt_constants_are_externalized() -> None:
    offenders: list[str] = []

    for directory in PRODUCTION_DIRS:
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for marker in FORBIDDEN_MARKERS:
                if marker in source:
                    offenders.append(f"{path}: {marker}")

    assert offenders == []
