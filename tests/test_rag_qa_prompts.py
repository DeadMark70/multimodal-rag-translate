from pathlib import Path

from core.prompt_loader import get_default_prompt_registry


def test_rag_qa_prompt_config_contains_required_prompt_keys() -> None:
    registry = get_default_prompt_registry()

    assert registry.get("plain_rag_answer").version >= 1
    assert registry.get("advanced_rag_answer").version >= 1
    assert registry.get("visual_tool_instruction").version >= 1
    assert registry.get("visual_verification_synthesis").version >= 1


def test_plain_rag_answer_prompt_requires_current_variables() -> None:
    prompt = get_default_prompt_registry().get("plain_rag_answer")

    assert prompt.required_variables == (
        "context_text",
        "graph_section",
        "history_section",
        "question",
        "intent_constraints",
    )


def test_advanced_rag_answer_prompt_requires_current_variables() -> None:
    prompt = get_default_prompt_registry().get("advanced_rag_answer")

    assert prompt.required_variables == (
        "context_text",
        "graph_section",
        "history_section",
        "question",
        "intent_constraints",
        "visual_instruction",
    )


def test_visual_tool_instruction_prompt_has_no_required_variables() -> None:
    prompt = get_default_prompt_registry().get("visual_tool_instruction")

    assert prompt.required_variables == ()


def test_visual_synthesis_prompt_requires_current_variables() -> None:
    prompt = get_default_prompt_registry().get("visual_verification_synthesis")

    assert prompt.required_variables == (
        "context",
        "question",
        "initial_response",
        "verification_results",
    )


def test_rag_qa_service_no_long_prompt_constants() -> None:
    source = Path("data_base/RAG_QA_service.py").read_text(encoding="utf-8")

    assert "PLAIN_RAG_PROMPT_TEMPLATE = " not in source
    assert "ADVANCED_RAG_PROMPT_TEMPLATE = " not in source
    assert "VISUAL_TOOL_INSTRUCTION = " not in source


def test_visual_synthesis_prompt_formats_current_runtime_payload() -> None:
    formatted = get_default_prompt_registry().format(
        "visual_verification_synthesis",
        context="context",
        question="question",
        initial_response="initial",
        verification_results="results",
    )

    assert "context" in formatted
    assert "question" in formatted
    assert "results" in formatted
    assert "initial" in formatted
