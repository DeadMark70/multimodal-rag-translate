from __future__ import annotations

from pathlib import Path

from core.prompt_loader import get_graph_rag_prompt_registry


EXPECTED_PROMPTS: dict[str, tuple[str, ...]] = {
    "entity_extraction": ("text",),
    "relation_extraction": ("text", "entities"),
    "one_pass_extraction": ("text",),
    "relevance_check": ("question", "title", "summary"),
    "community_answer": ("question", "title", "summary", "entities"),
    "global_synthesis": ("question", "community_answers"),
    "entity_identification": ("question",),
    "community_summary": ("entities_and_relations",),
    "parent_community": ("child_summaries",),
    "generic_router": (
        "question",
        "stage_hint",
        "task_type_hint",
        "prefer_global",
        "prefer_local",
        "has_communities",
    ),
}


SOURCE_MARKERS = (
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
)


def test_graph_rag_prompt_registry_has_expected_keys() -> None:
    registry = get_graph_rag_prompt_registry()
    assert set(registry._prompts) == set(EXPECTED_PROMPTS)


def test_graph_rag_prompt_required_variables_match_expected() -> None:
    registry = get_graph_rag_prompt_registry()

    for key, expected_variables in EXPECTED_PROMPTS.items():
        assert registry.get(key).required_variables == expected_variables


def test_graph_rag_source_files_no_longer_define_inline_prompt_constants() -> None:
    source_files = [
        Path("graph_rag/extractor.py"),
        Path("graph_rag/global_search.py"),
        Path("graph_rag/local_search.py"),
        Path("graph_rag/community_builder.py"),
        Path("graph_rag/generic_mode.py"),
    ]

    for path in source_files:
        source = path.read_text(encoding="utf-8")
        for marker in SOURCE_MARKERS:
            assert marker not in source, f"{path} still contains {marker}"


def test_graph_rag_prompt_format_smoke() -> None:
    registry = get_graph_rag_prompt_registry()

    entity_prompt = registry.format("entity_extraction", text="alpha")
    relation_prompt = registry.format(
        "relation_extraction",
        text="alpha",
        entities="- A (concept)",
    )
    community_answer_prompt = registry.format(
        "community_answer",
        question="What is the result?",
        title="社群 1",
        summary="summary",
        entities="- A\n- B",
    )
    synthesis_prompt = registry.format(
        "global_synthesis",
        question="What is the result?",
        community_answers="[A]\nanswer",
    )
    router_prompt = registry.format(
        "generic_router",
        question="What is the result?",
        stage_hint="none",
        task_type_hint="none",
        prefer_global="false",
        prefer_local="true",
        has_communities="true",
    )

    assert "alpha" in entity_prompt
    assert "alpha" in relation_prompt
    assert "What is the result?" in community_answer_prompt
    assert "[A]" in synthesis_prompt
    assert "has_communities" in router_prompt
