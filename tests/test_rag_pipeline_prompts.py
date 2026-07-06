from core.prompt_loader import get_rag_pipeline_prompt_registry


def test_rag_pipeline_prompt_registry_contains_expected_prompts() -> None:
    registry = get_rag_pipeline_prompt_registry()

    hyde = registry.get("hyde")
    multi_query = registry.get("multi_query")
    proposition = registry.get("proposition")
    context_enrichment = registry.get("context_enrichment")

    assert hyde.required_variables == ("question",)
    assert multi_query.required_variables == ("question",)
    assert proposition.required_variables == ("text",)
    assert context_enrichment.required_variables == (
        "document_title",
        "chunk_content",
    )

    assert hyde.template.count("{question}") == 1
    assert multi_query.template.count("{question}") == 1
    assert proposition.template.count("{text}") == 1
    assert context_enrichment.template.count("{document_title}") == 1
    assert context_enrichment.template.count("{chunk_content}") == 1

    assert "What is RAG?" in registry.format("hyde", question="What is RAG?")
    assert "What is RAG?" in registry.format("multi_query", question="What is RAG?")
    assert "Atomic proposition text" in registry.format(
        "proposition",
        text="Atomic proposition text",
    )
    assert "Doc title" in registry.format(
        "context_enrichment",
        document_title="Doc title",
        chunk_content="Chunk content",
    )
