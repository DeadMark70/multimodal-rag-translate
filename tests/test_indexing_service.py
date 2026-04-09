from unittest.mock import AsyncMock, patch

import pytest

from data_base.indexing_service import (
    DEFAULT_INDEXING_PROFILE,
    DEFAULT_PRODUCTION_INDEXING_PROFILE,
    index_markdown_document,
)


@pytest.mark.asyncio
async def test_index_markdown_document_defaults_to_recursive_baseline() -> None:
    with patch(
        "data_base.indexing_service.add_markdown_to_knowledge_base",
        new=AsyncMock(return_value="retriever"),
    ) as add_markdown:
        result = await index_markdown_document(
            user_id="user-1",
            markdown_text="# Demo",
            pdf_title="Demo",
            doc_id="doc-1",
        )

    assert result == "retriever"
    assert DEFAULT_INDEXING_PROFILE == "recursive_baseline"
    add_markdown.assert_awaited_once_with(
        user_id="user-1",
        markdown_text="# Demo",
        pdf_title="Demo",
        doc_id="doc-1",
        k_retriever=3,
        chunking_method="recursive",
        enable_context_enrichment=False,
    )


@pytest.mark.asyncio
async def test_index_markdown_document_semantic_contextual_profile_enables_advanced_indexing() -> None:
    with patch(
        "data_base.indexing_service.add_markdown_to_knowledge_base",
        new=AsyncMock(return_value="semantic-retriever"),
    ) as add_markdown:
        result = await index_markdown_document(
            user_id="user-1",
            markdown_text="# Demo",
            pdf_title="Demo",
            doc_id="doc-1",
            k_retriever=5,
            indexing_profile=DEFAULT_PRODUCTION_INDEXING_PROFILE,
        )

    assert result == "semantic-retriever"
    assert DEFAULT_PRODUCTION_INDEXING_PROFILE == "semantic_contextual"
    add_markdown.assert_awaited_once_with(
        user_id="user-1",
        markdown_text="# Demo",
        pdf_title="Demo",
        doc_id="doc-1",
        k_retriever=5,
        chunking_method="semantic",
        enable_context_enrichment=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("indexing_profile", "enable_proposition_indexing"),
    [
        ("hierarchical_parent_child", False),
        ("hierarchical_parent_child_proposition", True),
    ],
)
async def test_index_markdown_document_hierarchical_profiles_route_to_parent_child_indexing(
    indexing_profile: str,
    enable_proposition_indexing: bool,
) -> None:
    with patch(
        "data_base.indexing_service.add_markdown_with_hierarchical_indexing",
        new=AsyncMock(return_value=7),
    ) as add_hierarchical:
        result = await index_markdown_document(
            user_id="user-1",
            markdown_text="# Demo",
            pdf_title="Demo",
            doc_id="doc-1",
            indexing_profile=indexing_profile,
        )

    assert result == 7
    add_hierarchical.assert_awaited_once_with(
        user_id="user-1",
        markdown_text="# Demo",
        pdf_title="Demo",
        doc_id="doc-1",
        enable_proposition_indexing=enable_proposition_indexing,
    )


@pytest.mark.asyncio
async def test_index_markdown_document_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unsupported indexing_profile"):
        await index_markdown_document(
            user_id="user-1",
            markdown_text="# Demo",
            pdf_title="Demo",
            doc_id="doc-1",
            indexing_profile="unknown-profile",  # type: ignore[arg-type]
        )
