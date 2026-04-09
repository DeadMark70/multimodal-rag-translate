"""Production indexing orchestration for RAG/vector writes."""

from __future__ import annotations

from typing import Literal

from multimodal_rag.schemas import ExtractedDocument

from data_base.vector_store_manager import (
    add_markdown_to_knowledge_base,
    add_markdown_with_hierarchical_indexing,
    add_visual_summaries_to_knowledge_base,
    index_extracted_document,
)

IndexingProfile = Literal[
    "recursive_baseline",
    "semantic_contextual",
    "hierarchical_parent_child",
    "hierarchical_parent_child_proposition",
]

DEFAULT_INDEXING_PROFILE: IndexingProfile = "recursive_baseline"
DEFAULT_PRODUCTION_INDEXING_PROFILE: IndexingProfile = "semantic_contextual"


async def index_markdown_document(
    *,
    user_id: str,
    markdown_text: str,
    pdf_title: str,
    doc_id: str,
    k_retriever: int = 3,
    indexing_profile: IndexingProfile = DEFAULT_INDEXING_PROFILE,
) -> object:
    """Index one markdown document through the production orchestration path."""
    if indexing_profile == "recursive_baseline":
        return await add_markdown_to_knowledge_base(
            user_id=user_id,
            markdown_text=markdown_text,
            pdf_title=pdf_title,
            doc_id=doc_id,
            k_retriever=k_retriever,
            chunking_method="recursive",
            enable_context_enrichment=False,
        )

    if indexing_profile == "semantic_contextual":
        return await add_markdown_to_knowledge_base(
            user_id=user_id,
            markdown_text=markdown_text,
            pdf_title=pdf_title,
            doc_id=doc_id,
            k_retriever=k_retriever,
            chunking_method="semantic",
            enable_context_enrichment=True,
        )

    if indexing_profile == "hierarchical_parent_child":
        return await add_markdown_with_hierarchical_indexing(
            user_id=user_id,
            markdown_text=markdown_text,
            pdf_title=pdf_title,
            doc_id=doc_id,
            enable_proposition_indexing=False,
        )

    if indexing_profile == "hierarchical_parent_child_proposition":
        return await add_markdown_with_hierarchical_indexing(
            user_id=user_id,
            markdown_text=markdown_text,
            pdf_title=pdf_title,
            doc_id=doc_id,
            enable_proposition_indexing=True,
        )

    raise ValueError(f"Unsupported indexing_profile: {indexing_profile}")


def index_visual_summaries(
    *,
    user_id: str,
    doc_id: str,
    elements: list,
) -> int:
    """Index summarized visual elements through the production orchestration path."""
    return add_visual_summaries_to_knowledge_base(
        user_id=user_id,
        doc_id=doc_id,
        elements=elements,
    )


def index_extracted_document_content(*, user_id: str, doc: ExtractedDocument) -> None:
    """Index a multimodal extracted document through the production orchestration path."""
    index_extracted_document(user_id=user_id, doc=doc)
