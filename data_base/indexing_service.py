"""Production indexing orchestration for RAG/vector writes."""

from __future__ import annotations

from multimodal_rag.schemas import ExtractedDocument

from data_base.vector_store_manager import (
    add_markdown_to_knowledge_base,
    add_visual_summaries_to_knowledge_base,
    index_extracted_document,
)


async def index_markdown_document(
    *,
    user_id: str,
    markdown_text: str,
    pdf_title: str,
    doc_id: str,
    k_retriever: int = 3,
) -> object:
    """Index one markdown document through the production orchestration path."""
    return await add_markdown_to_knowledge_base(
        user_id=user_id,
        markdown_text=markdown_text,
        pdf_title=pdf_title,
        doc_id=doc_id,
        k_retriever=k_retriever,
    )


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
