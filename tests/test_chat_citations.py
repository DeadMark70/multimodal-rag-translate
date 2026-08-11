"""Truthful citation-builder contract tests."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from data_base.citations import build_source_details


@pytest.mark.asyncio
async def test_citation_uses_retrieved_text_and_metadata_only() -> None:
    documents = [
        Document(
            page_content="  original paragraph from the PDF  ",
            metadata={
                "doc_id": "doc-1",
                "file_name": "paper.pdf",
                "page": 3,
                "relevance_score": 0.82,
                "bbox": [0.1, 0.2, 0.8, 0.4],
            },
        )
    ]
    with patch(
        "data_base.citations.fetch_document_filenames",
        new=AsyncMock(return_value={"doc-1": "db-name.pdf"}),
    ):
        result = await build_source_details(documents, ["doc-1"])

    assert [item.model_dump() for item in result] == [
        {
            "doc_id": "doc-1",
            "filename": "paper.pdf",
            "page": 3,
            "snippet": "original paragraph from the PDF",
            "score": 0.82,
            "bbox": (0.1, 0.2, 0.8, 0.4),
        }
    ]


@pytest.mark.asyncio
async def test_invalid_precision_degrades_without_fabrication() -> None:
    document = Document(
        page_content="source text",
        metadata={
            "doc_id": "doc-1",
            "page": 0,
            "score": 2.5,
            "bbox": "[1,2,3,4]",
        },
    )
    with patch(
        "data_base.citations.fetch_document_filenames",
        new=AsyncMock(return_value={"doc-1": "paper.pdf"}),
    ):
        result = await build_source_details([document], ["doc-1"])

    assert result[0].filename == "paper.pdf"
    assert result[0].page is None
    assert result[0].score is None
    assert result[0].bbox is None
    assert result[0].snippet == "source text"


@pytest.mark.asyncio
async def test_missing_document_becomes_source_only_without_answer_fallback() -> None:
    with patch(
        "data_base.citations.fetch_document_filenames",
        new=AsyncMock(return_value={"doc-2": "missing.pdf"}),
    ):
        result = await build_source_details([], ["doc-2"])

    assert result[0].model_dump() == {
        "doc_id": "doc-2",
        "filename": "missing.pdf",
        "page": None,
        "snippet": None,
        "score": None,
        "bbox": None,
    }


@pytest.mark.asyncio
async def test_same_document_chunks_with_different_pages_are_preserved() -> None:
    documents = [
        Document(
            page_content="shared evidence",
            metadata={"doc_id": "doc-1", "page": 2},
        ),
        Document(
            page_content="shared evidence",
            metadata={"doc_id": "doc-1", "page": 7},
        ),
    ]
    with patch(
        "data_base.citations.fetch_document_filenames",
        new=AsyncMock(return_value={"doc-1": "one.pdf"}),
    ):
        result = await build_source_details(documents, ["doc-1"])

    assert [(item.doc_id, item.page, item.snippet) for item in result] == [
        ("doc-1", 2, "shared evidence"),
        ("doc-1", 7, "shared evidence"),
    ]


@pytest.mark.asyncio
async def test_same_document_chunks_with_different_snippets_are_preserved() -> None:
    documents = [
        Document(
            page_content="first passage",
            metadata={"doc_id": "doc-1", "page": 3},
        ),
        Document(
            page_content="second passage",
            metadata={"doc_id": "doc-1", "page": 3},
        ),
    ]
    with patch(
        "data_base.citations.fetch_document_filenames",
        new=AsyncMock(return_value={"doc-1": "one.pdf"}),
    ):
        result = await build_source_details(documents, ["doc-1"])

    assert [(item.doc_id, item.page, item.snippet) for item in result] == [
        ("doc-1", 3, "first passage"),
        ("doc-1", 3, "second passage"),
    ]


@pytest.mark.asyncio
async def test_exact_duplicate_chunks_keep_first_retrieval_occurrence() -> None:
    documents = [
        Document(page_content="first doc-2 chunk", metadata={"doc_id": "doc-2"}),
        Document(page_content="first doc-1 chunk", metadata={"doc_id": "doc-1"}),
        Document(page_content="first doc-2 chunk", metadata={"doc_id": "doc-2"}),
        Document(page_content="unrequested", metadata={"doc_id": "doc-3"}),
    ]
    with patch(
        "data_base.citations.fetch_document_filenames",
        new=AsyncMock(return_value={"doc-1": "one.pdf", "doc-2": "two.pdf"}),
    ):
        result = await build_source_details(documents, ["doc-1", "doc-2"])

    assert [(item.doc_id, item.snippet) for item in result] == [
        ("doc-2", "first doc-2 chunk"),
        ("doc-1", "first doc-1 chunk"),
    ]
