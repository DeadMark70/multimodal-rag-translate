from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from data_base.rag_crag import (
    classify_crag_retrieval,
    run_corrective_retrieval,
)


def test_crag_classification_is_deterministic_for_empty_and_populated_retrieval() -> None:
    empty = classify_crag_retrieval([])
    populated = classify_crag_retrieval(
        [Document(page_content="evidence", metadata={"doc_id": "doc-1"})]
    )

    assert (empty.status, empty.reason) == ("insufficient", "no_documents")
    assert (populated.status, populated.reason) == ("judge_required", "documents_present")


@pytest.mark.asyncio
async def test_crag_keeps_initial_documents_when_optional_judge_is_unavailable() -> None:
    documents = [Document(page_content="evidence", metadata={"doc_id": "doc-1"})]
    rewrite = AsyncMock()
    retrieve = AsyncMock()

    result = await run_corrective_retrieval(
        question="question",
        documents=documents,
        retriever=object(),
        judge=None,
        rewrite_mode="hyde",
        hyde_transformer=rewrite,
        query_executor=retrieve,
    )

    assert result.documents == documents
    assert result.status == "accepted"
    assert result.correction_applied is False
    rewrite.assert_not_awaited()
    retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_crag_rewrites_retrieves_fuses_and_scopes_rejected_documents() -> None:
    initial = [Document(page_content="weak", metadata={"doc_id": "doc-1"})]
    accepted = Document(page_content="accepted", metadata={"doc_id": "doc-1"})
    rejected = Document(page_content="rejected", metadata={"doc_id": "doc-2"})
    judge = AsyncMock(return_value=False)
    rewrite = AsyncMock(return_value=["question", "variant"])
    retrieve = AsyncMock(return_value=[[accepted], [rejected, accepted]])
    progress = AsyncMock()
    retriever = object()

    result = await run_corrective_retrieval(
        question="question",
        documents=initial,
        retriever=retriever,
        judge=judge,
        rewrite_mode="multi_query",
        doc_ids=["doc-1"],
        target_k=8,
        multi_query_transformer=rewrite,
        query_executor=retrieve,
        progress_callback=progress,
    )

    assert result.documents == [accepted]
    assert result.status == "corrected"
    assert result.correction_applied is True
    judge.assert_awaited_once_with(question="question", documents=initial)
    rewrite.assert_awaited_once_with(
        "question", enabled=True, phase="retrieval_rewrite"
    )
    retrieve.assert_awaited_once_with(retriever, ["question", "variant"])
    progress.assert_awaited()
