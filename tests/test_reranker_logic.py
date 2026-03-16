"""Focused regression tests for reranker selection behavior."""

# Standard library
from unittest.mock import AsyncMock
from unittest.mock import MagicMock, patch

# Third-party
import pytest
from langchain_core.documents import Document

from data_base.reranker import DocumentReranker, rerank_documents


@pytest.fixture(autouse=True)
def reset_reranker_singleton():
    """Reset singleton state between tests."""
    DocumentReranker._instance = None
    DocumentReranker._model = None
    DocumentReranker._model_name = None
    DocumentReranker._device = None
    DocumentReranker._init_error = None
    DocumentReranker._device_reason = None
    yield
    DocumentReranker._instance = None
    DocumentReranker._model = None
    DocumentReranker._model_name = None
    DocumentReranker._device = None
    DocumentReranker._init_error = None
    DocumentReranker._device_reason = None


def test_reranker_singleton() -> None:
    """Verify that DocumentReranker stays singleton-backed."""
    mock_model = MagicMock()
    with patch("data_base.reranker.torch.cuda.is_available", return_value=False):
        with patch("data_base.reranker.AutoModel.from_pretrained", return_value=mock_model) as mock_from_pretrained:
            r1 = DocumentReranker()
            r2 = DocumentReranker()

    assert r1 is r2
    assert mock_from_pretrained.call_count == 1


def test_reranking_logic() -> None:
    """Verify that reranking correctly reorders documents based on scores."""
    mock_model = MagicMock()
    mock_model.rerank.return_value = [
        {"index": 1, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.5},
        {"index": 0, "relevance_score": 0.1},
    ]

    reranker = object.__new__(DocumentReranker)
    DocumentReranker._instance = reranker
    DocumentReranker._model = mock_model
    DocumentReranker._model_name = "jinaai/jina-reranker-v3"
    DocumentReranker._device = "cpu"

    query = "SwinUNETR優勢"
    docs = [
        Document(page_content="雜訊內容 A", metadata={"id": "noise1"}),
        Document(page_content="SwinUNETR 核心架構優勢...", metadata={"id": "relevant"}),
        Document(page_content="雜訊內容 B", metadata={"id": "noise2"}),
    ]

    reranked = reranker.rerank(query, docs, top_k=3)

    assert reranked[0].metadata["id"] == "relevant"
    assert reranked[1].metadata["id"] == "noise2"
    assert reranked[2].metadata["id"] == "noise1"


def test_reranker_not_initialized() -> None:
    """Verify fallback when reranker is not initialized."""
    docs = [Document(page_content="A"), Document(page_content="B")]

    result = rerank_documents("query", docs, enabled=True)
    assert result == docs


def test_generation_rerank_prefers_non_noise_docs() -> None:
    """Known noise docs should only backfill after non-noise candidates."""
    from data_base.RAG_QA_service import _rerank_documents_for_generation

    mock_model = MagicMock()
    mock_model.rerank.return_value = [
        {"index": 0, "relevance_score": 0.99},
        {"index": 1, "relevance_score": 0.96},
        {"index": 2, "relevance_score": 0.95},
    ]

    reranker = object.__new__(DocumentReranker)
    DocumentReranker._instance = reranker
    DocumentReranker._model = mock_model
    DocumentReranker._model_name = "jinaai/jina-reranker-v3"
    DocumentReranker._device = "cpu"

    docs = [
        Document(page_content="SAM benchmark details", metadata={"id": "noise-1", "file_name": "sam-paper.pdf"}),
        Document(page_content="Relevant SwinUNETR comparison", metadata={"id": "relevant-1"}),
        Document(page_content="Another relevant nnU-Net comparison", metadata={"id": "relevant-2"}),
    ]

    selected = _rerank_documents_for_generation("比較 SwinUNETR 與 nnU-Net", docs, target_k=2)

    assert [doc.metadata["id"] for doc in selected] == ["relevant-1", "relevant-2"]


def test_generation_rerank_keeps_noise_when_query_requests_it() -> None:
    """Noise heuristics should not suppress docs when the query explicitly asks for them."""
    from data_base.RAG_QA_service import _rerank_documents_for_generation

    mock_model = MagicMock()
    mock_model.rerank.return_value = [
        {"index": 0, "relevance_score": 0.99},
        {"index": 1, "relevance_score": 0.96},
    ]

    reranker = object.__new__(DocumentReranker)
    DocumentReranker._instance = reranker
    DocumentReranker._model = mock_model
    DocumentReranker._model_name = "jinaai/jina-reranker-v3"
    DocumentReranker._device = "cpu"

    docs = [
        Document(page_content="SAM benchmark details", metadata={"id": "noise-1", "file_name": "sam-paper.pdf"}),
        Document(page_content="Relevant SwinUNETR comparison", metadata={"id": "relevant-1"}),
    ]

    selected = _rerank_documents_for_generation("請比較 SAM 與 SegVol", docs, target_k=2)

    assert [doc.metadata["id"] for doc in selected] == ["noise-1", "relevant-1"]


def test_reranker_inference_retries_on_cpu_after_cuda_oom() -> None:
    """Inference-time CUDA OOM should reload the model on CPU and retry once."""
    cpu_model = MagicMock()
    cpu_model.rerank.return_value = [{"index": 1, "relevance_score": 0.8}]

    gpu_model = MagicMock()
    gpu_model.rerank.side_effect = RuntimeError("CUDA out of memory")

    reranker = object.__new__(DocumentReranker)
    DocumentReranker._instance = reranker
    DocumentReranker._model = gpu_model
    DocumentReranker._model_name = "jinaai/jina-reranker-v3"
    DocumentReranker._device = "cuda"

    docs = [
        Document(page_content="Noise", metadata={"id": "noise"}),
        Document(page_content="Relevant", metadata={"id": "relevant"}),
    ]

    with patch.object(DocumentReranker, "_load_model", return_value=cpu_model):
        reranked = reranker.rerank("query", docs, top_k=1)

    assert [doc.metadata["id"] for doc in reranked] == ["relevant"]
    assert DocumentReranker.runtime_metadata()["reranker_device"] == "cpu"
    assert DocumentReranker.runtime_metadata()["reranker_reason"] == "cuda_oom_fallback"


def test_rag_answer_question_caps_rerank_candidate_count() -> None:
    """Rerank-enabled retrieval should not pull the old 50-document candidate set."""
    from data_base import RAG_QA_service

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="answer", usage_metadata={}))
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="Doc 1", metadata={})]

    with patch("data_base.RAG_QA_service.get_llm", return_value=mock_llm), patch(
        "data_base.RAG_QA_service.get_user_retriever",
        return_value=mock_retriever,
    ) as mock_get_retriever:
        import asyncio

        asyncio.run(
            RAG_QA_service.rag_answer_question(
                question="Explain",
                user_id="user-1",
                enable_reranking=True,
            )
        )

    assert mock_get_retriever.call_args.kwargs["k"] == 20
