"""Contracts for the generic retrieval-to-generation boundary."""

from langchain_core.documents import Document

from data_base.RAG_QA_service import RAGResult
from data_base.rag_pipeline_schemas import GeneratedRagAnswer, RagRetrievalResult


def test_retrieval_result_keeps_evidence_separate_from_generation_data() -> None:
    document = Document(
        page_content="Evidence from the source.",
        metadata={"doc_id": "doc-1", "retrieval_rank": 1},
    )

    result = RagRetrievalResult(
        documents=[document],
        source_doc_ids=["doc-1"],
        context="Context assembled from retrieved evidence.",
        metadata={"query_origin": "original", "expanded_queries": ["question"]},
        images=[{"image_id": "image-1"}],
    )

    assert result.documents == [document]
    assert result.source_doc_ids == ["doc-1"]
    assert result.context == "Context assembled from retrieved evidence."
    assert result.metadata["query_origin"] == "original"
    assert result.images == [{"image_id": "image-1"}]
    assert not hasattr(result, "answer")


def test_generated_answer_carries_only_generation_projection() -> None:
    result = GeneratedRagAnswer(
        answer="Supported answer.",
        usage={"total_tokens": 12},
        thought_process="Synthesized from supplied context.",
        tool_calls=[{"name": "visual_verification"}],
        agent_trace={"mode": "advanced"},
        visual_verification_meta={"iterations": 1},
    )

    assert result.answer == "Supported answer."
    assert result.usage == {"total_tokens": 12}
    assert result.thought_process == "Synthesized from supplied context."
    assert result.tool_calls == [{"name": "visual_verification"}]
    assert result.agent_trace == {"mode": "advanced"}
    assert result.visual_verification_meta == {"iterations": 1}
    assert not hasattr(result, "documents")


def test_legacy_rag_result_constructor_and_fields_remain_compatible() -> None:
    document = Document(page_content="Legacy evidence", metadata={"doc_id": "doc-1"})

    result = RAGResult("Legacy answer", ["doc-1"], [document])

    assert result.answer == "Legacy answer"
    assert result.source_doc_ids == ["doc-1"]
    assert result.documents == [document]
    assert result.usage == {}
    assert result.thought_process is None
    assert result.tool_calls == []
    assert result.agent_trace is None
    assert result.visual_verification_meta is None
