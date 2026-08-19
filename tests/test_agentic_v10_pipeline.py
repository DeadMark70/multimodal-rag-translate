"""Focused coverage for the evaluation-only Agentic RAG v10 path."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from data_base.agentic_v10.subquery_decomposer import SubQueryItem, _fallback_subqueries
from data_base.agentic_v10.subquery_pipeline_service import AgenticV10PipelineService
from evaluation.export_schemas import ExportCampaignRequest
from evaluation.export_service import _project_agentic_v10


def test_v10_fallback_produces_multiple_queries() -> None:
    items = _fallback_subqueries("Compare SAMed versus MedSAM on abdominal CT")
    assert len(items) >= 2
    assert any("SAMed" in item.query for item in items)


@pytest.mark.asyncio
async def test_v10_pipeline_packs_raw_top1_and_same_document_neighbors() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="model architecture", focus="架構", target_entity="Model"),
        SubQueryItem(id="SQ2", query="model benchmark", focus="實驗", target_entity="Benchmark"),
    ])
    reranker = MagicMock()
    first = Document(page_content="Architecture evidence", metadata={"doc_id": "doc-1", "page_number": 2, "chunk_index_in_page": 1})
    second = Document(page_content="Benchmark evidence", metadata={"doc_id": "doc-2", "page_number": 4, "chunk_index_in_page": 1})
    first_previous = Document(page_content="Architecture preceding context", metadata={"doc_id": "doc-1", "page_number": 2, "chunk_index_in_page": 0})
    first_next = Document(page_content="Architecture following context", metadata={"doc_id": "doc-1", "page_number": 2, "chunk_index_in_page": 2})
    second_previous = Document(page_content="Benchmark preceding context", metadata={"doc_id": "doc-2", "page_number": 4, "chunk_index_in_page": 0})
    second_next = Document(page_content="Benchmark following context", metadata={"doc_id": "doc-2", "page_number": 4, "chunk_index_in_page": 2})
    reranker.rerank_with_scores.side_effect = [[(first, 0.9)], [(second, 0.8)]]
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=reranker)
    synthesis_llm = MagicMock()
    synthesis_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="Grounded answer",
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    ))
    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(side_effect=[MagicMock(documents=[first]), MagicMock(documents=[second])])),
        patch("data_base.agentic_v10.subquery_pipeline_service.load_user_vector_documents_async", new=AsyncMock(return_value=[first_previous, first, first_next, second_previous, second, second_next])),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm", return_value=synthesis_llm) as get_llm,
    ):
        result = await service.execute(question="Compare the models", user_id="user-1")
    trace = result.agent_trace or {}
    v10 = trace["agentic_v10"]
    assert trace["agentic_execution_version"] == "v10"
    assert len(v10["branches"]) == 2
    assert v10["branches"][0]["raw_candidates"][0]["content"] == "Architecture evidence"
    synthesis_prompt = v10["synthesis"]["prompt_messages"][1]["content"]
    assert "Architecture evidence" in synthesis_prompt
    assert "Architecture preceding context" in synthesis_prompt
    assert "Architecture following context" in synthesis_prompt
    assert "Benchmark evidence" in synthesis_prompt
    assert "Benchmark preceding context" in synthesis_prompt
    assert "Benchmark following context" in synthesis_prompt
    assert v10["context_pack"]["strategy"] == "top1_raw_same_document_neighbors"
    assert v10["context_pack"]["top1_evidence_count"] == 2
    assert v10["context_pack"]["neighbor_evidence_count"] == 4
    neighbor_mapping = next(
        item
        for item in v10["source_document_mapping"]
        if item["document"]["content"] == "Architecture preceding context"
    )
    assert neighbor_mapping["context_origin"] == "same_document_neighbor"
    assert neighbor_mapping["neighbor_of_reference_ids"] == ["[Ref 1]"]
    assert result.usage["total_tokens"] == 10
    assert get_llm.call_args.kwargs == {"purpose": "synthesizer"}
    assert reranker.rerank_with_scores.call_count == 2
    assert [call.kwargs["top_k"] for call in reranker.rerank_with_scores.call_args_list] == [1, 1]
    assert reranker.rerank.call_count == 0


@pytest.mark.asyncio
async def test_v10_neighbor_lookup_failure_keeps_top1_raw_chunk() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="query", focus="focus", target_entity="entity")
    ])
    evidence = Document(page_content="Only raw evidence", metadata={"doc_id": "doc-1", "page_number": 1, "chunk_index_in_page": 0})
    reranker = MagicMock()
    reranker.rerank_with_scores.return_value = [(evidence, 0.9)]
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=reranker)
    synthesis_llm = MagicMock()
    synthesis_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Answer", usage_metadata={}))

    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(return_value=MagicMock(documents=[evidence]))),
        patch("data_base.agentic_v10.subquery_pipeline_service.load_user_vector_documents_async", new=AsyncMock(side_effect=RuntimeError("offline"))),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm", return_value=synthesis_llm),
    ):
        result = await service.execute(question="Question", user_id="user-1")

    v10 = result.agent_trace["agentic_v10"]
    assert v10["context_pack"]["neighbor_lookup_error"] == "RuntimeError"
    assert v10["context_pack"]["neighbor_evidence_count"] == 0
    assert "Only raw evidence" in v10["synthesis"]["prompt_messages"][1]["content"]


@pytest.mark.asyncio
async def test_v10_does_not_invoke_a_map_model() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="query", focus="focus", target_entity="entity")
    ])
    evidence = Document(page_content="Raw evidence", metadata={"doc_id": "doc-1", "page_number": 1, "chunk_index_in_page": 0})
    reranker = MagicMock()
    reranker.rerank_with_scores.return_value = [(evidence, 0.9)]
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=reranker)
    synthesis_llm = MagicMock()
    synthesis_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Answer", usage_metadata={}))

    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(return_value=MagicMock(documents=[evidence]))),
        patch("data_base.agentic_v10.subquery_pipeline_service.load_user_vector_documents_async", new=AsyncMock(return_value=[evidence])),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm", return_value=synthesis_llm) as get_llm,
    ):
        result = await service.execute(question="Question", user_id="user-1")

    assert "Raw evidence" in result.agent_trace["agentic_v10"]["context_pack"]["rendered_context"]
    assert get_llm.call_count == 1
    assert get_llm.call_args.kwargs == {"purpose": "synthesizer"}


@pytest.mark.asyncio
async def test_v10_returns_transparent_partial_when_no_evidence_exists() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="missing evidence", focus="缺失", target_entity="Missing"),
        SubQueryItem(id="SQ2", query="still missing", focus="缺失", target_entity="Missing"),
    ])
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=MagicMock())
    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(return_value=MagicMock(documents=[]))),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm") as get_llm,
    ):
        result = await service.execute(question="Unknown subject", user_id="user-1")
    assert result.agent_trace["response_status"] == "qualified_partial"
    assert result.agent_trace["agentic_v10"]["context_pack"]["top1_evidence_count"] == 0
    assert result.agent_trace["agentic_v10"]["context_pack"]["neighbor_evidence_count"] == 0
    assert "沒有可用" in result.answer
    get_llm.assert_not_called()


@pytest.mark.asyncio
async def test_v10_keeps_failed_retrieval_branches_in_partial_trace() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="missing evidence", focus="缺失", target_entity="Missing"),
        SubQueryItem(id="SQ2", query="still missing", focus="缺失", target_entity="Missing"),
    ])
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=MagicMock())
    with patch(
        "data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async",
        new=AsyncMock(side_effect=RuntimeError("offline")),
    ):
        result = await service.execute(question="Unknown subject", user_id="user-1")

    trace = result.agent_trace or {}
    assert trace["response_status"] == "qualified_partial"
    assert {branch["retrieval_error"] for branch in trace["agentic_v10"]["branches"]} == {"RuntimeError"}


def test_v10_export_keeps_raw_trace_behind_existing_export_switches() -> None:
    trace = MagicMock(
        execution_profile="agentic_eval_v10_top1_raw_same_document_neighbors",
        agentic_v10={
            "schema_version": "4",
            "response_status": "complete",
            "decomposition": {"sub_queries": [{"id": "SQ1"}]},
            "branches": [{"subquery_id": "SQ1"}],
            "deduplicated_evidence": [{"content": "evidence"}],
            "context_pack": {
                "strategy": "top1_raw_same_document_neighbors",
                "neighbor_evidence_count": 2,
            },
            "synthesis": {"prompt_messages": [{"content": "full prompt"}]},
        },
    )

    summary = _project_agentic_v10(trace, ExportCampaignRequest())
    assert summary is not None
    assert summary["summary"]["subquery_count"] == 1
    assert summary["summary"]["context_strategy"] == "top1_raw_same_document_neighbors"
    assert summary["summary"]["neighbor_evidence_count"] == 2
    assert "raw_payload" not in summary

    raw = _project_agentic_v10(
        trace,
        ExportCampaignRequest(
            include_raw_trace_payloads=True,
            include_full_prompts=True,
        ),
    )
    assert raw is not None
    assert raw["raw_payload"]["synthesis"]["prompt_messages"][0]["content"] == "full prompt"
    assert raw["raw_payload"]["context_pack"]["strategy"] == "top1_raw_same_document_neighbors"
    assert json.loads(json.dumps(raw)) == raw
