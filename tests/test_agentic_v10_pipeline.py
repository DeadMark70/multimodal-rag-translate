"""Focused coverage for the evaluation-only Agentic RAG v10 path."""

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
async def test_v10_pipeline_records_full_branch_trace() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="model architecture", focus="架構", target_entity="Model"),
        SubQueryItem(id="SQ2", query="model benchmark", focus="實驗", target_entity="Benchmark"),
    ])
    reranker = MagicMock()
    first = Document(page_content="Architecture evidence", metadata={"doc_id": "doc-1", "page": 2})
    second = Document(page_content="Benchmark evidence", metadata={"doc_id": "doc-2", "page": 4})
    reranker.rerank_with_scores.side_effect = [[(first, 0.9)], [(second, 0.8)]]
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=reranker)
    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(side_effect=[MagicMock(documents=[first]), MagicMock(documents=[second])])),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm") as get_llm,
    ):
        response = MagicMock(content="Grounded answer", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
        get_llm.return_value.ainvoke = AsyncMock(return_value=response)
        result = await service.execute(question="Compare the models", user_id="user-1")
    trace = result.agent_trace or {}
    v10 = trace["agentic_v10"]
    assert trace["agentic_execution_version"] == "v10"
    assert len(v10["branches"]) == 2
    assert v10["branches"][0]["raw_candidates"][0]["content"] == "Architecture evidence"
    assert v10["synthesis"]["prompt_messages"]
    assert result.usage["total_tokens"] == 10
    assert reranker.rerank_with_scores.call_count == 2
    assert [call.kwargs["top_k"] for call in reranker.rerank_with_scores.call_args_list] == [2, 2]
    assert reranker.rerank.call_count == 0


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
        execution_profile="agentic_v10_subquery_sequential_rerank_top2",
        agentic_v10={
            "schema_version": "1",
            "response_status": "qualified_answer",
            "decomposition": {"sub_queries": [{"id": "SQ1"}]},
            "branches": [{"subquery_id": "SQ1"}],
            "deduplicated_evidence": [{"content": "evidence"}],
            "synthesis": {"prompt_messages": [{"content": "full prompt"}]},
        },
    )

    summary = _project_agentic_v10(trace, ExportCampaignRequest())
    assert summary is not None
    assert summary["summary"]["subquery_count"] == 1
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
