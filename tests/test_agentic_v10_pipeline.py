"""Focused coverage for the evaluation-only Agentic RAG v10 path."""

import asyncio
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
async def test_v10_pipeline_maps_cards_in_parallel_and_final_uses_cards_only() -> None:
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
    active = 0
    peak_active = 0

    async def map_invoke(messages: list[dict[str, str]]) -> dict[str, object]:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {
            "parsed": {
                "status": "summarized",
                "supported_findings": [
                    {"statement": "Directly supported finding"}
                ],
                "missing_or_unsupported": [],
            },
            "raw": MagicMock(usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}),
            "parsing_error": None,
        }

    map_llm = MagicMock()
    map_llm.ainvoke = AsyncMock(side_effect=map_invoke)
    summary_llm = MagicMock()
    summary_llm.with_structured_output.return_value = map_llm
    synthesis_llm = MagicMock()
    synthesis_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="Grounded answer",
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    ))
    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(side_effect=[MagicMock(documents=[first]), MagicMock(documents=[second])])),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm", side_effect=[summary_llm, synthesis_llm]) as get_llm,
    ):
        result = await service.execute(question="Compare the models", user_id="user-1")
    trace = result.agent_trace or {}
    v10 = trace["agentic_v10"]
    assert trace["agentic_execution_version"] == "v10"
    assert len(v10["branches"]) == 2
    assert v10["branches"][0]["raw_candidates"][0]["content"] == "Architecture evidence"
    assert peak_active == 2
    assert summary_llm.with_structured_output.call_args.kwargs["method"] == "json_schema"
    map_schema = summary_llm.with_structured_output.call_args.args[0]
    assert map_schema.__name__ == "MapSubqueryEvidenceCard"
    assert "reference_ids" not in json.dumps(map_schema.model_json_schema())
    assert v10["branches"][0]["map"]["status"] == "summarized"
    assert v10["context_pack"]["evidence_cards"][0]["supported_findings"][0]["reference_ids"] == ["[Ref 1]"]
    synthesis_prompt = v10["synthesis"]["prompt_messages"][1]["content"]
    assert "Directly supported finding" in synthesis_prompt
    assert "Architecture evidence" not in synthesis_prompt
    assert result.usage["total_tokens"] == 24
    assert get_llm.call_args_list[0].kwargs == {"purpose": "summary"}
    assert get_llm.call_args_list[1].kwargs == {"purpose": "synthesizer"}
    assert reranker.rerank_with_scores.call_count == 2
    assert [call.kwargs["top_k"] for call in reranker.rerank_with_scores.call_args_list] == [1, 1]
    assert reranker.rerank.call_count == 0


@pytest.mark.asyncio
async def test_v10_map_failure_preserves_raw_chunk_for_final() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="query", focus="focus", target_entity="entity")
    ])
    evidence = Document(page_content="Only raw fallback evidence", metadata={"doc_id": "doc-1"})
    reranker = MagicMock()
    reranker.rerank_with_scores.return_value = [(evidence, 0.9)]
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=reranker)
    map_llm = MagicMock()
    map_llm.ainvoke = AsyncMock(side_effect=RuntimeError("bad schema"))
    summary_llm = MagicMock()
    summary_llm.with_structured_output.return_value = map_llm
    synthesis_llm = MagicMock()
    synthesis_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Answer", usage_metadata={}))

    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(return_value=MagicMock(documents=[evidence]))),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm", side_effect=[summary_llm, synthesis_llm]),
    ):
        result = await service.execute(question="Question", user_id="user-1")

    v10 = result.agent_trace["agentic_v10"]
    card = v10["context_pack"]["evidence_cards"][0]
    assert card["status"] == "raw_fallback"
    assert card["reference_ids"] == ["[Ref 1]"]
    assert "Only raw fallback evidence" in card["raw_evidence_block"]
    assert v10["map_stage"]["fallback_count"] == 1
    assert v10["branches"][0]["map"]["failure_diagnostic"] == "RuntimeError: bad schema"
    assert "Only raw fallback evidence" in v10["synthesis"]["prompt_messages"][1]["content"]


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
    assert all(card["status"] == "no_evidence" for card in result.agent_trace["agentic_v10"]["context_pack"]["evidence_cards"])
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
        execution_profile="agentic_v10_top1_map_reduce_backend_refs",
        agentic_v10={
            "schema_version": "3",
            "response_status": "complete",
            "decomposition": {"sub_queries": [{"id": "SQ1"}]},
            "branches": [{"subquery_id": "SQ1"}],
            "deduplicated_evidence": [{"content": "evidence"}],
            "map_stage": {"card_count": 1, "fallback_count": 1, "failure_count": 1},
            "synthesis": {"prompt_messages": [{"content": "full prompt"}]},
        },
    )

    summary = _project_agentic_v10(trace, ExportCampaignRequest())
    assert summary is not None
    assert summary["summary"]["subquery_count"] == 1
    assert summary["summary"]["map_fallback_count"] == 1
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
    assert "rendered_evidence_cards" not in raw["raw_payload"].get("context_pack", {})
    assert json.loads(json.dumps(raw)) == raw
