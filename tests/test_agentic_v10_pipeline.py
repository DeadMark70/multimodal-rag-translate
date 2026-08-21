"""Focused coverage for the evaluation-only Agentic RAG v10 path."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from data_base.agentic_v10.subquery_decomposer import (
    SubQueryDecompositionResponse,
    SubQueryItem,
    _fallback_subqueries,
)
from data_base.agentic_v10.subquery_pipeline_service import (
    AgenticV10PipelineService,
    CoverageAuditResponse,
    EntityCriterionMatrixCell,
    ExtractiveEvidenceLedgerEntry,
)
from evaluation.export_schemas import ExportCampaignRequest
from evaluation.export_service import _project_agentic_v10


def test_v10_fallback_produces_multiple_queries() -> None:
    items = _fallback_subqueries("Compare SAMed versus MedSAM on abdominal CT")
    assert len(items) >= 2
    assert len(items) <= 3
    assert any("SAMed" in item.query for item in items)


def test_v10_decomposition_schema_rejects_more_than_three_queries() -> None:
    items = [
        {"id": f"SQ{index}", "query": f"query {index}", "focus": "focus"}
        for index in range(1, 5)
    ]
    with pytest.raises(ValueError, match="at most 3"):
        SubQueryDecompositionResponse.model_validate({"sub_queries": items})


def test_v10_audit_validation_requires_each_requirement_and_source() -> None:
    audit = CoverageAuditResponse.model_validate({
        "needs_drill_down": 0,
        "answer": "Incomplete answer",
        "requirements": [
            {"id": "R1", "entity": "A", "criterion": "first"},
            {"id": "R2", "entity": "B", "criterion": "second"},
        ],
        "entity_criterion_matrix": [
            {"requirement_id": "R1", "coverage": "supported", "reference_ids": ["[Ref 1]"]},
        ],
        "extractive_evidence_ledger": [
            {"reference_id": "[Ref 1]", "requirement_ids": ["R1"]},
        ],
        "priority_gap": None,
    })

    result = AgenticV10PipelineService._validate_coverage_audit(
        audit=audit,
        valid_reference_ids={"[Ref 1]"},
    )

    assert result["validated"] is False
    assert result["failure_reason"] == "incomplete_matrix"


def test_v10_audit_validation_normalizes_bracketless_reference_ids() -> None:
    audit = CoverageAuditResponse.model_validate({
        "needs_drill_down": 1,
        "answer": None,
        "requirements": [{"id": "R1", "entity": "A", "criterion": "first"}],
        "entity_criterion_matrix": [
            {"requirement_id": "R1", "coverage": "partial", "reference_ids": ["Ref 1"]},
        ],
        "extractive_evidence_ledger": [
            {"reference_id": "Ref 1", "requirement_ids": ["R1"]},
        ],
        "priority_gap": {
            "requirement_id": "R1",
            "missing_information": "missing value",
            "retrieval_query": "A missing value",
        },
    })

    result = AgenticV10PipelineService._validate_coverage_audit(
        audit=audit,
        valid_reference_ids={"[Ref 1]"},
    )

    assert result["validated"] is True
    assert audit.entity_criterion_matrix[0].reference_ids == ["[Ref 1]"]
    assert audit.extractive_evidence_ledger[0].reference_id == "[Ref 1]"
    assert len(result["normalized_references"]) == 2


def test_v10_audit_validation_discards_empty_ledger_entries() -> None:
    audit = CoverageAuditResponse.model_validate({
        "needs_drill_down": 0,
        "answer": "Grounded answer [Ref 1]",
        "requirements": [{"id": "R1", "entity": "A", "criterion": "first"}],
        "entity_criterion_matrix": [
            {"requirement_id": "R1", "coverage": "supported", "reference_ids": ["[Ref 1]"]},
        ],
        "extractive_evidence_ledger": [
            {"reference_id": "[Ref 1]", "requirement_ids": ["R1"]},
            {"reference_id": "[Ref 2]", "requirement_ids": []},
        ],
        "priority_gap": None,
    })

    result = AgenticV10PipelineService._validate_coverage_audit(
        audit=audit,
        valid_reference_ids={"[Ref 1]", "[Ref 2]"},
    )

    assert result["validated"] is True
    assert result["discarded_empty_ledger_reference_ids"] == ["[Ref 2]"]
    assert [entry.reference_id for entry in audit.extractive_evidence_ledger] == ["[Ref 1]"]


def test_v10_drilldown_synthesis_prompt_v2_requires_grounded_complete_answers() -> None:
    messages = AgenticV10PipelineService._drilldown_synthesis_messages(
        question="What does Table 3 report?",
        matrix=[
            EntityCriterionMatrixCell(
                requirement_id="R1",
                coverage="supported",
                reference_ids=["[Ref 1]"],
            )
        ],
        ledger=[
            ExtractiveEvidenceLedgerEntry(
                reference_id="[Ref 1]",
                requirement_ids=["R1"],
            )
        ],
        context_text="[Ref 1] Table 3 evidence",
    )

    system_prompt = messages[0]["content"]
    assert "matrix 與 ledger 只用於列出回答義務和來源索引" in system_prompt
    assert "依原問題中明確要求的順序逐項回答" in system_prompt
    assert "不得將僅適用於某範圍的公式外推到其他範圍" in system_prompt
    assert "較窄範圍的證據不得支持較廣範圍的結論" in system_prompt
    assert messages[1]["content"].count("[Ref 1]") >= 2


@pytest.mark.asyncio
async def test_v10_audit_answer_uses_native_schema_and_raw_b_context() -> None:
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
    audit_llm = MagicMock()
    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value={
        "parsed": CoverageAuditResponse.model_validate({
            "needs_drill_down": 0,
            "answer": "Grounded answer [Ref 1]",
            "requirements": [
                {"id": "R1", "entity": "Model", "criterion": "architecture"},
                {"id": "R2", "entity": "Model", "criterion": "benchmark"},
            ],
            "entity_criterion_matrix": [
                {"requirement_id": "R1", "coverage": "supported", "reference_ids": ["[Ref 1]"]},
                {"requirement_id": "R2", "coverage": "supported", "reference_ids": ["[Ref 4]"]},
            ],
            "extractive_evidence_ledger": [
                {"reference_id": "[Ref 1]", "requirement_ids": ["R1"]},
                {"reference_id": "[Ref 4]", "requirement_ids": ["R2"]},
            ],
            "priority_gap": None,
        }),
        "raw": MagicMock(
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        ),
    })
    audit_llm.with_structured_output.return_value = structured_llm
    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(side_effect=[MagicMock(documents=[first]), MagicMock(documents=[second])])),
        patch("data_base.agentic_v10.subquery_pipeline_service.load_user_vector_documents_async", new=AsyncMock(return_value=[first_previous, first, first_next, second_previous, second, second_next])),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm", return_value=audit_llm) as get_llm,
    ):
        result = await service.execute(question="Compare the models", user_id="user-1")
    trace = result.agent_trace or {}
    v10 = trace["agentic_v10"]
    assert trace["agentic_execution_version"] == "v10"
    assert len(v10["branches"]) == 2
    assert v10["branches"][0]["raw_candidates"][0]["content"] == "Architecture evidence"
    audit_prompt = v10["coverage_audit"]["prompt_messages"][1]["content"]
    assert "Architecture evidence" in audit_prompt
    assert "Architecture preceding context" in audit_prompt
    assert "Architecture following context" in audit_prompt
    assert "Benchmark evidence" in audit_prompt
    assert "Benchmark preceding context" in audit_prompt
    assert "Benchmark following context" in audit_prompt
    assert v10["context_pack"]["strategy"] == (
        "initial_top1_raw_same_document_neighbors_drilldown_top2_no_neighbors"
    )
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
    audit_llm.with_structured_output.assert_called_once_with(
        CoverageAuditResponse,
        method="json_schema",
        include_raw=True,
    )
    assert v10["coverage_audit"]["route"] == "audit_answer"
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
async def test_v10_uses_one_structured_audit_without_map_calls() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="query", focus="focus", target_entity="entity")
    ])
    evidence = Document(page_content="Raw evidence", metadata={"doc_id": "doc-1", "page_number": 1, "chunk_index_in_page": 0})
    reranker = MagicMock()
    reranker.rerank_with_scores.return_value = [(evidence, 0.9)]
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=reranker)
    audit_llm = MagicMock()
    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value={
        "parsed": CoverageAuditResponse.model_validate({
            "needs_drill_down": 0,
            "answer": "Answer [Ref 1]",
            "requirements": [{"id": "R1", "entity": "entity", "criterion": "focus"}],
            "entity_criterion_matrix": [{"requirement_id": "R1", "coverage": "supported", "reference_ids": ["[Ref 1]"]}],
            "extractive_evidence_ledger": [{"reference_id": "[Ref 1]", "requirement_ids": ["R1"]}],
            "priority_gap": None,
        }),
        "raw": MagicMock(usage_metadata={}),
    })
    audit_llm.with_structured_output.return_value = structured_llm

    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(return_value=MagicMock(documents=[evidence]))),
        patch("data_base.agentic_v10.subquery_pipeline_service.load_user_vector_documents_async", new=AsyncMock(return_value=[evidence])),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm", return_value=audit_llm) as get_llm,
    ):
        result = await service.execute(question="Question", user_id="user-1")

    assert "Raw evidence" in result.agent_trace["agentic_v10"]["context_pack"]["rendered_context"]
    assert get_llm.call_count == 1
    assert get_llm.call_args.kwargs == {"purpose": "synthesizer"}
    assert result.agent_trace["agentic_v10"]["drill_down"] is None


@pytest.mark.asyncio
async def test_v10_runs_one_drill_down_and_final_only_receives_ledger_sources() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="initial query", focus="initial", target_entity="Model")
    ])
    initial = Document(page_content="Initial supported evidence", metadata={"doc_id": "doc-1", "page_number": 1, "chunk_index_in_page": 1})
    initial_neighbor = Document(page_content="Unused initial neighbor", metadata={"doc_id": "doc-1", "page_number": 1, "chunk_index_in_page": 0})
    drill = Document(page_content="Missing table values", metadata={"doc_id": "doc-2", "page_number": 2, "chunk_index_in_page": 1})
    drill_second = Document(page_content="Missing table conditions", metadata={"doc_id": "doc-3", "page_number": 3, "chunk_index_in_page": 1})
    reranker = MagicMock()
    reranker.rerank_with_scores.side_effect = [
        [(initial, 0.9)],
        [(drill, 0.8), (drill_second, 0.7)],
    ]
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=reranker)
    audit_llm = MagicMock()
    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value={
        "parsed": CoverageAuditResponse.model_validate({
            "needs_drill_down": 1,
            "answer": None,
            "requirements": [{"id": "R1", "entity": "Model", "criterion": "Table 3 values"}],
            "entity_criterion_matrix": [{"requirement_id": "R1", "coverage": "partial", "reference_ids": ["[Ref 1]"]}],
            "extractive_evidence_ledger": [{"reference_id": "[Ref 1]", "requirement_ids": ["R1"]}],
            "priority_gap": {
                "requirement_id": "R1",
                "missing_information": "Table 3 values and conditions",
                "retrieval_query": "Model Table 3 values conditions",
            },
        }),
        "raw": MagicMock(usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}),
    })
    audit_llm.with_structured_output.return_value = structured_llm
    final_llm = MagicMock()
    final_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="Final grounded answer [Ref 1] [Ref 3]",
        usage_metadata={"input_tokens": 9, "output_tokens": 4, "total_tokens": 13},
    ))
    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(side_effect=[MagicMock(documents=[initial]), MagicMock(documents=[drill, drill_second])])),
        patch("data_base.agentic_v10.subquery_pipeline_service.load_user_vector_documents_async", new=AsyncMock(return_value=[initial_neighbor, initial])),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm", side_effect=[audit_llm, final_llm]),
    ):
        result = await service.execute(question="What are the Table 3 values?", user_id="user-1")

    v10 = result.agent_trace["agentic_v10"]
    assert v10["coverage_audit"]["route"] == "conditional_drill_down"
    assert v10["drill_down"]["query"] == "Model Table 3 values conditions"
    assert v10["drill_down"]["selection_strategy"] == "rerank_top2_no_neighbors"
    assert v10["drill_down"]["rerank_top_k"] == 2
    assert v10["drill_down"]["neighbor_expansion_enabled"] is False
    assert v10["drill_down"]["new_evidence_count"] == 2
    assert reranker.rerank_with_scores.call_count == 2
    assert [call.kwargs["top_k"] for call in reranker.rerank_with_scores.call_args_list] == [1, 2]
    final_prompt = v10["synthesis"]["prompt_messages"][1]["content"]
    assert "Initial supported evidence" in final_prompt
    assert "Unused initial neighbor" not in final_prompt
    assert "Missing table values" in final_prompt
    assert "Missing table conditions" in final_prompt
    assert "Drill table caption" not in final_prompt
    assert "Drill condition detail" not in final_prompt
    assert [
        item["rerank_rank"]
        for item in v10["drill_down"]["source_document_mapping"]
    ] == [1, 2]
    assert result.usage["total_tokens"] == 20
    assert result.agent_trace["response_status"] == "complete"


@pytest.mark.asyncio
async def test_v10_invalid_audit_reference_falls_back_to_raw_b_synthesis() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="query", focus="focus", target_entity="entity")
    ])
    evidence = Document(page_content="Raw evidence", metadata={"doc_id": "doc-1", "page_number": 1, "chunk_index_in_page": 0})
    reranker = MagicMock()
    reranker.rerank_with_scores.return_value = [(evidence, 0.9)]
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=reranker)
    audit_llm = MagicMock()
    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value={
        "parsed": CoverageAuditResponse.model_validate({
            "needs_drill_down": 0,
            "answer": "Bad [Ref 99]",
            "requirements": [{"id": "R1", "entity": "entity", "criterion": "focus"}],
            "entity_criterion_matrix": [{"requirement_id": "R1", "coverage": "supported", "reference_ids": ["[Ref 99]"]}],
            "extractive_evidence_ledger": [{"reference_id": "[Ref 99]", "requirement_ids": ["R1"]}],
            "priority_gap": None,
        }),
        "raw": MagicMock(usage_metadata={}),
    })
    audit_llm.with_structured_output.return_value = structured_llm
    fallback_llm = MagicMock()
    fallback_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Fallback answer", usage_metadata={}))
    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(return_value=MagicMock(documents=[evidence]))),
        patch("data_base.agentic_v10.subquery_pipeline_service.load_user_vector_documents_async", new=AsyncMock(return_value=[evidence])),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm", side_effect=[audit_llm, fallback_llm]),
    ):
        result = await service.execute(question="Question", user_id="user-1")

    assert result.answer == "Fallback answer"
    assert result.agent_trace["agentic_v10"]["coverage_audit"]["route"] == "audit_fallback_raw_synthesis"
    assert result.agent_trace["agentic_v10"]["coverage_audit"]["reference_validation"]["failure_reason"] == "invalid_matrix_reference"


@pytest.mark.asyncio
async def test_v10_drill_down_without_new_evidence_returns_qualified_partial() -> None:
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=[
        SubQueryItem(id="SQ1", query="query", focus="focus", target_entity="entity")
    ])
    evidence = Document(page_content="Existing partial evidence", metadata={"doc_id": "doc-1", "page_number": 1, "chunk_index_in_page": 0})
    reranker = MagicMock()
    reranker.rerank_with_scores.return_value = [(evidence, 0.9)]
    service = AgenticV10PipelineService(decomposer=decomposer, reranker=reranker)
    audit_llm = MagicMock()
    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value={
        "parsed": CoverageAuditResponse.model_validate({
            "needs_drill_down": 1,
            "answer": None,
            "requirements": [{"id": "R1", "entity": "entity", "criterion": "missing metric"}],
            "entity_criterion_matrix": [{"requirement_id": "R1", "coverage": "partial", "reference_ids": ["[Ref 1]"]}],
            "extractive_evidence_ledger": [{"reference_id": "[Ref 1]", "requirement_ids": ["R1"]}],
            "priority_gap": {"requirement_id": "R1", "missing_information": "metric", "retrieval_query": "entity metric"},
        }),
        "raw": MagicMock(usage_metadata={}),
    })
    audit_llm.with_structured_output.return_value = structured_llm
    final_llm = MagicMock()
    final_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Partial answer [Ref 1]", usage_metadata={}))
    with (
        patch("data_base.agentic_v10.subquery_pipeline_service.get_user_retriever_async", new=AsyncMock(return_value=MagicMock())),
        patch("data_base.agentic_v10.subquery_pipeline_service.retrieve_hybrid_documents", new=AsyncMock(side_effect=[MagicMock(documents=[evidence]), MagicMock(documents=[])])),
        patch("data_base.agentic_v10.subquery_pipeline_service.load_user_vector_documents_async", new=AsyncMock(return_value=[evidence])),
        patch("data_base.agentic_v10.subquery_pipeline_service.get_llm", side_effect=[audit_llm, final_llm]),
    ):
        result = await service.execute(question="Question", user_id="user-1")

    assert result.answer == "Partial answer [Ref 1]"
    assert result.agent_trace["response_status"] == "qualified_partial"
    assert result.agent_trace["agentic_v10"]["drill_down"]["new_evidence_count"] == 0


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
        execution_profile="agentic_eval_v10_top1_neighbors_conditional_drilldown",
        agentic_v10={
            "schema_version": "4",
            "response_status": "complete",
            "decomposition": {"sub_queries": [{"id": "SQ1"}]},
            "branches": [{"subquery_id": "SQ1"}],
            "deduplicated_evidence": [{"content": "evidence"}],
            "context_pack": {
                "strategy": "top1_raw_same_document_neighbors_conditional_drilldown",
                "neighbor_evidence_count": 2,
            },
            "coverage_audit": {
                "route": "conditional_drill_down",
                "unresolved_requirement_count": 1,
            },
            "drill_down": {"attempted": True, "new_evidence_count": 3},
            "synthesis": {"prompt_messages": [{"content": "full prompt"}]},
        },
    )

    summary = _project_agentic_v10(trace, ExportCampaignRequest())
    assert summary is not None
    assert summary["summary"]["subquery_count"] == 1
    assert summary["summary"]["context_strategy"] == "top1_raw_same_document_neighbors_conditional_drilldown"
    assert summary["summary"]["neighbor_evidence_count"] == 2
    assert summary["summary"]["coverage_audit_route"] == "conditional_drill_down"
    assert summary["summary"]["drill_down_count"] == 1
    assert summary["summary"]["drill_down_new_evidence_count"] == 3
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
    assert raw["raw_payload"]["context_pack"]["strategy"] == "top1_raw_same_document_neighbors_conditional_drilldown"
    assert json.loads(json.dumps(raw)) == raw
