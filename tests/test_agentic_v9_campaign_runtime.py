"""Production-adapter coverage for the Agentic v9 campaign path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from evaluation.agentic_v9_campaign_runtime import AgenticV9CampaignRuntime
from evaluation.agentic_v9_admission import V9AdmissionContract
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
    SourceLocator,
)
from data_base.agentic_v9.visual_evidence_extractor import VisualEvidenceExtractionResult


class _Provider:
    def __init__(self) -> None:
        self.ainvoke = AsyncMock(side_effect=self._respond)

    async def _respond(self, messages):
        payload = json.loads(messages[-1]["content"])
        packet = payload["packed_evidence_packets"][0]
        return SimpleNamespace(
                content={
                    "supported_findings": [
                        {
                            "slot_id": packet["slot_ids"][0],
                            "statement": packet["statement"],
                            "evidence_ids": [packet["evidence_id"]],
                        }
                    ],
                    "unresolved_requirements": [],
                },
                usage_metadata={"input_tokens": 12, "output_tokens": 7},
            )


class _InvalidProvider:
    def __init__(self) -> None:
        self.ainvoke = AsyncMock(
            return_value=SimpleNamespace(
                content={"answer": "Unsupported provider prose."},
                usage_metadata={"input_tokens": 12, "output_tokens": 7},
            )
        )


def _setup() -> dict[str, object]:
    return {
        "max_input_tokens": 4096,
        "max_output_tokens": 256,
        "thinking_mode": False,
    }


async def _identity_reference_resolver(
    _user_id: str, references: list[str]
) -> dict[str, str]:
    """Keep unit tests independent of the production document repository."""
    return {reference: reference for reference in references}


@pytest.mark.asyncio
async def test_v9_campaign_runtime_runs_core_and_emits_real_evidence_trace() -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="The source reports a score of 0.91.",
                metadata={"doc_id": "doc-1", "page_number": 2, "chunk_id": "chunk-1"},
            )
        ]
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="attempt-trace-1",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert v9["query_contract"]["resolved_source_scope"]["authorized_doc_ids"] == [
        "doc-1"
    ]
    assert v9["evidence_packets"]
    assert v9["slot_resolutions"]
    assert v9["sufficiency"]["response_status"] == "complete"
    assert result.documents
    retrieve_documents.assert_awaited()
    provider.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_v9_campaign_runtime_resolves_filename_scope_to_canonical_document_id() -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="The source reports a score of 0.91.",
                metadata={"doc_id": "doc-1", "page_number": 2, "chunk_id": "chunk-1"},
            )
        ]
    )

    async def resolve_references(_user_id: str, references: list[str]) -> dict[str, str]:
        assert references == ["paper.pdf"]
        return {"paper.pdf": "doc-1"}

    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=resolve_references,
    )

    result = await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=["paper.pdf"],
        setup_snapshot=_setup(),
        trace_id="attempt-trace-filename-scope",
    )

    assert result.agent_trace["agentic_v9"]["query_contract"]["resolved_source_scope"]["authorized_doc_ids"] == ["doc-1"]
    assert result.agent_trace["agentic_v9"]["query_contract"]["resolved_source_scope"]["requested_doc_ids"] == ["doc-1"]
    assert result.agent_trace["response_status"] == "complete"
    retrieve_documents.assert_awaited()


@pytest.mark.asyncio
async def test_v9_runtime_rejects_incompatible_setup_before_provider_or_retrieval() -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={"thinking_mode": False},
        trace_id="attempt-trace-incompatible",
    )

    assert result.agent_trace["response_status"] == "configuration_incompatible"
    assert result.agent_trace["agentic_v9"]["configuration_incompatible"]["stage"] == "pre_route"
    assert result.documents == []
    retrieve_documents.assert_not_awaited()
    provider.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_v9_runtime_repeats_feasibility_after_contract_before_retrieval(
    monkeypatch,
) -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="exact_structured",
        intent="extract a table value",
        required_slots=[RequiredSlot(slot_id="S1", description="table value")],
        visual_required=True,
        evidence_extraction_required=True,
        max_llm_calls=2,
        runtime_token_budget=40_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )

    # The contract requires visual + evidence + final provider work but permits
    # only two calls.  It must be rejected before retrieval starts.
    result = await runtime.execute(
        question="What is the table score?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="attempt-trace-post-contract",
    )

    incompatible = result.agent_trace["agentic_v9"]["configuration_incompatible"]
    assert incompatible["stage"] == "post_contract"
    assert result.agent_trace["response_status"] == "configuration_incompatible"
    retrieve_documents.assert_not_awaited()
    provider.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_required_graph_locator_is_executed_and_recorded_before_complete_answer(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="The relation is source-bound.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    graph_locator = AsyncMock(
        return_value=SimpleNamespace(
            documents=[document],
            resolved_source_documents=[document],
            resolved_source_doc_ids=["doc-1"],
            resolved_source_chunk_ids=["chunk-1"],
            candidate_item_ids=["graph-item-1"],
            resolved_item_ids=["graph-item-1"],
            scope_approved_item_ids=["graph-item-1"],
            scored_item_ids=["graph-item-1"],
            packed_item_ids=["graph-item-1"],
            route="local-first",
            path="source_expand",
            fallback=None,
            graph_latency_ms=7,
        )
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"], resolved_doc_ids=["doc-1"], authorized_doc_ids=["doc-1"]
    )
    contract = QueryContract(
        route="graph_relational",
        intent="relation",
        required_slots=[RequiredSlot(slot_id="S1", description="relation")],
        graph_policy="required_locator",
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=4,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        graph_locator=graph_locator,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What relation is recorded?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="required-graph-trace",
    )

    assert result.agent_trace["response_status"] == "complete"
    assert result.agent_trace["agentic_v9"]["graph_execution"]["state"] == "executed"
    graph_locator.assert_awaited_once()


@pytest.mark.asyncio
async def test_required_graph_locator_without_source_evidence_is_insufficient(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="The vector result is not graph evidence.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    graph_locator = AsyncMock(
        return_value=SimpleNamespace(
            documents=[document],
            resolved_source_documents=[],
            resolved_source_doc_ids=[],
            resolved_source_chunk_ids=[],
            candidate_item_ids=["graph-item-1"],
            resolved_item_ids=[],
            scope_approved_item_ids=[],
            scored_item_ids=[],
            packed_item_ids=[],
            route="local-first",
            path="source_expand",
            fallback="no_source_bound_graph_evidence",
            graph_latency_ms=7,
        )
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"], resolved_doc_ids=["doc-1"], authorized_doc_ids=["doc-1"]
    )
    contract = QueryContract(
        route="graph_relational",
        intent="relation",
        required_slots=[RequiredSlot(slot_id="S1", description="relation")],
        graph_policy="required_locator",
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=4,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        graph_locator=graph_locator,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What relation is recorded?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="missing-required-graph-trace",
    )

    graph_execution = result.agent_trace["agentic_v9"]["graph_execution"]
    assert result.agent_trace["response_status"] == "insufficient"
    assert graph_execution["state"] == "required_but_not_satisfied"
    assert graph_execution["failure_reason"] == "no_source_bound_graph_evidence"
    assert result.agent_trace["agentic_v9"]["slot_resolutions"][0]["status"] != "supported"


@pytest.mark.asyncio
async def test_required_visual_evidence_is_recorded_before_complete_answer(monkeypatch) -> None:
    provider = _Provider()
    document = Document(
        page_content="Table 1 reports the result.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"], resolved_doc_ids=["doc-1"], authorized_doc_ids=["doc-1"]
    )
    contract = QueryContract(
        route="exact_structured",
        intent="table value",
        required_slots=[RequiredSlot(slot_id="S1", description="table value")],
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    async def extract_visual(task, _documents, _question, _controller):
        return VisualEvidenceExtractionResult(
            packets=(
                EvidencePacket(
                    schema_version="1",
                    evidence_id="visual-evidence-1",
                    task_id=task.task_id,
                    round_id=task.round_id,
                    query_id=task.query_id,
                    slot_ids=list(task.target_slot_ids),
                    statement="The table reports 0.91.",
                    support_type="direct",
                    source=EvidenceSource(
                        doc_id="doc-1", chunk_id="chunk-1", asset_id="asset-1"
                    ),
                    scope=EvidenceScope(),
                    locator=SourceLocator(pdf_page_index=1, table_id="table-1"),
                    validation_status="deterministic_valid",
                ),
            )
        )

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        visual_extractor=extract_visual,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is in the table?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="required-visual-trace",
    )

    assert result.agent_trace["response_status"] == "complete"
    assert result.agent_trace["agentic_v9"]["visual_execution"]["state"] == "executed"


@pytest.mark.asyncio
async def test_missing_required_visual_evidence_is_insufficient(monkeypatch) -> None:
    provider = _Provider()
    document = Document(
        page_content="Table 1 reports the result.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"], resolved_doc_ids=["doc-1"], authorized_doc_ids=["doc-1"]
    )
    contract = QueryContract(
        route="exact_structured",
        intent="table value",
        required_slots=[RequiredSlot(slot_id="S1", description="table value")],
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        visual_extractor=AsyncMock(return_value=VisualEvidenceExtractionResult()),
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is in the table?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="missing-required-visual-trace",
    )

    visual = result.agent_trace["agentic_v9"]["visual_execution"]
    assert result.agent_trace["response_status"] == "insufficient"
    assert visual["state"] == "required_but_not_satisfied"
    assert visual["failure_reason"] == "no_eligible_visual_evidence"
    assert result.agent_trace["agentic_v9"]["slot_resolutions"][0]["status"] == (
        "explicitly_unavailable"
    )


@pytest.mark.asyncio
async def test_invalid_final_provider_output_uses_deterministic_sections() -> None:
    provider = _InvalidProvider()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="The source reports a score of 0.91.",
                    metadata={
                        "doc_id": "doc-1",
                        "page_number": 2,
                        "chunk_id": "chunk-1",
                    },
                )
            ]
        ),
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="invalid-final-provider",
    )

    assert "Unsupported provider prose" not in result.answer
    assert "Supported conclusions" in result.answer
    assert "Unresolved/unverifiable requirements" in result.answer
    assert result.agent_trace["response_status"] == "insufficient"
