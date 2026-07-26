"""Production-adapter coverage for the Agentic v9 campaign path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from evaluation.agentic_v9_campaign_runtime import (
    AgenticV9CampaignRuntime,
    _evidence_packets_for_results,
)
from evaluation.agentic_v9_admission import V9AdmissionContract
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    QueryContract,
    RagRetrievalResult,
    RequiredSlot,
    ResolvedSourceScope,
    RetrievalTask,
    SourceLocator,
    TaskRetrievalResult,
)
from data_base.agentic_v9.visual_evidence_extractor import (
    VisualEvidenceExtractionResult,
)
from data_base.agentic_v9.execution_policy import V9ExecutionPolicyRuntime


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


class _FailingProvider:
    def __init__(self) -> None:
        self.ainvoke = AsyncMock(side_effect=RuntimeError("provider unavailable"))


class _ContractProvider:
    def __init__(self) -> None:
        self.ainvoke = AsyncMock(
            return_value=SimpleNamespace(
                content=json.dumps(
                    {
                        "selected_route": "single_lookup",
                        "slots": [
                            {
                                "description": "Retrieve the requested source-bound fact.",
                                "source_name_hints": ["paper.pdf"],
                                "authorized_source_doc_ids": ["doc-1"],
                                "locator_hints": [],
                                "expected_answer_type": "text",
                                "depends_on_slot_ids": [],
                                "visual_policy": "never",
                            }
                        ],
                        "route_reason": "One ambiguous source-bound request.",
                        "confidence": 0.75,
                    }
                ),
                usage_metadata={"input_tokens": 8, "output_tokens": 4},
            )
        )


class _ReserveCutoffRuntime(V9ExecutionPolicyRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.reserve_checks = 0

    def has_final_reserve(self, deadline) -> bool:
        self.reserve_checks += 1
        return self.reserve_checks <= 2


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


async def _async_value(value):
    return value


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
async def test_runtime_plans_ambiguity_once_and_persists_exact_v2_contract() -> None:
    planning_provider = _ContractProvider()
    final_provider = _Provider()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="The requested source-bound fact is stated here.",
                    metadata={
                        "doc_id": "doc-1",
                        "page_number": 1,
                        "chunk_id": "chunk-1",
                    },
                )
            ]
        ),
        provider_factory=lambda purpose: (
            planning_provider
            if purpose == "atomic_contract_planning"
            else final_provider
        ),
        document_reference_resolver=lambda _user_id, _references: _async_value(
            {"paper.pdf": "doc-1"}
        ),
    )

    result = await runtime.execute(
        question="Please investigate this unclear request.",
        user_id="user-a",
        authorized_doc_ids=["paper.pdf"],
        setup_snapshot={**_setup(), "max_llm_calls": 5},
        trace_id="ambiguous-contract-once",
    )

    contract = result.agent_trace["agentic_v9"]["query_contract"]
    assert planning_provider.ainvoke.await_count == 1
    assert contract["contract_version"] == "2"
    assert contract["slot_plan_status"] == "complete"
    assert contract["slot_semantics"] == "heuristic_experimental"
    assert contract["atomic_completeness"] is None
    assert contract["atomic_completeness_reason"] == "atomic_slot_matching_experimental"
    assert contract["route_decision"]["decision_source"] == "llm_planner"
    assert contract["route_decision"]["planner_call_used"] is True
    assert contract["required_slots"][0]["authorized_source_doc_ids"] == ["doc-1"]
    assert result.agent_trace["response_status"] != "complete"


@pytest.mark.asyncio
async def test_degraded_runtime_slot_plan_cannot_return_complete() -> None:
    invalid_planning_provider = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=SimpleNamespace(
                content="not-json",
                usage_metadata={"input_tokens": 1, "output_tokens": 1},
            )
        )
    )
    final_provider = _Provider()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="A source-bound fact is present.",
                    metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
                )
            ]
        ),
        provider_factory=lambda purpose: (
            invalid_planning_provider
            if purpose == "atomic_contract_planning"
            else final_provider
        ),
        document_reference_resolver=lambda _user_id, _references: _async_value(
            {"paper.pdf": "doc-1"}
        ),
    )

    result = await runtime.execute(
        question="Please investigate this unclear request.",
        user_id="user-a",
        authorized_doc_ids=["paper.pdf"],
        setup_snapshot={**_setup(), "max_llm_calls": 5},
        trace_id="degraded-contract",
    )

    assert (
        result.agent_trace["agentic_v9"]["query_contract"]["slot_plan_status"]
        == "degraded"
    )
    assert result.agent_trace["response_status"] != "complete"


@pytest.mark.asyncio
async def test_v9_campaign_runtime_resolves_filename_scope_to_canonical_document_id() -> (
    None
):
    provider = _Provider()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="The source reports a score of 0.91.",
                metadata={"doc_id": "doc-1", "page_number": 2, "chunk_id": "chunk-1"},
            )
        ]
    )

    async def resolve_references(
        _user_id: str, references: list[str]
    ) -> dict[str, str]:
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

    assert result.agent_trace["agentic_v9"]["query_contract"]["resolved_source_scope"][
        "authorized_doc_ids"
    ] == ["doc-1"]
    assert result.agent_trace["agentic_v9"]["query_contract"]["resolved_source_scope"][
        "requested_doc_ids"
    ] == ["doc-1"]
    assert result.agent_trace["response_status"] == "qualified_partial"
    retrieve_documents.assert_awaited()


@pytest.mark.asyncio
async def test_v9_runtime_rejects_incompatible_setup_before_provider_or_retrieval() -> (
    None
):
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
    assert (
        result.agent_trace["agentic_v9"]["configuration_incompatible"]["stage"]
        == "pre_route"
    )
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
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
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
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
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
    assert (
        result.agent_trace["agentic_v9"]["slot_resolutions"][0]["status"] != "supported"
    )


@pytest.mark.asyncio
async def test_required_visual_evidence_is_recorded_before_complete_answer(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="Table 1 reports the result.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
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
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
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
async def test_missing_preferred_visual_evidence_does_not_block_text_completion(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="Table 1 reports the result.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        contract_version="2",
        route="exact_structured",
        intent="table value",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="table value",
                visual_policy="preferred",
            )
        ],
        visual_requested=True,
        visual_required=False,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
        slot_plan_status="complete",
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    visual_extractor = AsyncMock(return_value=VisualEvidenceExtractionResult())
    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract",
        admission,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        visual_extractor=visual_extractor,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is in the table?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="preferred-visual-trace",
    )

    visual_extractor.assert_awaited_once()
    assert result.agent_trace["response_status"] == "qualified_partial"
    assert result.agent_trace["agentic_v9"]["visual_execution"]["state"] == (
        "attempted_without_evidence"
    )
    assert result.agent_trace["agentic_v9"]["slot_resolutions"][0]["status"] == (
        "supported"
    )


@pytest.mark.asyncio
async def test_required_visual_failure_only_downgrades_required_policy_slots(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="The source contains both requested facts.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        contract_version="2",
        route="exact_structured",
        intent="two facts",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="required table fact",
                visual_policy="required",
            ),
            RequiredSlot(
                slot_id="S2",
                description="preferred table fact",
                visual_policy="preferred",
            ),
        ],
        visual_requested=True,
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
        slot_plan_status="complete",
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    visual_target_slot_ids: list[str] = []

    async def extract_visual(task, _documents, _question, _controller):
        visual_target_slot_ids.extend(task.target_slot_ids)
        return VisualEvidenceExtractionResult(
            packets=(
                EvidencePacket(
                    schema_version="1",
                    evidence_id="preferred-slot-visual",
                    task_id=task.task_id,
                    round_id=task.round_id,
                    query_id=task.query_id,
                    slot_ids=["S2"],
                    statement="The preferred slot has visual evidence.",
                    support_type="direct",
                    source=EvidenceSource(
                        doc_id="doc-1",
                        chunk_id="chunk-1",
                        asset_id="asset-preferred",
                    ),
                    scope=EvidenceScope(),
                    locator=SourceLocator(
                        pdf_page_index=1,
                        table_id="table-preferred",
                    ),
                    validation_status="deterministic_valid",
                ),
            )
        )

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract",
        admission,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        visual_extractor=extract_visual,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What are the two facts?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="mixed-visual-policy-trace",
    )

    resolutions = {
        row["slot_id"]: row["status"]
        for row in result.agent_trace["agentic_v9"]["slot_resolutions"]
    }
    visual = result.agent_trace["agentic_v9"]["visual_execution"]
    assert visual_target_slot_ids == ["S1", "S2"]
    assert visual["state"] == "executed"
    assert visual["supported_slot_ids"] == ["S2"]
    assert resolutions == {"S1": "explicitly_unavailable", "S2": "supported"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("later_asset_present", "expected_required_status"),
    [(True, "supported"), (False, "explicitly_unavailable")],
)
async def test_multidocument_visual_call_aggregates_sources_and_binds_packets_per_slot(
    monkeypatch,
    later_asset_present: bool,
    expected_required_status: str,
) -> None:
    provider = _Provider()
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-a", "doc-z"],
        resolved_doc_ids=["doc-a", "doc-z"],
        authorized_doc_ids=["doc-a", "doc-z"],
    )
    contract = QueryContract(
        contract_version="2",
        route="multi_document_exact",
        intent="two source-bound visual facts",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="preferred first-document table fact",
                authorized_source_doc_ids=["doc-a"],
                locator_hints=["Table A"],
                visual_policy="preferred",
            ),
            RequiredSlot(
                slot_id="S2",
                description="required later-document table fact",
                authorized_source_doc_ids=["doc-z"],
                locator_hints=["Table Z"],
                visual_policy="required",
            ),
        ],
        locator_hints=["table"],
        visual_requested=True,
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=2,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
        slot_plan_status="complete",
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    async def retrieve_documents(_user_id, _query, doc_ids):
        if len(doc_ids) != 1:
            return []
        doc_id = doc_ids[0]
        has_asset = doc_id == "doc-a" or later_asset_present
        metadata = {
            "doc_id": doc_id,
            "chunk_id": f"chunk-{doc_id}",
            "page_number": 1,
            "table_id": "Table A" if doc_id == "doc-a" else "Table Z",
        }
        if has_asset:
            metadata.update(
                {
                    "asset_id": f"asset-{doc_id}",
                    "page_image_base64": "aW1hZ2U=",
                    "page_width": 10,
                    "page_height": 10,
                }
            )
        return [
            Document(
                page_content=f"{doc_id} contains its requested fact.",
                metadata=metadata,
            )
        ]

    visual_calls: list[list[tuple[str, list[str]]]] = []

    async def extract_visual(task, documents, _question, _controller):
        asset_bindings = [
            (
                document.metadata["doc_id"],
                document.metadata["visual_slot_ids"],
            )
            for document in documents
            if document.metadata.get("page_image_base64")
        ]
        visual_calls.append(asset_bindings)
        asset_doc_ids = [doc_id for doc_id, _slot_ids in asset_bindings]

        def packet(
            *,
            evidence_id: str,
            doc_id: str,
            slot_id: str,
            table_id: str,
        ) -> EvidencePacket:
            return EvidencePacket(
                schema_version="1",
                evidence_id=evidence_id,
                task_id=task.task_id,
                round_id=task.round_id,
                query_id=task.query_id,
                slot_ids=[slot_id],
                statement=f"{doc_id} visual fact.",
                support_type="direct",
                source=EvidenceSource(
                    doc_id=doc_id,
                    chunk_id=f"chunk-{doc_id}",
                    asset_id=f"asset-{doc_id}",
                ),
                scope=EvidenceScope(),
                locator=SourceLocator(
                    pdf_page_index=1,
                    table_id=table_id,
                ),
                validation_status="deterministic_valid",
            )

        packets = [
            packet(
                evidence_id="wrong-source-visual",
                doc_id="doc-a",
                slot_id="S2",
                table_id="Table A",
            )
        ]
        if "doc-z" in asset_doc_ids:
            packets.append(
                packet(
                    evidence_id="later-source-visual",
                    doc_id="doc-z",
                    slot_id="S2",
                    table_id="Table Z",
                )
            )
        return VisualEvidenceExtractionResult(packets=tuple(packets))

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract",
        admission,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        visual_extractor=extract_visual,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What are the two table facts?",
        user_id="user-a",
        authorized_doc_ids=["doc-a", "doc-z"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id=f"multidoc-visual-{later_asset_present}",
    )

    assert visual_calls == [
        [
            ("doc-a", ["S1"]),
            *([("doc-z", ["S2"])] if later_asset_present else []),
        ]
    ]
    v9 = result.agent_trace["agentic_v9"]
    visual_evidence_ids = {
        packet["evidence_id"]
        for packet in v9["evidence_packets"]
        if packet["source"].get("asset_id")
    }
    assert "wrong-source-visual" not in visual_evidence_ids
    assert ("later-source-visual" in visual_evidence_ids) is later_asset_present
    resolutions = {row["slot_id"]: row["status"] for row in v9["slot_resolutions"]}
    assert resolutions["S1"] == "supported"
    assert resolutions["S2"] == expected_required_status


@pytest.mark.asyncio
async def test_text_evidence_outside_atomic_slot_authorized_ids_cannot_support_it(
    monkeypatch,
) -> None:
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-a", "doc-b"],
        resolved_doc_ids=["doc-a", "doc-b"],
        authorized_doc_ids=["doc-a", "doc-b"],
        source_name_to_doc_ids={"Alpha.pdf": ["doc-a"], "Beta.pdf": ["doc-b"]},
    )
    contract = QueryContract(
        contract_version="2",
        route="exact_structured",
        intent="one source-bound fact",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="Report the Alpha result.",
                source_name_hints=["Alpha.pdf"],
                authorized_source_doc_ids=["doc-a"],
                locator_hints=["Table 1"],
            )
        ],
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=2,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
        slot_plan_status="complete",
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    async def retrieve_documents(_user_id, _query, _doc_ids):
        return [
            Document(
                page_content="Beta contains a plausible but unauthorized result.",
                metadata={
                    "doc_id": "doc-b",
                    "chunk_id": "chunk-b",
                    "page_number": 1,
                },
            )
        ]

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract",
        admission,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: _Provider(),
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is the Alpha result?",
        user_id="user-a",
        authorized_doc_ids=["doc-a", "doc-b"],
        setup_snapshot=_setup(),
        trace_id="atomic-slot-source-filter",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert v9["evidence_packets"] == []
    assert v9["slot_resolutions"][0]["status"] == "not_found"
    assert result.agent_trace["response_status"] == "insufficient"


@pytest.mark.asyncio
async def test_source_name_only_slot_rejects_a_different_globally_authorized_doc(
    monkeypatch,
) -> None:
    scope = ResolvedSourceScope(
        authorized_doc_ids=["doc-a", "doc-b"],
        source_name_to_doc_ids={"Alpha.pdf": ["doc-a"], "Beta.pdf": ["doc-b"]},
    )
    contract = QueryContract(
        contract_version="2",
        route="exact_structured",
        intent="source-name-bound fact",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="Report Alpha's result.",
                source_name_hints=["Alpha.pdf"],
            )
        ],
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=2,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
        slot_plan_status="complete",
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract",
        admission,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="Beta has a plausible result.",
                    metadata={"doc_id": "doc-b", "chunk_id": "chunk-b"},
                )
            ]
        ),
        provider_factory=lambda _purpose: _Provider(),
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is Alpha's result?",
        user_id="user-a",
        authorized_doc_ids=["doc-a", "doc-b"],
        setup_snapshot=_setup(),
        trace_id="source-name-only-filter",
    )

    assert result.agent_trace["agentic_v9"]["evidence_packets"] == []


@pytest.mark.asyncio
async def test_same_document_chunk_with_wrong_locator_cannot_support_slot(
    monkeypatch,
) -> None:
    scope = ResolvedSourceScope(authorized_doc_ids=["doc-a"])
    contract = QueryContract(
        contract_version="2",
        route="exact_structured",
        intent="table-bound fact",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="Report the Table 3 result.",
                authorized_source_doc_ids=["doc-a"],
                locator_hints=["Table 3"],
            )
        ],
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=2,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
        slot_plan_status="complete",
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract",
        admission,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="A result from the wrong table.",
                    metadata={
                        "doc_id": "doc-a",
                        "chunk_id": "chunk-a",
                        "table_id": "Table 4",
                    },
                )
            ]
        ),
        provider_factory=lambda _purpose: _Provider(),
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is the Table 3 result?",
        user_id="user-a",
        authorized_doc_ids=["doc-a"],
        setup_snapshot=_setup(),
        trace_id="wrong-locator-filter",
    )

    assert result.agent_trace["agentic_v9"]["evidence_packets"] == []


def test_v1_locator_hint_accepts_ordinary_retrieved_chunk_without_metadata() -> None:
    scope = ResolvedSourceScope(authorized_doc_ids=["doc-a"])
    contract = QueryContract(
        contract_version="1",
        route="exact_structured",
        intent="table-bound fact",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="Report the Table 3 result.",
                authorized_source_doc_ids=["doc-a"],
                locator_hints=["Table 3"],
            )
        ],
        resolved_source_scope=scope,
    )
    task = RetrievalTask(
        task_id="Q:round-1:S1",
        round_id="round-1",
        query_id="Q",
        query="Table 3 result",
        target_slot_ids=["S1"],
        source_scope=scope,
        locator_hints=["Table 3"],
    )
    results = (
        TaskRetrievalResult(
            task_id=task.task_id,
            retrieval=RagRetrievalResult(
                retrieval_id="retrieval",
                chunks=[
                    {
                        "doc_id": "doc-a",
                        "chunk_id": "chunk-a",
                        "text": "An ordinary retrieved result.",
                    }
                ],
            ),
        ),
    )

    packets = _evidence_packets_for_results(
        results=results,
        contract=contract,
        trace_id="trace",
        tasks_by_id={task.task_id: task},
    )

    assert [packet.slot_ids for packet in packets] == [["S1"]]
    assert packets[0].source.doc_id == "doc-a"


def test_v2_locator_hint_rejects_ordinary_retrieved_chunk_without_metadata() -> None:
    scope = ResolvedSourceScope(authorized_doc_ids=["doc-a"])
    contract = QueryContract(
        contract_version="2",
        route="exact_structured",
        intent="table-bound fact",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="Report the Table 3 result.",
                authorized_source_doc_ids=["doc-a"],
                locator_hints=["Table 3"],
            )
        ],
        resolved_source_scope=scope,
    )
    task = RetrievalTask(
        task_id="Q:round-1:S1",
        round_id="round-1",
        query_id="Q",
        query="Table 3 result",
        target_slot_ids=["S1"],
        source_scope=scope,
        locator_hints=["Table 3"],
    )
    results = (
        TaskRetrievalResult(
            task_id=task.task_id,
            retrieval=RagRetrievalResult(
                retrieval_id="retrieval",
                chunks=[
                    {
                        "doc_id": "doc-a",
                        "chunk_id": "chunk-a",
                        "text": "An ordinary retrieved result.",
                    }
                ],
            ),
        ),
    )

    packets = _evidence_packets_for_results(
        results=results,
        contract=contract,
        trace_id="trace",
        tasks_by_id={task.task_id: task},
    )

    assert packets == []


def test_grouped_task_chunk_is_bound_only_to_its_matching_atomic_slot() -> None:
    scope = ResolvedSourceScope(authorized_doc_ids=["doc-a"])
    contract = QueryContract(
        contract_version="2",
        route="multi_document_exact",
        intent="two table facts",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="Table 3 fact.",
                authorized_source_doc_ids=["doc-a"],
                locator_hints=["Table 3"],
            ),
            RequiredSlot(
                slot_id="S2",
                description="Table 4 fact.",
                authorized_source_doc_ids=["doc-a"],
                locator_hints=["Table 4"],
            ),
        ],
        resolved_source_scope=scope,
    )
    task = RetrievalTask(
        task_id="Q:round-1:group",
        round_id="round-1",
        query_id="Q",
        query="table facts",
        target_slot_ids=["S1", "S2"],
        source_scope=scope,
        locator_hints=["Table 3", "Table 4"],
    )
    results = (
        TaskRetrievalResult(
            task_id=task.task_id,
            retrieval=RagRetrievalResult(
                retrieval_id="retrieval",
                chunks=[
                    {
                        "doc_id": "doc-a",
                        "chunk_id": "chunk-a",
                        "text": "Only Table 3 is present.",
                        "table_id": "Table 3",
                    }
                ],
            ),
        ),
    )

    packets = _evidence_packets_for_results(
        results=results,
        contract=contract,
        trace_id="trace",
        tasks_by_id={task.task_id: task},
    )

    assert [packet.slot_ids for packet in packets] == [["S1"]]
    assert packets[0].locator.table_id == "Table 3"


@pytest.mark.asyncio
async def test_q16_repair_trace_persists_constraints_evidence_and_stop_reason(
    monkeypatch,
) -> None:
    scope = ResolvedSourceScope(
        requested_doc_ids=["odes", "ukan"],
        requested_source_names=["ODES.pdf", "Implicit-U-KAN2.0.pdf"],
        resolved_doc_ids=["odes", "ukan"],
        authorized_doc_ids=["odes", "ukan"],
        source_name_to_doc_ids={
            "ODES.pdf": ["odes"],
            "Implicit-U-KAN2.0.pdf": ["ukan"],
        },
    )
    contract = QueryContract(
        contract_version="2",
        route="multi_document_exact",
        intent="retrieve Q16 atomic facts",
        required_slots=[
            RequiredSlot(
                slot_id="S3",
                description="Transcribe the ODES equation.",
                entity_ids=["ODES"],
                source_name_hints=["ODES.pdf"],
                authorized_source_doc_ids=["odes"],
                locator_hints=["Equation 2"],
                expected_answer_type="equation",
            ),
            RequiredSlot(
                slot_id="S5",
                description="Report the U-KAN metric.",
                entity_ids=["Implicit-U-KAN2.0"],
                source_name_hints=["Implicit-U-KAN2.0.pdf"],
                authorized_source_doc_ids=["ukan"],
                locator_hints=["Table 3"],
                expected_answer_type="number",
            ),
            RequiredSlot(
                slot_id="S7",
                description="State the theorem boundary.",
                entity_ids=["Implicit-U-KAN2.0"],
                source_name_hints=["Implicit-U-KAN2.0.pdf"],
                authorized_source_doc_ids=["ukan"],
                locator_hints=["Theorem 1"],
            ),
        ],
        max_retrieval_rounds=1,
        max_repair_rounds=1,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
        slot_plan_status="complete",
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    calls_by_locator: dict[str, int] = {}
    retrieval_calls: list[tuple[str, list[str]]] = []

    async def retrieve_documents(_user_id, query, doc_ids):
        retrieval_calls.append((query, list(doc_ids)))
        locator = next(
            value for value in ("Equation 2", "Table 3", "Theorem 1") if value in query
        )
        calls_by_locator[locator] = calls_by_locator.get(locator, 0) + 1
        if locator in {"Equation 2", "Theorem 1"} and calls_by_locator[locator] == 1:
            return []
        doc_id = doc_ids[0]
        locator_metadata = (
            {"formula_id": locator}
            if locator == "Equation 2"
            else {"table_id": locator}
            if locator == "Table 3"
            else {"section": locator}
        )
        return [
            Document(
                page_content=f"Located evidence for {locator}.",
                metadata={
                    "doc_id": doc_id,
                    "chunk_id": f"chunk-{locator}",
                    "page_number": 2,
                    **locator_metadata,
                },
            )
        ]

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract",
        admission,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: _Provider(),
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Gold answers are SECRET-ODES and SECRET-THEOREM.",
        user_id="user-a",
        authorized_doc_ids=["odes", "ukan"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="Q16-repair-trace",
    )

    repair = result.agent_trace["agentic_v9"]["repairs"][0]
    assert len(result.agent_trace["agentic_v9"]["repairs"]) == 1
    assert repair["repair_round_index"] == 1
    assert [task["target_slot_ids"] for task in repair["tasks"]] == [
        ["S3"],
        ["S7"],
    ]
    assert [task["source_scope"]["authorized_doc_ids"] for task in repair["tasks"]] == [
        ["odes"],
        ["ukan"],
    ]
    assert [task["locator_hints"] for task in repair["tasks"]] == [
        ["Equation 2"],
        ["Theorem 1"],
    ]
    assert repair["resulting_evidence_ids"]
    assert repair["stop_reason"] == "evidence_complete"
    repair_queries = [query for query, _doc_ids in retrieval_calls[3:]]
    assert all("SECRET-ODES" not in query for query in repair_queries)
    assert all("SECRET-THEOREM" not in query for query in repair_queries)


@pytest.mark.asyncio
async def test_runtime_persists_terminal_reserve_reason_after_repair(
    monkeypatch,
) -> None:
    scope = ResolvedSourceScope(authorized_doc_ids=["doc-a"])
    contract = QueryContract(
        contract_version="2",
        route="multi_document_exact",
        intent="missing atomic fact",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="Locate the missing fact.",
                authorized_source_doc_ids=["doc-a"],
            )
        ],
        max_retrieval_rounds=1,
        max_repair_rounds=2,
        max_llm_calls=2,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
        slot_plan_status="complete",
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract",
        admission,
    )
    policy_runtime = _ReserveCutoffRuntime()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[]),
        provider_factory=lambda _purpose: _Provider(),
        policy_runtime=policy_runtime,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is the missing fact?",
        user_id="user-a",
        authorized_doc_ids=["doc-a"],
        setup_snapshot=_setup(),
        trace_id="repair-reserve-terminal",
    )

    repairs = result.agent_trace["agentic_v9"]["repairs"]
    assert len(repairs) == 1
    assert repairs[0]["tasks"]
    assert repairs[0]["stop_reason"] == "final_budget_protected"


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


@pytest.mark.asyncio
async def test_budgeted_final_provider_failure_uses_deterministic_sections() -> None:
    provider = _FailingProvider()
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
        trace_id="failed-final-provider",
    )

    assert "Final generation was unavailable" not in result.answer
    assert "Supported conclusions" in result.answer
    assert "Unresolved/unverifiable requirements" in result.answer
    assert result.agent_trace["response_status"] == "qualified_partial"
    assert result.agent_trace["agentic_v9"]["metrics"]["final_generation_count"] == 0
