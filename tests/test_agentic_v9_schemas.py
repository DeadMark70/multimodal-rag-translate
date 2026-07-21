"""Contract tests for the evidence-first Agentic v9 schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    FinalClaim,
    GraphPolicy,
    QueryContract,
    RagRetrievalResult,
    ResolvedSourceScope,
    SlotResolution,
    SourceLocator,
    SufficiencyReport,
    TaskRetrievalResult,
    V9ExecutionResult,
    V9ExecutionRequest,
    default_graph_policy,
)
from evaluation.trace_schemas import AgentTraceDetail, summarize_agent_trace


def _evidence_packet(*, support_type: str = "direct") -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id="evidence-1",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["slot-1"],
        statement="The reported score is 0.9079.",
        support_type=support_type,
        source=EvidenceSource(
            doc_id="doc-1", chunk_id="chunk-1", document_name="paper.pdf"
        ),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(
            pdf_page_index=4, printed_page_label="5", table_id="table-2"
        ),
        raw_value=Decimal("0.9079"),
        normalized_value=Decimal("90.79"),
        unit="percent",
    )


def test_evidence_packet_requires_positive_provenance_and_disallows_missing_support() -> (
    None
):
    packet = _evidence_packet()

    assert packet.normalized_value == Decimal("90.79")
    assert packet.locator.pdf_page_index == 4
    assert packet.model_dump(mode="json")["raw_value"] == "0.9079"

    with pytest.raises(ValidationError):
        _evidence_packet(support_type="missing")

    with pytest.raises(ValidationError):
        EvidencePacket(
            schema_version="1",
            evidence_id="evidence-1",
            task_id="task-1",
            round_id="round-1",
            query_id="query-1",
            slot_ids=["slot-1"],
            statement="The reported score is 0.9079.",
            support_type="direct",
            scope=EvidenceScope(),
            locator=SourceLocator(pdf_page_index=4),
        )


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("single_lookup", "never"),
        ("bounded_compare", "never"),
        ("exact_structured", "locator_fallback"),
        ("multi_document_exact", "locator_fallback"),
        ("multi_hop", "locator_fallback"),
        ("graph_relational", "required_locator"),
    ],
)
def test_route_graph_policy_defaults_are_frozen(
    route: str, expected: GraphPolicy
) -> None:
    assert default_graph_policy(route) == expected


def test_slot_resolution_and_sufficiency_keep_absence_separate_from_evidence() -> None:
    report = SufficiencyReport(
        evidence_complete=False,
        answerable=True,
        response_status="qualified_partial",
        supported_slot_ids=["slot-1"],
        explicitly_unavailable_slot_ids=["slot-2"],
        not_found_slot_ids=["slot-3"],
        conflicted_slot_ids=["slot-4"],
        stop_reason="repair_budget_exhausted",
    )

    assert report.evidence_complete is False
    assert report.answerable is True
    assert (
        SlotResolution(slot_id="slot-2", status="explicitly_unavailable").status
        == "explicitly_unavailable"
    )

    with pytest.raises(ValidationError):
        SlotResolution(slot_id="slot-2", status="missing")


@pytest.mark.parametrize(
    ("status", "evidence_ids"),
    [
        ("supported", []),
        ("conflicted", []),
        ("conflicted", ["evidence-1"]),
        ("explicitly_unavailable", ["evidence-1"]),
        ("not_found", ["evidence-1"]),
    ],
)
def test_slot_resolution_rejects_incoherent_evidence_links(
    status: str, evidence_ids: list[str]
) -> None:
    with pytest.raises(ValidationError):
        SlotResolution(slot_id="slot-1", status=status, evidence_ids=evidence_ids)


def test_slot_resolution_accepts_only_positive_evidence_for_supported_or_conflicted() -> None:
    assert SlotResolution(
        slot_id="slot-1", status="supported", evidence_ids=["evidence-1"]
    ).evidence_ids == ["evidence-1"]
    assert SlotResolution(
        slot_id="slot-1",
        status="conflicted",
        evidence_ids=["evidence-1", "evidence-2"],
    ).status == "conflicted"


@pytest.mark.parametrize(
    "report",
    [
        {
            "evidence_complete": True,
            "answerable": True,
            "response_status": "qualified_partial",
            "explicitly_unavailable_slot_ids": ["slot-1"],
        },
        {
            "evidence_complete": True,
            "answerable": True,
            "response_status": "qualified_partial",
            "not_found_slot_ids": ["slot-1"],
        },
        {
            "evidence_complete": False,
            "answerable": True,
            "response_status": "complete",
        },
        {
            "evidence_complete": True,
            "answerable": False,
            "response_status": "complete",
        },
        {
            "evidence_complete": True,
            "answerable": True,
            "response_status": "complete",
            "conflicted_slot_ids": ["slot-1"],
        },
    ],
)
def test_sufficiency_report_rejects_internally_incoherent_completion(
    report: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        SufficiencyReport(**report)


def test_request_serializes_only_requested_sources_and_scope_resolver_owns_authorization() -> (
    None
):
    request = V9ExecutionRequest(
        question="What is the reported score?",
        requested_doc_ids=["doc-1"],
        requested_source_names=["paper.pdf"],
        history=[{"role": "user", "content": "Use the paper."}],
        setup_snapshot={"model_name": "gemini-test", "max_output_tokens": 512},
        trace_id="trace-1",
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=request.requested_doc_ids,
        requested_source_names=request.requested_source_names,
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
        rejected_source_names=[],
    )

    payload = request.model_dump(mode="json")
    assert payload["requested_doc_ids"] == ["doc-1"]
    assert "authorized_doc_ids" not in payload
    assert "user_id" not in payload
    assert scope.authorized_doc_ids == ["doc-1"]


@pytest.mark.parametrize(
    ("route", "expected_graph_policy"),
    [
        ("single_lookup", "never"),
        ("bounded_compare", "never"),
        ("exact_structured", "locator_fallback"),
        ("multi_document_exact", "locator_fallback"),
        ("multi_hop", "locator_fallback"),
        ("graph_relational", "required_locator"),
    ],
)
def test_query_contract_applies_the_model_default_graph_policy_for_each_route(
    route: str, expected_graph_policy: GraphPolicy
) -> None:
    contract = QueryContract(route=route, intent="test routing")

    assert contract.graph_policy == expected_graph_policy


@pytest.mark.parametrize("extra_field", ["user_id", "authorized_doc_ids"])
def test_request_rejects_adapter_injected_authorization_fields(extra_field: str) -> None:
    payload = {
        "question": "What is the reported score?",
        "trace_id": "trace-1",
        extra_field: "not-allowed",
    }

    with pytest.raises(ValidationError):
        V9ExecutionRequest(**payload)


def test_final_claim_rejects_evidence_only_scope_constraint_support_type() -> None:
    with pytest.raises(ValidationError):
        FinalClaim(
            claim_id="claim-1",
            statement="The frozen source scope cannot establish this claim.",
            support_type="scope_constraint",
        )


def test_trace_execution_version_is_backward_compatible_and_summary_preserves_it() -> (
    None
):
    detail = AgentTraceDetail(
        trace_id="trace-1",
        campaign_id="campaign-1",
        campaign_result_id="result-1",
        question_id="question-1",
        question="What is the reported score?",
        mode="agentic",
        run_number=1,
        trace_status="completed",
        created_at=datetime.now(timezone.utc),
    )

    assert detail.agentic_execution_version == "v8"
    assert summarize_agent_trace(detail).agentic_execution_version == "v8"

    serialized = detail.model_dump(mode="json")
    restored = AgentTraceDetail.model_validate(serialized)
    assert restored.agentic_execution_version == "v8"
    assert summarize_agent_trace(restored).agentic_execution_version == "v8"


def test_retrieval_and_execution_results_preserve_the_evidence_only_boundary() -> None:
    retrieval = RagRetrievalResult(
        retrieval_id="retrieval-1", chunks=[{"chunk_id": "chunk-1"}]
    )
    task_result = TaskRetrievalResult(task_id="task-1", retrieval=retrieval)
    execution = V9ExecutionResult(trace_id="trace-1", task_results=[task_result])

    assert execution.task_results[0].retrieval.chunks[0]["chunk_id"] == "chunk-1"
    assert "answer" not in task_result.model_dump()
