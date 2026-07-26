"""Verified final-answer contracts for the Agentic v9 evidence path."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from data_base.agentic_v9.final_answer import generate_final_answer
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    FinalAnswerResult,
    QueryContract,
    RequiredSlot,
    SlotResolution,
    SourceLocator,
    SufficiencyReport,
)


def _contract() -> QueryContract:
    return QueryContract(
        route="exact_structured",
        intent="Report the score.",
        required_slots=[RequiredSlot(slot_id="score", description="Reported score")],
    )


def _packet(evidence_id: str = "E1") -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["score"],
        statement="The reported score is 0.91.",
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", document_name="paper.pdf"),
        scope=EvidenceScope(metric="Dice"),
        locator=SourceLocator(pdf_page_index=4, table_id="Table 1"),
        raw_value=Decimal("0.91"),
        normalized_value=Decimal("0.91"),
    )


class _RecordingInvoker:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def invoke(
        self, *, phase: str, purpose: str, messages: list[dict[str, Any]]
    ) -> Any:
        self.calls.append({"phase": phase, "purpose": purpose, "messages": messages})
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_structured_supported_finding_is_persisted_as_slot_bound_claim() -> None:
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "The reported score is 0.91.",
                    "evidence_ids": ["E1"],
                }
            ],
            "unresolved_requirements": [],
        }
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.claims[0].slot_id == "score"
    assert result.claims[0].evidence_ids == ["E1"]


@pytest.mark.asyncio
async def test_final_answer_uses_only_packed_evidence_and_renders_versioned_citations() -> (
    None
):
    invoker = _RecordingInvoker(
        SimpleNamespace(
            content={
                "supported_findings": [
                    {
                        "slot_id": "score",
                        "statement": "The score is 0.91.",
                        "evidence_ids": ["E1"],
                    }
                ],
                "unresolved_requirements": [],
            }
        )
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.final_generation_count == 1
    assert result.used_evidence_ids == ["E1"]
    assert "[v1:paper.pdf p.5, Table 1; E1]" in result.answer
    assert [(call["phase"], call["purpose"]) for call in invoker.calls] == [
        ("final_answer", "final_answer")
    ]
    assert "E1" in str(invoker.calls[0]["messages"])


@pytest.mark.asyncio
async def test_false_finding_is_absent_from_claims_and_supported_conclusions() -> None:
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "The score is 0.99.",
                    "evidence_ids": ["E1"],
                }
            ],
            "unresolved_requirements": [],
        }
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.final_generation_count == 1
    assert result.claims == []
    assert "0.99" not in result.answer
    assert len(invoker.calls) == 1


@pytest.mark.asyncio
async def test_structured_findings_do_not_request_a_second_provider_generation() -> (
    None
):
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "The reported score is 0.91.",
                    "evidence_ids": ["E1"],
                }
            ],
            "unresolved_requirements": [],
        },
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.final_generation_count == 1
    assert result.claims[0].support_type == "direct"
    assert [(call["phase"], call["purpose"]) for call in invoker.calls] == [
        ("final_answer", "final_answer"),
    ]


@pytest.mark.asyncio
async def test_unpacked_evidence_cannot_support_a_final_claim() -> None:
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "The score is 0.91.",
                    "evidence_ids": ["E2"],
                }
            ],
            "unresolved_requirements": [],
        }
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.claims == []
    assert result.used_evidence_ids == []
    assert result.response_status == "insufficient"


@pytest.mark.asyncio
async def test_final_answer_accepts_the_typed_packer_packet_projection() -> None:
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "The score is 0.91.",
                    "evidence_ids": ["E1"],
                }
            ],
            "unresolved_requirements": [],
        }
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=SimpleNamespace(packets=(_packet(),)),
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.claims[0].qualified_reason is None
    assert result.response_status == "complete"


@pytest.mark.asyncio
async def test_false_legacy_final_claim_is_excluded_from_persistence_and_rendering() -> (
    None
):
    invoker = _RecordingInvoker(
        FinalAnswerResult(
            response_status="complete",
            answer="The score is 0.99.",
            claims=[
                {
                    "claim_id": "claim-1",
                    "statement": "The score is 0.99.",
                    "support_type": "direct",
                    "evidence_ids": ["E1"],
                }
            ],
            used_evidence_ids=["E1"],
            final_generation_count=1,
        )
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.final_generation_count == 1
    assert result.claims == []
    assert "0.99" not in result.answer


@pytest.mark.asyncio
async def test_only_the_fixed_no_claim_final_fallback_bypasses_draft_validation() -> (
    None
):
    invoker = _RecordingInvoker(
        FinalAnswerResult(
            response_status="qualified_partial",
            answer="Untrusted uncited partial.",
            final_generation_count=0,
        )
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.final_generation_count == 1
    assert result.response_status == "insufficient"
    assert "Supported conclusions" in result.answer
    assert "Unresolved/unverifiable requirements" in result.answer


@pytest.mark.asyncio
async def test_fixed_no_claim_final_fallback_remains_a_qualified_partial() -> None:
    fallback_answer = (
        "Final generation was unavailable; evidence is returned as a qualified partial."
    )
    invoker = _RecordingInvoker(
        FinalAnswerResult(
            response_status="qualified_partial",
            answer=fallback_answer,
            final_generation_count=0,
        )
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.response_status == "qualified_partial"
    assert fallback_answer not in result.answer
    assert "Supported conclusions" in result.answer
    assert "Unresolved/unverifiable requirements" in result.answer
    assert result.claims == []
    assert result.used_evidence_ids == []
    assert result.final_generation_count == 0


@pytest.mark.asyncio
async def test_missing_optional_slot_does_not_downgrade_complete_required_sufficiency() -> (
    None
):
    contract = QueryContract(
        route="exact_structured",
        intent="Report the required score.",
        required_slots=[
            RequiredSlot(slot_id="score", description="Reported score"),
            RequiredSlot(
                slot_id="optional-note",
                description="Optional context",
                required=False,
            ),
        ],
    )
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "The reported score is 0.91.",
                    "evidence_ids": ["E1"],
                }
            ],
            "unresolved_requirements": [],
        }
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=contract,
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"]),
            SlotResolution(slot_id="optional-note", status="not_found"),
        ],
        sufficiency_report=SufficiencyReport(
            evidence_complete=True,
            answerable=True,
            response_status="complete",
            supported_slot_ids=["score"],
        ),
        llm_invoker=invoker,
    )

    assert result.response_status == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,evidence_ids",
    [
        ("not_found", []),
        ("conflicted", ["E1", "E2"]),
        ("explicitly_unavailable", []),
    ],
)
async def test_findings_for_non_supported_slots_are_rejected(
    status: str, evidence_ids: list[str]
) -> None:
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "Maybe the score is 0.91.",
                    "evidence_ids": evidence_ids or ["E1"],
                }
            ],
            "unresolved_requirements": [],
        }
    )
    packets = [_packet(), _packet("E2")]

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=packets,
        slot_resolutions=[
            SlotResolution(slot_id="score", status=status, evidence_ids=evidence_ids)
        ],
        llm_invoker=invoker,
    )

    assert result.claims == []
    assert "Maybe the score" not in result.answer
    assert "score" in result.answer


@pytest.mark.asyncio
async def test_unknown_slot_and_cross_slot_evidence_are_rejected() -> None:
    other_slot_packet = _packet()
    other_slot_packet.slot_ids = ["other"]
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "unknown",
                    "statement": "Unknown finding.",
                    "evidence_ids": ["E1"],
                },
                {
                    "slot_id": "score",
                    "statement": "Cross-slot finding.",
                    "evidence_ids": ["E1"],
                },
            ],
            "unresolved_requirements": [],
        }
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[other_slot_packet],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.claims == []
    assert "Unknown finding" not in result.answer
    assert "Cross-slot finding" not in result.answer


@pytest.mark.asyncio
async def test_missing_required_unresolved_row_is_built_deterministically() -> None:
    invoker = _RecordingInvoker(
        {"supported_findings": [], "unresolved_requirements": []}
    )

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[],
        slot_resolutions=[
            SlotResolution(
                slot_id="score",
                status="not_found",
                reason="No source-bound evidence was found.",
            )
        ],
        llm_invoker=invoker,
    )

    assert "Supported conclusions" in result.answer
    assert "Unresolved/unverifiable requirements" in result.answer
    assert "score: No source-bound evidence was found." in result.answer


@pytest.mark.asyncio
async def test_q14_unsupported_segmentanybone_finding_is_never_rendered() -> None:
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "source",
                    "statement": "SegmentAnyBone嚗皜穿?",
                    "evidence_ids": ["E1"],
                }
            ],
            "unresolved_requirements": [],
        }
    )
    contract = QueryContract(
        route="exact_structured",
        intent="Identify the source requirement.",
        required_slots=[
            RequiredSlot(slot_id="source", description="Source requirement")
        ],
    )

    result = await generate_final_answer(
        question="Which source establishes the requirement?",
        contract=contract,
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(
                slot_id="source",
                status="not_found",
                reason="Required source evidence was not found.",
            )
        ],
        llm_invoker=invoker,
    )

    assert "SegmentAnyBone" not in result.answer
    assert "source: Required source evidence was not found." in result.answer
