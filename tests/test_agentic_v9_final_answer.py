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
async def test_final_answer_uses_only_packed_evidence_and_renders_versioned_citations() -> (
    None
):
    invoker = _RecordingInvoker(
        SimpleNamespace(
            content={
                "answer": "The score is 0.91.",
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "statement": "The score is 0.91.",
                        "support_type": "direct",
                        "evidence_ids": ["E1"],
                    }
                ],
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
async def test_invalid_claim_is_qualified_without_a_second_final_generation() -> None:
    invoker = _RecordingInvoker(
        {
            "answer": "The score is 0.99.",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "statement": "The score is 0.99.",
                    "support_type": "direct",
                    "evidence_ids": ["E1"],
                }
            ],
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
    assert result.claims[0].support_type == "qualified"
    assert "does not match cited exact evidence" in result.claims[0].qualified_reason
    assert len(invoker.calls) == 1


@pytest.mark.asyncio
async def test_high_risk_prose_uses_one_batched_verifier_and_qualifies_rejected_claim() -> (
    None
):
    invoker = _RecordingInvoker(
        {
            "answer": "The approach is best.",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "statement": "The approach is best.",
                    "support_type": "comparative_inference",
                    "premise_evidence_ids": ["E1"],
                }
            ],
        },
        {
            "verdicts": [
                {"claim_id": "claim-1", "supported": False, "reason": "not established"}
            ]
        },
    )

    result = await generate_final_answer(
        question="Which approach is best?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.final_generation_count == 1
    assert result.claims[0].support_type == "qualified"
    assert result.claims[0].qualified_reason == "not established"
    assert [(call["phase"], call["purpose"]) for call in invoker.calls] == [
        ("final_answer", "final_answer"),
        ("claim_verifier", "claim_verifier"),
    ]


@pytest.mark.asyncio
async def test_unpacked_evidence_cannot_support_a_final_claim() -> None:
    invoker = _RecordingInvoker(
        {
            "answer": "The score is 0.91.",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "statement": "The score is 0.91.",
                    "support_type": "direct",
                    "evidence_ids": ["E2"],
                }
            ],
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
            "answer": "The score is 0.91.",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "statement": "The score is 0.91.",
                    "support_type": "direct",
                    "evidence_ids": ["E1"],
                }
            ],
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
async def test_direct_final_result_is_treated_as_an_untrusted_draft_and_qualified() -> (
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
    assert result.claims[0].support_type == "qualified"
    assert "does not match cited exact evidence" in result.claims[0].qualified_reason
    assert "[v1:paper.pdf p.5, Table 1; E1]" in result.answer


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
    assert result.answer == ""


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
    assert result.answer == fallback_answer
    assert result.claims == []
    assert result.used_evidence_ids == []
    assert result.final_generation_count == 0
