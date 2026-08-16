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
    FinalClaim,
    QueryContract,
    RequiredSlot,
    SlotResolution,
    SourceLocator,
    SynthesisObligation,
)


def _contract() -> QueryContract:
    return QueryContract(
        route="exact_structured",
        intent="Report the score.",
        required_slots=[RequiredSlot(slot_id="score", description="Reported score")],
    )


def _two_slot_contract() -> QueryContract:
    return QueryContract(
        route="bounded_compare",
        intent="Compare two reported findings.",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Finding for A"),
            RequiredSlot(slot_id="S2", description="Finding for B"),
        ],
    )


def _packet(
    evidence_id: str = "E1",
    slot_id: str = "score",
    statement: str = "The reported score is 0.91.",
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=[slot_id],
        statement=statement,
        support_type="direct",
        source=EvidenceSource(
            doc_id="doc-1",
            document_name="paper.pdf",
            source_span_hash=f"hash-{evidence_id}",
        ),
        scope=EvidenceScope(metric="Dice"),
        locator=SourceLocator(pdf_page_index=4, table_id="Table 1"),
        raw_value=Decimal("0.91"),
        normalized_value=Decimal("0.91"),
        extractor_version="v9-deterministic-1",
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
                "supported_findings": [
                    {
                        "slot_id": "score",
                        "statement": "reported score is 0.91.",
                        "support_type": "direct",
                        "evidence_ids": ["E1"],
                        "premise_evidence_ids": [],
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
async def test_final_answer_sends_registered_v2_synthesis_prompt_to_provider() -> None:
    invoker = _RecordingInvoker(
        SimpleNamespace(
            content={
                "supported_findings": [
                    {
                        "slot_id": "score",
                        "statement": "reported score is 0.91.",
                        "support_type": "direct",
                        "evidence_ids": ["E1"],
                        "premise_evidence_ids": [],
                    }
                ],
                "synthesized_findings": [],
                "unresolved_requirements": [],
                "unresolved_obligations": [],
            }
        )
    )

    await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    system_message = next(
        message["content"]
        for message in invoker.calls[0]["messages"]
        if message["role"] == "system"
    )
    assert "Do not infer a rounding method or precision" in system_message
    assert "Distinguish source-stated facts from derived conclusions" in system_message
    assert (
        "Evidence insufficiency belongs in unresolved_requirements or "
        "unresolved_obligations" in system_message
    )
    assert "all direct premise evidence IDs required by its dependencies" in system_message


@pytest.mark.asyncio
async def test_invalid_claim_is_qualified_without_a_second_final_generation() -> None:
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "The score is 0.99.",
                    "support_type": "direct",
                    "evidence_ids": ["E1"],
                    "premise_evidence_ids": [],
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
    assert result.claims[0].support_type == "qualified"
    assert "does not match cited exact evidence" in result.claims[0].qualified_reason
    assert len(invoker.calls) == 1


@pytest.mark.asyncio
async def test_high_risk_prose_uses_one_batched_verifier_and_qualifies_rejected_claim() -> (
    None
):
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "The approach is best.",
                    "support_type": "comparative_inference",
                    "evidence_ids": [],
                    "premise_evidence_ids": ["E1"],
                }
            ],
            "unresolved_requirements": [],
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
    assert result.response_status == "insufficient"
    assert result.used_evidence_ids == []
    assert [(call["phase"], call["purpose"]) for call in invoker.calls] == [
        ("final_answer", "final_answer"),
        ("claim_verifier", "claim_verifier"),
    ]


@pytest.mark.asyncio
async def test_unpacked_evidence_cannot_support_a_final_claim() -> None:
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "The score is 0.91.",
                    "support_type": "direct",
                    "evidence_ids": ["E2"],
                    "premise_evidence_ids": [],
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

    assert result.claims[0].qualified_reason == (
        "claim_references_unpacked_or_unknown_evidence"
    )
    assert result.used_evidence_ids == []
    assert result.response_status == "insufficient"


@pytest.mark.asyncio
async def test_final_answer_accepts_the_typed_packer_packet_projection() -> None:
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "reported score is 0.91.",
                    "support_type": "direct",
                    "evidence_ids": ["E1"],
                    "premise_evidence_ids": [],
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
async def test_direct_final_result_is_rejected_as_an_untrusted_legacy_envelope() -> (
    None
):
    invoker = _RecordingInvoker(
        FinalAnswerResult(
            response_status="complete",
            answer="The score is 0.99.",
            claims=[
                FinalClaim(
                    claim_id="claim-1",
                    slot_id="score",
                    statement="The score is 0.99.",
                    support_type="direct",
                    evidence_ids=["E1"],
                )
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

    assert result.final_generation_count == 0
    assert result.claims == []
    assert result.response_status == "insufficient"
    assert result.used_evidence_ids == []


@pytest.mark.asyncio
async def test_untrusted_no_claim_final_result_does_not_bypass_draft_validation() -> None:
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

    assert result.final_generation_count == 0
    assert result.response_status == "insufficient"
    assert result.answer == "Final generation was unavailable; no verified answer was produced."

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


@pytest.mark.asyncio
async def test_multi_slot_complete_requires_an_accepted_finding_for_every_slot() -> None:
    result = await generate_final_answer(
        question="Compare A and B.",
        contract=_two_slot_contract(),
        packed_packets=[
            _packet("E1", "S1", "Finding A."),
            _packet("E2", "S2", "Finding B."),
        ],
        slot_resolutions=[
            SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
            SlotResolution(slot_id="S2", status="supported", evidence_ids=["E2"]),
        ],
        llm_invoker=_RecordingInvoker(
            {
                "supported_findings": [
                    {
                        "slot_id": "S1",
                        "statement": "Finding A.",
                        "support_type": "direct",
                        "evidence_ids": ["E1"],
                        "premise_evidence_ids": [],
                    }
                ],
                "unresolved_requirements": [],
            }
        ),
    )

    assert result.response_status == "qualified_partial"
    assert {claim.slot_id for claim in result.claims if not claim.qualified_reason} == {
        "S1"
    }
    assert result.used_evidence_ids == ["E1"]
    assert "S2" in result.answer


@pytest.mark.asyncio
@pytest.mark.parametrize("claim_slot", [None, "unknown", "S2"])
async def test_final_finding_cannot_escape_its_required_slot_authorization(
    claim_slot: str | None,
) -> None:
    finding = {
        "statement": "Finding A.",
        "support_type": "direct",
        "evidence_ids": ["E1"],
        "premise_evidence_ids": [],
    }
    if claim_slot is not None:
        finding["slot_id"] = claim_slot
    result = await generate_final_answer(
        question="Compare A and B.",
        contract=_two_slot_contract(),
        packed_packets=[_packet("E1", "S1", "Finding A.")],
        slot_resolutions=[
            SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
            SlotResolution(slot_id="S2", status="not_found"),
        ],
        llm_invoker=_RecordingInvoker(
            {
                "supported_findings": [finding],
                "unresolved_requirements": [],
            }
        ),
    )

    assert result.response_status == "insufficient"
    assert result.claims == []
    assert result.used_evidence_ids == []


@pytest.mark.asyncio
async def test_complete_answer_omits_an_empty_unresolved_section() -> None:
    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=_RecordingInvoker(
            {
                "supported_findings": [
                    {
                        "slot_id": "score",
                        "statement": "The reported score is 0.91.",
                        "support_type": "direct",
                        "evidence_ids": ["E1"],
                        "premise_evidence_ids": [],
                    }
                ],
                "unresolved_requirements": [],
            }
        ),
    )

    assert result.response_status == "complete"
    assert "Unresolved" not in result.answer
    assert "Unable to confirm" not in result.answer


@pytest.mark.asyncio
async def test_invalid_final_json_fails_closed_without_retrying_generation() -> None:
    invoker = _RecordingInvoker("not-json")

    result = await generate_final_answer(
        question="What is the reported score?",
        contract=_contract(),
        packed_packets=[_packet()],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.response_status == "insufficient"
    assert result.final_generation_count == 0
    assert len(invoker.calls) == 1


@pytest.mark.asyncio
async def test_synthesized_finding_creates_obligation_bound_claim_with_derived_support_type() -> None:
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare Method A and B DSC",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Method A DSC"),
            RequiredSlot(slot_id="S2", description="Method B DSC"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare Method A vs B DSC",
                depends_on_slot_ids=["S1", "S2"],
            )
        ],
    )
    p1 = _packet("E1", "S1", "Method A achieved 85.5% DSC.")
    p2 = _packet("E2", "S2", "Method B achieved 82.0% DSC.")
    resolutions = [
        SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
        SlotResolution(slot_id="S2", status="supported", evidence_ids=["E2"]),
    ]

    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "S1",
                    "statement": "Method A achieved 85.5% DSC.",
                    "evidence_ids": ["E1"],
                    "premise_evidence_ids": [],
                },
                {
                    "slot_id": "S2",
                    "statement": "Method B achieved 82.0% DSC.",
                    "evidence_ids": ["E2"],
                    "premise_evidence_ids": [],
                },
            ],
            "synthesized_findings": [
                {
                    "obligation_id": "O1",
                    "statement": "Method A achieved higher DSC than Method B (85.5% vs 82.0%).",
                    "premise_evidence_ids": ["E1", "E2"],
                }
            ],
            "unresolved_requirements": [],
            "unresolved_obligations": [],
        },
        # verifier response for O1 comparative_inference claim
        {
            "verdicts": [
                {
                    "claim_id": "claim-3",
                    "supported": True,
                    "reason": None,
                }
            ]
        },
    )

    result = await generate_final_answer(
        question="Which method has higher DSC?",
        contract=contract,
        packed_packets=[p1, p2],
        slot_resolutions=resolutions,
        llm_invoker=invoker,
    )

    assert result.final_generation_count == 1
    assert result.response_status == "complete"
    assert len(result.claims) == 3
    # Check direct claims
    direct_claims = [c for c in result.claims if c.slot_id is not None]
    assert len(direct_claims) == 2
    assert {c.slot_id for c in direct_claims} == {"S1", "S2"}
    assert all(c.obligation_id is None for c in direct_claims)
    assert all(c.support_type == "direct" for c in direct_claims)

    # Check synthesized claim
    syn_claims = [c for c in result.claims if c.obligation_id is not None]
    assert len(syn_claims) == 1
    assert syn_claims[0].obligation_id == "O1"
    assert syn_claims[0].slot_id is None
    assert syn_claims[0].support_type == "comparative_inference"
    assert set(syn_claims[0].premise_evidence_ids) == {"E1", "E2"}

    assert set(result.used_evidence_ids) == {"E1", "E2"}


@pytest.mark.asyncio
async def test_synthesized_finding_with_missing_premise_closure_is_rejected() -> None:
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare Method A and B DSC",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Method A DSC"),
            RequiredSlot(slot_id="S2", description="Method B DSC"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare Method A vs B DSC",
                depends_on_slot_ids=["S1", "S2"],
            )
        ],
    )
    p1 = _packet("E1", "S1", "Method A achieved 85.5% DSC.")
    resolutions = [
        SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
        SlotResolution(slot_id="S2", status="not_found"),
    ]

    # Model tries to synthesize O1 using only E1 (missing S2 dependency premise)
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "S1",
                    "statement": "Method A achieved 85.5% DSC.",
                    "evidence_ids": ["E1"],
                    "premise_evidence_ids": [],
                },
            ],
            "synthesized_findings": [
                {
                    "obligation_id": "O1",
                    "statement": "Method A achieved higher DSC.",
                    "premise_evidence_ids": ["E1"],
                }
            ],
            "unresolved_requirements": [
                {"slot_id": "S2", "reason": "No evidence found for S2"}
            ],
            "unresolved_obligations": [
                {"obligation_id": "O1", "reason": "Cannot compare without Method B DSC"}
            ],
        }
    )

    result = await generate_final_answer(
        question="Which method has higher DSC?",
        contract=contract,
        packed_packets=[p1],
        slot_resolutions=resolutions,
        llm_invoker=invoker,
    )

    # Keep the candidate and qualify it so the missing dependency is visible.
    obligation_claim = next(
        c for c in result.claims if c.obligation_id == "O1"
    )
    assert obligation_claim.qualified_reason == (
        "missing_obligation_dependency_closure"
    )
    assert result.unresolved_obligations[0].obligation_id == "O1"
    assert result.response_status == "qualified_partial"


@pytest.mark.asyncio
async def test_final_answer_batches_paraphrase_and_obligation_verification_once() -> None:
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare scores and decoder structure.",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Method A score"),
            RequiredSlot(slot_id="S2", description="Method B score"),
            RequiredSlot(slot_id="S3", description="Decoder structure"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="aggregation",
                description="Arithmetic score difference",
                depends_on_slot_ids=["S1", "S2"],
            ),
            SynthesisObligation(
                obligation_id="O2",
                kind="qualification",
                description="Rounded decoder metric",
                depends_on_slot_ids=["S3"],
            ),
        ],
    )
    packets = [
        _packet("E1", "S1", "Method A score is 85."),
        _packet("E2", "S2", "Method B score is 80."),
        _packet("E3", "S3", "The decoder consists of two stages."),
    ]
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "S1",
                    "statement": "Method A score is 85.",
                    "evidence_ids": ["E1"],
                    "premise_evidence_ids": [],
                },
                {
                    "slot_id": "S3",
                    "statement": "The decoder has two stages.",
                    "evidence_ids": ["E3"],
                    "premise_evidence_ids": [],
                },
            ],
            "synthesized_findings": [
                {
                    "obligation_id": "O1",
                    "statement": "Method A exceeds Method B by 5 points.",
                    "premise_evidence_ids": ["E1", "E2"],
                },
                {
                    "obligation_id": "O2",
                    "statement": "The decoder rounds to two stages.",
                    "premise_evidence_ids": ["E3"],
                },
            ],
            "unresolved_requirements": [],
            "unresolved_obligations": [],
        },
        {
            "verdicts": [
                {"claim_id": "claim-2", "supported": True, "reason": None},
                {"claim_id": "claim-3", "supported": True, "reason": None},
                {
                    "claim_id": "claim-4",
                    "supported": False,
                    "reason": "rounding_method_not_stated",
                },
            ]
        },
    )

    result = await generate_final_answer(
        question="Compare the reported scores and decoder structure.",
        contract=contract,
        packed_packets=packets,
        slot_resolutions=[
            SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
            SlotResolution(slot_id="S2", status="supported", evidence_ids=["E2"]),
            SlotResolution(slot_id="S3", status="supported", evidence_ids=["E3"]),
        ],
        llm_invoker=invoker,
    )

    accepted_ids = {
        claim.claim_id
        for claim in result.claims
        if claim.qualified_reason is None
    }
    rounding_claim = next(
        claim for claim in result.claims if claim.obligation_id == "O2"
    )
    assert [call["purpose"] for call in invoker.calls] == [
        "final_answer",
        "claim_verifier",
    ]
    assert result.claim_verifier_call_count == 1
    assert accepted_ids == {"claim-1", "claim-2", "claim-3"}
    assert rounding_claim.qualified_reason == "rounding_method_not_stated"
    assert result.response_status == "qualified_partial"


@pytest.mark.asyncio
async def test_verifier_unavailable_leaves_all_pending_claims_unresolved() -> None:
    invoker = _RecordingInvoker(
        {
            "supported_findings": [
                {
                    "slot_id": "score",
                    "statement": "The decoder has two stages.",
                    "evidence_ids": ["E1"],
                    "premise_evidence_ids": [],
                }
            ],
            "unresolved_requirements": [],
        }
    )

    result = await generate_final_answer(
        question="What is the decoder structure?",
        contract=_contract(),
        packed_packets=[_packet(statement="The decoder consists of two stages.")],
        slot_resolutions=[
            SlotResolution(slot_id="score", status="supported", evidence_ids=["E1"])
        ],
        llm_invoker=invoker,
    )

    assert result.claim_verifier_call_count == 1
    assert result.claims[0].qualified_reason == "claim_verifier_unavailable_or_invalid"
    assert result.used_evidence_ids == []
    assert result.response_status == "insufficient"
