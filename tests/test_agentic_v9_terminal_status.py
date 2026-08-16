"""Tests for deterministic terminal status calculation and backend observability."""

from __future__ import annotations

import json

import pytest

from data_base.agentic_v9.final_answer import (
    FinalAnswerDraft,
    FinalAnswerRenderer,
    _response_status,
)
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    FinalClaim,
    QueryContract,
    RequiredSlot,
    SlotResolution,
    SourceLocator,
    SupportedFinding,
    SynthesizedFinding,
    SynthesisObligation,
    UnresolvedObligation,
    UnresolvedRequirement,
)


from types import SimpleNamespace


class _DummyInvoker:
    def __init__(self, draft: FinalAnswerDraft) -> None:
        self.draft = draft

    async def invoke(self, **kwargs: object) -> object:
        if kwargs.get("phase") == "claim_verifier":
            claim_ids: list[str] = []
            messages = kwargs.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    content = message.get("content")
                    if not isinstance(content, str):
                        continue
                    try:
                        payload = json.loads(content)
                    except json.JSONDecodeError:
                        continue
                    for row in payload.get("claims", []):
                        claim = row.get("claim", {})
                        claim_id = claim.get("claim_id")
                        if isinstance(claim_id, str) and claim_id not in claim_ids:
                            claim_ids.append(claim_id)
            return SimpleNamespace(
                content={
                    "verdicts": [
                        {"claim_id": claim_id, "supported": True, "reason": None}
                        for claim_id in claim_ids
                    ]
                }
            )
        return SimpleNamespace(content=self.draft.model_dump_json())


def _make_packet(
    evidence_id: str,
    slot_ids: list[str],
    statement: str,
    *,
    doc_id: str = "doc-1",
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="t-1",
        round_id="r-1",
        query_id="q-1",
        slot_ids=slot_ids,
        statement=statement,
        support_type="direct",
        source=EvidenceSource(
            doc_id=doc_id,
            chunk_id=f"c-{evidence_id}",
            source_span_hash="hash-12345",
        ),
        scope=EvidenceScope(),
        locator=SourceLocator(pdf_page_index=1),
        extractor_version="prose_curator_v1",
        validation_status="deterministic_valid",
    )


@pytest.mark.asyncio
async def test_terminal_status_all_slots_and_obligations_supported_is_complete() -> None:
    contract = QueryContract(
        contract_version="2",
        route="bounded_compare",
        intent="Compare Model A and Model B",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Model A score"),
            RequiredSlot(slot_id="S2", description="Model B score"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare scores of Model A and Model B",
                depends_on_slot_ids=["S1", "S2"],
            )
        ],
    )
    p1 = _make_packet("E1", ["S1"], "Model A score is 0.85.")
    p2 = _make_packet("E2", ["S2"], "Model B score is 0.92.")
    draft = FinalAnswerDraft(
        supported_findings=[
            SupportedFinding(
                slot_id="S1",
                statement="Model A score is 0.85.",
                evidence_ids=["E1"],
                premise_evidence_ids=[],
            ),
            SupportedFinding(
                slot_id="S2",
                statement="Model B score is 0.92.",
                evidence_ids=["E2"],
                premise_evidence_ids=[],
            ),
        ],
        synthesized_findings=[
            SynthesizedFinding(
                obligation_id="O1",
                statement="Model B outperforms Model A (0.92 vs 0.85).",
                premise_evidence_ids=["E1", "E2"],
            )
        ],
    )
    renderer = FinalAnswerRenderer(_DummyInvoker(draft))
    result = await renderer.render(
        question="Compare Model A and Model B",
        contract=contract,
        packed_packets=[p1, p2],
        slot_resolutions=[
            SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
            SlotResolution(slot_id="S2", status="supported", evidence_ids=["E2"]),
        ],
    )

    assert result.response_status == "complete"
    assert result.claim_verifier_call_count == 1
    assert result.used_evidence_ids == ["E1", "E2"]
    assert len(result.claims) == 3
    assert all(claim.qualified_reason is None for claim in result.claims)
    assert "Model A score is 0.85." in result.answer
    assert "Model B score is 0.92." in result.answer
    assert "Model B outperforms Model A" in result.answer


@pytest.mark.asyncio
async def test_terminal_status_missing_one_slot_is_qualified_partial() -> None:
    contract = QueryContract(
        contract_version="2",
        route="bounded_compare",
        intent="Compare Model A and Model B",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Model A score"),
            RequiredSlot(slot_id="S2", description="Model B score"),
        ],
    )
    p1 = _make_packet("E1", ["S1"], "Model A score is 0.85.")
    draft = FinalAnswerDraft(
        supported_findings=[
            SupportedFinding(
                slot_id="S1",
                statement="Model A score is 0.85.",
                evidence_ids=["E1"],
            )
        ],
        unresolved_requirements=[
            UnresolvedRequirement(
                slot_id="S2", reason="No score reported for Model B."
            )
        ],
    )
    renderer = FinalAnswerRenderer(_DummyInvoker(draft))
    result = await renderer.render(
        question="Compare Model A and Model B",
        contract=contract,
        packed_packets=[p1],
        slot_resolutions=[
            SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
            SlotResolution(
                slot_id="S2", status="not_found", reason="No chunk found"
            ),
        ],
    )

    assert result.response_status == "qualified_partial"
    assert result.used_evidence_ids == ["E1"]
    assert len(result.claims) == 1
    assert "- S2: No score reported for Model B." in result.answer


@pytest.mark.asyncio
async def test_terminal_status_missing_obligation_is_qualified_partial() -> None:
    contract = QueryContract(
        contract_version="2",
        route="bounded_compare",
        intent="Compare Model A and Model B",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Model A score"),
            RequiredSlot(slot_id="S2", description="Model B score"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare scores of Model A and Model B",
                depends_on_slot_ids=["S1", "S2"],
            )
        ],
    )
    p1 = _make_packet("E1", ["S1"], "Model A score is 0.85.")
    p2 = _make_packet("E2", ["S2"], "Model B score is 0.92.")
    draft = FinalAnswerDraft(
        supported_findings=[
            SupportedFinding(
                slot_id="S1",
                statement="Model A score is 0.85.",
                evidence_ids=["E1"],
            ),
            SupportedFinding(
                slot_id="S2",
                statement="Model B score is 0.92.",
                evidence_ids=["E2"],
            ),
        ],
        unresolved_obligations=[
            UnresolvedObligation(
                obligation_id="O1",
                reason="Cannot establish a conclusive comparison.",
            )
        ],
    )
    renderer = FinalAnswerRenderer(_DummyInvoker(draft))
    result = await renderer.render(
        question="Compare Model A and Model B",
        contract=contract,
        packed_packets=[p1, p2],
        slot_resolutions=[
            SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
            SlotResolution(slot_id="S2", status="supported", evidence_ids=["E2"]),
        ],
    )

    assert result.response_status == "qualified_partial"
    assert result.used_evidence_ids == ["E1", "E2"]
    assert "- O1: Cannot establish a conclusive comparison." in result.answer


@pytest.mark.asyncio
async def test_terminal_status_obligation_broken_dependency_closure_is_qualified_partial() -> None:
    contract = QueryContract(
        contract_version="2",
        route="bounded_compare",
        intent="Compare Model A and Model B",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Model A score"),
            RequiredSlot(slot_id="S2", description="Model B score"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare scores of Model A and Model B",
                depends_on_slot_ids=["S1", "S2"],
            )
        ],
    )
    p1 = _make_packet("E1", ["S1"], "Model A score is 0.85.")
    p2 = _make_packet("E2", ["S2"], "Model B score is 0.92.")
    # Obligation only references E1, missing S2's premise packet E2
    draft = FinalAnswerDraft(
        supported_findings=[
            SupportedFinding(
                slot_id="S1",
                statement="Model A score is 0.85.",
                evidence_ids=["E1"],
            ),
            SupportedFinding(
                slot_id="S2",
                statement="Model B score is 0.92.",
                evidence_ids=["E2"],
            ),
        ],
        synthesized_findings=[
            SynthesizedFinding(
                obligation_id="O1",
                statement="Model A is 0.85.",
                premise_evidence_ids=["E1"],
            )
        ],
    )
    renderer = FinalAnswerRenderer(_DummyInvoker(draft))
    result = await renderer.render(
        question="Compare Model A and Model B",
        contract=contract,
        packed_packets=[p1, p2],
        slot_resolutions=[
            SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
            SlotResolution(slot_id="S2", status="supported", evidence_ids=["E2"]),
        ],
    )

    assert result.response_status == "qualified_partial"
    assert "- O1: Obligation O1 (Compare scores of Model A and Model B) was not synthesized from verified premises." in result.answer


@pytest.mark.asyncio
async def test_terminal_status_slot_claim_verification_failure_is_qualified_partial() -> None:
    contract = QueryContract(
        contract_version="2",
        route="multi_hop",
        intent="Look up score and method",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Reported score"),
            RequiredSlot(slot_id="S2", description="Method name"),
        ],
    )
    p1 = _make_packet("E1", ["S1"], "The score is 0.85.")
    p2 = _make_packet("E2", ["S2"], "The method is UNet.")
    # S1 claims a number not present in E1, S2 claims valid statement
    draft = FinalAnswerDraft(
        supported_findings=[
            SupportedFinding(
                slot_id="S1",
                statement="The score is 999.0.",
                evidence_ids=["E1"],
            ),
            SupportedFinding(
                slot_id="S2",
                statement="The method is UNet.",
                evidence_ids=["E2"],
            ),
        ],
    )
    renderer = FinalAnswerRenderer(_DummyInvoker(draft))
    result = await renderer.render(
        question="Look up score and method",
        contract=contract,
        packed_packets=[p1, p2],
        slot_resolutions=[
            SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
            SlotResolution(slot_id="S2", status="supported", evidence_ids=["E2"]),
        ],
    )

    assert result.response_status == "qualified_partial"
    assert result.used_evidence_ids == ["E2"]
    assert len(result.claims) == 2
    failed_claim = next(c for c in result.claims if c.slot_id == "S1")
    assert failed_claim.qualified_reason == "claim does not match cited exact evidence"
    assert "- S1: No accepted final finding covered this required slot." in result.answer


@pytest.mark.asyncio
async def test_terminal_status_no_supported_claims_is_insufficient() -> None:
    contract = QueryContract(
        contract_version="2",
        route="single_lookup",
        intent="Look up score",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Reported score"),
        ],
    )
    draft = FinalAnswerDraft(
        unresolved_requirements=[
            UnresolvedRequirement(slot_id="S1", reason="No data")
        ],
    )
    renderer = FinalAnswerRenderer(_DummyInvoker(draft))
    result = await renderer.render(
        question="Look up score",
        contract=contract,
        packed_packets=[],
        slot_resolutions=[
            SlotResolution(slot_id="S1", status="not_found", reason="No data"),
        ],
    )

    assert result.response_status == "insufficient"
    assert result.used_evidence_ids == []
    assert result.claims == []


def test_response_status_reducer_pure_function() -> None:
    contract = QueryContract(
        contract_version="2",
        route="single_lookup",
        intent="Test",
        required_slots=[RequiredSlot(slot_id="S1", description="Slot 1")],
    )
    claim = FinalClaim(
        claim_id="c1",
        slot_id="S1",
        statement="Fact",
        support_type="direct",
        evidence_ids=["E1"],
    )
    status = _response_status(
        claims=[claim],
        contract=contract,
        slot_resolutions=[SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"])],
    )
    assert status == "complete"

    # With unresolved requirement
    status_partial = _response_status(
        claims=[claim],
        contract=contract,
        slot_resolutions=[SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"])],
        unresolved_requirements=[UnresolvedRequirement(slot_id="S2", reason="Missing")],
    )
    assert status_partial == "qualified_partial"

    # With zero accepted claims
    status_insufficient = _response_status(
        claims=[],
        contract=contract,
        slot_resolutions=[],
    )
    assert status_insufficient == "insufficient"
