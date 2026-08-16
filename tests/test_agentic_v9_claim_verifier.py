"""Deterministic claim gates and one-batch semantic verification tests."""

from __future__ import annotations

from decimal import Decimal
import json
from types import SimpleNamespace
from typing import Any

import pytest

from data_base.agentic_v9.claim_verifier import (
    ClaimVerifier,
    gate_claim_deterministically,
    numeric_tokens,
)
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    FinalClaim,
    QueryContract,
    RequiredSlot,
    SourceLocator,
    SynthesisObligation,
)


def _packet(
    evidence_id: str = "E1",
    statement: str = "The reported score is 91%.",
    *,
    support_type: str = "direct",
    slot_ids: list[str] | None = None,
    premise_evidence_ids: list[str] | None = None,
    validation_status: str = "deterministic_valid",
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=slot_ids or ["S1"],
        statement=statement,
        support_type=support_type,  # type: ignore[arg-type]
        source=EvidenceSource(
            doc_id="doc-1",
            document_name="paper.pdf",
            source_span_hash=f"hash-{evidence_id}",
        ),
        scope=EvidenceScope(metric="Dice"),
        locator=SourceLocator(pdf_page_index=4),
        raw_value=Decimal("91"),
        normalized_value=Decimal("91"),
        premise_evidence_ids=premise_evidence_ids or [],
        extractor_version="v9-deterministic-1",
        validation_status=validation_status,  # type: ignore[arg-type]
    )


def _direct_claim(statement: str, *, evidence_ids: list[str] | None = None) -> FinalClaim:
    return FinalClaim(
        claim_id="claim-1",
        slot_id="S1",
        statement=statement,
        support_type="direct",
        evidence_ids=evidence_ids or ["E1"],
    )


def _obligation_claim() -> FinalClaim:
    return FinalClaim(
        claim_id="claim-obligation",
        obligation_id="O1",
        statement="The decoder has two stages.",
        support_type="calculated",
        premise_evidence_ids=["E1", "E2"],
    )


def _contract() -> QueryContract:
    return QueryContract(
        route="exact_structured",
        intent="Report the decoder structure.",
        required_slots=[RequiredSlot(slot_id="S1", description="Decoder structure")],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="aggregation",
                description="Combine direct decoder premises.",
                depends_on_slot_ids=["S1"],
            )
        ],
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("33x fewer parameters", {("33", "ratio")}),
        ("33× fewer parameters", {("33", "ratio")}),
        ("33-fold fewer parameters", {("33", "ratio")}),
        ("12.50% reduction", {("12.5", "percent")}),
        ("Table 1", {("1", "scalar")}),
        ("Algorithm 2結果", {("2", "scalar")}),
        ("提升12.5%結果", {("12.5", "percent")}),
        ("Model2", set()),
        ("foo_12", set()),
        ("Method-v2", set()),
    ],
)
def test_numeric_tokens_preserve_ratio_and_percent_semantics(
    text: str, expected: set[tuple[str, str]]
) -> None:
    assert numeric_tokens(text) == expected


def test_direct_ratio_claim_cannot_match_a_different_ratio() -> None:
    claim = _direct_claim("The method uses 33x fewer parameters.")
    result = gate_claim_deterministically(
        claim,
        {"E1": _packet(statement="The method uses 34x fewer parameters.")},
    )

    assert result.status == "rejected"


def test_ascii_and_multiplication_ratio_tokens_normalize_identically() -> None:
    assert numeric_tokens("33x fewer parameters") == numeric_tokens(
        "33× fewer parameters"
    )


def test_direct_verbatim_span_is_accepted_without_verifier() -> None:
    result = gate_claim_deterministically(
        _direct_claim("reported score is 91%"),
        {"E1": _packet()},
    )

    assert result.status == "accepted"


def test_direct_paraphrase_with_valid_provenance_is_sent_to_verifier() -> None:
    result = gate_claim_deterministically(
        _direct_claim("the decoder has two stages"),
        {"E1": _packet(statement="The decoder consists of two stages.")},
    )

    assert result.status == "verify"


def test_obligation_with_complete_direct_premises_is_sent_to_verifier() -> None:
    result = gate_claim_deterministically(
        _obligation_claim(),
        {
            "E1": _packet(statement="The decoder has one stage."),
            "E2": _packet("E2", statement="The decoder has one more stage."),
        },
    )

    assert result.status == "verify"
    assert result.reason != "calculated_claim_lacks_calculated_evidence"


def test_unknown_or_unqualified_evidence_is_rejected_before_verifier() -> None:
    unknown = gate_claim_deterministically(
        _direct_claim("reported score is 91%", evidence_ids=["missing"]),
        {"E1": _packet()},
    )
    unqualified = gate_claim_deterministically(
        _direct_claim("reported score is 91%"),
        {"E1": _packet(validation_status="invalid")},
    )

    assert unknown.status == "rejected"
    assert unqualified.status == "rejected"


def test_whitespace_only_claim_is_rejected_before_verifier() -> None:
    result = gate_claim_deterministically(
        _direct_claim("   \t\n"),
        {"E1": _packet()},
    )

    assert result.status == "rejected"
    assert result.reason == "claim_statement_empty"


class _RecordingInvoker:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def invoke(
        self, *, phase: str, purpose: str, messages: list[dict[str, Any]]
    ) -> Any:
        self.calls.append({"phase": phase, "purpose": purpose, "messages": messages})
        return self.response


@pytest.mark.asyncio
async def test_verifier_sends_one_claim_scoped_batch_with_contract_targets() -> None:
    invoker = _RecordingInvoker(
        SimpleNamespace(
            content={
                "verdicts": [
                    {"claim_id": "claim-1", "supported": True, "reason": None}
                ]
            }
        )
    )
    claim = _direct_claim("the decoder has two stages")
    verdicts = await ClaimVerifier(invoker).verify(
        [claim],
        {"E1": _packet(statement="The decoder consists of two stages.")},
        contract=_contract(),
    )

    assert verdicts[claim.claim_id].supported is True
    assert len(invoker.calls) == 1
    payload = json.loads(invoker.calls[0]["messages"][1]["content"])
    row = payload["claims"][0]
    assert row["target_kind"] == "slot"
    assert row["target_description"] == "Decoder structure"
    assert [packet["evidence_id"] for packet in row["evidence_packets"]] == ["E1"]
    system_message = invoker.calls[0]["messages"][0]["content"].casefold()
    assert "one verdict per claim" in system_message
    assert "arithmetic" in system_message
    assert "rounding" in system_message
    assert "direct paraphrase" in system_message
    assert "ambiguous" in system_message
