"""Deterministic-first verification for final Agentic v9 claims."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_base.agentic_v9.schemas import EvidencePacket, FinalClaim, LlmInvoker


_NUMBERS = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![\w]|\.\d)")
_HIGH_RISK = re.compile(
    r"\b(?:outperform(?:s|ed|ing)?|better|best|highest|first|sota|"
    r"state[ -]of[ -]the[ -]art|caus(?:e|es|ed|al|ally)|safe|robust)\b",
    re.IGNORECASE,
)
_USABLE_STATUSES = {"deterministic_valid", "quote_bound", "derived_non_evidence"}


class ClaimVerdict(BaseModel):
    """One verification result; a false verdict never authorizes a claim."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    supported: bool
    reason: str | None = None


class ClaimVerificationResponse(BaseModel):
    """The strictly typed response accepted from the single verifier batch."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[ClaimVerdict] = Field(default_factory=list)


def requires_prose_verification(claim: FinalClaim) -> bool:
    """Identify claims deterministic checks cannot establish exactly."""
    return claim.support_type in {"comparative_inference", "qualified"} or bool(
        _HIGH_RISK.search(claim.statement)
    )


def verify_claim_deterministically(
    claim: FinalClaim, packets_by_id: Mapping[str, EvidencePacket]
) -> ClaimVerdict:
    """Check provenance, premise closure, and exact numeric facts without a model."""
    evidence_ids = list(
        dict.fromkeys([*claim.evidence_ids, *claim.premise_evidence_ids])
    )
    if not evidence_ids:
        return ClaimVerdict(
            claim_id=claim.claim_id, supported=False, reason="claim_has_no_evidence_ids"
        )
    packets = [packets_by_id.get(evidence_id) for evidence_id in evidence_ids]
    if any(packet is None for packet in packets):
        return ClaimVerdict(
            claim_id=claim.claim_id,
            supported=False,
            reason="claim_references_unpacked_or_unknown_evidence",
        )
    typed_packets = [packet for packet in packets if packet is not None]
    if any(
        packet.validation_status not in _USABLE_STATUSES
        or packet.support_type == "contradictory"
        for packet in typed_packets
    ):
        return ClaimVerdict(
            claim_id=claim.claim_id, supported=False, reason="invalid_evidence"
        )
    if not _has_premise_closure(typed_packets, packets_by_id):
        return ClaimVerdict(
            claim_id=claim.claim_id, supported=False, reason="missing_premise_closure"
        )
    if requires_prose_verification(claim):
        return ClaimVerdict(claim_id=claim.claim_id, supported=True)
    if claim.support_type == "calculated" and not any(
        packet.support_type == "calculated" for packet in typed_packets
    ):
        return ClaimVerdict(
            claim_id=claim.claim_id,
            supported=False,
            reason="calculated_claim_lacks_calculated_evidence",
        )
    if not _exact_numbers_are_supported(claim.statement, typed_packets):
        return ClaimVerdict(
            claim_id=claim.claim_id,
            supported=False,
            reason="claim does not match cited exact evidence",
        )
    return ClaimVerdict(claim_id=claim.claim_id, supported=True)


class ClaimVerifier:
    """Invoke the injected, already-budgeted verifier at most once per answer."""

    def __init__(self, llm_invoker: LlmInvoker) -> None:
        self._invoker = llm_invoker

    async def verify(
        self,
        claims: Sequence[FinalClaim],
        packets_by_id: Mapping[str, EvidencePacket],
    ) -> dict[str, ClaimVerdict]:
        """Verify all unresolved high-risk prose in one typed batch response."""
        if not claims:
            return {}
        evidence_ids = sorted(
            {
                evidence_id
                for claim in claims
                for evidence_id in [*claim.evidence_ids, *claim.premise_evidence_ids]
                if evidence_id in packets_by_id
            }
        )
        payload = json.dumps(
            {
                "claims": [claim.model_dump(mode="json") for claim in claims],
                "evidence_packets": [
                    packets_by_id[evidence_id].model_dump(mode="json")
                    for evidence_id in evidence_ids
                ],
                "response_schema": {
                    "verdicts": [
                        {
                            "claim_id": "string",
                            "supported": "boolean",
                            "reason": "string|null",
                        }
                    ]
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        try:
            response = await self._invoker.invoke(
                phase="claim_verifier",
                purpose="claim_verifier",
                messages=[
                    {
                        "role": "system",
                        "content": "Verify only the listed claims against the supplied evidence. Return JSON only.",
                    },
                    {"role": "user", "content": payload},
                ],
            )
            parsed = ClaimVerificationResponse.model_validate(
                _response_content(response)
            )
        except Exception:
            return {
                claim.claim_id: ClaimVerdict(
                    claim_id=claim.claim_id,
                    supported=False,
                    reason="claim_verifier_unavailable_or_invalid",
                )
                for claim in claims
            }
        allowed_ids = {claim.claim_id for claim in claims}
        verdicts = {
            verdict.claim_id: verdict
            for verdict in parsed.verdicts
            if verdict.claim_id in allowed_ids
        }
        return {
            claim.claim_id: verdicts.get(
                claim.claim_id,
                ClaimVerdict(
                    claim_id=claim.claim_id,
                    supported=False,
                    reason="claim_verifier_omitted_verdict",
                ),
            )
            for claim in claims
        }


def qualify_failed_claim(claim: FinalClaim, verdict: ClaimVerdict) -> FinalClaim:
    """Keep provenance but make failed content visibly non-assertive."""
    return claim.model_copy(
        update={
            "support_type": "qualified",
            "qualified_reason": verdict.reason or "claim_not_verified",
        }
    )


def _has_premise_closure(
    packets: Iterable[EvidencePacket], packets_by_id: Mapping[str, EvidencePacket]
) -> bool:
    required_ids = {
        premise_id for packet in packets for premise_id in packet.premise_evidence_ids
    }
    return required_ids.issubset(packets_by_id)


def _exact_numbers_are_supported(
    statement: str, packets: Iterable[EvidencePacket]
) -> bool:
    claim_numbers = {_normalize_number(value) for value in _NUMBERS.findall(statement)}
    if not claim_numbers:
        return any(statement.strip() == packet.statement.strip() for packet in packets)
    evidence_numbers = {
        _normalize_number(value)
        for packet in packets
        for value in _NUMBERS.findall(packet.statement)
    }
    return claim_numbers.issubset(evidence_numbers)


def _normalize_number(value: str) -> str:
    value = value.lstrip("+")
    return value.rstrip("0").rstrip(".") if "." in value else value


def _response_content(response: Any) -> Any:
    content = getattr(response, "content", response)
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    if isinstance(content, str):
        return json.loads(content)
    return content


__all__ = [
    "ClaimVerdict",
    "ClaimVerificationResponse",
    "ClaimVerifier",
    "qualify_failed_claim",
    "requires_prose_verification",
    "verify_claim_deterministically",
]
