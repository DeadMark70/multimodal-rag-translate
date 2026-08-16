"""Deterministic-first verification for final Agentic v9 claims."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_base.agentic_v9.evidence_validator import (
    is_qualified_evidence,
    normalize_source_span,
)
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    FinalClaim,
    LlmInvoker,
    QueryContract,
)


_NUMERIC_TOKEN = re.compile(
    r"(?<![\w.])(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))"
    r"\s*(?P<suffix>%|percent|x|×|-?fold)?(?![\w])",
    re.IGNORECASE,
)


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


ClaimGateStatus = Literal["accepted", "verify", "rejected"]


class ClaimGateResult(BaseModel):
    """The deterministic disposition before any semantic verifier call."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    status: ClaimGateStatus
    reason: str | None = None


def numeric_tokens(text: str) -> set[tuple[str, str]]:
    """Return normalized numeric values with their semantic suffix kinds."""
    tokens: set[tuple[str, str]] = set()
    for match in _NUMERIC_TOKEN.finditer(text):
        value = _normalize_decimal(match.group("value"))
        suffix = (match.group("suffix") or "").lower()
        kind = (
            "percent"
            if suffix in {"%", "percent"}
            else "ratio"
            if suffix in {"x", "×", "fold", "-fold"}
            else "scalar"
        )
        tokens.add((value, kind))
    return tokens


def gate_claim_deterministically(
    claim: FinalClaim, packets_by_id: Mapping[str, EvidencePacket]
) -> ClaimGateResult:
    """Apply structural, numeric, and exact-span checks before semantic review."""
    evidence_ids = list(
        dict.fromkeys([*claim.evidence_ids, *claim.premise_evidence_ids])
    )
    if not evidence_ids:
        return ClaimGateResult(
            claim_id=claim.claim_id,
            status="rejected",
            reason="claim_has_no_evidence_ids",
        )

    closure_ids = _collect_evidence_closure(evidence_ids, packets_by_id)
    if any(evidence_id not in packets_by_id for evidence_id in closure_ids):
        return ClaimGateResult(
            claim_id=claim.claim_id,
            status="rejected",
            reason="claim_references_unpacked_or_unknown_evidence",
        )

    typed_packets = [packets_by_id[evidence_id] for evidence_id in closure_ids]
    if any(
        not is_qualified_evidence(packet, packets_by_id)
        for packet in typed_packets
    ):
        return ClaimGateResult(
            claim_id=claim.claim_id,
            status="rejected",
            reason="invalid_evidence",
        )
    if not _has_premise_closure(typed_packets, packets_by_id):
        return ClaimGateResult(
            claim_id=claim.claim_id,
            status="rejected",
            reason="missing_premise_closure",
        )

    # Obligation claims always need semantic checking once their direct premises
    # are complete.  In particular, an aggregation obligation does not require
    # a pre-computed ``calculated`` packet.
    if claim.obligation_id is not None:
        return ClaimGateResult(
            claim_id=claim.claim_id,
            status="verify",
            reason="obligation_requires_semantic_verification",
        )

    cited_packets = [packets_by_id[evidence_id] for evidence_id in evidence_ids]
    claim_numbers = numeric_tokens(claim.statement)
    evidence_numbers = {
        token
        for packet in cited_packets
        for token in numeric_tokens(packet.statement)
    }
    if not claim_numbers.issubset(evidence_numbers):
        return ClaimGateResult(
            claim_id=claim.claim_id,
            status="rejected",
            reason="claim does not match cited exact evidence",
        )

    normalized_claim = normalize_source_span(claim.statement)
    if any(
        normalized_claim in normalize_source_span(packet.statement)
        for packet in cited_packets
    ):
        return ClaimGateResult(claim_id=claim.claim_id, status="accepted")
    return ClaimGateResult(
        claim_id=claim.claim_id,
        status="verify",
        reason="claim_requires_semantic_verification",
    )


class ClaimVerifier:
    """Invoke the injected, already-budgeted verifier at most once per answer."""

    def __init__(self, llm_invoker: LlmInvoker) -> None:
        self._invoker = llm_invoker

    async def verify(
        self,
        claims: Sequence[FinalClaim],
        packets_by_id: Mapping[str, EvidencePacket],
        *,
        contract: QueryContract,
    ) -> dict[str, ClaimVerdict]:
        """Verify all pending claims in one typed batch response."""
        if not claims:
            return {}

        obligation_by_id = {
            obligation.obligation_id: obligation
            for obligation in contract.synthesis_obligations
        }
        slot_by_id = {slot.slot_id: slot for slot in contract.required_slots}
        claim_rows: list[dict[str, Any]] = []
        for claim in claims:
            if claim.slot_id is not None:
                target_kind = "slot"
                target_description = slot_by_id.get(
                    claim.slot_id
                ).description if claim.slot_id in slot_by_id else ""
            else:
                target_kind = "obligation"
                target_description = obligation_by_id.get(
                    claim.obligation_id
                ).description if claim.obligation_id in obligation_by_id else ""
            evidence_ids = list(
                dict.fromkeys([*claim.evidence_ids, *claim.premise_evidence_ids])
            )
            evidence_closure_ids = _collect_evidence_closure(
                evidence_ids, packets_by_id
            )
            claim_rows.append(
                {
                    "claim": claim.model_dump(mode="json"),
                    "target_kind": target_kind,
                    "target_description": target_description,
                    "evidence_packets": [
                        packets_by_id[evidence_id].model_dump(mode="json")
                        for evidence_id in evidence_closure_ids
                        if evidence_id in packets_by_id
                    ],
                }
            )
        payload = json.dumps(
            {
                "claims": claim_rows,
                "contract": contract.model_dump(mode="json"),
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
                        "content": (
                            "Verify only the listed claims against the supplied evidence. "
                            "Return exactly one verdict per claim and JSON only. Support a claim "
                            "only from its supplied evidence_packets. Recompute arithmetic only "
                            "from cited direct premises. Do not infer a rounding method or a "
                            "precision assumption. Accept a direct paraphrase only when it is "
                            "entailed by the supplied evidence. Reject missing or ambiguous "
                            "support."
                        ),
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


def _collect_evidence_closure(
    evidence_ids: Sequence[str], packets_by_id: Mapping[str, EvidencePacket]
) -> list[str]:
    """Collect cited evidence and every declared premise ID once."""
    collected = list(dict.fromkeys(evidence_ids))
    index = 0
    while index < len(collected):
        packet = packets_by_id.get(collected[index])
        if packet is not None:
            for premise_id in packet.premise_evidence_ids:
                if premise_id not in collected:
                    collected.append(premise_id)
        index += 1
    return collected


def _normalize_decimal(value: str) -> str:
    try:
        normalized = Decimal(value).normalize()
    except InvalidOperation:
        return value.lstrip("+")
    if normalized == 0:
        return "0"
    return format(normalized, "f")


def _gate_as_verdict(gate: ClaimGateResult) -> ClaimVerdict:
    if gate.status == "accepted":
        return ClaimVerdict(claim_id=gate.claim_id, supported=True, reason=gate.reason)
    return ClaimVerdict(
        claim_id=gate.claim_id,
        supported=False,
        reason=gate.reason or "claim_not_verified",
    )


def gate_as_verdict(gate: ClaimGateResult) -> ClaimVerdict:
    """Project a deterministic gate result into the strict verdict model."""
    return _gate_as_verdict(gate)


def _response_content(response: Any) -> Any:
    content = getattr(response, "content", response)
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    if isinstance(content, str):
        return json.loads(content)
    return content


__all__ = [
    "ClaimGateResult",
    "ClaimGateStatus",
    "ClaimVerdict",
    "ClaimVerificationResponse",
    "ClaimVerifier",
    "gate_as_verdict",
    "gate_claim_deterministically",
    "numeric_tokens",
    "qualify_failed_claim",
]
