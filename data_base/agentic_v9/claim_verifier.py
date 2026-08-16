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
from data_base.agentic_v9.provider_boundary import provider_response_content
from data_base.agentic_v9.schemas import (
    BudgetExceededError,
    EvidencePacket,
    FinalClaim,
    LlmInvoker,
    QueryContract,
)


_NUMERIC_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.-])(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))"
    r"\s*(?P<suffix>%|percent|x|×|-?fold)?(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
_SAFE_GATE_REASONS = frozenset(
    {
        "claim_has_no_evidence_ids",
        "claim_references_unpacked_or_unknown_evidence",
        "invalid_evidence",
        "missing_premise_closure",
        "claim_statement_empty",
        "unknown_obligation",
        "missing_obligation_dependency_closure",
        "direct_claim_requires_direct_evidence",
        "claim does not match cited exact evidence",
        "claim_requires_semantic_verification",
        "obligation_requires_semantic_verification",
    }
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
    claim: FinalClaim,
    packets_by_id: Mapping[str, EvidencePacket],
    *,
    contract: QueryContract | None = None,
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

    normalized_claim = normalize_source_span(claim.statement)
    if not normalized_claim:
        return ClaimGateResult(
            claim_id=claim.claim_id,
            status="rejected",
            reason="claim_statement_empty",
        )

    # Obligation claims always need semantic checking once their direct premises
    # are complete.  In particular, an aggregation obligation does not require
    # a pre-computed ``calculated`` packet.
    if claim.obligation_id is not None:
        direct_packets = [
            packet for packet in typed_packets if packet.support_type == "direct"
        ]
        if contract is not None:
            obligation = next(
                (
                    item
                    for item in contract.synthesis_obligations
                    if item.obligation_id == claim.obligation_id
                ),
                None,
            )
            if obligation is None:
                return ClaimGateResult(
                    claim_id=claim.claim_id,
                    status="rejected",
                    reason="unknown_obligation",
                )
            covered_slot_ids = {
                slot_id
                for packet in direct_packets
                for slot_id in packet.slot_ids
            }
            if not set(obligation.depends_on_slot_ids).issubset(covered_slot_ids):
                return ClaimGateResult(
                    claim_id=claim.claim_id,
                    status="rejected",
                    reason="missing_obligation_dependency_closure",
                )
        return ClaimGateResult(
            claim_id=claim.claim_id,
            status="verify",
            reason="obligation_requires_semantic_verification",
        )

    cited_packets = [packets_by_id[evidence_id] for evidence_id in evidence_ids]
    if any(packet.support_type != "direct" for packet in cited_packets):
        return ClaimGateResult(
            claim_id=claim.claim_id,
            status="rejected",
            reason="direct_claim_requires_direct_evidence",
        )
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
        self._last_call_count = 0
        self._last_diagnostic_code: str | None = None

    @property
    def last_call_count(self) -> int:
        return self._last_call_count

    @property
    def last_diagnostic_code(self) -> str | None:
        return self._last_diagnostic_code

    async def verify(
        self,
        claims: Sequence[FinalClaim],
        packets_by_id: Mapping[str, EvidencePacket],
        *,
        contract: QueryContract,
    ) -> dict[str, ClaimVerdict]:
        """Verify all pending claims in one typed batch response."""
        self._last_call_count = 0
        self._last_diagnostic_code = None
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
        except BudgetExceededError:
            self._last_diagnostic_code = "budget_rejected"
            return _fail_closed_verdicts(claims, reason="claim_verifier_budget_rejected")
        except (json.JSONDecodeError, ValueError, TypeError):
            self._last_call_count = 1
            self._last_diagnostic_code = "invalid_provider_response"
            return _fail_closed_verdicts(claims, reason="claim_verifier_invalid_response")
        except Exception:
            self._last_call_count = 1
            self._last_diagnostic_code = "provider_failure"
            return _fail_closed_verdicts(claims, reason="claim_verifier_provider_failure")
        self._last_call_count = 1
        pending_ids = [claim.claim_id for claim in claims]
        verdict_ids = [verdict.claim_id for verdict in parsed.verdicts]
        if (
            len(pending_ids) != len(set(pending_ids))
            or len(verdict_ids) != len(set(verdict_ids))
            or len(pending_ids) != len(verdict_ids)
            or set(pending_ids) != set(verdict_ids)
        ):
            self._last_diagnostic_code = "invalid_provider_response"
            return _fail_closed_verdicts(claims, reason="claim_verifier_invalid_response")
        self._last_diagnostic_code = (
            "accepted" if all(verdict.supported for verdict in parsed.verdicts) else "claim_rejected"
        )
        return {verdict.claim_id: verdict for verdict in parsed.verdicts}


def qualify_failed_claim(claim: FinalClaim, verdict: ClaimVerdict) -> FinalClaim:
    """Keep provenance but make failed content visibly non-assertive."""
    reason = verdict.reason or "claim_not_verified"
    if (
        len(reason) > 96
        or (
            reason not in _SAFE_GATE_REASONS
            and not re.fullmatch(r"[A-Za-z0-9_.:-]+", reason)
        )
    ):
        reason = "claim_rejected"
    return claim.model_copy(
        update={
            "support_type": "qualified",
            "qualified_reason": reason,
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


def _fail_closed_verdicts(
    claims: Sequence[FinalClaim],
    *,
    reason: str = "claim_verifier_unavailable_or_invalid",
) -> dict[str, ClaimVerdict]:
    return {
        claim.claim_id: ClaimVerdict(
            claim_id=claim.claim_id,
            supported=False,
            reason=reason,
        )
        for claim in claims
    }


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
    return provider_response_content(response)


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
