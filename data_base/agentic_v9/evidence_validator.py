"""Deterministic source binding for curated Agentic v9 prose evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
import unicodedata
from typing import Literal

from data_base.agentic_v9.schemas import EvidencePacket, FinalClaim


ValidationStatus = Literal["deterministic_valid", "quote_bound", "invalid"]

_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])")
_HIGH_RISK = re.compile(
    r"\b(?:"
    r"outperform(?:s|ed|ing)?|better|best|highest|first|sota|"
    r"state[ -]of[ -]the[ -]art|caus(?:e|es|ed|al|ally)|safe|robust"
    r")\b",
    re.IGNORECASE,
)
_COMPARATIVE = re.compile(
    r"\b(?:outperform(?:s|ed|ing)?|better|best|highest)\b", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class EvidenceValidationResult:
    """The accepted packet or claim-only handoff for one submitted statement."""

    status: ValidationStatus
    packet: EvidencePacket | None = None
    final_claim: FinalClaim | None = None
    reason: str | None = None


def normalize_source_span(value: str) -> str:
    """Apply only Unicode and whitespace normalization before span matching."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def source_span_hash(value: str) -> str:
    """Return the stable SHA-256 identity of a normalized verbatim source span."""
    return sha256(normalize_source_span(value).encode("utf-8")).hexdigest()


def validate_deterministic_packet(
    packet: EvidencePacket, *, source_text: str
) -> EvidenceValidationResult:
    """Bind a deterministic source extraction to its exact source span."""
    span = _source_span(packet.statement, source_text)
    if span is None:
        return EvidenceValidationResult(
            status="invalid", reason="deterministic_statement_is_not_source_bound"
        )
    return EvidenceValidationResult(
        status="deterministic_valid",
        packet=_with_source_span(packet, span, validation_status="deterministic_valid"),
    )


def validate_prose_packet(
    candidate: EvidencePacket,
    *,
    source: EvidencePacket,
    source_text: str,
) -> EvidenceValidationResult:
    """Accept only source-scope-preserving extractive prose, never paraphrases."""
    if not _same_source_identity(candidate, source):
        return EvidenceValidationResult(status="invalid", reason="source_identity_rewritten")
    if candidate.scope != source.scope:
        return EvidenceValidationResult(status="invalid", reason="source_scope_rewritten")

    span = _source_span(candidate.statement, source_text)
    if span is None:
        return EvidenceValidationResult(
            status="invalid", reason="statement_is_not_a_verbatim_source_span"
        )
    if not set(_NUMBER.findall(normalize_source_span(candidate.statement))).issubset(
        _NUMBER.findall(span)
    ):
        return EvidenceValidationResult(status="invalid", reason="new_number_in_statement")

    bound = _with_source_span(candidate, span, validation_status="quote_bound")
    if _HIGH_RISK.search(bound.statement):
        return EvidenceValidationResult(
            status="quote_bound",
            final_claim=_high_risk_claim(bound, premise_evidence_id=source.evidence_id),
            reason="high_risk_abstraction_requires_final_claim",
        )
    return EvidenceValidationResult(status="quote_bound", packet=bound)


def _source_span(statement: str, source_text: str) -> str | None:
    normalized_statement = normalize_source_span(statement)
    normalized_source = normalize_source_span(source_text)
    if not normalized_statement or normalized_statement not in normalized_source:
        return None
    return normalized_statement


def _same_source_identity(candidate: EvidencePacket, source: EvidencePacket) -> bool:
    candidate_source = candidate.source.model_dump(exclude={"source_span_hash"})
    source_source = source.source.model_dump(exclude={"source_span_hash"})
    return candidate_source == source_source


def _with_source_span(
    packet: EvidencePacket, span: str, *, validation_status: ValidationStatus
) -> EvidencePacket:
    source = packet.source.model_copy(update={"source_span_hash": source_span_hash(span)})
    return packet.model_copy(
        update={"source": source, "validation_status": validation_status}
    )


def _high_risk_claim(packet: EvidencePacket, *, premise_evidence_id: str) -> FinalClaim:
    support_type = "comparative_inference" if _COMPARATIVE.search(packet.statement) else "qualified"
    return FinalClaim(
        claim_id=f"claim:{packet.evidence_id}",
        statement=packet.statement,
        support_type=support_type,
        premise_evidence_ids=[premise_evidence_id],
        qualified_reason="high_risk_abstraction_requires_claim_verification",
    )


__all__ = [
    "EvidenceValidationResult",
    "normalize_source_span",
    "source_span_hash",
    "validate_deterministic_packet",
    "validate_prose_packet",
]
