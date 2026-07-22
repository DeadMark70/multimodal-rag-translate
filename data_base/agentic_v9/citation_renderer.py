"""Stable, provenance-only citations for verified Agentic v9 claims."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from data_base.agentic_v9.schemas import EvidencePacket, FinalClaim


def render_evidence_citation(
    packet: EvidencePacket, *, citation_format_version: str = "1"
) -> str:
    """Render one packet locator without exposing its untrusted source text."""
    if citation_format_version != "1":
        raise ValueError(
            f"unsupported citation format version: {citation_format_version}"
        )
    document = packet.source.document_name or packet.source.doc_id
    locations: list[str] = []
    if packet.locator.printed_page_label:
        locations.append(f"p.{packet.locator.printed_page_label}")
    elif packet.locator.pdf_page_index is not None:
        locations.append(f"p.{packet.locator.pdf_page_index + 1}")
    if packet.locator.section:
        locations.append(packet.locator.section)
    if packet.locator.table_id:
        locations.append(packet.locator.table_id)
    if packet.locator.figure_id:
        locations.append(packet.locator.figure_id)
    locator = ", ".join(locations)
    return f"[v1:{document}{' ' + locator if locator else ''}; {packet.evidence_id}]"


def render_claim_citations(
    claim: FinalClaim,
    packets_by_id: Mapping[str, EvidencePacket],
    *,
    citation_format_version: str = "1",
) -> str:
    """Render citations in claim-ID order and fail closed for missing packets."""
    evidence_ids = list(
        dict.fromkeys([*claim.evidence_ids, *claim.premise_evidence_ids])
    )
    citations = [
        render_evidence_citation(
            packets_by_id[evidence_id], citation_format_version=citation_format_version
        )
        for evidence_id in evidence_ids
        if evidence_id in packets_by_id
    ]
    return " ".join(citations)


def render_verified_answer(
    claims: Iterable[FinalClaim],
    packets: Iterable[EvidencePacket],
    *,
    citation_format_version: str = "1",
) -> str:
    """Project only verified claims into a deterministic cited answer."""
    packets_by_id = {packet.evidence_id: packet for packet in packets}
    lines: list[str] = []
    for claim in claims:
        statement = claim.statement
        if claim.qualified_reason:
            statement = f"{statement} (Qualified: {claim.qualified_reason})"
        citations = render_claim_citations(
            claim,
            packets_by_id,
            citation_format_version=citation_format_version,
        )
        lines.append(f"{statement}{' ' + citations if citations else ''}")
    return "\n".join(lines)


__all__ = [
    "render_claim_citations",
    "render_evidence_citation",
    "render_verified_answer",
]
