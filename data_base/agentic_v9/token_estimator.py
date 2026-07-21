"""Deterministic, conservative token estimates for Agentic v9 evidence."""

from __future__ import annotations

import re

from data_base.agentic_v9.schemas import EvidencePacket


_TOKEN_UNITS = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|"
    r"[A-Za-z]+(?:['-][A-Za-z]+)*|"
    r"\d+(?:[.,]\d+)*|"
    r"[^\s]"
)


class TokenEstimator:
    """Estimate rendered v9 evidence without calling a provider tokenizer.

    The estimator deliberately counts every CJK character, word, numeric unit,
    and punctuation token.  This is deterministic and conservative enough to
    maintain the v9 input ceiling when a provider tokenizer is unavailable.
    """

    def estimate_text(self, text: str) -> int:
        """Return a non-negative estimate for text that will enter a prompt."""
        if not text:
            return 0
        return len(_TOKEN_UNITS.findall(text))

    def estimate_packet(self, packet: EvidencePacket) -> int:
        """Return the estimate for one indivisible rendered evidence packet."""
        return self.estimate_text(render_evidence_packet(packet))


def render_evidence_packet(packet: EvidencePacket) -> str:
    """Render provenance with the statement as one indivisible evidence unit."""
    slots = ",".join(packet.slot_ids)
    locator = _render_locator(packet)
    return (
        f"[{packet.evidence_id}] slots={slots} source={packet.source.doc_id} "
        f"locator={locator}\n{packet.statement}"
    )


def _render_locator(packet: EvidencePacket) -> str:
    locator = packet.locator
    parts: list[str] = []
    if locator.pdf_page_index is not None:
        parts.append(f"pdf_page={locator.pdf_page_index}")
    if locator.printed_page_label:
        parts.append(f"page={locator.printed_page_label}")
    if locator.section:
        parts.append(f"section={locator.section}")
    if locator.table_id:
        parts.append(f"table={locator.table_id}")
    if locator.figure_id:
        parts.append(f"figure={locator.figure_id}")
    if locator.bbox:
        parts.append("bbox=" + ",".join(str(value) for value in locator.bbox))
    return ";".join(parts)


def estimate_text_tokens(text: str) -> int:
    """Convenience entry point for callers that do not need an estimator object."""
    return TokenEstimator().estimate_text(text)


def estimate_evidence_packet_tokens(packet: EvidencePacket) -> int:
    """Convenience entry point for one atomic packet estimate."""
    return TokenEstimator().estimate_packet(packet)
