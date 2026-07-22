"""Typed campaign-to-agentic adapters.

This module deliberately translates identifiers and projections only.  It does
not select retrieval, prompt, source, or execution policy: those remain owned
by the versioned execution core and its injected adapters.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from langchain_core.documents import Document

from data_base.agentic_v9.schemas import EvidencePacket, FinalAnswerResult

AgenticExecutionVersion = Literal["v8", "v9"]


def campaign_execution_identity(
    identity: str, agentic_execution_version: AgenticExecutionVersion
) -> tuple[str, str, AgenticExecutionVersion]:
    """Map a public campaign identity to a core mode and explicit version."""
    normalized = str(identity).strip().lower()
    aliases: dict[str, tuple[str, AgenticExecutionVersion]] = {
        "naive": ("naive", "v8"),
        "naive-baseline": ("naive", "v8"),
        "agentic": ("agentic", agentic_execution_version),
        "agentic-v8": ("agentic", "v8"),
        "v8": ("agentic", "v8"),
        "agentic-v9": ("agentic", "v9"),
        "v9": ("agentic", "v9"),
        "agentic-v9-shadow": ("agentic", "v9"),
    }
    try:
        core_mode, version = aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported campaign execution identity: {identity}") from exc
    return normalized, core_mode, version


def used_evidence_documents(
    evidence_packets: Iterable[EvidencePacket],
    final_answer: FinalAnswerResult,
) -> list[Document]:
    """Project only evidence explicitly used by the verified final answer.

    RAGAS contexts must never be populated from merely retrieved or packed
    evidence.  Exact duplicate source spans collapse deterministically while
    preserving the final answer's cited evidence order.
    """
    by_id = {packet.evidence_id: packet for packet in evidence_packets}
    documents: list[Document] = []
    seen: set[tuple[str, str, str]] = set()
    for evidence_id in final_answer.used_evidence_ids:
        packet = by_id.get(evidence_id)
        if packet is None:
            continue
        identity = (
            packet.source.doc_id,
            packet.source.chunk_id or "",
            packet.statement.strip(),
        )
        if not identity[2] or identity in seen:
            continue
        seen.add(identity)
        documents.append(
            Document(
                page_content=packet.statement,
                metadata={
                    "evidence_id": packet.evidence_id,
                    "doc_id": packet.source.doc_id,
                    "chunk_id": packet.source.chunk_id,
                    "used_in_final_answer": True,
                },
            )
        )
    return documents


__all__ = ["campaign_execution_identity", "used_evidence_documents"]
