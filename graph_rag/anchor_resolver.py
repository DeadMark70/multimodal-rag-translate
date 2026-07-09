from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.documents import Document

from data_base.document_metadata import matches_document_id
from graph_rag.schemas import EvidenceAnchor

ResolutionStatus = Literal["resolved", "fuzzy_resolved", "unresolved", "stale"]
VerificationStatus = Literal["quote_match", "quote_mismatch", "hash_mismatch", "not_checked"]


class ChunkLookup(Protocol):
    def by_chunk_id(self, user_id: str, chunk_id: str) -> Document | None: ...

    def by_doc_and_index(self, user_id: str, doc_id: str, chunk_index: int) -> Document | None: ...

    def by_chunk_hash(self, user_id: str, doc_id: str, chunk_hash: str) -> Document | None: ...

    def fuzzy_by_quote(self, user_id: str, doc_id: str, quote: str) -> Document | None: ...


@dataclass(frozen=True, slots=True)
class AnchorResolutionResult:
    anchor: EvidenceAnchor
    document: Document | None
    resolution_status: ResolutionStatus
    verification_status: VerificationStatus
    reason: str


class ChunkAnchorResolver:
    def __init__(self, lookup: ChunkLookup) -> None:
        self._lookup = lookup

    def resolve(self, user_id: str, anchor: EvidenceAnchor) -> AnchorResolutionResult:
        saw_doc_id_mismatch = False

        if anchor.chunk_id:
            document = self._lookup.by_chunk_id(user_id, anchor.chunk_id)
            if document is not None:
                if not _document_matches_anchor(anchor, document):
                    saw_doc_id_mismatch = True
                else:
                    if anchor.chunk_hash and document.metadata.get("chunk_hash") != anchor.chunk_hash:
                        return AnchorResolutionResult(
                            anchor=anchor,
                            document=document,
                            resolution_status="stale",
                            verification_status="hash_mismatch",
                            reason="chunk_id_hash_mismatch",
                        )
                    return AnchorResolutionResult(
                        anchor=anchor,
                        document=document,
                        resolution_status="resolved",
                        verification_status=_verification_status(anchor, document),
                        reason="chunk_id",
                    )

        if anchor.chunk_index is not None:
            document = self._lookup.by_doc_and_index(user_id, anchor.doc_id, anchor.chunk_index)
            if document is not None:
                if not _document_matches_anchor(anchor, document):
                    saw_doc_id_mismatch = True
                else:
                    return AnchorResolutionResult(
                        anchor=anchor,
                        document=document,
                        resolution_status="resolved",
                        verification_status=_verification_status(anchor, document),
                        reason="doc_id_chunk_index",
                    )

        if anchor.chunk_hash:
            document = self._lookup.by_chunk_hash(user_id, anchor.doc_id, anchor.chunk_hash)
            if document is not None:
                if not _document_matches_anchor(anchor, document):
                    saw_doc_id_mismatch = True
                else:
                    return AnchorResolutionResult(
                        anchor=anchor,
                        document=document,
                        resolution_status="resolved",
                        verification_status=_verification_status(anchor, document),
                        reason="chunk_hash",
                    )

        if anchor.quote:
            document = self._lookup.fuzzy_by_quote(user_id, anchor.doc_id, anchor.quote)
            if document is not None:
                if not _document_matches_anchor(anchor, document):
                    saw_doc_id_mismatch = True
                else:
                    return AnchorResolutionResult(
                        anchor=anchor,
                        document=document,
                        resolution_status="fuzzy_resolved",
                        verification_status="quote_match",
                        reason="fuzzy_quote",
                    )

        if saw_doc_id_mismatch:
            return AnchorResolutionResult(
                anchor=anchor,
                document=None,
                resolution_status="unresolved",
                verification_status="not_checked",
                reason="doc_id_mismatch",
            )

        return AnchorResolutionResult(
            anchor=anchor,
            document=None,
            resolution_status="unresolved",
            verification_status="not_checked",
            reason="no_matching_chunk",
        )


def _verification_status(anchor: EvidenceAnchor, document: Document) -> VerificationStatus:
    if not anchor.quote:
        return "not_checked"
    if anchor.quote in document.page_content:
        return "quote_match"
    return "quote_mismatch"


def _document_matches_anchor(anchor: EvidenceAnchor, document: Document) -> bool:
    return matches_document_id(document.metadata, anchor.doc_id)
