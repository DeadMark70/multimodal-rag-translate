"""Thread-safe, provenance-preserving storage for Agentic v9 evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from threading import RLock
from typing import Any, Iterable, Mapping
import unicodedata

from data_base.agentic_v9.schemas import EvidencePacket


_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class EvidenceIdentity:
    """Source-local identity; content hashing is only a locator fallback."""

    doc_id: str
    chunk_id: str | None
    parent_id: str | None
    pdf_page_index: int | None
    asset_id: str | None
    normalized_hash: str | None = None

    @classmethod
    def from_packet(cls, packet: EvidencePacket) -> EvidenceIdentity:
        """Build identity from this packet's own provenance, never list metadata."""
        source = packet.source
        page = packet.locator.pdf_page_index
        has_precise_location = any(
            (
                source.chunk_id,
                source.parent_id,
                page is not None,
                source.asset_id,
            )
        )
        return cls(
            doc_id=source.doc_id,
            chunk_id=source.chunk_id,
            parent_id=source.parent_id,
            pdf_page_index=page,
            asset_id=source.asset_id,
            normalized_hash=None
            if has_precise_location
            else _normalized_statement_hash(packet.statement),
        )


@dataclass(frozen=True)
class EvidencePoolItem:
    """One retrieved evidence item and the retrieval data attached to that item."""

    packet: EvidencePacket
    metadata: Mapping[str, Any]
    retrieval_scores: Mapping[str, float]

    def __init__(
        self,
        packet: EvidencePacket,
        *,
        metadata: Mapping[str, Any] | None = None,
        retrieval_scores: Mapping[str, float] | None = None,
    ) -> None:
        object.__setattr__(self, "packet", packet)
        object.__setattr__(self, "metadata", deepcopy(dict(metadata or {})))
        object.__setattr__(
            self,
            "retrieval_scores",
            deepcopy(dict(retrieval_scores or {})),
        )

    @property
    def identity(self) -> EvidenceIdentity:
        """Return this item's provenance identity."""
        return EvidenceIdentity.from_packet(self.packet)


@dataclass(frozen=True)
class EvidencePoolEntry:
    """Canonical item for an identity plus every idempotently merged observation."""

    identity: EvidenceIdentity
    observations: tuple[EvidencePoolItem, ...]

    @property
    def item(self) -> EvidencePoolItem:
        """Return the stable canonical observation for this identity."""
        return self.observations[0]

    @property
    def packet(self) -> EvidencePacket:
        """Expose the canonical evidence packet."""
        return self.item.packet

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Expose canonical packet metadata without fabricating source fields."""
        return self.item.metadata

    @property
    def retrieval_scores(self) -> Mapping[str, float]:
        """Expose canonical retrieval scores exactly as supplied for that item."""
        return self.item.retrieval_scores


class EvidencePool:
    """Store v9 retrieved evidence with explicit lifecycle sets.

    Lifecycle transitions are deliberately explicit: adding evidence only marks it
    retrieved. Acceptance, context packing, final use, and rejection are separate
    caller-owned decisions so observability cannot infer an answer from retrieval.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._entries: dict[EvidenceIdentity, EvidencePoolEntry] = {}
        self._identity_by_evidence_id: dict[str, EvidenceIdentity] = {}
        self._accepted: set[EvidenceIdentity] = set()
        self._packed: set[EvidenceIdentity] = set()
        self._used: set[EvidenceIdentity] = set()
        self._rejected: dict[EvidenceIdentity, str | None] = {}

    def add(
        self,
        packet: EvidencePacket | EvidencePoolItem,
        *,
        metadata: Mapping[str, Any] | None = None,
        retrieval_scores: Mapping[str, float] | None = None,
    ) -> EvidencePoolEntry:
        """Add one retrieved item, merging source-identical observations safely."""
        item = (
            packet
            if isinstance(packet, EvidencePoolItem)
            else EvidencePoolItem(
                packet,
                metadata=metadata,
                retrieval_scores=retrieval_scores,
            )
        )
        identity = item.identity
        with self._lock:
            current = self._entries.get(identity)
            if current is None:
                entry = EvidencePoolEntry(identity=identity, observations=(item,))
            else:
                observations = _merge_observations(current.observations, item)
                entry = (
                    current
                    if observations == current.observations
                    else EvidencePoolEntry(identity=identity, observations=observations)
                )
            self._entries[identity] = entry
            for observation in entry.observations:
                self._identity_by_evidence_id[observation.packet.evidence_id] = identity
            return entry

    def add_many(
        self, items: Iterable[EvidencePacket | EvidencePoolItem]
    ) -> tuple[EvidencePoolEntry, ...]:
        """Add items and return their canonical entries in caller order."""
        return tuple(self.add(item) for item in items)

    def get(self, evidence_id: str) -> EvidencePoolEntry:
        """Look up an item by any observed evidence ID."""
        with self._lock:
            return self._entries[self._identity_for(evidence_id)]

    @property
    def retrieved_ids(self) -> tuple[str, ...]:
        """Canonical evidence IDs for all retrieved source identities."""
        with self._lock:
            return self._ids_for(self._entries)

    @property
    def accepted_ids(self) -> tuple[str, ...]:
        """Canonical evidence IDs explicitly accepted for use."""
        with self._lock:
            return self._ids_for(self._accepted)

    @property
    def packed_ids(self) -> tuple[str, ...]:
        """Canonical evidence IDs explicitly packed into a context."""
        with self._lock:
            return self._ids_for(self._packed)

    @property
    def used_ids(self) -> tuple[str, ...]:
        """Canonical evidence IDs explicitly used by a final claim."""
        with self._lock:
            return self._ids_for(self._used)

    @property
    def rejected_ids(self) -> tuple[str, ...]:
        """Canonical evidence IDs explicitly rejected from the pool."""
        with self._lock:
            return self._ids_for(self._rejected)

    def mark_accepted(self, evidence_id: str) -> None:
        """Record an explicit acceptance without changing packing or usage."""
        with self._lock:
            identity = self._identity_for(evidence_id)
            self._ensure_not_rejected(identity)
            self._accepted.add(identity)

    def mark_packed(self, evidence_id: str) -> None:
        """Record an explicit context-packing decision."""
        with self._lock:
            identity = self._identity_for(evidence_id)
            self._ensure_not_rejected(identity)
            self._packed.add(identity)

    def mark_used(self, evidence_id: str) -> None:
        """Record an explicit final-claim use decision."""
        with self._lock:
            identity = self._identity_for(evidence_id)
            self._ensure_not_rejected(identity)
            self._used.add(identity)

    def mark_rejected(self, evidence_id: str, *, reason: str | None = None) -> None:
        """Reject evidence and remove it from non-terminal lifecycle sets."""
        with self._lock:
            identity = self._identity_for(evidence_id)
            self._accepted.discard(identity)
            self._packed.discard(identity)
            self._used.discard(identity)
            self._rejected[identity] = reason

    def rejection_reason(self, evidence_id: str) -> str | None:
        """Return the persisted rejection reason, if this item was rejected."""
        with self._lock:
            return self._rejected.get(self._identity_for(evidence_id))

    def _identity_for(self, evidence_id: str) -> EvidenceIdentity:
        try:
            return self._identity_by_evidence_id[evidence_id]
        except KeyError as exc:
            raise KeyError(f"unknown evidence ID: {evidence_id}") from exc

    def _ids_for(
        self, identities: Iterable[EvidenceIdentity] | Mapping[EvidenceIdentity, Any]
    ) -> tuple[str, ...]:
        return tuple(
            sorted(self._entries[identity].packet.evidence_id for identity in identities)
        )

    def _ensure_not_rejected(self, identity: EvidenceIdentity) -> None:
        if identity in self._rejected:
            raise ValueError("rejected evidence cannot be accepted, packed, or used")


def _merge_observations(
    observations: tuple[EvidencePoolItem, ...], item: EvidencePoolItem
) -> tuple[EvidencePoolItem, ...]:
    """Deduplicate exact observations and sort all variants deterministically."""
    candidates = {*(_observation_key(value) for value in observations)}
    item_key = _observation_key(item)
    if item_key in candidates:
        return observations
    return tuple(sorted((*observations, item), key=_observation_key))


def _observation_key(item: EvidencePoolItem) -> str:
    """Serialize observable input into an order-independent canonical key."""
    return _stable_json(
        {
            "packet": item.packet.model_dump(mode="json"),
            "metadata": item.metadata,
            "retrieval_scores": item.retrieval_scores,
        }
    )


def _normalized_statement_hash(statement: str) -> str:
    """Hash normalized text only for packets without a precise source locator."""
    normalized = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", statement)).strip()
    return sha256(normalized.casefold().encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    """Create deterministic sort material for metadata-bearing observations."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


__all__ = [
    "EvidenceIdentity",
    "EvidencePool",
    "EvidencePoolEntry",
    "EvidencePoolItem",
]
