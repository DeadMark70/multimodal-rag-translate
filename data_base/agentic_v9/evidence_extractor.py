"""Deterministic-first, source-bound evidence packet extraction for v9."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import json
import re
from typing import Any, Literal

from core.prompt_loader import format_agentic_rag_prompt
from data_base.agentic_v9.evidence_pool import EvidencePoolEntry, EvidencePoolItem
from data_base.agentic_v9.provider_boundary import provider_response_text
from data_base.agentic_v9.schemas import (
    BudgetExceededError,
    EvidencePacket,
    FinalClaim,
    LlmInvoker,
    QueryContract,
    RequiredSlot,
)
from data_base.agentic_v9.evidence_validator import (
    validate_deterministic_packet,
    validate_prose_packet,
)
from data_base.agentic_v9.slot_constraints import structured_locator_state


_NUMBER = re.compile(r"(?<![\w.])([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:\s*([A-Za-z%µ]+))?(?![\w.])")
_THEOREM_RANGE = re.compile(r"(?:Theorem\s+\d+\s*:\s*)?\b([A-Za-z])\s+(?:in|∈)\s*\[[^\]]+\]", re.IGNORECASE)
_FORMULA = re.compile(r"\b[A-Za-z][A-Za-z_]*\s*=\s*[^.\n]+")
_TABLE_ROW = re.compile(r"\bTable\s+\d+\s*\|[^.\n]+", re.IGNORECASE)
_ENUMERATION = re.compile(r"\(a\)[^.\n]*(?:;\s*\(b\)[^.\n]*)+", re.IGNORECASE)

QualificationStatus = Literal[
    "not_attempted",
    "deterministic",
    "provider_qualified",
    "no_match",
    "provider_failed",
    "invalid_response",
]


@dataclass(frozen=True)
class EvidenceQualificationOutcome:
    """Safe result of one deterministic-plus-provider qualification pass."""

    packets: tuple[EvidencePacket, ...]
    status: QualificationStatus
    failure_code: str | None = None
    provider_call_attempted: bool = False
    provider_response_received: bool = False
    qualification_unknown_source_id_count: int = 0
    qualification_unauthorized_source_slot_count: int = 0
    qualification_statement_not_verbatim_count: int = 0


@dataclass
class _QualificationRejectionCounts:
    unknown_source_id: int = 0
    unauthorized_source_slot: int = 0
    statement_not_verbatim: int = 0


class EvidenceExtractor:
    """Extract typed packets before one optional, budgeted prose-curation call."""

    def __init__(self, budgeted_invoker: LlmInvoker | None = None) -> None:
        self._invoker = budgeted_invoker
        self._final_claims: list[FinalClaim] = []

    @property
    def final_claims(self) -> tuple[FinalClaim, ...]:
        """Return high-risk prose reserved for the final-claim verifier."""
        return tuple(self._final_claims)

    def extract_deterministic(
        self,
        contract: QueryContract,
        pool: Iterable[EvidencePacket | EvidencePoolItem | EvidencePoolEntry],
    ) -> list[EvidencePacket]:
        """Return exact source-derived evidence without a model invocation."""
        items = _as_items(pool)
        packets: list[EvidencePacket] = []
        for slot in contract.required_slots:
            matching = _matched_items_for_slot(slot, items)
            packets.extend(extract_numeric_packets(slot=slot, items=matching))
            packets.extend(_extract_structured_packets(slot, matching))
        return _deduplicate_packets(packets)

    async def extract(
        self,
        contract: QueryContract,
        pool: Iterable[EvidencePacket | EvidencePoolItem | EvidencePoolEntry],
        *,
        repairs_complete: bool,
        question: str = "",
    ) -> list[EvidencePacket]:
        """Return packets while preserving the historical list-only interface."""
        outcome = await self.extract_with_outcome(
            contract,
            pool,
            repairs_complete=repairs_complete,
            question=question,
        )
        return list(outcome.packets)

    async def extract_with_outcome(
        self,
        contract: QueryContract,
        pool: Iterable[EvidencePacket | EvidencePoolItem | EvidencePoolEntry],
        *,
        repairs_complete: bool,
        question: str = "",
    ) -> EvidenceQualificationOutcome:
        """Finish deterministic work and report one provider pass honestly."""
        self._final_claims.clear()
        items = _as_items(pool)
        accepted = [
            item.packet for item in items if _is_prevalidated_packet(item.packet)
        ]
        packets = _deduplicate_packets(
            [*accepted, *self.extract_deterministic(contract, items)]
        )
        unresolved = [
            slot
            for slot in contract.required_slots
            if slot.slot_id not in _covered_slots(packets)
        ]
        eligible_ids_by_slot = {
            slot.slot_id: {
                item.packet.evidence_id
                for item in _eligible_items_for_slot(slot, items)
            }
            for slot in unresolved
        }
        curated_items = _items_for_evidence_ids(
            items,
            {
                evidence_id
                for evidence_ids in eligible_ids_by_slot.values()
                for evidence_id in evidence_ids
            },
        )
        if (
            not repairs_complete
            or not unresolved
            or not curated_items
            or self._invoker is None
        ):
            return EvidenceQualificationOutcome(
                packets=tuple(packets),
                status="deterministic" if packets else "not_attempted",
            )

        # A malformed batch is terminal: this stage never spends a second repair call.
        try:
            response = await self._curate_once(
                question=question or contract.intent,
                slots=unresolved,
                items=curated_items,
                eligible_ids_by_slot=eligible_ids_by_slot,
            )
        except BudgetExceededError:
            return EvidenceQualificationOutcome(
                packets=tuple(packets),
                status="not_attempted",
                failure_code="budget_not_admitted",
            )
        except Exception:
            return EvidenceQualificationOutcome(
                packets=tuple(packets),
                status="provider_failed",
                failure_code="provider_attempt_failed",
                provider_call_attempted=True,
            )

        rejection_counts = _QualificationRejectionCounts()
        curated = _parse_curated_packets(
            response,
            slots=unresolved,
            items=curated_items,
            eligible_ids_by_slot=eligible_ids_by_slot,
            final_claims=self._final_claims,
            rejection_counts=rejection_counts,
        )
        if curated is None:
            return EvidenceQualificationOutcome(
                packets=tuple(packets),
                status="invalid_response",
                failure_code="invalid_provider_response",
                provider_call_attempted=True,
                provider_response_received=True,
                qualification_unknown_source_id_count=(
                    rejection_counts.unknown_source_id
                ),
                qualification_unauthorized_source_slot_count=(
                    rejection_counts.unauthorized_source_slot
                ),
                qualification_statement_not_verbatim_count=(
                    rejection_counts.statement_not_verbatim
                ),
            )
        combined = tuple(_deduplicate_packets([*packets, *curated]))
        return EvidenceQualificationOutcome(
            packets=combined,
            status="provider_qualified" if curated else "no_match",
            provider_call_attempted=True,
            provider_response_received=True,
            qualification_unknown_source_id_count=(
                rejection_counts.unknown_source_id
            ),
            qualification_unauthorized_source_slot_count=(
                rejection_counts.unauthorized_source_slot
            ),
            qualification_statement_not_verbatim_count=(
                rejection_counts.statement_not_verbatim
            ),
        )

    async def _curate_once(
        self,
        *,
        question: str,
        slots: Sequence[RequiredSlot],
        items: Sequence[EvidencePoolItem],
        eligible_ids_by_slot: Mapping[str, set[str]],
    ) -> Any:
        source_evidence = _render_source_evidence(items, eligible_ids_by_slot)
        messages = [
            {
                "role": "user",
                "content": format_agentic_rag_prompt(
                    "evidence_extract",
                    question=question,
                    unresolved_slots=json.dumps(
                        [slot.model_dump(mode="json") for slot in slots], ensure_ascii=False
                    ),
                    source_evidence=source_evidence,
                ),
            }
        ]
        return await self._invoker.invoke(
            phase="evidence_extract",
            purpose="evidence_extraction",
            messages=messages,
        )


def extract_numeric_packets(
    *, slot: RequiredSlot, items: Iterable[EvidencePacket | EvidencePoolItem | EvidencePoolEntry]
) -> list[EvidencePacket]:
    """Extract exact numeric literals and adjacent units from slot-bound source text."""
    packets: list[EvidencePacket] = []
    for item in _items_for_slot(slot, _as_items(items)):
        source_text = _source_text(item)
        for index, match in enumerate(_NUMBER.finditer(source_text)):
            if source_text[max(0, match.start() - 8) : match.start()].casefold().endswith("table "):
                continue
            literal, unit = match.groups()
            result = validate_deterministic_packet(
                _derived_packet(
                    item.packet,
                    evidence_id=f"det:{item.packet.evidence_id}:number:{index}",
                    slot_ids=[slot.slot_id],
                    statement=source_text,
                    raw_value=Decimal(literal),
                    normalized_value=Decimal(literal),
                    unit=unit,
                    display_precision=_precision(literal),
                    extractor_version="v9-deterministic-1",
                ),
                source_text=source_text,
            )
            if result.packet is not None:
                packets.append(result.packet)
    return packets


def calculate_difference(
    *, slot: RequiredSlot, left: EvidencePacket, right: EvidencePacket
) -> EvidencePacket:
    """Create a calculated packet with direct, explicit premise evidence IDs."""
    if not _is_validated_direct_premise(left) or not _is_validated_direct_premise(right):
        raise ValueError(
            "difference calculation requires validated span-hashed direct premises"
        )
    left_value = left.raw_value if left.raw_value is not None else _first_numeric_value(left)
    right_value = right.raw_value if right.raw_value is not None else _first_numeric_value(right)
    if left_value is None or right_value is None:
        raise ValueError("difference calculation requires direct numeric premises")
    value = left_value - right_value
    scale = max(_decimal_places(left_value), _decimal_places(right_value))
    rendered = f"{value:.{scale}f}"
    return _derived_packet(
        left,
        evidence_id=f"calc:{left.evidence_id}:{right.evidence_id}:difference",
        slot_ids=[slot.slot_id],
        statement=f"Difference between {left.evidence_id} and {right.evidence_id}: {rendered}",
        support_type="calculated",
        raw_value=Decimal(rendered),
        normalized_value=Decimal(rendered),
        calculation_operation="difference",
        premise_evidence_ids=[left.evidence_id, right.evidence_id],
        display_precision=scale,
        extractor_version="v9-deterministic-1",
        validation_status="derived_non_evidence",
    )


def _extract_structured_packets(
    slot: RequiredSlot, items: Sequence[EvidencePoolItem]
) -> list[EvidencePacket]:
    packets: list[EvidencePacket] = []
    patterns = (_THEOREM_RANGE, _FORMULA, _TABLE_ROW, _ENUMERATION)
    for item in items:
        source_text = _source_text(item)
        for pattern_index, pattern in enumerate(patterns):
            for match_index, match in enumerate(pattern.finditer(source_text)):
                result = validate_deterministic_packet(
                    _derived_packet(
                        item.packet,
                        evidence_id=(
                            f"det:{item.packet.evidence_id}:structured:{pattern_index}:{match_index}"
                        ),
                        slot_ids=[slot.slot_id],
                        statement=match.group(0).strip(),
                        extractor_version="v9-deterministic-1",
                    ),
                    source_text=source_text,
                )
                if result.packet is not None:
                    packets.append(result.packet)
    return packets


def _as_items(
    pool: Iterable[EvidencePacket | EvidencePoolItem | EvidencePoolEntry],
) -> list[EvidencePoolItem]:
    result: list[EvidencePoolItem] = []
    for value in pool:
        if isinstance(value, EvidencePoolItem):
            result.append(value)
        elif isinstance(value, EvidencePoolEntry):
            result.append(value.item)
        elif isinstance(value, EvidencePacket):
            result.append(EvidencePoolItem(value))
        else:
            raise TypeError("evidence extraction requires typed v9 evidence pool items")
    return result


def _items_for_slot(slot: RequiredSlot, items: Sequence[EvidencePoolItem]) -> list[EvidencePoolItem]:
    return [item for item in items if slot.slot_id in item.packet.slot_ids]


def _matched_items_for_slot(
    slot: RequiredSlot, items: Sequence[EvidencePoolItem]
) -> list[EvidencePoolItem]:
    """Permit deterministic extraction only from a verified structured locator."""
    matching = _items_for_slot(slot, items)
    if not any(_locator_state(slot, item) == "matched" for item in matching):
        return []
    return [item for item in matching if _locator_state(slot, item) == "matched"]


def _eligible_items_for_slot(
    slot: RequiredSlot, items: Sequence[EvidencePoolItem]
) -> list[EvidencePoolItem]:
    """Keep ordinary and unavailable candidates, never contradicting locators."""
    return [
        item
        for item in _items_for_slot(slot, items)
        if _locator_state(slot, item) != "mismatched"
    ]


def _locator_state(slot: RequiredSlot, item: EvidencePoolItem) -> str:
    return structured_locator_state(slot.locator_hints, _locator_metadata(item))


def _locator_metadata(item: EvidencePoolItem) -> dict[str, Any]:
    """Merge retrieved locator metadata with the packet's persisted locator."""
    metadata = dict(item.packet.locator.model_dump(mode="python"))
    metadata.update(
        {
            key: value
            for key, value in item.metadata.items()
            if key in {"section", "table_id", "figure_id", "formula_id"}
            and value is not None
        }
    )
    return metadata


def _items_for_evidence_ids(
    items: Sequence[EvidencePoolItem], evidence_ids: set[str]
) -> list[EvidencePoolItem]:
    return [item for item in items if item.packet.evidence_id in evidence_ids]


def _source_text(item: EvidencePoolItem) -> str:
    for key in ("text", "content", "raw_text"):
        candidate = item.metadata.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return item.packet.statement


def _derived_packet(source: EvidencePacket, **updates: Any) -> EvidencePacket:
    defaults: dict[str, Any] = {
        "support_type": "direct",
        "raw_value": None,
        "normalized_value": None,
        "unit": None,
        "calculation_operation": None,
        "premise_evidence_ids": [],
        "display_precision": None,
        "rounding_mode": None,
        "prompt_version": None,
    }
    defaults.update(updates)
    return source.model_copy(update=defaults)


def _precision(literal: str) -> int:
    return len(literal.partition(".")[2])


def _decimal_places(value: Decimal) -> int:
    return max(0, -value.as_tuple().exponent)


def _first_numeric_value(packet: EvidencePacket) -> Decimal | None:
    match = _NUMBER.search(packet.statement)
    return Decimal(match.group(1)) if match else None


def _is_validated_direct_premise(packet: EvidencePacket) -> bool:
    return (
        packet.support_type == "direct"
        and packet.validation_status in {"deterministic_valid", "quote_bound"}
        and packet.source.source_span_hash is not None
    )


def _covered_slots(packets: Iterable[EvidencePacket]) -> set[str]:
    return {slot_id for packet in packets for slot_id in packet.slot_ids}


def _render_source_evidence(
    items: Sequence[EvidencePoolItem],
    eligible_ids_by_slot: Mapping[str, set[str]],
) -> str:
    """Render source spans with their slot-specific authorization boundary."""
    return "\n".join(
        (
            f"{item.packet.evidence_id} [eligible slots: {','.join(slot_ids)}]: "
            f"{_source_text(item)}"
        )
        for item in items
        if (
            slot_ids := [
                slot_id
                for slot_id, evidence_ids in eligible_ids_by_slot.items()
                if item.packet.evidence_id in evidence_ids
            ]
        )
    )


def _is_prevalidated_packet(packet: EvidencePacket) -> bool:
    if packet.support_type == "contradictory":
        return False
    if packet.validation_status in {"quote_bound", "derived_non_evidence"}:
        return True
    return packet.validation_status == "deterministic_valid" and (
        packet.extractor_version is not None
        or packet.source.source_span_hash is not None
    )


def _parse_curated_packets(
    response: Any,
    *,
    slots: Sequence[RequiredSlot],
    items: Sequence[EvidencePoolItem],
    eligible_ids_by_slot: Mapping[str, set[str]],
    rejection_counts: _QualificationRejectionCounts,
    final_claims: list[FinalClaim] | None = None,
) -> list[EvidencePacket] | None:
    content = getattr(response, "content", response)
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    if not isinstance(content, (str, Mapping)):
        content = provider_response_text(response)
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return None
    if not isinstance(content, Mapping) or set(content) != {"packets"}:
        return None
    raw_packets = content.get("packets")
    if not isinstance(raw_packets, list):
        return None
    valid_slots = {slot.slot_id for slot in slots}
    by_id = {item.packet.evidence_id: item for item in items}
    packets: list[EvidencePacket] = []
    for raw in raw_packets:
        if not isinstance(raw, Mapping) or set(raw) != {"source_evidence_id", "slot_ids", "statement"}:
            continue
        source_id, slot_ids, statement = (
            raw["source_evidence_id"], raw["slot_ids"], raw["statement"]
        )
        if (
            not isinstance(source_id, str)
            or not isinstance(statement, str)
            or not statement.strip()
            or not isinstance(slot_ids, list)
            or not all(isinstance(slot_id, str) for slot_id in slot_ids)
            or not slot_ids
        ):
            continue
        if source_id not in by_id:
            rejection_counts.unknown_source_id += 1
            continue
        if not set(slot_ids).issubset(valid_slots) or any(
            source_id not in eligible_ids_by_slot.get(slot_id, set())
            for slot_id in slot_ids
        ):
            rejection_counts.unauthorized_source_slot += 1
            continue
        item = by_id[source_id]
        result = validate_prose_packet(
            _derived_packet(
                item.packet,
                evidence_id=f"curated:{source_id}:{':'.join(slot_ids)}",
                slot_ids=list(dict.fromkeys(slot_ids)),
                statement=statement,
                extractor_version="v9-prose-curator-1",
                prompt_version="1",
            ),
            source=item.packet,
            source_text=_source_text(item),
        )
        if result.packet is not None:
            packets.append(result.packet)
        elif result.final_claim is not None and final_claims is not None:
            final_claims.append(result.final_claim)
        elif result.reason == "statement_is_not_a_verbatim_source_span":
            rejection_counts.statement_not_verbatim += 1
    return packets


def _deduplicate_packets(packets: Iterable[EvidencePacket]) -> list[EvidencePacket]:
    unique: dict[str, EvidencePacket] = {}
    for packet in packets:
        unique.setdefault(packet.evidence_id, packet)
    return list(unique.values())


async def extract_evidence_packets(
    contract: QueryContract,
    pool: Iterable[EvidencePacket | EvidencePoolItem | EvidencePoolEntry],
    budget: LlmInvoker | None = None,
    *,
    repairs_complete: bool = True,
    question: str = "",
) -> list[EvidencePacket]:
    """Convenience entry point with an injected, already-budgeted invoker."""
    return await EvidenceExtractor(budget).extract(
        contract, pool, repairs_complete=repairs_complete, question=question
    )


__all__ = [
    "EvidenceQualificationOutcome",
    "EvidenceExtractor",
    "calculate_difference",
    "extract_evidence_packets",
    "extract_numeric_packets",
]
