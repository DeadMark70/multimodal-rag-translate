"""Deterministic-first, source-bound evidence packet extraction for v9."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
import json
import re
from typing import Any

from core.prompt_loader import format_agentic_rag_prompt
from data_base.agentic_v9.evidence_pool import EvidencePoolEntry, EvidencePoolItem
from data_base.agentic_v9.schemas import (
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


_NUMBER = re.compile(r"(?<![\w.])([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:\s*([A-Za-z%µ]+))?(?![\w.])")
_THEOREM_RANGE = re.compile(r"(?:Theorem\s+\d+\s*:\s*)?\b([A-Za-z])\s+(?:in|∈)\s*\[[^\]]+\]", re.IGNORECASE)
_FORMULA = re.compile(r"\b[A-Za-z][A-Za-z_]*\s*=\s*[^.\n]+")
_TABLE_ROW = re.compile(r"\bTable\s+\d+\s*\|[^.\n]+", re.IGNORECASE)
_ENUMERATION = re.compile(r"\(a\)[^.\n]*(?:;\s*\(b\)[^.\n]*)+", re.IGNORECASE)
_COMPARATIVE = re.compile(r"\b(compare|comparison|versus|\bvs\.?|better|outperform|which performs)\b", re.IGNORECASE)


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
            matching = _items_for_slot(slot, items)
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
        """Finish deterministic work, then curate the remaining prose slots once."""
        self._final_claims.clear()
        items = _as_items(pool)
        packets = self.extract_deterministic(contract, items)
        unresolved = [
            slot
            for slot in contract.required_slots
            if slot.slot_id not in _covered_slots(packets)
            and not _COMPARATIVE.search(slot.description)
        ]
        if not repairs_complete or not unresolved or self._invoker is None:
            return packets

        # A malformed batch is terminal: this stage never spends a second repair call.
        curated = await self._curate_once(
            question=question or contract.intent,
            slots=unresolved,
            items=items,
        )
        return _deduplicate_packets([*packets, *curated])

    async def _curate_once(
        self,
        *,
        question: str,
        slots: Sequence[RequiredSlot],
        items: Sequence[EvidencePoolItem],
    ) -> list[EvidencePacket]:
        source_evidence = _render_source_evidence(items)
        messages = [
            {
                "role": "system",
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
        try:
            response = await self._invoker.invoke(
                phase="evidence_extract",
                purpose="evidence_extraction",
                messages=messages,
            )
        except Exception:
            return []
        return _parse_curated_packets(
            response,
            slots=slots,
            items=items,
            final_claims=self._final_claims,
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


def _covered_slots(packets: Iterable[EvidencePacket]) -> set[str]:
    return {slot_id for packet in packets for slot_id in packet.slot_ids}


def _render_source_evidence(items: Sequence[EvidencePoolItem]) -> str:
    return "\n".join(
        f"{item.packet.evidence_id}: {_source_text(item)}" for item in items
    )


def _parse_curated_packets(
    response: Any,
    *,
    slots: Sequence[RequiredSlot],
    items: Sequence[EvidencePoolItem],
    final_claims: list[FinalClaim] | None = None,
) -> list[EvidencePacket]:
    content = getattr(response, "content", response)
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return []
    if not isinstance(content, Mapping) or set(content) != {"packets"}:
        return []
    raw_packets = content.get("packets")
    if not isinstance(raw_packets, list):
        return []
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
            or not set(slot_ids).issubset(valid_slots)
            or source_id not in by_id
        ):
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
    "EvidenceExtractor",
    "calculate_difference",
    "extract_evidence_packets",
    "extract_numeric_packets",
]
