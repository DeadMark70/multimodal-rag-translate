"""Bounded, provenance-safe evidence packing for Agentic v9 final prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from data_base.agentic_v9.schemas import EvidencePacket, QueryContract, RequiredSlot
from data_base.agentic_v9.token_estimator import (
    PromptTokenEstimate,
    TokenEstimator,
    render_evidence_packet,
)


@dataclass(frozen=True, slots=True)
class PackedEvidenceContext:
    """The complete bounded evidence projection consumed by a final v9 phase."""

    packets: tuple[EvidencePacket, ...]
    rendered_text: str
    estimated_input_tokens: int
    dropped_packet_ids: tuple[str, ...]
    tokens_by_slot: dict[str, int]
    tokens_by_source: dict[str, int]
    input_token_budget: int
    failure_reason: str | None = None
    prompt_estimate: PromptTokenEstimate | None = None

    @property
    def is_packable(self) -> bool:
        """Whether a final phase may consume this context."""
        return self.failure_reason is None


@dataclass(frozen=True, slots=True)
class _Candidate:
    packet: EvidencePacket
    estimate: int
    quality: float
    index: int


class EvidenceContextPacker:
    """Select whole evidence packets without exceeding the final input budget."""

    def __init__(
        self,
        *,
        setup_input_ceiling: int,
        remaining_runtime_tokens: int,
        final_output_reserve: int,
        thinking_token_reserve: int = 0,
        instruction: str = "",
        question: str = "",
        contract: object | None = None,
        history: Sequence[object] | None = None,
        image_tokens: int = 0,
        schema: object | None = None,
        safety_margin_tokens: int | None = None,
        estimator: TokenEstimator | None = None,
    ) -> None:
        for name, value in (
            ("setup_input_ceiling", setup_input_ceiling),
            ("remaining_runtime_tokens", remaining_runtime_tokens),
            ("final_output_reserve", final_output_reserve),
            ("thinking_token_reserve", thinking_token_reserve),
            ("image_tokens", image_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self._estimator = estimator or TokenEstimator()
        prompt_estimator = self._estimator
        if not callable(getattr(prompt_estimator, "estimate_prompt", None)):
            # Packet-only test/deterministic estimators predate final prompt
            # accounting; keep their explicit packet budget authoritative.
            prompt_estimator = TokenEstimator(base_safety_margin_tokens=0)
        self._prompt_estimator: TokenEstimator = prompt_estimator  # type: ignore[assignment]
        self._fixed_prompt_estimate = self._prompt_estimator.estimate_prompt(
            instruction=instruction,
            question=question,
            contract=contract,
            history=history,
            image_tokens=image_tokens,
            schema=schema,
            safety_margin_tokens=safety_margin_tokens,
        )
        available_input_tokens = min(
            setup_input_ceiling,
            max(
                remaining_runtime_tokens
                - final_output_reserve
                - thinking_token_reserve,
                0,
            ),
        )
        self._input_token_budget = max(
            available_input_tokens - self._fixed_prompt_estimate.fixed_overhead_tokens,
            0,
        )

    @property
    def input_token_budget(self) -> int:
        """The Setup and runtime constrained evidence budget."""
        return self._input_token_budget

    def pack(
        self,
        packets: Sequence[EvidencePacket] | Iterable[EvidencePacket],
        *,
        required_slots: Sequence[RequiredSlot] | QueryContract | None = None,
        quality_by_evidence_id: Mapping[str, float] | None = None,
        derived_claim_premise_evidence_ids: Sequence[str] | None = None,
    ) -> PackedEvidenceContext:
        """Pack whole packets, failing closed when answerable mandatory slots lose.

        Duplicate source chunks/spans are reduced before selection.  Required
        slots that have available evidence are covered first; only then are
        remaining candidates considered by quality and source diversity.
        """
        source_packets = tuple(packets)
        candidates, deduplicated_drops = self._deduplicate(
            source_packets, quality_by_evidence_id or {}
        )
        if bool(getattr(self._estimator, "must_fail_closed", False)):
            return self._failed_context(
                source_packets,
                deduplicated_drops,
                reason="provider_estimate_error_exceeded",
            )
        required = self._required_slots(required_slots)
        selected: list[_Candidate] = []
        selected_ids: set[str] = set()
        used_tokens = 0
        candidates_by_id = {
            candidate.packet.evidence_id: candidate for candidate in candidates
        }

        for slot in required:
            if not slot.required:
                continue
            slot_candidates = [
                candidate for candidate in candidates if slot.slot_id in candidate.packet.slot_ids
            ]
            # A slot with no positive evidence is not answerable here; absence
            # remains owned by SlotResolution, not fabricated by the packer.
            if not slot_candidates:
                continue
            if any(
                candidate.packet.evidence_id in selected_ids
                for candidate in slot_candidates
            ):
                continue
            best_closure = next(
                (
                    closure
                    for candidate in sorted(slot_candidates, key=self._quality_order)
                    if (
                        closure := self._premise_closure(
                            candidate, candidates_by_id, selected_ids
                        )
                    )
                    is not None
                    and used_tokens + sum(item.estimate for item in closure)
                    <= self._input_token_budget
                ),
                None,
            )
            if best_closure is None:
                return self._failed_context(source_packets, deduplicated_drops)
            selected.extend(best_closure)
            selected_ids.update(item.packet.evidence_id for item in best_closure)
            used_tokens += sum(item.estimate for item in best_closure)

        for premise_id in derived_claim_premise_evidence_ids or ():
            candidate = candidates_by_id.get(premise_id)
            closure = (
                self._premise_closure(candidate, candidates_by_id, selected_ids)
                if candidate is not None
                else None
            )
            if (
                closure is None
                or used_tokens + sum(item.estimate for item in closure)
                > self._input_token_budget
            ):
                return self._failed_context(
                    source_packets,
                    deduplicated_drops,
                    reason="derived_claim_premises_cannot_fit",
                )
            selected.extend(closure)
            selected_ids.update(item.packet.evidence_id for item in closure)
            used_tokens += sum(item.estimate for item in closure)

        source_counts = self._source_counts(selected)
        remaining = [
            candidate
            for candidate in candidates
            if candidate.packet.evidence_id not in selected_ids
        ]
        while remaining:
            candidate = min(
                remaining,
                key=lambda item: (
                    source_counts.get(item.packet.source.doc_id, 0),
                    *self._quality_order(item),
                ),
            )
            remaining.remove(candidate)
            closure = self._premise_closure(candidate, candidates_by_id, selected_ids)
            if (
                closure is None
                or used_tokens + sum(item.estimate for item in closure)
                > self._input_token_budget
            ):
                continue
            selected.extend(closure)
            selected_ids.update(item.packet.evidence_id for item in closure)
            used_tokens += sum(item.estimate for item in closure)
            for item in closure:
                source_counts[item.packet.source.doc_id] = (
                    source_counts.get(item.packet.source.doc_id, 0) + 1
                )

        selected_packets = tuple(candidate.packet for candidate in selected)
        selected_token_counts = {
            candidate.packet.evidence_id: candidate.estimate for candidate in selected
        }
        dropped = tuple(
            packet.evidence_id
            for packet in source_packets
            if packet.evidence_id not in selected_ids
        )
        prompt_estimate = self._fixed_prompt_estimate.with_evidence(used_tokens)
        return PackedEvidenceContext(
            packets=selected_packets,
            rendered_text="\n\n".join(render_evidence_packet(packet) for packet in selected_packets),
            # This legacy field is the packed evidence metric.  The complete
            # provider input projection remains available separately so fixed
            # prompt overhead does not distort evidence distributions.
            estimated_input_tokens=used_tokens,
            dropped_packet_ids=dropped,
            tokens_by_slot=self._tokens_by_slot(selected, selected_token_counts),
            tokens_by_source=self._tokens_by_source(selected, selected_token_counts),
            input_token_budget=self._input_token_budget,
            prompt_estimate=prompt_estimate,
        )

    def _deduplicate(
        self,
        packets: tuple[EvidencePacket, ...],
        quality_by_evidence_id: Mapping[str, float],
    ) -> tuple[list[_Candidate], set[str]]:
        winners: dict[tuple[str, ...], _Candidate] = {}
        dropped: set[str] = set()
        for index, packet in enumerate(packets):
            candidate = _Candidate(
                packet=packet,
                estimate=self._estimator.estimate_packet(packet),
                quality=self._quality(quality_by_evidence_id.get(packet.evidence_id)),
                index=index,
            )
            identity = self._source_identity(packet)
            previous = winners.get(identity)
            if previous is None:
                winners[identity] = candidate
            elif self._quality_order(candidate) < self._quality_order(previous):
                winners[identity] = candidate
                dropped.add(previous.packet.evidence_id)
            else:
                dropped.add(packet.evidence_id)
        return sorted(winners.values(), key=lambda item: item.index), dropped

    @staticmethod
    def _premise_closure(
        candidate: _Candidate,
        candidates_by_id: Mapping[str, _Candidate],
        selected_ids: set[str],
    ) -> list[_Candidate] | None:
        """Return premise-first transitive closure, or ``None`` when unsafe.

        A calculated packet and a derived claim's explicit premise packet must
        never be rendered without all of their direct (and nested) premises.
        Missing IDs and cycles are provenance failures, not invitations to
        synthesize a replacement packet.
        """
        closure: list[_Candidate] = []
        visiting: set[str] = set()
        included = set(selected_ids)

        def visit(item: _Candidate) -> bool:
            evidence_id = item.packet.evidence_id
            if evidence_id in included:
                return True
            if evidence_id in visiting:
                return False
            visiting.add(evidence_id)
            for premise_id in item.packet.premise_evidence_ids:
                premise = candidates_by_id.get(premise_id)
                if premise is None or not visit(premise):
                    return False
            visiting.remove(evidence_id)
            included.add(evidence_id)
            closure.append(item)
            return True

        return closure if visit(candidate) else None

    @staticmethod
    def _source_identity(packet: EvidencePacket) -> tuple[str, ...]:
        source = packet.source
        if source.chunk_id:
            return ("chunk", source.doc_id, source.chunk_id)
        if source.source_span_hash:
            return ("span", source.doc_id, source.source_span_hash)
        return ("packet", packet.evidence_id)

    @staticmethod
    def _quality(value: object) -> float:
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    @staticmethod
    def _quality_order(candidate: _Candidate) -> tuple[float, int, str]:
        return (-candidate.quality, candidate.index, candidate.packet.evidence_id)

    @staticmethod
    def _required_slots(
        required_slots: Sequence[RequiredSlot] | QueryContract | None,
    ) -> tuple[RequiredSlot, ...]:
        if required_slots is None:
            return ()
        if isinstance(required_slots, QueryContract):
            return tuple(required_slots.required_slots)
        return tuple(required_slots)

    @staticmethod
    def _source_counts(candidates: Sequence[_Candidate]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in candidates:
            doc_id = candidate.packet.source.doc_id
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

    @staticmethod
    def _tokens_by_slot(
        candidates: Sequence[_Candidate], token_counts: Mapping[str, int]
    ) -> dict[str, int]:
        totals: dict[str, int] = {}
        for candidate in candidates:
            tokens = token_counts[candidate.packet.evidence_id]
            for slot_id in candidate.packet.slot_ids:
                totals[slot_id] = totals.get(slot_id, 0) + tokens
        return totals

    @staticmethod
    def _tokens_by_source(
        candidates: Sequence[_Candidate], token_counts: Mapping[str, int]
    ) -> dict[str, int]:
        totals: dict[str, int] = {}
        for candidate in candidates:
            doc_id = candidate.packet.source.doc_id
            totals[doc_id] = totals.get(doc_id, 0) + token_counts[candidate.packet.evidence_id]
        return totals

    def _failed_context(
        self,
        source_packets: Sequence[EvidencePacket],
        deduplicated_drops: set[str],
        *,
        reason: str = "mandatory_evidence_cannot_fit",
    ) -> PackedEvidenceContext:
        dropped = tuple(
            packet.evidence_id
            for packet in source_packets
            if packet.evidence_id not in deduplicated_drops
        )
        return PackedEvidenceContext(
            packets=(),
            rendered_text="",
            estimated_input_tokens=0,
            dropped_packet_ids=tuple(sorted(set(dropped) | deduplicated_drops)),
            tokens_by_slot={},
            tokens_by_source={},
            input_token_budget=self._input_token_budget,
            failure_reason=reason,
            prompt_estimate=self._fixed_prompt_estimate,
        )


def pack_evidence_context(
    packets: Sequence[EvidencePacket] | Iterable[EvidencePacket],
    *,
    setup_input_ceiling: int,
    remaining_runtime_tokens: int,
    final_output_reserve: int,
    thinking_token_reserve: int = 0,
    instruction: str = "",
    question: str = "",
    contract: object | None = None,
    history: Sequence[object] | None = None,
    image_tokens: int = 0,
    schema: object | None = None,
    safety_margin_tokens: int | None = None,
    required_slots: Sequence[RequiredSlot] | QueryContract | None = None,
    quality_by_evidence_id: Mapping[str, float] | None = None,
    derived_claim_premise_evidence_ids: Sequence[str] | None = None,
    estimator: TokenEstimator | None = None,
) -> PackedEvidenceContext:
    """Pack evidence through the stable functional v9 boundary."""
    return EvidenceContextPacker(
        setup_input_ceiling=setup_input_ceiling,
        remaining_runtime_tokens=remaining_runtime_tokens,
        final_output_reserve=final_output_reserve,
        thinking_token_reserve=thinking_token_reserve,
        instruction=instruction,
        question=question,
        contract=contract,
        history=history,
        image_tokens=image_tokens,
        schema=schema,
        safety_margin_tokens=safety_margin_tokens,
        estimator=estimator,
    ).pack(
        packets,
        required_slots=required_slots,
        quality_by_evidence_id=quality_by_evidence_id,
        derived_claim_premise_evidence_ids=derived_claim_premise_evidence_ids,
    )
