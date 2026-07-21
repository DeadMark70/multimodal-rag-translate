"""Focused contracts for bounded Agentic v9 evidence packing."""

from __future__ import annotations

from decimal import Decimal

from data_base.agentic_v9.context_packer import EvidenceContextPacker
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    RequiredSlot,
    SourceLocator,
)
from data_base.agentic_v9.token_estimator import PromptTokenEstimate, TokenEstimator


def _packet(
    evidence_id: str,
    *,
    slot_ids: list[str] | None = None,
    statement: str = "A bounded, atomic evidence statement.",
    doc_id: str = "doc-1",
    chunk_id: str | None = None,
    source_span_hash: str | None = None,
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=slot_ids or ["slot-1"],
        statement=statement,
        support_type="direct",
        source=EvidenceSource(
            doc_id=doc_id,
            chunk_id=chunk_id,
            source_span_hash=source_span_hash,
        ),
        scope=EvidenceScope(metric="Dice"),
        locator=SourceLocator(pdf_page_index=0),
        raw_value=Decimal("0.91"),
    )


class _FixedEstimator:
    def __init__(self, values: dict[str, int]) -> None:
        self._values = values

    def estimate_packet(self, packet: EvidencePacket) -> int:
        return self._values[packet.evidence_id]


def test_token_estimator_is_positive_and_conservative_for_text() -> None:
    estimator = TokenEstimator()

    assert estimator.estimate_text("") == 0
    assert estimator.estimate_text("中英文 mixed 0.91") >= 5


def test_packer_keeps_best_atomic_evidence_for_each_answerable_required_slot() -> None:
    packets = [
        _packet("low", statement="low quality", chunk_id="chunk-low"),
        _packet("best", statement="best quality", chunk_id="chunk-best"),
        _packet("optional", slot_ids=["slot-2"], statement="optional", chunk_id="c2"),
    ]
    packer = EvidenceContextPacker(
        setup_input_ceiling=200,
        remaining_runtime_tokens=200,
        final_output_reserve=20,
    )

    result = packer.pack(
        packets,
        required_slots=[RequiredSlot(slot_id="slot-1", description="score")],
        quality_by_evidence_id={"low": 0.1, "best": 0.9, "optional": 0.8},
    )

    assert result.failure_reason is None
    assert [packet.evidence_id for packet in result.packets][:1] == ["best"]
    assert "[best]" in result.rendered_text
    assert result.estimated_input_tokens <= 180
    assert result.tokens_by_slot["slot-1"] > 0
    assert result.tokens_by_source["doc-1"] > 0


def test_packer_drops_the_whole_atomic_packet_when_it_does_not_fit() -> None:
    packet = _packet("large", statement="atomic " * 100, chunk_id="chunk-large")
    packer = EvidenceContextPacker(
        setup_input_ceiling=10,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
    )

    result = packer.pack([packet])

    assert result.packets == ()
    assert result.dropped_packet_ids == ("large",)
    assert result.rendered_text == ""
    assert result.estimated_input_tokens == 0


def test_packer_persists_drops_when_packets_are_supplied_as_an_iterator() -> None:
    packet = _packet("iterator-drop", statement="atomic " * 100, chunk_id="iterator")
    packer = EvidenceContextPacker(
        setup_input_ceiling=10,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
    )

    result = packer.pack(packet for packet in [packet])

    assert result.dropped_packet_ids == ("iterator-drop",)


def test_packer_fails_closed_and_records_drops_when_mandatory_evidence_cannot_fit() -> None:
    packet = _packet("must-fit", statement="atomic " * 100, chunk_id="chunk-required")
    packer = EvidenceContextPacker(
        setup_input_ceiling=10,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
    )

    result = packer.pack(
        [packet],
        required_slots=[RequiredSlot(slot_id="slot-1", description="required")],
    )

    assert result.failure_reason == "mandatory_evidence_cannot_fit"
    assert result.packets == ()
    assert result.dropped_packet_ids == ("must-fit",)
    assert result.tokens_by_slot == {}
    assert result.tokens_by_source == {}


def test_packer_uses_the_best_fitting_mandatory_evidence_packet() -> None:
    packets = [
        _packet("too-large", statement="best but too large", chunk_id="large"),
        _packet("fits", statement="next best and fits", chunk_id="fits"),
    ]
    packer = EvidenceContextPacker(
        setup_input_ceiling=2,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
        estimator=_FixedEstimator({"too-large": 3, "fits": 2}),  # type: ignore[arg-type]
    )

    result = packer.pack(
        packets,
        required_slots=[RequiredSlot(slot_id="slot-1", description="required")],
        quality_by_evidence_id={"too-large": 1.0, "fits": 0.5},
    )

    assert result.failure_reason is None
    assert [packet.evidence_id for packet in result.packets] == ["fits"]
    assert result.dropped_packet_ids == ("too-large",)


def test_packer_deduplicates_chunk_spans_and_prefers_a_new_source_after_required_coverage() -> None:
    packets = [
        _packet("required", chunk_id="required", statement="required"),
        _packet("duplicate-low", chunk_id="shared", statement="duplicate low"),
        _packet("duplicate-best", chunk_id="shared", statement="duplicate best"),
        _packet("other-source", doc_id="doc-2", chunk_id="other", statement="other"),
    ]
    packer = EvidenceContextPacker(
        setup_input_ceiling=200,
        remaining_runtime_tokens=200,
        final_output_reserve=0,
    )

    result = packer.pack(
        packets,
        required_slots=[RequiredSlot(slot_id="slot-1", description="required")],
        quality_by_evidence_id={
            "required": 1.0,
            "duplicate-low": 0.1,
            "duplicate-best": 0.8,
            "other-source": 0.7,
        },
    )

    assert "duplicate-low" in result.dropped_packet_ids
    assert "duplicate-best" in [packet.evidence_id for packet in result.packets]
    assert "other-source" in [packet.evidence_id for packet in result.packets]
    assert set(result.tokens_by_source) == {"doc-1", "doc-2"}


def test_complete_prompt_estimate_leaves_only_the_remaining_budget_for_evidence() -> None:
    packer = EvidenceContextPacker(
        setup_input_ceiling=100,
        remaining_runtime_tokens=160,
        final_output_reserve=30,
        thinking_token_reserve=20,
        instruction="Use only cited evidence.",
        question="比較中英文表格中的 Dice 分數。",
        contract={"required_slots": ["slot-1"]},
        history=[{"role": "user", "content": "Earlier question"}],
        image_tokens=7,
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        safety_margin_tokens=5,
    )

    result = packer.pack([_packet("fits", statement="evidence", chunk_id="fits")])

    assert isinstance(result.prompt_estimate, PromptTokenEstimate)
    assert result.prompt_estimate.instruction > 0
    assert result.prompt_estimate.question > 0
    assert result.prompt_estimate.contract > 0
    assert result.prompt_estimate.history > 0
    assert result.prompt_estimate.image == 7
    assert result.prompt_estimate.schema > 0
    assert result.prompt_estimate.safety_margin == 5
    assert result.input_token_budget == max(
        min(100, 160 - 30 - 20) - result.prompt_estimate.fixed_overhead_tokens,
        0,
    )
    assert result.estimated_input_tokens == result.prompt_estimate.total_tokens


def test_packer_includes_transitive_premises_for_calculated_and_derived_claim_evidence() -> None:
    root = _packet("root", statement="root", chunk_id="root")
    intermediate = _packet("intermediate", statement="intermediate", chunk_id="intermediate")
    calculated = _packet(
        "calculated", statement="calculated", chunk_id="calculated"
    )
    intermediate.premise_evidence_ids = ["root"]
    calculated.support_type = "calculated"
    calculated.premise_evidence_ids = ["intermediate"]
    derived_premise = _packet(
        "derived-premise", statement="derived", chunk_id="derived"
    )
    packer = EvidenceContextPacker(
        setup_input_ceiling=100,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
        estimator=_FixedEstimator(
            {"root": 2, "intermediate": 2, "calculated": 2, "derived-premise": 2}
        ),  # type: ignore[arg-type]
    )

    result = packer.pack(
        [calculated, intermediate, root, derived_premise],
        required_slots=[RequiredSlot(slot_id="slot-1", description="calculation")],
        derived_claim_premise_evidence_ids=["derived-premise"],
    )

    assert [packet.evidence_id for packet in result.packets][:4] == [
        "root",
        "intermediate",
        "calculated",
        "derived-premise",
    ]


def test_calibration_persists_provider_error_increases_margin_then_fails_closed() -> None:
    estimator = TokenEstimator(
        base_safety_margin_tokens=1,
        excessive_error_ratio=0.20,
        fail_closed_after_excessive_errors=2,
    )
    estimator.record_provider_input_tokens(
        estimated_input_tokens=10, provider_input_tokens=20
    )

    assert estimator.provider_input_errors[-1].error_tokens == 10
    assert estimator.safety_margin_tokens > 1

    estimator.record_provider_input_tokens(
        estimated_input_tokens=10, provider_input_tokens=20
    )
    packer = EvidenceContextPacker(
        setup_input_ceiling=100,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
        estimator=estimator,
    )

    result = packer.pack([_packet("unsafe", chunk_id="unsafe")])

    assert result.failure_reason == "provider_estimate_error_exceeded"
    assert result.dropped_packet_ids == ("unsafe",)


def test_estimator_is_conservative_for_structured_and_image_prompt_content() -> None:
    estimator = TokenEstimator()

    assert estimator.estimate_text(r"\\frac{Dice_{zh}}{0.91}") >= 10
    assert estimator.estimate_json({"中文": ["value", 0.91]}) >= 10
    assert estimator.estimate_table("| 指標 | Dice |\n| --- | --- |\n| A | 0.91 |") >= 20
    assert estimator.estimate_image(width=512, height=512) > 0
