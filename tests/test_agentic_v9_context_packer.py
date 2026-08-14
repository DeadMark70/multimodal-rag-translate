"""Focused contracts for bounded Agentic v9 evidence packing."""

from __future__ import annotations

from decimal import Decimal

from data_base.agentic_v9.context_packer import (
    EvidenceContextPacker,
    FinalContextSelectionPolicy,
)
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    QueryContract,
    RequiredSlot,
    ResponseConstraint,
    SourceLocator,
    SynthesisObligation,
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
    asset_id: str | None = None,
    raw_value: Decimal | None = Decimal("0.91"),
    table_id: str | None = None,
    figure_id: str | None = None,
    section: str | None = None,
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
            asset_id=asset_id,
        ),
        scope=EvidenceScope(metric="Dice"),
        locator=SourceLocator(
            pdf_page_index=0,
            table_id=table_id,
            figure_id=figure_id,
            section=section,
        ),
        raw_value=raw_value,
    )


class _FixedEstimator:
    def __init__(self, values: dict[str, int]) -> None:
        self._values = values

    def estimate_packet(self, packet: EvidencePacket) -> int:
        return self._values[packet.evidence_id]


def _soft_policy(*, preferred_max_packets: int = 8) -> FinalContextSelectionPolicy:
    return FinalContextSelectionPolicy(
        version="soft_final_pack_r1",
        preferred_max_packets=preferred_max_packets,
        new_source_bonus=0.05,
        near_duplicate_penalty=0.08,
        visual_without_visual_intent_penalty=0.03,
    )


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
    assert result.estimated_input_tokens == result.prompt_estimate.evidence
    assert result.prompt_estimate.total_tokens == (
        result.estimated_input_tokens
        + result.prompt_estimate.fixed_overhead_tokens
    )


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


def test_soft_policy_uses_rerank_quality_before_source_diversity() -> None:
    """A diversity bonus cannot turn into a source quota."""
    required = _packet("required", doc_id="doc-a", chunk_id="required")
    high_quality = _packet(
        "high-quality", doc_id="doc-a", chunk_id="high", slot_ids=["optional"]
    )
    unseen_source = _packet(
        "unseen-source", doc_id="doc-b", chunk_id="unseen", slot_ids=["optional"]
    )
    packer = EvidenceContextPacker(
        setup_input_ceiling=100,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
        estimator=_FixedEstimator(
            {"required": 1, "high-quality": 1, "unseen-source": 1}
        ),  # type: ignore[arg-type]
    )

    result = packer.pack(
        [required, high_quality, unseen_source],
        required_slots=[RequiredSlot(slot_id="slot-1", description="required")],
        quality_by_evidence_id={
            "required": 0.2,
            "high-quality": 1.0,
            "unseen-source": 0.10,
        },
        selection_policy=_soft_policy(preferred_max_packets=2),
    )

    high_quality_decision = next(
        decision
        for decision in result.selection_decisions
        if decision.evidence_id == "high-quality"
    )
    assert [packet.evidence_id for packet in result.packets] == ["required", "high-quality"]
    assert result.selection_policy_version == "soft_final_pack_r1"
    assert high_quality_decision.source_bonus == 0.0
    assert high_quality_decision.utility == 0.92


def test_soft_policy_only_removes_exact_source_duplicates() -> None:
    """Near duplicates remain eligible and are only omitted for the soft limit."""
    first = _packet("first", statement="Dice is 0.91 for the held-out split.", chunk_id="a")
    second = _packet("second", statement="Dice is 0.91 for the held-out split.", chunk_id="b")
    packer = EvidenceContextPacker(
        setup_input_ceiling=100,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
        estimator=_FixedEstimator({"first": 1, "second": 1}),  # type: ignore[arg-type]
    )

    result = packer.pack(
        [first, second],
        quality_by_evidence_id={"first": 1.0, "second": 0.9},
        selection_policy=_soft_policy(preferred_max_packets=1),
    )

    second_decision = next(
        decision for decision in result.selection_decisions if decision.evidence_id == "second"
    )
    assert [packet.evidence_id for packet in result.packets] == ["first"]
    assert second_decision.selected is False
    assert second_decision.redundancy_penalty == 0.08
    assert second_decision.reason == "preferred_packet_limit"


def test_soft_policy_keeps_unique_numeric_or_structured_evidence() -> None:
    """Different numeric/table provenance is not treated as a near duplicate."""
    first = _packet(
        "first",
        statement="Dice score for the experiment.",
        chunk_id="a",
        raw_value=Decimal("0.91"),
        table_id="Table 1",
    )
    second = _packet(
        "second",
        statement="Dice score for the experiment.",
        chunk_id="b",
        raw_value=Decimal("0.92"),
        table_id="Table 2",
    )
    packer = EvidenceContextPacker(
        setup_input_ceiling=100,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
        estimator=_FixedEstimator({"first": 1, "second": 1}),  # type: ignore[arg-type]
    )

    result = packer.pack(
        [first, second],
        quality_by_evidence_id={"first": 1.0, "second": 0.9},
        selection_policy=_soft_policy(preferred_max_packets=2),
    )

    second_decision = next(
        decision for decision in result.selection_decisions if decision.evidence_id == "second"
    )
    assert [packet.evidence_id for packet in result.packets] == ["first", "second"]
    assert second_decision.redundancy_penalty == 0.0


def test_soft_policy_downweights_typed_visual_packet_without_excluding_it() -> None:
    """Text-only questions may downweight visual evidence but cannot exclude it."""
    visual = _packet("visual", chunk_id="visual", asset_id="asset-1")
    text = _packet("text", chunk_id="text")
    packer = EvidenceContextPacker(
        setup_input_ceiling=100,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
    )

    result = packer.pack(
        [visual, text],
        quality_by_evidence_id={"visual": 1.0, "text": 0.9},
        selection_policy=_soft_policy(preferred_max_packets=1),
    )

    visual_decision = next(
        decision for decision in result.selection_decisions if decision.evidence_id == "visual"
    )
    assert [packet.evidence_id for packet in result.packets] == ["visual"]
    assert visual_decision.visual_penalty == 0.03
    assert visual_decision.utility == 1.02


def test_required_packet_closure_overrides_preferred_packet_limit() -> None:
    """A required packet's transitive premises remain answerable above the soft cap."""
    packets = [
        _packet(f"premise-{index}", slot_ids=["premise"], chunk_id=f"premise-{index}")
        for index in range(8)
    ]
    required = _packet("required", chunk_id="required")
    required.premise_evidence_ids = [packets[-1].evidence_id]
    for index in range(7, 0, -1):
        packets[index].premise_evidence_ids = [packets[index - 1].evidence_id]
    packets.append(required)
    packer = EvidenceContextPacker(
        setup_input_ceiling=100,
        remaining_runtime_tokens=100,
        final_output_reserve=0,
        estimator=_FixedEstimator({packet.evidence_id: 1 for packet in packets}),  # type: ignore[arg-type]
    )

    result = packer.pack(
        packets,
        required_slots=[RequiredSlot(slot_id="slot-1", description="required")],
        quality_by_evidence_id={"required": 1.0},
        selection_policy=_soft_policy(),
    )

    assert result.failure_reason is None
    assert len(result.packets) == 9
    assert any(
        decision.reason == "required_evidence_over_preferred_limit"
        for decision in result.selection_decisions
    )


def test_synthesis_obligations_and_constraints_never_appear_in_context_packing() -> None:
    contract = QueryContract(
        contract_version="2",
        route="bounded_compare",
        intent="Compare Model A and Model B.",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Model A score"),
            RequiredSlot(slot_id="S2", description="Model B score"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare scores",
                depends_on_slot_ids=["S1", "S2"],
            )
        ],
        response_constraints=[
            ResponseConstraint(
                constraint_id="C1",
                kind="output_format",
                description="Table output",
            )
        ],
    )
    packets = [
        _packet("p1", slot_ids=["S1"], statement="Model A score is 0.95"),
        _packet("p2", slot_ids=["S2"], statement="Model B score is 0.90"),
    ]
    packer = EvidenceContextPacker(
        setup_input_ceiling=500,
        remaining_runtime_tokens=500,
        final_output_reserve=50,
    )

    result = packer.pack(packets, required_slots=contract)

    assert result.failure_reason is None
    assert set(result.tokens_by_slot.keys()) == {"S1", "S2"}
    assert "O1" not in result.tokens_by_slot
    assert "C1" not in result.tokens_by_slot

