from data_base.agentic_v9.schemas import RequiredSlot
from data_base.agentic_v9.slot_constraints import slot_content_matches_chunk


def _numeric_slot(slot_id: str, description: str) -> RequiredSlot:
    return RequiredSlot(
        slot_id=slot_id,
        description=description,
        entity_ids=["Implicit-U-KAN2.0"],
        expected_answer_type="number",
    )


def test_colocated_peers_without_unique_descriptor_fail_closed() -> None:
    first = _numeric_slot("S5", "Report the metric.")
    second = _numeric_slot("S6", "Report the metric.")

    assert not slot_content_matches_chunk(
        slot=first,
        peer_slots=[second],
        text="The metric is 0.8.",
    )
    assert not slot_content_matches_chunk(
        slot=second,
        peer_slots=[first],
        text="The metric is 0.8.",
    )


def test_colocated_peer_requires_its_name_to_be_locally_associated_with_value() -> None:
    ukan = _numeric_slot("S5", "Report the U-KAN metric.")
    proposed = _numeric_slot("S6", "Report the proposed method metric.")
    text = "U-KAN metric is 0.8; the proposed method is listed."

    assert slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text=text,
    )
    assert not slot_content_matches_chunk(
        slot=proposed,
        peer_slots=[ukan],
        text=text,
    )


def test_numeric_colocated_peer_without_answer_value_is_unsupported() -> None:
    ukan = _numeric_slot("S5", "Report the U-KAN metric.")
    proposed = _numeric_slot("S6", "Report the proposed method metric.")

    assert not slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text="Table 3 discusses the U-KAN metric.",
    )
