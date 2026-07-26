import pytest

from data_base.agentic_v9.schemas import RequiredSlot
from data_base.agentic_v9.slot_constraints import slot_content_matches_chunk


def _numeric_slot(
    slot_id: str,
    description: str,
    *,
    entity_ids: list[str] | None = None,
    locator_hints: list[str] | None = None,
) -> RequiredSlot:
    return RequiredSlot(
        slot_id=slot_id,
        description=description,
        entity_ids=entity_ids or ["Implicit-U-KAN2.0"],
        locator_hints=locator_hints or [],
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


def test_q16_noise_condition_is_not_treated_as_ukan_dice() -> None:
    ukan = _numeric_slot(
        "S5",
        "Retrieve the U-KAN Dice at noise level 0.4.",
        locator_hints=["Table 3"],
    )
    proposed = _numeric_slot(
        "S6",
        "Retrieve the proposed-method Dice at noise level 0.4.",
        locator_hints=["Table 3"],
    )

    assert not slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text=("U-KAN was evaluated at noise level 0.4; no Dice result was reported."),
    )


def test_q16_noise_condition_preserves_independent_ukan_dice_value() -> None:
    ukan = _numeric_slot(
        "S5",
        "Retrieve the U-KAN Dice at noise level 0.4.",
        locator_hints=["Table 3"],
    )
    proposed = _numeric_slot(
        "S6",
        "Retrieve the proposed-method Dice at noise level 0.4.",
        locator_hints=["Table 3"],
    )

    assert slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text=(
            "At noise level 0.4, U-KAN Dice was 0.81; the proposed method is listed."
        ),
    )
    assert slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text=("At noise level 0.4, U-KAN Dice was 0.4; the proposed method is listed."),
    )


def test_unrelated_case_count_is_not_a_numeric_dice_result() -> None:
    ukan = _numeric_slot(
        "S5",
        "Retrieve the U-KAN Dice at noise level 0.4.",
    )
    proposed = _numeric_slot(
        "S6",
        "Retrieve the proposed-method Dice at noise level 0.4.",
    )

    assert not slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text=(
            "U-KAN was evaluated on 10 cases at noise level 0.4; "
            "the Dice analysis is pending."
        ),
    )


@pytest.mark.parametrize(
    "text",
    [
        "U-KAN Dice result of 10 cases at noise level 0.4.",
        "U-KAN Dice was 10 cases.",
    ],
    ids=["result-of-cases", "was-cases"],
)
def test_dice_rejects_semantically_linked_case_count(text: str) -> None:
    ukan = _numeric_slot("S5", "Retrieve the U-KAN Dice.")
    proposed = _numeric_slot("S6", "Retrieve the proposed-method Dice.")

    assert not slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text=text,
    )


@pytest.mark.parametrize(
    "value_with_unit",
    [
        "10 folds",
        "10 subjects",
        "10 samples",
        "10 images",
        "10 patients",
        "10 epochs",
        "10 iterations",
        "10 runs",
        "10 seeds",
        "10 years",
        "10 pages",
        "10 tables",
        "10 figures",
        "91%",
        "91 percent",
    ],
)
def test_dimensionless_dice_rejects_count_and_percent_units(
    value_with_unit: str,
) -> None:
    ukan = _numeric_slot("S5", "Retrieve the U-KAN Dice.")
    proposed = _numeric_slot("S6", "Retrieve the proposed-method Dice.")

    assert not slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text=f"U-KAN Dice was {value_with_unit}.",
    )


def test_locator_wording_does_not_request_a_table_count() -> None:
    ukan = _numeric_slot(
        "S5",
        "Retrieve the U-KAN Dice in Table 3.",
        locator_hints=["Table 3"],
    )
    proposed = _numeric_slot(
        "S6",
        "Retrieve the proposed-method Dice in Table 3.",
        locator_hints=["Table 3"],
    )

    assert not slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text="U-KAN Dice was 10 tables.",
    )


@pytest.mark.parametrize(
    ("ukan_description", "proposed_description", "text"),
    [
        (
            "Retrieve the U-KAN Dice.",
            "Retrieve the proposed-method Dice.",
            "U-KAN Dice was 0.81; the proposed method is listed.",
        ),
        (
            "Retrieve the U-KAN accuracy percentage.",
            "Retrieve the proposed-method accuracy percentage.",
            "U-KAN accuracy was 91%; the proposed method is listed.",
        ),
        (
            "Retrieve the U-KAN accuracy as a percentage.",
            "Retrieve the proposed-method accuracy as a percentage.",
            "U-KAN accuracy was 91 percent; the proposed method is listed.",
        ),
        (
            "Retrieve the U-KAN patient count.",
            "Retrieve the proposed-method patient count.",
            "U-KAN patient count was 10 patients; the proposed method is listed.",
        ),
        (
            "Retrieve the U-KAN training duration in years.",
            "Retrieve the proposed-method training duration in years.",
            "U-KAN training duration was 5 years; the proposed method is listed.",
        ),
    ],
    ids=[
        "dimensionless",
        "percent-symbol",
        "percent-word",
        "requested-count",
        "requested-years",
    ],
)
def test_numeric_result_allows_explicitly_requested_unit(
    ukan_description: str,
    proposed_description: str,
    text: str,
) -> None:
    ukan = _numeric_slot("S5", ukan_description)
    proposed = _numeric_slot("S6", proposed_description)

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


@pytest.mark.parametrize(
    "text",
    [
        "U-KAN Dice: 0.81; the proposed method is listed.",
        "U-KAN scored 0.81; the proposed method is listed.",
    ],
    ids=["label", "scored"],
)
def test_numeric_result_requires_explicit_ukan_association(text: str) -> None:
    ukan = _numeric_slot("S5", "Retrieve the U-KAN Dice.")
    proposed = _numeric_slot("S6", "Retrieve the proposed-method Dice.")

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


@pytest.mark.parametrize(
    "text",
    [
        "U-KAN Dice was 0.81, while the proposed-method Dice was not reported.",
        (
            "U-KAN Dice was 0.81, while the proposed-method Dice was 0.75 "
            "but not reported."
        ),
    ],
    ids=["peer-without-value", "peer-negated-value"],
)
def test_unavailable_peer_does_not_negate_ukan_result(text: str) -> None:
    ukan = _numeric_slot("S5", "Retrieve the U-KAN Dice.")
    proposed = _numeric_slot("S6", "Retrieve the proposed-method Dice.")

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


def test_negated_numeric_result_is_not_answer_evidence() -> None:
    ukan = _numeric_slot(
        "S5",
        "Retrieve the U-KAN Dice at noise level 0.4.",
    )
    proposed = _numeric_slot(
        "S6",
        "Retrieve the proposed-method Dice at noise level 0.4.",
    )

    assert not slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text="At noise level 0.4, U-KAN Dice 0.81 was not reported.",
    )
    assert not slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text=("U-KAN Dice was 0.81; however, the result was not reported."),
    )


@pytest.mark.parametrize(
    ("ukan", "proposed", "text"),
    [
        (
            _numeric_slot(
                "S5",
                "Retrieve U-KAN Dice from the 2024 evaluation.",
            ),
            _numeric_slot(
                "S6",
                "Retrieve proposed-method Dice from the 2024 evaluation.",
            ),
            "The 2024 evaluation included U-KAN; Dice was unavailable.",
        ),
        (
            _numeric_slot(
                "S5",
                "Retrieve U-KAN Dice for model version 2.0.",
            ),
            _numeric_slot(
                "S6",
                "Retrieve proposed-method Dice for model version 2.0.",
            ),
            "U-KAN model version 2.0 was evaluated; no Dice was given.",
        ),
        (
            _numeric_slot(
                "S5",
                "Retrieve U-KAN Dice.",
                entity_ids=["U-KAN2.0"],
            ),
            _numeric_slot(
                "S6",
                "Retrieve proposed-method Dice.",
                entity_ids=["U-KAN2.0"],
            ),
            "U-KAN model 2.0 was evaluated; Dice was unavailable.",
        ),
        (
            _numeric_slot(
                "S5",
                "Retrieve U-KAN Dice using 5 folds.",
            ),
            _numeric_slot(
                "S6",
                "Retrieve proposed-method Dice using 5 folds.",
            ),
            "U-KAN using 5 folds had no reported Dice result.",
        ),
        (
            _numeric_slot(
                "S5",
                "Retrieve U-KAN Dice at noise level .4.",
            ),
            _numeric_slot(
                "S6",
                "Retrieve proposed-method Dice at noise level .4.",
            ),
            "U-KAN at noise level 0.40 had no reported Dice result.",
        ),
        (
            _numeric_slot(
                "S5",
                "Retrieve U-KAN Dice.",
                locator_hints=["Table 3"],
            ),
            _numeric_slot(
                "S6",
                "Retrieve proposed-method Dice.",
                locator_hints=["Table 3"],
            ),
            "Table 3 lists U-KAN, but its Dice result is unavailable.",
        ),
    ],
    ids=[
        "year",
        "model-version",
        "entity-version",
        "fold-count",
        "decimal-variant",
        "locator",
    ],
)
def test_slot_condition_number_variants_are_not_answer_signals(
    ukan: RequiredSlot,
    proposed: RequiredSlot,
    text: str,
) -> None:
    assert not slot_content_matches_chunk(
        slot=ukan,
        peer_slots=[proposed],
        text=text,
    )
