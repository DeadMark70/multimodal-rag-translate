import pytest

from data_base.agentic_v9.schemas import RequiredSlot, SlotCondition
from data_base.agentic_v9.slot_constraints import slot_content_matches_chunk


def _numeric_slot(
    slot_id: str,
    description: str,
    *,
    entity_ids: list[str] | None = None,
    locator_hints: list[str] | None = None,
    requested_measure: str | None = None,
    expected_result_unit: str | None = None,
    conditions: list[SlotCondition] | None = None,
) -> RequiredSlot:
    if entity_ids is None:
        entity_ids = (
            ["proposed method"] if "proposed" in description.casefold() else ["U-KAN"]
        )
    normalized = description.casefold()
    if requested_measure is None:
        if "dice" in normalized:
            requested_measure = "Dice"
        elif "accuracy" in normalized:
            requested_measure = "accuracy"
        elif "patient count" in normalized:
            requested_measure = "patient count"
        elif "training duration" in normalized:
            requested_measure = "training duration"
        else:
            requested_measure = "metric"
    if expected_result_unit is None:
        if "percentage" in normalized:
            expected_result_unit = "percent"
        elif "patient count" in normalized:
            expected_result_unit = "patients"
        elif "in years" in normalized:
            expected_result_unit = "years"
        else:
            expected_result_unit = "dimensionless"
    return RequiredSlot(
        slot_id=slot_id,
        description=description,
        entity_ids=entity_ids,
        locator_hints=locator_hints or [],
        requested_measure=requested_measure,
        expected_answer_type="number",
        expected_result_unit=expected_result_unit,
        conditions=conditions or [],
    )


def test_dimensionless_dice_rejects_condition_counts_patients_and_years() -> None:
    slot = _numeric_slot(
        "S5",
        "Retrieve U-KAN Dice under the requested condition.",
        conditions=[SlotCondition(field="noise_level", operator="=", value="0.4")],
    )

    for text in (
        "U-KAN Dice was 10 cases at noise level 0.4.",
        "U-KAN Dice was patient count 10 at noise level 0.4.",
        "U-KAN Dice was year 2024 at noise level 0.4.",
        "U-KAN was evaluated at noise level 0.4; Dice was not reported.",
    ):
        assert not slot_content_matches_chunk(
            slot=slot,
            peer_slots=[],
            text=text,
        )


def test_dice_result_binds_measure_value_unit_and_noise_condition() -> None:
    slot = _numeric_slot(
        "S5",
        "Retrieve U-KAN Dice under the requested condition.",
        conditions=[SlotCondition(field="noise_level", operator="=", value="0.4")],
    )

    assert slot_content_matches_chunk(
        slot=slot,
        peer_slots=[],
        text="U-KAN Dice was 0.81 at noise level 0.4.",
    )
    assert not slot_content_matches_chunk(
        slot=slot,
        peer_slots=[],
        text="U-KAN result was 0.81 at noise level 0.4.",
    )


@pytest.mark.parametrize(
    ("slot", "text"),
    [
        (
            _numeric_slot(
                "count",
                "Retrieve the U-KAN patient count.",
                requested_measure="patient count",
                expected_result_unit="patients",
            ),
            "U-KAN patient count was 10 patients.",
        ),
        (
            _numeric_slot(
                "year",
                "Retrieve the U-KAN publication year.",
                requested_measure="publication year",
                expected_result_unit="year",
            ),
            "U-KAN publication year was 2024.",
        ),
    ],
    ids=["patient-count", "publication-year"],
)
def test_requested_patient_count_and_year_are_supported(
    slot: RequiredSlot,
    text: str,
) -> None:
    assert slot_content_matches_chunk(slot=slot, peer_slots=[], text=text)


@pytest.mark.parametrize(
    ("answer_type", "requested_measure", "positive", "negative"),
    [
        (
            "equation",
            "regional impurity",
            "Regional impurity = p(1-p).",
            "Regional impurity is discussed.",
        ),
        (
            "definition",
            "A complement",
            "A complement is defined as the values outside A.",
            "A complement appears in the theorem.",
        ),
        (
            "range",
            "m",
            "The range for m is 1 <= m < n.",
            "The theorem mentions m.",
        ),
        (
            "categorical",
            "mask type",
            "The mask type is semantic.",
            "The mask type appears in the source.",
        ),
        (
            "boolean",
            "prompt-free inference",
            "Prompt-free inference is supported.",
            "Prompt-free inference is discussed.",
        ),
        (
            "comparison",
            "method performance",
            "Method performance is better for A than B.",
            "Method performance for A and B is listed.",
        ),
        (
            "explanation",
            "penalty reason",
            "The penalty reason is higher overlap because adjacent teeth touch.",
            "The penalty reason is stated.",
        ),
        (
            "list",
            "processing branches",
            "The processing branches are convolution, state space, and skip.",
            "The processing branches are described.",
        ),
    ],
)
def test_nonnumeric_result_types_use_type_appropriate_checks(
    answer_type: str,
    requested_measure: str,
    positive: str,
    negative: str,
) -> None:
    slot = RequiredSlot(
        slot_id="S1",
        description=f"Retrieve the {requested_measure}.",
        requested_measure=requested_measure,
        expected_answer_type=answer_type,
    )

    assert slot_content_matches_chunk(
        slot=slot,
        peer_slots=[],
        text=positive,
    )
    assert not slot_content_matches_chunk(
        slot=slot,
        peer_slots=[],
        text=negative,
    )


def test_existing_v2_numeric_payload_without_result_role_fails_closed() -> None:
    slot = RequiredSlot.model_validate(
        {
            "slot_id": "S5",
            "description": "Retrieve U-KAN Dice at noise level 0.4.",
            "entity_ids": ["U-KAN"],
            "expected_answer_type": "number",
        }
    )

    assert not slot_content_matches_chunk(
        slot=slot,
        peer_slots=[],
        text="U-KAN Dice was 0.81 at noise level 0.4.",
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
        "U-KAN Dice scored 0.81; the proposed method is listed.",
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
