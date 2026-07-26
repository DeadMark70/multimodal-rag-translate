"""Question-only deterministic atomic contract planning."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from data_base.agentic_v9.contract_planner import QuestionContractPlanner


QUESTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "golden"
    / "agentic_v9_questions_v2.json"
)


def _questions() -> dict[str, dict]:
    rows = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    return {row["id"]: row for row in rows}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question_id", "expected_route", "minimum_slots"),
    [
        ("Q5", "exact_structured", 3),
        ("Q7", "bounded_compare", 2),
        ("Q11", "multi_hop", 2),
        ("Q14", "multi_document_exact", 3),
        ("Q16", "multi_document_exact", 7),
    ],
)
async def test_formal_questions_decompose_into_atomic_answer_free_slots(
    question_id: str,
    expected_route: str,
    minimum_slots: int,
) -> None:
    case = _questions()[question_id]

    contract = await QuestionContractPlanner().plan(
        question=case["question"],
        authorized_source_names=case["source_docs"],
        authorized_source_doc_ids=[f"doc-{index}" for index, _ in enumerate(case["source_docs"], 1)],
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 8192},
    )

    assert contract.contract_version == "2"
    assert contract.slot_plan_status == "complete"
    assert contract.route == expected_route
    assert len(contract.required_slots) >= minimum_slots
    assert len(contract.required_slots) <= 8
    assert [slot.slot_id for slot in contract.required_slots] == [
        f"S{index}" for index in range(1, len(contract.required_slots) + 1)
    ]
    assert all(slot.description.strip() for slot in contract.required_slots)
    assert all(
        slot.authorized_source_doc_ids for slot in contract.required_slots
    )


@pytest.mark.asyncio
async def test_q16_has_seven_ordered_slots_without_expected_numeric_answers() -> None:
    case = _questions()["Q16"]

    contract = await QuestionContractPlanner().plan(
        question=case["question"],
        authorized_source_names=case["source_docs"],
        authorized_source_doc_ids=["gepar", "odes", "ukan"],
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 8192},
    )

    assert [slot.slot_id for slot in contract.required_slots] == [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
    ]
    descriptions = [slot.description for slot in contract.required_slots]
    assert ["penalty" in descriptions[0].casefold(), "reason" in descriptions[1].casefold()] == [True, True]
    assert "equation" in descriptions[2].casefold()
    assert "|a^c" in descriptions[3].casefold()
    assert "u-kan" in descriptions[4].casefold()
    assert "proposed" in descriptions[5].casefold()
    assert "theorem" in descriptions[6].casefold()
    for forbidden in ("0.179", "0.4064", "0.9079", "0 ≤", "0 <="):
        assert all(forbidden not in description for description in descriptions)


@pytest.mark.asyncio
async def test_numbered_parallel_source_and_locator_clauses_split_stably() -> None:
    contract = await QuestionContractPlanner().plan(
        question=(
            "Using Alpha.pdf and Beta.pdf: 1. report the U-KAN Dice and proposed "
            "method Dice from Table 3; 2. give Equation 2; 3. explain Theorem 1 "
            "and Appendix D."
        ),
        authorized_source_names=["Alpha.pdf", "Beta.pdf"],
        authorized_source_doc_ids=["alpha", "beta"],
        setup_policy={"max_llm_calls": 5},
    )

    assert len(contract.required_slots) >= 5
    all_locators = {
        locator
        for slot in contract.required_slots
        for locator in slot.locator_hints
    }
    assert {"Table 3", "Equation 2", "Theorem 1", "Appendix D"} <= all_locators
    assert contract.route_decision is not None
    assert "numbered_subquestions" in contract.route_decision.matched_rules
    assert "parallel_values" in contract.route_decision.matched_rules
    assert "multiple_named_sources" in contract.route_decision.matched_rules


def test_planner_api_cannot_accept_question_snapshot_or_gold_fields() -> None:
    parameters = inspect.signature(QuestionContractPlanner.plan).parameters
    assert set(parameters) == {
        "self",
        "question",
        "authorized_source_names",
        "authorized_source_doc_ids",
        "setup_policy",
    }

    snapshot = {
        "question": "What is the reported value?",
        "ground_truth": "secret",
        "key_points": ["secret"],
        "atomic_facts": [{"text": "secret"}],
        "expected_evidence": [{"answer": "secret"}],
    }
    with pytest.raises(TypeError):
        QuestionContractPlanner().plan(
            question_snapshot=snapshot,
            authorized_source_names=["paper.pdf"],
            authorized_source_doc_ids=["doc-1"],
            setup_policy={},
        )
