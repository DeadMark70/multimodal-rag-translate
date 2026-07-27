"""Question-only deterministic atomic contract planning."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from data_base.agentic_v9 import contract_planner
from data_base.agentic_v9.contract_planner import QuestionContractPlanner
from data_base.agentic_v9.schemas import BudgetExceededError
from data_base.agentic_v9.slot_constraints import structured_locator_state


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
    "question_id",
    ["Q5", "Q7", "Q11", "Q14"],
)
async def test_formal_questions_use_bounded_experimental_answer_free_slots(
    question_id: str,
) -> None:
    case = _questions()[question_id]
    doc_ids = [f"doc-{index}" for index, _ in enumerate(case["source_docs"], 1)]

    contract = await QuestionContractPlanner().plan(
        question=case["question"],
        authorized_source_names=case["source_docs"],
        authorized_source_doc_ids=doc_ids,
        authorized_source_name_to_doc_ids={
            name: [doc_id]
            for name, doc_id in zip(case["source_docs"], doc_ids, strict=True)
        },
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 8192},
    )

    assert contract.contract_version == "2"
    assert contract.slot_semantics == "heuristic_experimental"
    assert contract.atomic_completeness is None
    assert contract.slot_plan_status in {"complete", "degraded"}
    assert 1 <= len(contract.required_slots) <= 8
    assert [slot.slot_id for slot in contract.required_slots] == [
        f"S{index}" for index in range(1, len(contract.required_slots) + 1)
    ]
    assert all(slot.description.strip() for slot in contract.required_slots)
    assert all(slot.authorized_source_doc_ids for slot in contract.required_slots)


@pytest.mark.asyncio
async def test_production_contract_does_not_require_strict_evidence_extraction() -> (
    None
):
    case = _questions()["Q5"]
    doc_ids = [f"doc-{index}" for index, _ in enumerate(case["source_docs"], 1)]

    contract = await QuestionContractPlanner().plan(
        question=case["question"],
        authorized_source_names=case["source_docs"],
        authorized_source_doc_ids=doc_ids,
        authorized_source_name_to_doc_ids={
            name: [doc_id]
            for name, doc_id in zip(case["source_docs"], doc_ids, strict=True)
        },
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 8192},
    )

    assert contract.evidence_extraction_required is False


@pytest.mark.asyncio
async def test_q16_uses_generic_experimental_planning_without_benchmark_bundle() -> (
    None
):
    planner_source = (
        Path(__file__).resolve().parents[1]
        / "data_base"
        / "agentic_v9"
        / "contract_planner.py"
    ).read_text(encoding="utf-8")
    assert "q16_structured_bundle" not in planner_source
    assert "_known_question_slots" not in planner_source
    for benchmark_name in (
        "gepar3d",
        "segmentanybone",
        "weak-mamba-unet",
        "swinunetr",
    ):
        assert benchmark_name not in planner_source.casefold()

    case = _questions()["Q16"]

    contract = await QuestionContractPlanner().plan(
        question=case["question"],
        authorized_source_names=case["source_docs"],
        authorized_source_doc_ids=["gepar", "odes", "ukan"],
        authorized_source_name_to_doc_ids={
            name: [doc_id]
            for name, doc_id in zip(
                case["source_docs"], ["gepar", "odes", "ukan"], strict=True
            )
        },
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 8192},
    )

    assert contract.slot_semantics == "heuristic_experimental"
    assert contract.atomic_completeness is None
    assert contract.route_decision is not None
    assert "q16_structured_bundle" not in contract.route_decision.matched_rules
    serialized = contract.model_dump_json()
    for forbidden in ("0.179", "0.4064", "0.9079", "0 ≤", "0 <="):
        assert forbidden not in serialized
    for forbidden in (
        "tooth 1 to tooth 32 penalty",
        "regional impurity equation",
        "|A^c(x,y)|",
    ):
        assert forbidden not in serialized


def test_deterministic_route_source_contains_no_known_corpus_or_model_identifiers() -> (
    None
):
    source = inspect.getsource(contract_planner._deterministic_route).casefold()

    for identifier in (
        "css",
        "gepar3d",
        "implicit-u-kan",
        "mednext",
        "medsam",
        "miccss",
        "nnmamba",
        "odes",
        "sam-med3d",
        "samed",
        "segmentanybone",
        "segvol",
        "semi-mamba-unet",
        "swinunetr",
        "u-kan",
        "weak-mamba-unet",
    ):
        assert identifier not in source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("corpus_question", "generic_question"),
    [
        (
            "Explain how nnMamba MICCSS reconstructs the input.",
            "Explain how Model-A Module-B reconstructs the input.",
        ),
        (
            "Explain SAMed and MedSAM behavior.",
            "Explain Model-A and Model-B behavior.",
        ),
    ],
)
async def test_model_named_and_generic_equivalent_questions_route_equivalently(
    corpus_question: str,
    generic_question: str,
) -> None:
    async def plan(question: str):
        return await QuestionContractPlanner().plan(
            question=question,
            authorized_source_names=["paper.pdf"],
            authorized_source_doc_ids=["doc-1"],
            authorized_source_name_to_doc_ids={"paper.pdf": ["doc-1"]},
            setup_policy={},
        )

    named = await plan(corpus_question)
    generic = await plan(generic_question)

    assert (
        named.route,
        named.slot_plan_status,
        named.route_decision.decision_source,
    ) == (
        generic.route,
        generic.slot_plan_status,
        generic.route_decision.decision_source,
    )


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
        authorized_source_name_to_doc_ids={
            "Alpha.pdf": ["alpha"],
            "Beta.pdf": ["beta"],
        },
        setup_policy={"max_llm_calls": 5},
    )

    assert len(contract.required_slots) >= 5
    all_locators = {
        locator for slot in contract.required_slots for locator in slot.locator_hints
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
        "authorized_source_name_to_doc_ids",
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


def test_exact_section_locator_extraction_rejects_truncated_prose() -> None:
    assert contract_planner._exact_locators("Retrieve Section Overview.") == [
        "Section Overview"
    ]
    assert contract_planner._exact_locators("Section explains.") == []
    assert contract_planner._exact_locators("Section explains architecture.") == []


def test_exact_locator_extraction_preserves_dotted_identifiers() -> None:
    assert contract_planner._exact_locators(
        "Compare Section 3.2, Table 3.2, Figure 1.2, and Equation 2.1."
    ) == ["Section 3.2", "Table 3.2", "Figure 1.2", "Equation 2.1"]


def test_extracted_dotted_locator_matches_only_its_exact_metadata() -> None:
    locator = contract_planner._exact_locators("Retrieve Table 3.2.")[0]

    assert structured_locator_state(
        [locator], {"table_id": "Table 3.2"}
    ) == "matched"
    assert structured_locator_state(
        [locator], {"table_id": "Table 3.3"}
    ) == "mismatched"


def test_exact_locator_extraction_rejects_prose_without_swallowing_punctuation() -> None:
    assert contract_planner._exact_locators("Retrieve Table 3.2. Then summarize.") == [
        "Table 3.2"
    ]
    assert contract_planner._exact_locators("Table discusses ablation results.") == []


class _PlannerInvoker:
    def __init__(self, response: object = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def invoke(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_authoritative_source_mapping_survives_reordered_canonical_ids() -> None:
    contract = await QuestionContractPlanner().plan(
        question="From nnMamba.pdf, report the value in Table 2.",
        authorized_source_names=["nnMamba.pdf", "Other.pdf"],
        authorized_source_doc_ids=["doc-a", "doc-z"],
        authorized_source_name_to_doc_ids={
            "nnMamba.pdf": ["doc-z"],
            "Other.pdf": ["doc-a"],
        },
        setup_policy={},
    )

    assert contract.required_slots[0].source_name_hints == ["nnMamba.pdf"]
    assert contract.required_slots[0].authorized_source_doc_ids == ["doc-z"]


@pytest.mark.asyncio
async def test_multisource_direct_planner_without_mapping_degrades_without_zip_pairing() -> (
    None
):
    contract = await QuestionContractPlanner().plan(
        question="From nnMamba.pdf, report the value in Table 2.",
        authorized_source_names=["nnMamba.pdf", "Other.pdf"],
        authorized_source_doc_ids=["doc-a", "doc-z"],
        setup_policy={},
    )

    assert contract.slot_plan_status == "degraded"
    assert contract.route_decision.decision_source == "safe_fallback"
    assert (
        contract.route_decision.fallback_reason
        == "authoritative_source_mapping_missing"
    )
    assert contract.required_slots[0].source_name_hints == [
        "nnMamba.pdf",
        "Other.pdf",
    ]
    assert contract.required_slots[0].authorized_source_doc_ids == [
        "doc-a",
        "doc-z",
    ]


@pytest.mark.asyncio
async def test_ambiguous_question_uses_one_contract_planning_call() -> None:
    invoker = _PlannerInvoker(
        {
            "content": json.dumps(
                {
                    "selected_route": "bounded_compare",
                    "slots": [
                        {
                            "description": "Compare the requested source-bound claims.",
                            "source_name_hints": ["paper.pdf"],
                            "authorized_source_doc_ids": ["doc-1"],
                            "locator_hints": [],
                            "expected_answer_type": "comparison",
                            "depends_on_slot_ids": [],
                            "visual_policy": "never",
                        }
                    ],
                    "route_reason": "The question has an ambiguous comparison.",
                    "confidence": 0.8,
                }
            )
        }
    )

    contract = await QuestionContractPlanner(llm_invoker=invoker).plan(
        question="Please help me understand how these claims relate.",
        authorized_source_names=["paper.pdf"],
        authorized_source_doc_ids=["doc-1"],
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 512},
    )

    assert len(invoker.calls) == 1
    assert invoker.calls[0]["phase"] == "contract_planning"
    assert contract.route_decision.decision_source == "llm_planner"
    assert contract.route_decision.planner_call_used is True
    assert contract.slot_plan_status == "complete"


@pytest.mark.asyncio
async def test_ambiguity_prompt_contains_authoritative_source_mapping() -> None:
    invoker = _PlannerInvoker(
        {
            "content": json.dumps(
                {
                    "selected_route": "single_lookup",
                    "slots": [
                        {
                            "description": "Retrieve the requested fact.",
                            "source_name_hints": ["nnMamba.pdf"],
                            "authorized_source_doc_ids": ["doc-z"],
                            "locator_hints": [],
                            "expected_answer_type": "text",
                            "depends_on_slot_ids": [],
                            "visual_policy": "never",
                        }
                    ],
                    "route_reason": "One source-bound lookup.",
                    "confidence": 0.8,
                }
            )
        }
    )

    contract = await QuestionContractPlanner(llm_invoker=invoker).plan(
        question="Please investigate this unclear request.",
        authorized_source_names=["nnMamba.pdf", "Other.pdf"],
        authorized_source_doc_ids=["doc-a", "doc-z"],
        authorized_source_name_to_doc_ids={
            "nnMamba.pdf": ["doc-z"],
            "Other.pdf": ["doc-a"],
        },
        setup_policy={},
    )

    prompt = invoker.calls[0]["messages"][0]["content"]
    assert '"nnMamba.pdf": ["doc-z"]' in prompt
    assert '"Other.pdf": ["doc-a"]' in prompt
    assert contract.slot_plan_status == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("route_reason", "The answer is 0.9079."),
        ("locator_hints", ["0.9079"]),
        ("locator_hints", ["Section explains"]),
        ("locator_hints", ["Section explains architecture"]),
        ("depends_on_slot_ids", ["S99"]),
    ],
)
async def test_planner_authored_answer_values_and_invalid_dependencies_degrade(
    field: str, value: object
) -> None:
    slot = {
        "description": "Retrieve the requested source-bound fact.",
        "source_name_hints": ["paper.pdf"],
        "authorized_source_doc_ids": ["doc-1"],
        "locator_hints": [],
        "expected_answer_type": "text",
        "depends_on_slot_ids": [],
        "visual_policy": "never",
    }
    payload = {
        "selected_route": "single_lookup",
        "slots": [slot],
        "route_reason": "The request needs one lookup.",
        "confidence": 0.8,
    }
    if field in slot:
        slot[field] = value
    else:
        payload[field] = value
    invoker = _PlannerInvoker({"content": json.dumps(payload)})

    contract = await QuestionContractPlanner(llm_invoker=invoker).plan(
        question="Please investigate this unclear request.",
        authorized_source_names=["paper.pdf"],
        authorized_source_doc_ids=["doc-1"],
        setup_policy={},
    )

    assert contract.slot_plan_status == "degraded"
    assert contract.route_decision.fallback_reason == "invalid_planner_output"


@pytest.mark.asyncio
async def test_planner_rejects_authorized_id_paired_with_wrong_source_name() -> None:
    invoker = _PlannerInvoker(
        {
            "content": json.dumps(
                {
                    "selected_route": "single_lookup",
                    "slots": [
                        {
                            "description": "Retrieve the requested fact.",
                            "source_name_hints": ["nnMamba.pdf"],
                            "authorized_source_doc_ids": ["doc-a"],
                            "locator_hints": [],
                            "expected_answer_type": "text",
                            "depends_on_slot_ids": [],
                            "visual_policy": "never",
                        }
                    ],
                    "route_reason": "One source-bound lookup.",
                    "confidence": 0.8,
                }
            )
        }
    )

    contract = await QuestionContractPlanner(llm_invoker=invoker).plan(
        question="Please investigate this unclear request.",
        authorized_source_names=["nnMamba.pdf", "Other.pdf"],
        authorized_source_doc_ids=["doc-a", "doc-z"],
        authorized_source_name_to_doc_ids={
            "nnMamba.pdf": ["doc-z"],
            "Other.pdf": ["doc-a"],
        },
        setup_policy={},
    )

    assert contract.slot_plan_status == "degraded"
    assert contract.route_decision.fallback_reason == "unauthorized_source_expansion"


@pytest.mark.asyncio
async def test_preferred_visual_slot_is_requested_but_not_required() -> None:
    contract = await QuestionContractPlanner().plan(
        question="What is the table score?",
        authorized_source_names=["paper.pdf"],
        authorized_source_doc_ids=["doc-1"],
        setup_policy={},
    )

    assert contract.required_slots[0].visual_policy == "preferred"
    assert contract.visual_requested is True
    assert contract.visual_required is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error", "fallback_reason"),
    [
        ({"content": "not-json"}, None, "invalid_planner_output"),
        (None, TimeoutError("slow"), "planner_timeout"),
        (None, BudgetExceededError("budget"), "planner_budget_rejected"),
        (
            {
                "content": json.dumps(
                    {
                        "selected_route": "single_lookup",
                        "slots": [
                            {
                                "description": "Retrieve the requested fact.",
                                "source_name_hints": ["outside.pdf"],
                                "authorized_source_doc_ids": ["outside"],
                                "locator_hints": [],
                                "expected_answer_type": "text",
                                "depends_on_slot_ids": [],
                                "visual_policy": "never",
                            }
                        ],
                        "route_reason": "Expanded scope.",
                        "confidence": 0.7,
                    }
                )
            },
            None,
            "unauthorized_source_expansion",
        ),
    ],
)
async def test_planner_failures_return_degraded_safe_fallback(
    response: object, error: Exception | None, fallback_reason: str
) -> None:
    invoker = _PlannerInvoker(response=response, error=error)

    contract = await QuestionContractPlanner(llm_invoker=invoker).plan(
        question="Please investigate this unclear request.",
        authorized_source_names=["paper.pdf"],
        authorized_source_doc_ids=["doc-1"],
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 512},
    )

    assert len(invoker.calls) == 1
    assert contract.route_decision.decision_source == "safe_fallback"
    assert contract.route_decision.fallback_reason == fallback_reason
    assert contract.slot_plan_status == "degraded"
    assert contract.required_slots[0].authorized_source_doc_ids == ["doc-1"]
