"""Focused contracts for compiling v9 retrieval-only task plans."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from data_base.agentic_v9.retrieval_tasks import RetrievalTaskCompiler
from data_base.agentic_v9.schemas import (
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
)


def _scope(*doc_ids: str) -> ResolvedSourceScope:
    return ResolvedSourceScope(
        requested_doc_ids=list(doc_ids),
        resolved_doc_ids=list(doc_ids),
        authorized_doc_ids=list(doc_ids),
    )


def _contract(
    *,
    route: str,
    entities: list[str],
    locator_hints: list[str],
    scope: ResolvedSourceScope,
    visual_required: bool = False,
) -> QueryContract:
    return QueryContract(
        route=route,
        intent="Retrieve source-bound evidence only.",
        entities=entities,
        locator_hints=locator_hints,
        required_slots=[
            RequiredSlot(
                slot_id="slot-main",
                description="Resolve the requested source-bound fact.",
                locator_hints=locator_hints,
            ),
            RequiredSlot(
                slot_id="slot-qualification",
                description="Resolve scope and qualification constraints.",
                locator_hints=locator_hints,
            ),
        ],
        visual_required=visual_required,
        max_retrieval_rounds=2,
        max_llm_calls=1,
        runtime_token_budget=1,
        resolved_source_scope=scope,
    )


def test_q9_compiles_bounded_a_b_tasks_before_a_dependent_qualification_task() -> None:
    contract = _contract(
        route="bounded_compare",
        entities=["SwinUNETR", "nnU-Net"],
        locator_hints=["table"],
        scope=_scope("swin", "nnunet"),
    )

    plan = RetrievalTaskCompiler().compile(
        question="In 3D medical image segmentation, which performs better: SwinUNETR or nnU-Net?",
        query_id="Q9",
        contract=contract,
    )

    assert [task.round_id for task in plan.tasks] == ["round-1", "round-1", "round-2"]
    assert [task.query for task in plan.tasks[:2]] == [
        "SwinUNETR: In 3D medical image segmentation, which performs better: SwinUNETR or nnU-Net?",
        "nnU-Net: In 3D medical image segmentation, which performs better: SwinUNETR or nnU-Net?",
    ]
    assert all(task.target_slot_ids == ["slot-main"] for task in plan.tasks[:2])
    assert plan.tasks[2].target_slot_ids == ["slot-qualification"]
    assert plan.tasks[2].depends_on_task_ids == [
        "Q9:round-1:compare-a",
        "Q9:round-1:compare-b",
    ]
    assert plan.tasks[2].graph_policy == "never"
    assert all(task.source_scope.authorized_doc_ids for task in plan.tasks)


def test_q15_preserves_asset_locators_and_visual_policy() -> None:
    contract = _contract(
        route="exact_structured",
        entities=["Polyp-SAM", "CVC-ClinicDB"],
        locator_hints=["figure", "table"],
        scope=_scope("polyp-sam"),
        visual_required=True,
    )

    plan = RetrievalTaskCompiler().compile(
        question="What are Polyp-SAM Figure 1(b) CVC-ClinicDB mIoU and the Table 1 batch size?",
        query_id="Q15",
        contract=contract,
    )

    task = plan.tasks[0]
    assert task.target_slot_ids == ["slot-main", "slot-qualification"]
    assert task.locator_hints == ["figure", "table"]
    assert task.visual_required is True
    assert task.graph_policy == "locator_fallback"
    assert task.source_group_id == "source-group-1"


def test_q16_partitions_authorized_sources_into_deterministic_source_groups() -> None:
    contract = _contract(
        route="multi_document_exact",
        entities=["GEPAR3D", "ODES", "Implicit-U-KAN2.0"],
        locator_hints=["appendix", "formula", "table"],
        scope=_scope("gepar", "odes", "ukan"),
    )

    plan = RetrievalTaskCompiler().compile(
        question="Retrieve GEPAR3D, ODES, and Implicit U-KAN2.0 penalties, formula, noise data, and theorem boundary.",
        query_id="Q16",
        contract=contract,
    )

    assert [task.source_group_id for task in plan.tasks] == [
        "source-group-1",
        "source-group-2",
        "source-group-3",
        "source-group-4",
    ]
    assert [task.source_scope.authorized_doc_ids for task in plan.tasks[:3]] == [
        ["gepar"],
        ["odes"],
        ["ukan"],
    ]
    assert plan.tasks[3].source_scope.authorized_doc_ids == ["gepar", "odes", "ukan"]
    assert plan.tasks[3].depends_on_task_ids == [
        "Q16:round-1:source-group-1",
        "Q16:round-1:source-group-2",
        "Q16:round-1:source-group-3",
    ]
    assert plan.tasks[3].target_slot_ids == ["slot-qualification"]


@pytest.mark.parametrize(
    ("query_id", "entities"),
    [
        ("Q1", ["SwinUNETR", "MedNeXt", "nnMamba"]),
        ("Q2", ["MedSAM", "SAM-Med3D", "MedSAM-2"]),
    ],
)
def test_q1_q2_multi_hop_tasks_have_round_two_dependencies(
    query_id: str, entities: list[str]
) -> None:
    contract = _contract(
        route="multi_hop",
        entities=entities,
        locator_hints=["source passage"],
        scope=_scope("doc-a", "doc-b", "doc-c"),
    )

    plan = RetrievalTaskCompiler().compile(
        question=f"Compare {' / '.join(entities)}.",
        query_id=query_id,
        contract=contract,
    )

    initial_tasks = plan.tasks[:3]
    dependent_task = plan.tasks[3]
    assert [task.round_id for task in initial_tasks] == ["round-1"] * 3
    assert all(task.target_slot_ids == ["slot-main"] for task in initial_tasks)
    assert dependent_task.round_id == "round-2"
    assert dependent_task.target_slot_ids == ["slot-qualification"]
    assert dependent_task.depends_on_task_ids == [task.task_id for task in initial_tasks]
    assert dependent_task.graph_policy == "locator_fallback"


def test_compiler_fails_closed_without_an_authorized_scope() -> None:
    contract = _contract(
        route="single_lookup",
        entities=["nnU-Net"],
        locator_hints=["source passage"],
        scope=ResolvedSourceScope(),
    )

    with pytest.raises(ValueError, match="authorized source scope"):
        RetrievalTaskCompiler().compile(
            question="What is nnU-Net?", query_id="Q10", contract=contract
        )


def test_tasks_are_typed_evidence_only_without_an_answer_field() -> None:
    contract = _contract(
        route="single_lookup",
        entities=["nnU-Net"],
        locator_hints=["source passage"],
        scope=_scope("nnunet"),
    )

    plan = RetrievalTaskCompiler().compile(
        question="What is the nnU-Net recipe?", query_id="Q10", contract=contract
    )

    assert "answer" not in plan.model_dump()
    assert all("answer" not in task.model_dump() for task in plan.tasks)
    with pytest.raises(ValidationError):
        type(plan.tasks[0])(**plan.tasks[0].model_dump(), answer="not permitted")
