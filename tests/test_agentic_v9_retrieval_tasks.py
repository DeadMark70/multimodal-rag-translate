"""Focused contracts for compiling v9 retrieval-only task plans."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from data_base.agentic_v9.retrieval_tasks import RetrievalTaskCompiler
from data_base.agentic_v9.schemas import (
    ComparisonPlan,
    ComparisonSubject,
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


def test_comparison_overlay_compiles_one_subject_bound_task_per_subject() -> None:
    scope = _scope("nnmamba-doc", "mednext-doc")
    comparison = ComparisonPlan(
        subjects=[
            ComparisonSubject(
                subject_id="nnmamba",
                display_name="nnMamba",
                aliases=["Mamba model"],
                retrieval_query="nnMamba parameters FLOPs efficiency",
            ),
            ComparisonSubject(
                subject_id="efficientmednext_l",
                display_name="EfficientMedNeXt-L",
                aliases=["Efficient MedNeXt L"],
                retrieval_query="EfficientMedNeXt-L parameters FLOPs efficiency",
            ),
        ],
        dimensions=["parameters", "FLOPs", "computational efficiency"],
    )
    contract = QueryContract(
        route="exact_structured",
        intent="Compare bounded source evidence.",
        required_slots=[
            RequiredSlot(
                slot_id=f"comparison-subject:{subject.subject_id}",
                description=f"Find evidence for {subject.display_name}.",
                entity_ids=[subject.display_name, *subject.aliases],
                expected_answer_type="comparison",
            )
            for subject in comparison.subjects
        ],
        comparison_plan=comparison,
        max_retrieval_rounds=1,
        max_repair_rounds=1,
        max_llm_calls=4,
        runtime_token_budget=40_000,
        resolved_source_scope=scope,
    )

    plan = RetrievalTaskCompiler().compile(
        question="Does nnMamba have higher efficiency than EfficientMedNeXt-L?",
        query_id="Q4",
        contract=contract,
    )

    assert [
        (task.subject_id, task.target_slot_ids, task.query) for task in plan.tasks
    ] == [
        (
            "nnmamba",
            ["comparison-subject:nnmamba"],
            "nnMamba parameters FLOPs efficiency",
        ),
        (
            "efficientmednext_l",
            ["comparison-subject:efficientmednext_l"],
            "EfficientMedNeXt-L parameters FLOPs efficiency",
        ),
    ]
    assert all(task.round_id == "round-1" for task in plan.tasks)
    assert all(task.depends_on_task_ids == [] for task in plan.tasks)
    assert all(task.source_scope == scope for task in plan.tasks)
    assert all(task.graph_policy == "locator_fallback" for task in plan.tasks)


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


def test_q16_compiles_atomic_source_and_locator_groups_without_answer_text() -> None:
    scope = ResolvedSourceScope(
        requested_doc_ids=["gepar", "odes", "ukan"],
        requested_source_names=["GEPAR3D.pdf", "ODES.pdf", "Implicit-U-KAN2.0.pdf"],
        resolved_doc_ids=["gepar", "odes", "ukan"],
        authorized_doc_ids=["gepar", "odes", "ukan"],
        source_name_to_doc_ids={
            "GEPAR3D.pdf": ["gepar"],
            "ODES.pdf": ["odes"],
            "Implicit-U-KAN2.0.pdf": ["ukan"],
        },
    )
    contract = QueryContract(
        contract_version="2",
        route="multi_document_exact",
        intent="Retrieve seven atomic facts.",
        entities=["GEPAR3D", "ODES", "Implicit-U-KAN2.0"],
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="Identify the GEPAR3D penalty.",
                entity_ids=["GEPAR3D"],
                source_name_hints=["GEPAR3D.pdf"],
                authorized_source_doc_ids=["gepar"],
                locator_hints=["penalty definition"],
            ),
            RequiredSlot(
                slot_id="S3",
                description="Transcribe the ODES equation.",
                entity_ids=["ODES"],
                source_name_hints=["ODES.pdf"],
                authorized_source_doc_ids=["odes"],
                locator_hints=["Equation 2"],
                expected_answer_type="equation",
            ),
            RequiredSlot(
                slot_id="S5",
                description="Report the U-KAN metric.",
                entity_ids=["Implicit-U-KAN2.0"],
                source_name_hints=["Implicit-U-KAN2.0.pdf"],
                authorized_source_doc_ids=["ukan"],
                locator_hints=["Table 3"],
                expected_answer_type="number",
            ),
            RequiredSlot(
                slot_id="S6",
                description="Report the proposed method metric.",
                entity_ids=["Implicit-U-KAN2.0"],
                source_name_hints=["Implicit-U-KAN2.0.pdf"],
                authorized_source_doc_ids=["ukan"],
                locator_hints=["Table 3"],
                expected_answer_type="number",
            ),
            RequiredSlot(
                slot_id="S7",
                description="State the theorem boundary.",
                entity_ids=["Implicit-U-KAN2.0"],
                source_name_hints=["Implicit-U-KAN2.0.pdf"],
                authorized_source_doc_ids=["ukan"],
                locator_hints=["Theorem 1"],
            ),
        ],
        max_retrieval_rounds=2,
        max_llm_calls=1,
        runtime_token_budget=1,
        resolved_source_scope=scope,
        slot_plan_status="complete",
    )

    plan = RetrievalTaskCompiler().compile(
        question="Expected ODES answer is SECRET-GOLD and U-KAN is 0.9079.",
        query_id="Q16",
        contract=contract,
    )

    groups = {tuple(task.target_slot_ids): task for task in plan.tasks}
    assert groups[("S3",)].source_scope.authorized_doc_ids == ["odes"]
    assert groups[("S3",)].locator_hints == ["Equation 2"]
    assert groups[("S5", "S6")].source_scope.authorized_doc_ids == ["ukan"]
    assert groups[("S5", "S6")].locator_hints == ["Table 3"]
    assert groups[("S7",)].source_scope.authorized_doc_ids == ["ukan"]
    assert groups[("S7",)].locator_hints == ["Theorem 1"]
    assert all("SECRET-GOLD" not in task.query for task in plan.tasks)
    assert all("0.9079" not in task.query for task in plan.tasks)
    assert "ODES.pdf" in groups[("S3",)].query
    assert "Transcribe the ODES equation." in groups[("S3",)].query


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
