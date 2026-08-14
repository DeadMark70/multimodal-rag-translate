"""Contracts for bounded, source-authorized Agentic v9 evidence repair."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from data_base.agentic_v9.repair import build_repair_plan
from data_base.agentic_v9.schemas import (
    ComparisonPlan,
    ComparisonSubject,
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
    ResponseConstraint,
    SlotResolution,
    SourceLocator,
    SufficiencyReport,
    SynthesisObligation,
)
from data_base.agentic_v9.sufficiency_gate import (
    SufficiencyEvaluation,
    evaluate_sufficiency,
)
from evaluation.trace_schemas import AgenticV9TracePayload


def _contract(*, route: str = "exact_structured", repair_rounds: int = 1) -> QueryContract:
    return QueryContract(
        route=route,
        intent="Retrieve source-bound evidence only.",
        entities=["global entity that is not the original question"],
        locator_hints=["Appendix"],
        required_slots=[
            RequiredSlot(
                slot_id="theorem-range",
                description="Theorem 1 m range",
                entity_ids=["GEPAR3D"],
                locator_hints=["Theorem 1"],
            ),
            RequiredSlot(
                slot_id="noise-score",
                description="noise robustness score",
                entity_ids=["ODES"],
                locator_hints=["Table 3"],
            ),
        ],
        max_repair_rounds=repair_rounds,
        resolved_source_scope=ResolvedSourceScope(
            requested_doc_ids=["gepar", "odes"],
            resolved_doc_ids=["gepar", "odes"],
            authorized_doc_ids=["gepar", "odes"],
        ),
    )


def _packet_for_slot(
    evidence_id: str,
    slot_id: str,
    doc_id: str,
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id=f"task:{slot_id}",
        round_id="round-1",
        query_id="q",
        slot_ids=[slot_id],
        statement=f"Evidence for {slot_id}.",
        support_type="direct",
        source=EvidenceSource(doc_id=doc_id, chunk_id=evidence_id),
        scope=EvidenceScope(),
        locator=SourceLocator(section="retrieved_context"),
        validation_status="deterministic_valid",
    )


def test_repair_query_derives_only_from_missing_slot_entity_and_locator() -> None:
    contract = _contract()
    sufficiency = evaluate_sufficiency(contract, [])

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="query-17",
        repair_round_index=1,
        final_budget_available=True,
    )

    assert len(plan.tasks) == 2
    assert plan.tasks[0].target_slot_ids == ["theorem-range"]
    assert plan.tasks[0].query == "GEPAR3D Theorem 1 m range Theorem 1"
    assert plan.tasks[1].query == "ODES noise robustness score Table 3"
    assert "original question" not in " ".join(task.query for task in plan.tasks)
    assert all(task.source_scope.authorized_doc_ids == ["gepar", "odes"] for task in plan.tasks)
    assert all("answer" not in task.model_dump() for task in plan.tasks)


def test_repair_stops_at_route_or_contract_round_cap() -> None:
    contract = _contract(route="bounded_compare", repair_rounds=2)
    sufficiency = evaluate_sufficiency(contract, [])

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="query-18",
        repair_round_index=2,
        final_budget_available=True,
    )

    assert plan.tasks == []
    assert plan.stop_reason == "repair_round_cap_reached"


def test_comparison_repair_targets_only_the_missing_subject() -> None:
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-a", "doc-b"],
        resolved_doc_ids=["doc-a", "doc-b"],
        authorized_doc_ids=["doc-a", "doc-b"],
    )
    comparison = ComparisonPlan(
        subjects=[
            ComparisonSubject(
                subject_id="nnmamba",
                display_name="nnMamba",
                aliases=["Mamba model"],
                retrieval_query="nnMamba efficiency",
            ),
            ComparisonSubject(
                subject_id="efficientmednext_l",
                display_name="EfficientMedNeXt-L",
                aliases=["Efficient MedNeXt L"],
                retrieval_query="EfficientMedNeXt-L efficiency",
            ),
        ],
        dimensions=["parameters", "FLOPs"],
    )
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare efficiency.",
        required_slots=[
            RequiredSlot(
                slot_id="comparison-subject:nnmamba",
                description="Find nnMamba efficiency evidence.",
                entity_ids=["nnMamba"],
            ),
            RequiredSlot(
                slot_id="comparison-subject:efficientmednext_l",
                description="Find EfficientMedNeXt-L efficiency evidence.",
                entity_ids=["EfficientMedNeXt-L"],
            ),
        ],
        comparison_plan=comparison,
        max_repair_rounds=5,
        resolved_source_scope=scope,
    )
    sufficiency = SufficiencyEvaluation(
        slot_resolutions=(
            SlotResolution(
                slot_id="comparison-subject:nnmamba",
                status="supported",
                evidence_ids=["evidence-a"],
            ),
            SlotResolution(
                slot_id="comparison-subject:efficientmednext_l",
                status="not_found",
            ),
        ),
        report=SufficiencyReport(
            evidence_complete=False,
            answerable=True,
            response_status="qualified_partial",
            supported_slot_ids=["comparison-subject:nnmamba"],
            not_found_slot_ids=["comparison-subject:efficientmednext_l"],
        ),
        repairable_slot_ids=("comparison-subject:efficientmednext_l",),
    )

    repair = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="q4",
        repair_round_index=1,
        final_budget_available=True,
    )

    assert len(repair.tasks) == 1
    assert repair.tasks[0].subject_id == "efficientmednext_l"
    assert repair.tasks[0].target_slot_ids == [
        "comparison-subject:efficientmednext_l"
    ]
    assert "EfficientMedNeXt-L" in repair.tasks[0].query
    assert "Efficient MedNeXt L" in repair.tasks[0].query
    assert "parameters" in repair.tasks[0].query
    assert "FLOPs" in repair.tasks[0].query
    assert repair.tasks[0].source_scope == scope


def test_comparison_repair_has_exactly_one_round_and_skips_complete_coverage() -> None:
    scope = ResolvedSourceScope(authorized_doc_ids=["doc-a"])
    comparison = ComparisonPlan(
        subjects=[
            ComparisonSubject(
                subject_id="a",
                display_name="Model A",
                retrieval_query="Model A efficiency",
            ),
            ComparisonSubject(
                subject_id="b",
                display_name="Model B",
                retrieval_query="Model B efficiency",
            ),
        ],
        dimensions=["efficiency"],
    )
    contract = QueryContract(
        route="multi_document_exact",
        intent="Compare models.",
        required_slots=[
            RequiredSlot(
                slot_id=f"comparison-subject:{subject.subject_id}",
                description=f"Find {subject.display_name}.",
            )
            for subject in comparison.subjects
        ],
        comparison_plan=comparison,
        max_repair_rounds=5,
        resolved_source_scope=scope,
    )
    missing = evaluate_sufficiency(contract, [])

    first_round = build_repair_plan(
        contract=contract,
        sufficiency=missing,
        query_id="q",
        repair_round_index=1,
        final_budget_available=True,
    )

    assert len(first_round.tasks) == 1
    assert first_round.tasks[0].subject_id == "a"

    capped = build_repair_plan(
        contract=contract,
        sufficiency=missing,
        query_id="q",
        repair_round_index=2,
        final_budget_available=True,
    )

    assert capped.tasks == []
    assert capped.stop_reason == "repair_round_cap_reached"

    complete_packets = [
        _packet_for_slot("e-a", "comparison-subject:a", "doc-a"),
        _packet_for_slot("e-b", "comparison-subject:b", "doc-a"),
    ]
    complete = build_repair_plan(
        contract=contract,
        sufficiency=evaluate_sufficiency(contract, complete_packets),
        query_id="q",
        repair_round_index=1,
        final_budget_available=True,
    )
    assert complete.tasks == []
    assert complete.stop_reason == "no_repairable_slots"


def test_repair_never_runs_when_the_final_budget_is_not_protected() -> None:
    contract = _contract()
    sufficiency = evaluate_sufficiency(contract, [])

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="query-19",
        repair_round_index=1,
        final_budget_available=False,
    )

    assert plan.tasks == []
    assert plan.stop_reason == "final_budget_protected"


def test_single_lookup_never_repairs_even_when_a_slot_is_missing() -> None:
    contract = _contract(route="single_lookup", repair_rounds=1)
    sufficiency = evaluate_sufficiency(contract, [])

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="query-20",
        repair_round_index=1,
        final_budget_available=True,
    )

    assert plan.tasks == []
    assert plan.stop_reason == "repair_round_cap_reached"


def test_atomic_repair_groups_only_required_not_found_slots_by_constraints() -> None:
    scope = ResolvedSourceScope(
        requested_doc_ids=["gepar", "odes", "ukan"],
        resolved_doc_ids=["gepar", "odes", "ukan"],
        authorized_doc_ids=["gepar", "odes", "ukan"],
        source_name_to_doc_ids={
            "GEPAR3D.pdf": ["gepar"],
            "ODES.pdf": ["odes"],
            "Implicit-U-KAN2.0.pdf": ["ukan"],
        },
    )
    slots = [
        RequiredSlot(
            slot_id="S1",
            description="already supported",
            authorized_source_doc_ids=["gepar"],
        ),
        RequiredSlot(
            slot_id="S2",
            description="conflicted value",
            authorized_source_doc_ids=["odes"],
        ),
        RequiredSlot(
            slot_id="S3",
            description="explicitly unavailable appendix",
            authorized_source_doc_ids=["odes"],
        ),
        RequiredSlot(
            slot_id="S4",
            description="U-KAN metric",
            entity_ids=["Implicit-U-KAN2.0"],
            source_name_hints=["Implicit-U-KAN2.0.pdf"],
            authorized_source_doc_ids=["ukan"],
            locator_hints=["Table 3"],
        ),
        RequiredSlot(
            slot_id="S5",
            description="proposed method metric",
            entity_ids=["Implicit-U-KAN2.0"],
            source_name_hints=["Implicit-U-KAN2.0.pdf"],
            authorized_source_doc_ids=["ukan"],
            locator_hints=["Table 3"],
        ),
        RequiredSlot(
            slot_id="S6",
            description="theorem boundary",
            entity_ids=["Implicit-U-KAN2.0"],
            source_name_hints=["Implicit-U-KAN2.0.pdf"],
            authorized_source_doc_ids=["ukan"],
            locator_hints=["Theorem 1"],
        ),
    ]
    contract = QueryContract(
        contract_version="2",
        route="multi_document_exact",
        intent="atomic repair",
        required_slots=slots,
        max_repair_rounds=5,
        resolved_source_scope=scope,
    )
    resolutions = (
        SlotResolution(
            slot_id="S1", status="supported", evidence_ids=["evidence-supported"]
        ),
        SlotResolution(
            slot_id="S2",
            status="conflicted",
            evidence_ids=["evidence-a", "evidence-b"],
        ),
        SlotResolution(slot_id="S3", status="explicitly_unavailable"),
        SlotResolution(slot_id="S4", status="not_found"),
        SlotResolution(slot_id="S5", status="not_found"),
        SlotResolution(slot_id="S6", status="not_found"),
    )
    sufficiency = SufficiencyEvaluation(
        slot_resolutions=resolutions,
        report=SufficiencyReport(
            evidence_complete=False,
            answerable=True,
            response_status="qualified_partial",
            supported_slot_ids=["S1"],
            conflicted_slot_ids=["S2"],
            explicitly_unavailable_slot_ids=["S3"],
            not_found_slot_ids=["S4", "S5", "S6"],
        ),
        repairable_slot_ids=("S1", "S2", "S3", "S4", "S5", "S6"),
    )

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="Q16",
        repair_round_index=1,
        final_budget_available=True,
    )

    assert [task.target_slot_ids for task in plan.tasks] == [["S4", "S5"], ["S6"]]
    assert all(task.source_scope.authorized_doc_ids == ["ukan"] for task in plan.tasks)
    assert [task.locator_hints for task in plan.tasks] == [
        ["Table 3"],
        ["Theorem 1"],
    ]
    assert "U-KAN metric" in plan.tasks[0].query
    assert "proposed method metric" in plan.tasks[0].query
    assert "theorem boundary" in plan.tasks[1].query
    assert all(
        excluded not in task.target_slot_ids
        for task in plan.tasks
        for excluded in ("S1", "S2", "S3")
    )
    assert plan.resulting_evidence_ids == []


def test_repair_has_an_absolute_two_round_cap() -> None:
    contract = _contract(route="multi_document_exact", repair_rounds=5)
    sufficiency = evaluate_sufficiency(contract, [])

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="query-absolute-cap",
        repair_round_index=3,
        final_budget_available=True,
    )

    assert plan.tasks == []
    assert plan.stop_reason == "repair_round_cap_reached"


def test_persisted_trace_rejects_executed_repair_without_stop_reason() -> None:
    contract = _contract()
    plan = build_repair_plan(
        contract=contract,
        sufficiency=evaluate_sufficiency(contract, []),
        query_id="query-trace",
        repair_round_index=1,
        final_budget_available=True,
    )

    with pytest.raises(ValidationError, match="persisted stop reason"):
        AgenticV9TracePayload(repairs=[plan])


def test_equivalent_locator_and_term_variants_group_before_two_task_cap() -> None:
    scope = ResolvedSourceScope(authorized_doc_ids=["ukan"])
    contract = QueryContract(
        contract_version="2",
        route="multi_document_exact",
        intent="repair ordered atomic facts",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="first Table 3 metric",
                entity_ids=["U-KAN", "Metric"],
                authorized_source_doc_ids=["ukan"],
                locator_hints=[" Table 3 "],
            ),
            RequiredSlot(
                slot_id="S2",
                description="second Table 3 metric",
                entity_ids=[" metric ", "u-kan"],
                authorized_source_doc_ids=["ukan"],
                locator_hints=["table   3"],
            ),
            RequiredSlot(
                slot_id="S3",
                description="Theorem 1 boundary",
                entity_ids=["U-KAN"],
                authorized_source_doc_ids=["ukan"],
                locator_hints=["Theorem 1"],
            ),
            RequiredSlot(
                slot_id="S4",
                description="Appendix A qualification",
                entity_ids=["U-KAN"],
                authorized_source_doc_ids=["ukan"],
                locator_hints=["Appendix A"],
            ),
        ],
        max_repair_rounds=2,
        resolved_source_scope=scope,
    )
    sufficiency = evaluate_sufficiency(contract, [])

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="canonical-grouping",
        repair_round_index=1,
        final_budget_available=True,
    )

    assert [task.target_slot_ids for task in plan.tasks] == [
        ["S1", "S2"],
        ["S3"],
    ]
    assert plan.tasks[0].locator_hints == ["Table 3"]


def test_atomic_comparison_repair_targets_only_missing_slots_of_first_missing_subject() -> None:
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-a", "doc-b"],
        resolved_doc_ids=["doc-a", "doc-b"],
        authorized_doc_ids=["doc-a", "doc-b"],
        source_name_to_doc_ids={
            "DocA.pdf": ["doc-a"],
            "DocB.pdf": ["doc-b"],
        },
    )
    contract = QueryContract(
        contract_version="2",
        route="bounded_compare",
        intent="Compare Model A and Model B.",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="Model A parameters",
                entity_ids=["Model A"],
                authorized_source_doc_ids=["doc-a"],
                locator_hints=["Table 1"],
            ),
            RequiredSlot(
                slot_id="S2",
                description="Model A FLOPs",
                entity_ids=["Model A"],
                authorized_source_doc_ids=["doc-a"],
                locator_hints=["Table 1"],
            ),
            RequiredSlot(
                slot_id="S3",
                description="Model B parameters",
                entity_ids=["Model B"],
                authorized_source_doc_ids=["doc-b"],
                locator_hints=["Table 2"],
            ),
            RequiredSlot(
                slot_id="S4",
                description="Model B FLOPs",
                entity_ids=["Model B"],
                authorized_source_doc_ids=["doc-b"],
                locator_hints=["Table 2"],
            ),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Synthesize comparison",
                depends_on_slot_ids=["S1", "S2", "S3", "S4"],
            )
        ],
        response_constraints=[
            ResponseConstraint(
                constraint_id="C1",
                kind="output_format",
                description="Present in markdown table",
            )
        ],
        comparison_plan=ComparisonPlan(
            subjects=[
                ComparisonSubject(
                    subject_id="model_a",
                    display_name="Model A",
                    retrieval_query="Model A parameters FLOPs",
                    evidence_slot_ids=["S1", "S2"],
                ),
                ComparisonSubject(
                    subject_id="model_b",
                    display_name="Model B",
                    retrieval_query="Model B parameters FLOPs",
                    evidence_slot_ids=["S3", "S4"],
                ),
            ],
            dimensions=["parameters", "FLOPs"],
        ),
        max_repair_rounds=1,
        resolved_source_scope=scope,
    )
    sufficiency = SufficiencyEvaluation(
        slot_resolutions=(
            SlotResolution(slot_id="S1", status="supported", evidence_ids=["e1"]),
            SlotResolution(slot_id="S2", status="supported", evidence_ids=["e2"]),
            SlotResolution(slot_id="S3", status="supported", evidence_ids=["e3"]),
            SlotResolution(slot_id="S4", status="not_found"),
        ),
        report=SufficiencyReport(
            evidence_complete=False,
            answerable=True,
            response_status="qualified_partial",
            supported_slot_ids=["S1", "S2", "S3"],
            not_found_slot_ids=["S4"],
        ),
        repairable_slot_ids=("S4",),
    )

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="q-repair-atomic",
        repair_round_index=1,
        final_budget_available=True,
    )

    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.subject_id == "model_b"
    assert task.target_slot_ids == ["S4"]
    assert "O1" not in task.target_slot_ids
    assert "C1" not in task.target_slot_ids
    assert task.source_scope.authorized_doc_ids == ["doc-b"]
    assert "Model B" in task.query
    assert "Table 2" in task.locator_hints

