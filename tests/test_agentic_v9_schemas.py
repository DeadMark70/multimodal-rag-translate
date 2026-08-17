"""Contract tests for the evidence-first Agentic v9 schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from data_base.agentic_v9.schemas import (
    ComparisonPlan,
    ComparisonSubject,
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    FinalAnswerDraft,
    FinalAnswerResult,
    FinalClaim,
    GraphPolicy,
    QueryContract,
    RagRetrievalResult,
    ResolvedSourceScope,
    RequiredSlot,
    ResponseConstraint,
    RouteDecision,
    SlotResolution,
    SourceLocator,
    SupportedFinding,
    SufficiencyReport,
    SynthesisObligation,
    TaskRetrievalResult,
    UnresolvedRequirement,
    V9ExecutionMetrics,
    V9ExecutionResult,
    V9ExecutionRequest,
    default_graph_policy,
    validate_active_atomic_contract,
)
from evaluation.trace_schemas import (
    AgentTraceDetail,
    EvaluationRoutingDecision,
    summarize_agent_trace,
)


def _evidence_packet(*, support_type: str = "direct") -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id="evidence-1",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["slot-1"],
        statement="The reported score is 0.9079.",
        support_type=support_type,
        source=EvidenceSource(
            doc_id="doc-1", chunk_id="chunk-1", document_name="paper.pdf"
        ),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(
            pdf_page_index=4, printed_page_label="5", table_id="table-2"
        ),
        raw_value=Decimal("0.9079"),
        normalized_value=Decimal("90.79"),
        unit="percent",
    )


def test_evidence_packet_requires_positive_provenance_and_disallows_missing_support() -> (
    None
):
    packet = _evidence_packet()

    assert packet.normalized_value == Decimal("90.79")
    assert packet.locator.pdf_page_index == 4
    assert packet.model_dump(mode="json")["raw_value"] == "0.9079"

    with pytest.raises(ValidationError):
        _evidence_packet(support_type="missing")

    with pytest.raises(ValidationError):
        EvidencePacket(
            schema_version="1",
            evidence_id="evidence-1",
            task_id="task-1",
            round_id="round-1",
            query_id="query-1",
            slot_ids=["slot-1"],
            statement="The reported score is 0.9079.",
            support_type="direct",
            scope=EvidenceScope(),
            locator=SourceLocator(pdf_page_index=4),
        )


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("single_lookup", "never"),
        ("bounded_compare", "never"),
        ("exact_structured", "locator_fallback"),
        ("multi_document_exact", "locator_fallback"),
        ("multi_hop", "locator_fallback"),
        ("graph_relational", "required_locator"),
    ],
)
def test_route_graph_policy_defaults_are_frozen(
    route: str, expected: GraphPolicy
) -> None:
    assert default_graph_policy(route) == expected


def test_slot_resolution_and_sufficiency_keep_absence_separate_from_evidence() -> None:
    report = SufficiencyReport(
        evidence_complete=False,
        answerable=True,
        response_status="qualified_partial",
        supported_slot_ids=["slot-1"],
        explicitly_unavailable_slot_ids=["slot-2"],
        not_found_slot_ids=["slot-3"],
        conflicted_slot_ids=["slot-4"],
        stop_reason="repair_budget_exhausted",
    )

    assert report.evidence_complete is False
    assert report.answerable is True
    assert (
        SlotResolution(slot_id="slot-2", status="explicitly_unavailable").status
        == "explicitly_unavailable"
    )

    with pytest.raises(ValidationError):
        SlotResolution(slot_id="slot-2", status="missing")


@pytest.mark.parametrize(
    ("status", "evidence_ids"),
    [
        ("supported", []),
        ("conflicted", []),
        ("conflicted", ["evidence-1"]),
        ("explicitly_unavailable", ["evidence-1"]),
        ("not_found", ["evidence-1"]),
    ],
)
def test_slot_resolution_rejects_incoherent_evidence_links(
    status: str, evidence_ids: list[str]
) -> None:
    with pytest.raises(ValidationError):
        SlotResolution(slot_id="slot-1", status=status, evidence_ids=evidence_ids)


def test_slot_resolution_accepts_only_positive_evidence_for_supported_or_conflicted() -> (
    None
):
    assert SlotResolution(
        slot_id="slot-1", status="supported", evidence_ids=["evidence-1"]
    ).evidence_ids == ["evidence-1"]
    assert (
        SlotResolution(
            slot_id="slot-1",
            status="conflicted",
            evidence_ids=["evidence-1", "evidence-2"],
        ).status
        == "conflicted"
    )


@pytest.mark.parametrize(
    "report",
    [
        {
            "evidence_complete": True,
            "answerable": True,
            "response_status": "qualified_partial",
            "explicitly_unavailable_slot_ids": ["slot-1"],
        },
        {
            "evidence_complete": True,
            "answerable": True,
            "response_status": "qualified_partial",
            "not_found_slot_ids": ["slot-1"],
        },
        {
            "evidence_complete": False,
            "answerable": True,
            "response_status": "complete",
        },
        {
            "evidence_complete": True,
            "answerable": False,
            "response_status": "complete",
        },
        {
            "evidence_complete": True,
            "answerable": True,
            "response_status": "complete",
            "conflicted_slot_ids": ["slot-1"],
        },
    ],
)
def test_sufficiency_report_rejects_internally_incoherent_completion(
    report: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SufficiencyReport(**report)


def test_request_serializes_only_requested_sources_and_scope_resolver_owns_authorization() -> (
    None
):
    request = V9ExecutionRequest(
        question="What is the reported score?",
        requested_doc_ids=["doc-1"],
        requested_source_names=["paper.pdf"],
        history=[{"role": "user", "content": "Use the paper."}],
        setup_snapshot={"model_name": "gemini-test", "max_output_tokens": 512},
        trace_id="trace-1",
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=request.requested_doc_ids,
        requested_source_names=request.requested_source_names,
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
        rejected_source_names=[],
    )

    payload = request.model_dump(mode="json")
    assert payload["requested_doc_ids"] == ["doc-1"]
    assert "authorized_doc_ids" not in payload
    assert "user_id" not in payload
    assert scope.authorized_doc_ids == ["doc-1"]


@pytest.mark.parametrize(
    ("route", "expected_graph_policy"),
    [
        ("single_lookup", "never"),
        ("bounded_compare", "never"),
        ("exact_structured", "locator_fallback"),
        ("multi_document_exact", "locator_fallback"),
        ("multi_hop", "locator_fallback"),
        ("graph_relational", "required_locator"),
    ],
)
def test_query_contract_applies_the_model_default_graph_policy_for_each_route(
    route: str, expected_graph_policy: GraphPolicy
) -> None:
    contract = QueryContract(route=route, intent="test routing")

    assert contract.graph_policy == expected_graph_policy


def test_query_contract_rejects_a_runtime_budget_without_provider_call_budget() -> None:
    with pytest.raises(ValidationError):
        QueryContract(
            route="single_lookup",
            intent="Find the reported score.",
            max_llm_calls=0,
            runtime_token_budget=1,
        )


def test_query_contract_v2_carries_atomic_slot_and_route_provenance() -> None:
    route_decision = RouteDecision(
        selected_route="multi_document_exact",
        decision_source="deterministic",
        matched_rules=["numbered_subquestions", "multiple_named_sources"],
        candidate_routes=["multi_document_exact", "exact_structured"],
        route_reason="Multiple exact requirements span named sources.",
        planner_call_used=False,
        confidence=1.0,
    )
    slot = RequiredSlot(
        slot_id="S1",
        description="Retrieve the reported metric.",
        source_name_hints=["paper.pdf"],
        authorized_source_doc_ids=["doc-1"],
        expected_answer_type="number",
        depends_on_slot_ids=[],
        visual_policy="preferred",
    )

    contract = QueryContract(
        contract_version="2",
        route="multi_document_exact",
        intent="Resolve each requested fact.",
        required_slots=[slot],
        route_decision=route_decision,
        slot_plan_status="complete",
    )

    assert contract.slot_semantics == "heuristic_experimental"
    assert contract.atomic_completeness is None
    assert contract.atomic_completeness_reason == "atomic_slot_matching_experimental"
    assert contract.slot_plan_status == "complete"
    assert contract.route_decision == route_decision
    assert contract.required_slots[0].model_dump() == {
        "slot_id": "S1",
        "description": "Retrieve the reported metric.",
        "required": True,
        "entity_ids": [],
        "locator_hints": [],
        "source_name_hints": ["paper.pdf"],
        "authorized_source_doc_ids": ["doc-1"],
        "expected_answer_type": "number",
        "depends_on_slot_ids": [],
        "visual_policy": "preferred",
    }


def test_query_contract_v2_missing_slot_plan_status_stays_missing() -> None:
    contract = QueryContract(
        contract_version="2",
        route="single_lookup",
        intent="Resolve one atomic source-bound fact.",
        required_slots=[RequiredSlot(slot_id="S1", description="Retrieve the fact.")],
    )

    assert contract.slot_semantics == "heuristic_experimental"
    assert contract.atomic_completeness is None
    assert contract.atomic_completeness_reason == "atomic_slot_matching_experimental"
    assert contract.slot_plan_status is None


def test_query_contract_v1_projects_legacy_generic_atomic_completeness_na() -> None:
    contract = QueryContract(
        route="single_lookup",
        intent="Read the legacy generic fact.",
        required_slots=[RequiredSlot(slot_id="fact", description="generic fact")],
    )

    assert contract.contract_version == "1"
    assert contract.slot_semantics == "legacy_generic"
    assert contract.slot_plan_status is None
    assert contract.atomic_completeness is None
    assert contract.atomic_completeness_reason is None


def test_query_contract_v2_active_atomic_contract_shape_and_serialization() -> None:
    contract = QueryContract(
        contract_version="2",
        route="bounded_compare",
        intent="Compare two models from authorized sources",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Retrieve A latency."),
            RequiredSlot(slot_id="S2", description="Retrieve B latency."),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare the two reported latencies.",
                depends_on_slot_ids=["S1", "S2"],
            )
        ],
        response_constraints=[
            ResponseConstraint(
                constraint_id="C1",
                kind="prohibition",
                description="Do not claim a universal ranking.",
            )
        ],
        comparison_plan=ComparisonPlan(
            subjects=[
                ComparisonSubject(
                    subject_id="model_a",
                    display_name="Model A",
                    retrieval_query="Model A reported latency",
                    evidence_slot_ids=["S1"],
                ),
                ComparisonSubject(
                    subject_id="model_b",
                    display_name="Model B",
                    retrieval_query="Model B reported latency",
                    evidence_slot_ids=["S2"],
                ),
            ]
        ),
        slot_plan_status="complete",
        slot_plan_source="deterministic",
        slot_plan_confidence="high",
        slot_plan_fallback_reason=None,
        truncated_requirement_count=0,
    )
    assert contract.route == "bounded_compare"
    assert contract.comparison_plan is not None
    assert contract.comparison_plan.subjects[0].evidence_slot_ids == ["S1"]

    validated = validate_active_atomic_contract(contract)
    assert validated is contract

    dumped = contract.model_dump(mode="json")
    assert dumped["synthesis_obligations"][0]["obligation_id"] == "O1"
    assert dumped["synthesis_obligations"][0]["kind"] == "comparison"
    assert dumped["synthesis_obligations"][0]["depends_on_slot_ids"] == ["S1", "S2"]
    assert dumped["response_constraints"][0]["constraint_id"] == "C1"
    assert dumped["response_constraints"][0]["kind"] == "prohibition"
    assert dumped["slot_plan_source"] == "deterministic"
    assert dumped["slot_plan_confidence"] == "high"
    assert "slot_plan_fallback_reason" not in dumped
    assert dumped["truncated_requirement_count"] == 0
    assert dumped["comparison_plan"]["subjects"][0]["evidence_slot_ids"] == ["S1"]


def test_query_contract_rejects_duplicate_slot_ids() -> None:
    with pytest.raises(ValidationError):
        QueryContract(
            route="single_lookup",
            intent="Resolve duplicate slots",
            required_slots=[
                RequiredSlot(slot_id="S1", description="Slot 1"),
                RequiredSlot(slot_id="S1", description="Duplicate Slot 1"),
            ],
        )


def test_query_contract_rejects_obligation_referencing_unknown_slot() -> None:
    with pytest.raises(ValidationError):
        QueryContract(
            route="single_lookup",
            intent="Resolve slot with dangling obligation",
            required_slots=[RequiredSlot(slot_id="S1", description="Slot 1")],
            synthesis_obligations=[
                SynthesisObligation(
                    obligation_id="O1",
                    kind="comparison",
                    description="Obligation referencing missing S2",
                    depends_on_slot_ids=["S1", "S2"],
                )
            ],
        )


def test_query_contract_rejects_comparison_subject_referencing_unknown_slot() -> None:
    with pytest.raises(ValidationError):
        QueryContract(
            route="bounded_compare",
            intent="Compare models with invalid slot binding",
            required_slots=[RequiredSlot(slot_id="S1", description="Slot 1")],
            comparison_plan=ComparisonPlan(
                subjects=[
                    ComparisonSubject(
                        subject_id="model_a",
                        display_name="Model A",
                        retrieval_query="Model A evidence",
                        evidence_slot_ids=["S1"],
                    ),
                    ComparisonSubject(
                        subject_id="model_b",
                        display_name="Model B",
                        retrieval_query="Model B evidence",
                        evidence_slot_ids=["S2"],
                    ),
                ]
            ),
        )


def test_validate_active_atomic_contract_rejects_non_sequential_or_excessive_slots() -> None:
    with pytest.raises(ValueError, match="1 to 8 slots"):
        validate_active_atomic_contract(
            QueryContract(
                route="single_lookup",
                intent="Empty slots",
                required_slots=[],
            )
        )

    slots_9 = [
        RequiredSlot(slot_id=f"S{i}", description=f"Slot {i}") for i in range(1, 10)
    ]
    with pytest.raises(ValueError, match="1 to 8 slots"):
        validate_active_atomic_contract(
            QueryContract(
                route="multi_document_exact",
                intent="9 slots",
                required_slots=slots_9,
            )
        )

    with pytest.raises(ValueError, match="sequential slot IDs"):
        validate_active_atomic_contract(
            QueryContract(
                route="multi_document_exact",
                intent="Non sequential slots",
                required_slots=[
                    RequiredSlot(slot_id="S1", description="Slot 1"),
                    RequiredSlot(slot_id="S3", description="Slot 3"),
                ],
            )
        )

    with pytest.raises(ValueError, match="sequential slot IDs"):
        validate_active_atomic_contract(
            QueryContract(
                route="single_lookup",
                intent="Legacy slot ID",
                required_slots=[
                    RequiredSlot(slot_id="fact", description="Generic fact")
                ],
            )
        )


def test_validate_active_atomic_contract_requires_non_empty_comparison_binding() -> None:
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare models with unbound subjects",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Slot 1"),
            RequiredSlot(slot_id="S2", description="Slot 2"),
        ],
        comparison_plan=ComparisonPlan(
            subjects=[
                ComparisonSubject(
                    subject_id="model_a",
                    display_name="Model A",
                    retrieval_query="Model A query",
                    evidence_slot_ids=[],
                ),
                ComparisonSubject(
                    subject_id="model_b",
                    display_name="Model B",
                    retrieval_query="Model B query",
                    evidence_slot_ids=["S2"],
                ),
            ]
        ),
    )
    with pytest.raises(ValueError, match="must bind to at least one evidence slot"):
        validate_active_atomic_contract(contract)


def test_v9_execution_metrics_atomic_defaults_and_guards() -> None:
    metrics = V9ExecutionMetrics()
    assert metrics.atomic_planner_call_count == 0
    assert metrics.comparison_planner_call_count == 0
    assert metrics.slot_binding_method == "not_instrumented"
    assert metrics.semantic_qualification == "not_instrumented"

    with pytest.raises(ValidationError):
        V9ExecutionMetrics(atomic_planner_call_count=2)

    with pytest.raises(ValidationError):
        V9ExecutionMetrics(comparison_planner_call_count=1)  # type: ignore[arg-type]


def test_actual_routing_trace_has_first_class_route_provenance() -> None:
    decision = EvaluationRoutingDecision(
        routing_decision_id="route-1",
        run_id="run-1",
        campaign_id="campaign-1",
        selected_mode="agentic",
        analysis_type="actual",
        decision_source="safe_fallback",
        candidate_routes=["multi_hop", "exact_structured"],
        matched_rules=["mixed_requirements"],
        fallback_reason="planner_timeout",
        reason="Use the bounded safe route.",
        confidence=0.25,
        created_at=datetime.now(timezone.utc),
    )

    assert decision.decision_source == "safe_fallback"
    assert decision.candidate_routes == ["multi_hop", "exact_structured"]
    assert decision.matched_rules == ["mixed_requirements"]
    assert decision.fallback_reason == "planner_timeout"


@pytest.mark.parametrize("extra_field", ["user_id", "authorized_doc_ids"])
def test_request_rejects_adapter_injected_authorization_fields(
    extra_field: str,
) -> None:
    payload = {
        "question": "What is the reported score?",
        "trace_id": "trace-1",
        extra_field: "not-allowed",
    }

    with pytest.raises(ValidationError):
        V9ExecutionRequest(**payload)


def test_final_claim_rejects_evidence_only_scope_constraint_support_type() -> None:
    with pytest.raises(ValidationError):
        FinalClaim(
            claim_id="claim-1",
            statement="The frozen source scope cannot establish this claim.",
            support_type="scope_constraint",
        )


def test_structured_final_draft_has_strict_supported_and_unresolved_rows() -> None:
    draft = FinalAnswerDraft(
        supported_findings=[
            SupportedFinding(
                slot_id="score",
                statement="The reported score is 0.91.",
                support_type="calculated",
                evidence_ids=["E1"],
                premise_evidence_ids=["E-source"],
            )
        ],
        unresolved_requirements=[
            UnresolvedRequirement(slot_id="source", reason="Source was not found.")
        ],
    )

    assert draft.supported_findings[0].slot_id == "score"
    assert draft.supported_findings[0].support_type == "calculated"
    assert draft.supported_findings[0].premise_evidence_ids == ["E-source"]
    assert draft.unresolved_requirements[0].slot_id == "source"
    with pytest.raises(ValidationError):
        FinalAnswerDraft(answer="provider-authored prose")
    with pytest.raises(ValidationError):
        SupportedFinding(
            slot_id="score",
            statement="The reported score is 0.91.",
            evidence_ids=["E1"],
            qualifier="maybe",
        )


def test_existing_final_claim_and_result_payloads_remain_compatible() -> None:
    claim = FinalClaim.model_validate(
        {
            "claim_id": "claim-1",
            "statement": "The score is 0.91.",
            "support_type": "direct",
            "evidence_ids": ["E1"],
        }
    )
    result = FinalAnswerResult.model_validate(
        {
            "response_status": "complete",
            "answer": "The score is 0.91.",
            "claims": [claim.model_dump()],
            "used_evidence_ids": ["E1"],
            "final_generation_count": 1,
        }
    )

    assert claim.slot_id is None
    assert result.claims == [claim]


def test_trace_execution_version_is_backward_compatible_and_summary_preserves_it() -> (
    None
):
    detail = AgentTraceDetail(
        trace_id="trace-1",
        campaign_id="campaign-1",
        campaign_result_id="result-1",
        question_id="question-1",
        question="What is the reported score?",
        mode="agentic",
        run_number=1,
        trace_status="completed",
        created_at=datetime.now(timezone.utc),
    )

    assert detail.agentic_execution_version == "v8"
    assert summarize_agent_trace(detail).agentic_execution_version == "v8"

    serialized = detail.model_dump(mode="json")
    restored = AgentTraceDetail.model_validate(serialized)
    assert restored.agentic_execution_version == "v8"
    assert summarize_agent_trace(restored).agentic_execution_version == "v8"


def test_retrieval_and_execution_results_preserve_the_evidence_only_boundary() -> None:
    retrieval = RagRetrievalResult(
        retrieval_id="retrieval-1", chunks=[{"chunk_id": "chunk-1"}]
    )
    task_result = TaskRetrievalResult(task_id="task-1", retrieval=retrieval)
    execution = V9ExecutionResult(trace_id="trace-1", task_results=[task_result])

    assert execution.task_results[0].retrieval.chunks[0]["chunk_id"] == "chunk-1"
    assert "answer" not in task_result.model_dump()


def test_validate_active_atomic_contract_enforces_sequential_obligation_ids() -> None:
    from data_base.agentic_v9.schemas import (
        QueryContract,
        RequiredSlot,
        SynthesisObligation,
        validate_active_atomic_contract,
    )

    valid_contract = QueryContract(
        route="single_lookup",
        intent="Resolve obligations.",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Fact A"),
            RequiredSlot(slot_id="S2", description="Fact B"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare A and B",
                depends_on_slot_ids=["S1", "S2"],
            ),
            SynthesisObligation(
                obligation_id="O2",
                kind="aggregation",
                description="Aggregate A and B",
                depends_on_slot_ids=["S1", "S2"],
            ),
        ],
    )
    assert validate_active_atomic_contract(valid_contract) == valid_contract

    # Non-sequential obligation IDs (O1, O3) must be rejected
    invalid_contract = QueryContract(
        route="single_lookup",
        intent="Resolve obligations.",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Fact A"),
            RequiredSlot(slot_id="S2", description="Fact B"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare A and B",
                depends_on_slot_ids=["S1", "S2"],
            ),
            SynthesisObligation(
                obligation_id="O3",
                kind="aggregation",
                description="Aggregate A and B",
                depends_on_slot_ids=["S1", "S2"],
            ),
        ],
    )
    with pytest.raises(ValueError, match="sequential"):
        validate_active_atomic_contract(invalid_contract)


def test_validate_active_atomic_contract_requires_non_empty_obligation_dependencies() -> None:
    from pydantic import ValidationError
    from data_base.agentic_v9.schemas import SynthesisObligation

    with pytest.raises(ValidationError):
        SynthesisObligation(
            obligation_id="O1",
            kind="comparison",
            description="Compare nothing",
            depends_on_slot_ids=[],
        )


def test_validate_active_atomic_contract_rejects_synthesis_slot() -> None:
    from data_base.agentic_v9.schemas import (
        QueryContract,
        RequiredSlot,
        validate_active_atomic_contract,
    )

    invalid_contract = QueryContract(
        route="single_lookup",
        intent="Resolve obligations.",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="以 Table 1 精確數值重新計算相對效率比值",
            ),
        ],
    )
    with pytest.raises(ValueError, match="derived synthesis operation"):
        validate_active_atomic_contract(invalid_contract)
