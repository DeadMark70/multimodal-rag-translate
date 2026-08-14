"""Exact-shape contract tests for Evaluation Export Schema v2."""

from copy import deepcopy

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from evaluation.accounting_schemas import (
    CampaignResearchSummaryResponse,
    CostSummary,
    EvaluationOverheadSummary,
    LatencySummary,
    TokenBreakdown,
)
from evaluation.campaign_schemas import (
    AblationResponse,
    AgentBehaviorResponse,
    CampaignErrorsResponse,
    CampaignStageWarningsResponse,
    HumanVsAutoResponse,
    ResearchQuestionComparisonResponse,
    RouterAnalysisResponse,
)
from data_base.agentic_v9.schemas import (
    ComparisonPlan,
    ComparisonSubject,
    QueryContract,
    RequiredSlot,
    ResponseConstraint,
    SynthesisObligation,
    V9ExecutionMetrics,
)
from evaluation.export_schemas import (
    ExportAvailability,
    ExportCampaignIdentityV2,
    ExportCampaignRequest,
    ExportCampaignResponse,
    ExportDiagnosticsDataV2,
    ExportClaimV2,
    ExportContextPackV2,
    ExportEvidenceCoverageV2,
    ExportEvidenceReferenceV2,
    ExportGraphEventV2,
    ExportGraphEvidenceItemV2,
    ExportHumanRatingV2,
    ExportHumanEvalQueueV2,
    ExportHumanEvaluationDataV2,
    ExportLlmCallV2,
    ExportMetadataV2,
    ExportOverviewDataV2,
    ExportRedactionMetadata,
    ExportReleaseMetricsV2,
    ExportResultV2,
    ExportRetrievalChunkV2,
    ExportRetrievalEventV2,
    ExportRoutingDecisionV2,
    ExportRunLatencyV2,
    ExportRunObservabilityDataV2,
    ExportRunObservabilityV2,
    ExportRunSummaryV2,
    ExportRunV2,
    ExportSection,
    ExportSectionsV2,
    ExportToolCallV2,
    ExportTraceEventV2,
    ExportV9ComparisonSubjectV2,
    ExportV9ExecutionObservabilityV2,
)

NOW = datetime(2026, 8, 13, tzinfo=timezone.utc)


def _tokens() -> TokenBreakdown:
    return TokenBreakdown(
        input_tokens=10,
        output_text_tokens=5,
        reasoning_tokens=2,
        other_tokens=0,
        total_tokens=17,
        accounting_status="complete",
        phase_attribution_status="complete",
    )


def _aggregate(model_type):
    return model_type(
        campaign_id="campaign-1",
        analysis_unit="execution",
        sample_count=1,
        independent_question_count=1,
        repeat_count=1,
        sample_note="one execution",
        warnings=[],
        rows=[],
        summaries={},
    )


def _observability_data() -> ExportRunObservabilityDataV2:
    return ExportRunObservabilityDataV2(
        run_id="run-1",
        campaign_id="campaign-1",
        run_summary=ExportRunSummaryV2(
            run_id="run-1",
            campaign_id="campaign-1",
            question_id="question-1",
            mode="agentic-v9",
            repeat_number=1,
            answer_preview="An answer",
            latency_ms=12,
            total_tokens=17,
            accounting_status="complete",
            created_at=NOW,
        ),
        accounting_diagnostics=_tokens(),
        trace_events=[
            ExportTraceEventV2(
                event_id="event-1",
                run_id="run-1",
                campaign_id="campaign-1",
                span_id="span-1",
                event_type="generation",
                event_schema_version="1.0",
                sequence=1,
                stage_type="generation",
                stage_name="final_answer",
                started_at=NOW,
                ended_at=NOW,
                duration_ms=12,
                status="success",
                retry_count=0,
                payload={"nested": {"safe": [True, 1, "value"]}},
                created_at=NOW,
            )
        ],
        llm_calls=[
            ExportLlmCallV2(
                llm_call_id="call-1",
                run_id="run-1",
                campaign_id="campaign-1",
                span_id="span-1",
                provider="google",
                model_name="gemini",
                phase="final_answer",
                purpose="campaign_generation",
                reservation_id="reservation-1",
                provider_attempt=1,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=17,
                reasoning_tokens=2,
                other_tokens=0,
                estimated_cost_usd=0.01,
                estimated_cost_twd=0.3,
                latency_ms=12,
                status="success",
                prompt_hash="prompt-hash",
                response_hash="response-hash",
                prompt_capture_status="captured",
                full_prompt_capture_status="captured",
                prompt_preview="Question preview",
                full_prompt="Full captured prompt",
                created_at=NOW,
            )
        ],
        retrieval_events=[
            ExportRetrievalEventV2(
                retrieval_event_id="retrieval-1",
                run_id="run-1",
                campaign_id="campaign-1",
                span_id="span-1",
                query="test query",
                query_hash="query-hash",
                retriever_name="hybrid",
                top_k=5,
                result_count=1,
                latency_ms=3,
                created_at=NOW,
            )
        ],
        retrieval_chunks=[
            ExportRetrievalChunkV2(
                retrieval_chunk_id="retrieval-chunk-1",
                run_id="run-1",
                campaign_id="campaign-1",
                span_id="span-1",
                retrieval_event_id="retrieval-1",
                chunk_id="chunk-1",
                doc_id="doc-1",
                page_start=1,
                page_end=1,
                modality="text",
                rank_before_rerank=1,
                rank_after_rerank=1,
                dense_score=0.8,
                bm25_score=0.7,
                rerank_score=0.9,
                used_in_context=True,
                used_in_answer=True,
                expected_evidence_match=True,
                excerpt="Retrieved excerpt",
                content_hash="content-hash",
                provenance="measured",
                availability=ExportAvailability(status="complete"),
                created_at=NOW,
            )
        ],
        context_packs=[
            ExportContextPackV2(
                context_pack_id="context-pack-1",
                run_id="run-1",
                campaign_id="campaign-1",
                attempt_id="attempt-1",
                condition_id="guided",
                schema_version="1",
                span_id="span-1",
                input_chunk_count=1,
                packed_chunk_count=1,
                token_count=20,
                retrieved_but_not_packed_evidence=[
                    ExportEvidenceReferenceV2(
                        evidence_id="evidence-2", doc_id="doc-2", chunk_id="chunk-2"
                    )
                ],
                created_at=NOW,
            )
        ],
        tool_calls=[
            ExportToolCallV2(
                tool_call_id="tool-1",
                run_id="run-1",
                campaign_id="campaign-1",
                span_id="span-1",
                tool_name="retriever",
                action="search",
                latency_ms=3,
                status="success",
                created_at=NOW,
            )
        ],
        routing_decisions=[
            ExportRoutingDecisionV2(
                routing_decision_id="route-1",
                run_id="run-1",
                campaign_id="campaign-1",
                span_id="span-1",
                selected_mode="agentic-v9",
                analysis_type="actual",
                decision_source="deterministic",
                candidate_routes=["agentic-v9"],
                matched_rules=["complex_question"],
                fallback_reason=None,
                confidence=1,
                reason="matched rule",
                created_at=NOW,
            )
        ],
        graph_events=[
            ExportGraphEventV2(
                graph_event_id="graph-1",
                run_id="run-1",
                campaign_id="campaign-1",
                span_id="span-1",
                graph_query="entity query",
                graph_search_mode="local",
                graph_evidence_mode="provenance_gated",
                graph_route="graph",
                router_reason="entity match",
                graph_snapshot_version="snapshot-1",
                graph_schema_version="schema-1",
                graph_extraction_prompt_version="prompt-1",
                matched_entity_ids=["entity-1"],
                community_ids=[1],
                node_count=1,
                edge_count=1,
                path_count=1,
                graph_latency_ms=2,
                graph_context_tokens=5,
                graph_to_chunk_success_rate=1,
                graph_noise_ratio=0,
                created_at=NOW,
            )
        ],
        graph_evidence_items=[
            ExportGraphEvidenceItemV2(
                graph_evidence_item_id="graph-evidence-1",
                graph_event_id="graph-1",
                node_ids=["entity-1"],
                edge_ids=["edge-1"],
                relation_path=["related_to"],
                source_doc_ids=["doc-1"],
                source_chunk_ids=["chunk-1"],
                pages=[1],
                asset_ids=["asset-1"],
                confidence=1,
                provenance_status="full",
                used_as_locator=True,
                packed_in_context=True,
                used_in_answer=True,
                supported_claim_ids=["claim-1"],
                created_at=NOW,
            )
        ],
        graph_observability_status="recorded",
        claims=[
            ExportClaimV2(
                claim_id="claim-1",
                run_id="run-1",
                campaign_id="campaign-1",
                attempt_id="attempt-1",
                condition_id="guided",
                schema_version="1",
                span_id="span-1",
                claim_text="Supported claim",
                claim_type="factual",
                support_status="supported",
                evidence_refs=[
                    ExportEvidenceReferenceV2(
                        evidence_id="evidence-1", doc_id="doc-1", chunk_id="chunk-1"
                    )
                ],
                unsupported_reason=None,
                repair_action=None,
                post_repair_status="supported",
                extraction_status="recorded",
                created_at=NOW,
            )
        ],
        claim_extraction_status="recorded",
        human_ratings=[
            ExportHumanRatingV2(
                human_rating_id="rating-1",
                run_id="run-1",
                campaign_id="campaign-1",
                span_id="span-1",
                rater_id_hash="rater-hash",
                rubric_version="1",
                correctness_score=1,
                faithfulness_score=1,
                completeness_score=1,
                citation_quality_score=1,
                usefulness_score=1,
                comments="Strong answer",
                is_blinded=True,
                shown_mode_label=False,
                created_at=NOW,
            )
        ],
        evidence_coverage=[
            ExportEvidenceCoverageV2(
                atomic_fact_id="fact-1",
                fact_text="Grounded fact",
                retrieved=True,
                packed=True,
                mentioned=True,
                cited=True,
                expected_doc_ids=["doc-1"],
            )
        ],
        evidence_coverage_status="complete",
        agentic_v9=ExportV9ExecutionObservabilityV2(),
    )


def test_schema_v2_request_defaults_to_summary_only() -> None:
    request = ExportCampaignRequest()

    assert request.model_dump() == {
        "include_run_observability": False,
        "include_raw_trace_payloads": False,
        "include_prompt_previews": True,
        "include_full_prompts": False,
        "include_answers": True,
        "include_retrieved_excerpts": True,
        "format": "json",
    }


def test_schema_v2_has_exact_top_level_and_section_shapes() -> None:
    assert set(ExportCampaignResponse.model_fields) == {
        "schema_version",
        "export_metadata",
        "campaign",
        "sections",
        "runs",
    }
    assert set(ExportSectionsV2.model_fields) == {
        "overview",
        "question_analysis",
        "agent_behavior",
        "router_analysis",
        "ablation",
        "human_evaluation",
        "diagnostics",
    }
    assert ExportCampaignResponse.model_fields["schema_version"].default == "2.0"


def test_schema_v2_rejects_arbitrary_properties_in_every_named_section() -> None:
    response = _fully_populated_response().model_dump(mode="python")
    paths = [
        ("overview", "data", "research_summary"),
        ("question_analysis", "data"),
        ("agent_behavior", "data"),
        ("router_analysis", "data"),
        ("ablation", "data"),
        ("human_evaluation", "data", "comparison"),
        ("diagnostics", "data", "errors"),
        ("diagnostics", "data", "stage_warnings"),
    ]

    for path in paths:
        invalid = deepcopy(response)
        target = invalid["sections"]
        for key in path:
            target = target[key]
        target["unexpected"] = "must be rejected"
        with pytest.raises(ValidationError):
            ExportCampaignResponse.model_validate(invalid)


def test_schema_v2_v9_comparison_has_one_fixed_shape_and_rejects_extras() -> None:
    comparison = {
        "planner_status": "planned",
        "planner_latency_ms": 1.5,
        "planner_fallback_reason": None,
        "fallback_stage": None,
        "validation_issues": [{"path": "subjects.0", "type": "invalid"}],
        "is_comparison": True,
        "subjects": [
            {"subject_id": "subject-1", "display_name": "Subject", "aliases": []}
        ],
        "dimensions": ["quality"],
        "task_diagnostics": [
            {
                "task_id": "task-1",
                "subject_id": "subject-1",
                "query_hash": "hash",
                "query_preview": "query",
                "status": "executed",
                "fallback_reason": None,
                "candidate_count": 1,
                "pre_subject_limit_count": 1,
                "selected_count": 1,
                "selected": [{"doc_id": "doc-1", "chunk_id": "chunk-1"}],
            }
        ],
        "coverage_before_repair": ["subject-1"],
        "missing_before_repair": [],
        "repair_executed": False,
        "coverage_after_repair": ["subject-1"],
        "missing_after_repair": [],
        "final_status": "complete",
        "final_evidence_subjects": ["subject-1"],
        "final_evidence_count": 1,
        "final_evidence": [
            {
                "evidence_id": "evidence-1",
                "doc_id": "doc-1",
                "chunk_id": "chunk-1",
                "subject_ids": ["subject-1"],
            }
        ],
    }

    exported = ExportV9ExecutionObservabilityV2(comparison=comparison)
    assert exported.comparison is not None
    assert exported.comparison.final_evidence_count == 1

    with pytest.raises(ValidationError):
        ExportV9ExecutionObservabilityV2(
            comparison={**comparison, "provider_body": "must be rejected"}
        )


def test_schema_v2_allow_lists_identity_result_and_observability_rows() -> None:
    assert set(ExportCampaignIdentityV2.model_fields) == {
        "id",
        "name",
        "status",
        "benchmark_id",
        "modes",
        "repeat_count",
        "created_at",
        "updated_at",
    }
    forbidden_result_fields = {
        "token_usage",
        "question_snapshot",
        "model_config_snapshot",
        "system_version_snapshot",
        "derived_metrics",
        "error_message",
    }
    assert forbidden_result_fields.isdisjoint(ExportResultV2.model_fields)
    assert {"payload", "error"}.issubset(ExportTraceEventV2.model_fields)
    assert {
        "provider_body",
        "provider_error",
        "credentials",
        "authorization",
    }.isdisjoint(ExportLlmCallV2.model_fields)
    assert {"error", "payload"}.issubset(ExportLlmCallV2.model_fields)


def test_schema_v2_redacted_answer_and_reference_fields_accept_none() -> None:
    result = ExportResultV2(
        run_id="run-1",
        campaign_id="campaign-1",
        question_id="question-1",
        question="What happened?",
        mode="agentic-v9",
        run_number=1,
        repeat_number=1,
        condition_id=None,
        execution_profile="evaluation_v9",
        context_policy_version="context-v1",
        agentic_execution_version="v9",
        execution_identity="identity-v1",
        response_status="complete",
        status="completed",
        answer=None,
        ground_truth=None,
        ground_truth_short=None,
        contexts=None,
        source_doc_ids=["doc-1"],
        latency_ms=None,
        total_latency_ms=None,
        total_tokens=None,
        created_at=NOW,
    )

    assert result.answer is None
    assert result.ground_truth is None
    assert result.ground_truth_short is None
    assert result.contexts is None


def _fully_populated_response() -> ExportCampaignResponse:
    research_summary = CampaignResearchSummaryResponse(
        campaign_id="campaign-1",
        completed_run_count=1,
        total_run_count=1,
        failed_run_count=0,
        quality_status="complete",
        token_accounting_status="complete",
        pricing_status="complete",
        phase_attribution_status="complete",
        sample_count=1,
        latency=LatencySummary(mean_ms=12, sample_count=1),
        tokens=_tokens(),
        execution_cost=CostSummary(pricing_status="complete"),
        modes=[],
        evaluation_overhead=EvaluationOverheadSummary(
            tokens=_tokens(), pricing_status="complete"
        ),
    )
    availability = ExportAvailability(status="complete", reasons=[])
    sections = ExportSectionsV2(
        overview=ExportSection(
            availability=availability,
            data=ExportOverviewDataV2(
                research_summary=research_summary.model_dump(mode="python"),
                release_metrics=ExportSection(
                    availability=availability,
                    data=ExportReleaseMetricsV2(
                        benchmark_id="benchmark-1",
                        benchmark_kind="formal",
                        comparable=True,
                    ),
                ),
            ),
        ),
        question_analysis=ExportSection(
            availability=availability,
            data=_aggregate(ResearchQuestionComparisonResponse).model_dump(
                mode="python"
            ),
        ),
        agent_behavior=ExportSection(
            availability=availability,
            data=_aggregate(AgentBehaviorResponse).model_dump(mode="python"),
        ),
        router_analysis=ExportSection(
            availability=availability,
            data=_aggregate(RouterAnalysisResponse).model_dump(mode="python"),
        ),
        ablation=ExportSection(
            availability=availability,
            data=_aggregate(AblationResponse).model_dump(mode="python"),
        ),
        human_evaluation=ExportSection(
            availability=availability,
            data=ExportHumanEvaluationDataV2(
                comparison=_aggregate(HumanVsAutoResponse).model_dump(mode="python"),
                queue=ExportHumanEvalQueueV2(campaign_id="campaign-1", rows=[]),
            ),
        ),
        diagnostics=ExportSection(
            availability=availability,
            data=ExportDiagnosticsDataV2(
                errors=CampaignErrorsResponse(
                    campaign_id="campaign-1", rows=[]
                ).model_dump(mode="python"),
                stage_warnings=CampaignStageWarningsResponse(
                    campaign_id="campaign-1", rows=[]
                ).model_dump(mode="python"),
            ),
        ),
    )
    response = ExportCampaignResponse(
        export_metadata=ExportMetadataV2(
            exported_at=NOW,
            options=ExportCampaignRequest(),
            redaction=ExportRedactionMetadata(),
            availability_warnings=["legacy run has partial accounting"],
        ),
        campaign=ExportCampaignIdentityV2(
            id="campaign-1",
            name="Export fixture",
            status="completed",
            benchmark_id="benchmark-1",
            modes=["agentic-v9"],
            repeat_count=1,
            created_at=NOW,
            updated_at=NOW,
        ),
        sections=sections,
        runs=[
            ExportRunV2(
                result=ExportResultV2(
                    run_id="run-1",
                    campaign_id="campaign-1",
                    question_id="question-1",
                    question="What happened?",
                    mode="agentic-v9",
                    run_number=1,
                    repeat_number=1,
                    condition_id="guided",
                    execution_profile="evaluation_v9",
                    context_policy_version="context-v1",
                    agentic_execution_version="v9",
                    execution_identity="identity-v1",
                    response_status="complete",
                    status="completed",
                    answer="An answer",
                    ground_truth="A reference answer",
                    ground_truth_short="Reference",
                    contexts=["Retrieved excerpt"],
                    source_doc_ids=["doc-1"],
                    latency_ms=12,
                    total_latency_ms=15,
                    total_tokens=17,
                    created_at=NOW,
                ),
                ragas_metrics={"faithfulness": 0.9},
                accounting=_tokens(),
                latency=ExportRunLatencyV2(
                    latency_ms=12,
                    total_latency_ms=15,
                    started_at=NOW,
                    completed_at=NOW,
                ),
                observability=ExportRunObservabilityV2(
                    included=True,
                    availability=availability,
                    data=_observability_data(),
                ),
            )
        ],
    )

    return response


def test_schema_v2_constructs_a_fully_populated_response() -> None:
    dumped = _fully_populated_response().model_dump(mode="json")
    assert dumped["schema_version"] == "2.0"
    assert dumped["sections"]["overview"]["data"]["release_metrics"]["data"]
    assert dumped["runs"][0]["ragas_metrics"] == {"faithfulness": 0.9}
    assert dumped["runs"][0]["observability"]["data"]["trace_events"]
    assert dumped["export_metadata"]["redaction"] == {
        "provider_errors": "excluded",
        "stack_traces": "excluded",
        "credentials": "redacted",
    }


def test_export_v2_round_trips_active_atomic_contract_and_new_metrics() -> None:
    contract = QueryContract(
        contract_version="2",
        route="bounded_compare",
        intent="Compare model A and model B",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Model A accuracy"),
            RequiredSlot(slot_id="S2", description="Model B accuracy"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare S1 and S2",
                depends_on_slot_ids=["S1", "S2"],
            )
        ],
        response_constraints=[
            ResponseConstraint(
                constraint_id="C1",
                kind="output_format",
                description="Format in comparison table",
            )
        ],
        comparison_plan=ComparisonPlan(
            subjects=[
                ComparisonSubject(
                    subject_id="model_a",
                    display_name="Model A",
                    retrieval_query="model a accuracy",
                    evidence_slot_ids=["S1"],
                ),
                ComparisonSubject(
                    subject_id="model_b",
                    display_name="Model B",
                    retrieval_query="model b accuracy",
                    evidence_slot_ids=["S2"],
                ),
            ],
            dimensions=["accuracy"],
        ),
        slot_plan_source="deterministic",
        slot_plan_confidence="high",
        slot_plan_fallback_reason=None,
        truncated_requirement_count=0,
    )
    metrics = V9ExecutionMetrics(
        atomic_planner_call_count=1,
        comparison_planner_call_count=0,
        slot_binding_method="task_target_inherited",
        semantic_qualification="not_enabled",
        reserved_tokens=500,
        reconciled_tokens=500,
    )
    observability = ExportV9ExecutionObservabilityV2(
        contract=contract,
        metrics=metrics,
    )

    dumped = observability.model_dump(mode="json")
    loaded = ExportV9ExecutionObservabilityV2.model_validate(dumped)

    assert loaded.contract is not None
    assert loaded.contract.contract_version == "2"
    assert len(loaded.contract.required_slots) == 2
    assert [s.slot_id for s in loaded.contract.required_slots] == ["S1", "S2"]
    assert len(loaded.contract.synthesis_obligations) == 1
    assert loaded.contract.synthesis_obligations[0].obligation_id == "O1"
    assert loaded.contract.synthesis_obligations[0].depends_on_slot_ids == ["S1", "S2"]
    assert len(loaded.contract.response_constraints) == 1
    assert loaded.contract.response_constraints[0].constraint_id == "C1"
    assert loaded.contract.comparison_plan is not None
    assert loaded.contract.comparison_plan.subjects[0].evidence_slot_ids == ["S1"]
    assert loaded.contract.comparison_plan.subjects[1].evidence_slot_ids == ["S2"]
    assert loaded.contract.slot_plan_source == "deterministic"
    assert loaded.contract.slot_plan_confidence == "high"
    assert loaded.contract.truncated_requirement_count == 0
    assert loaded.metrics.atomic_planner_call_count == 1
    assert loaded.metrics.comparison_planner_call_count == 0
    assert loaded.metrics.slot_binding_method == "task_target_inherited"
    assert loaded.metrics.semantic_qualification == "not_enabled"


def test_export_v2_comparison_subject_preserves_evidence_slot_ids() -> None:
    subject = ExportV9ComparisonSubjectV2(
        subject_id="model_a",
        display_name="Model A",
        aliases=["A"],
        evidence_slot_ids=["S1"],
    )
    dumped = subject.model_dump(mode="json")
    loaded = ExportV9ComparisonSubjectV2.model_validate(dumped)
    assert loaded.evidence_slot_ids == ["S1"]


def test_export_v2_reads_historical_contract_without_additive_fields() -> None:
    legacy_payload = {
        "schema_version": "1",
        "contract": {
            "contract_version": "1",
            "route": "single_lookup",
            "intent": "Lookup metric",
            "required_slots": [{"slot_id": "slot-1", "description": "Metric"}],
        },
        "metrics": {
            "reconciled_tokens": 100,
        },
    }
    loaded = ExportV9ExecutionObservabilityV2.model_validate(legacy_payload)
    assert loaded.contract is not None
    assert loaded.contract.contract_version == "1"
    assert loaded.contract.synthesis_obligations == []
    assert loaded.contract.response_constraints == []
    assert loaded.contract.comparison_plan is None
    assert loaded.contract.slot_plan_source is None
    assert loaded.metrics.atomic_planner_call_count == 0
    assert loaded.metrics.comparison_planner_call_count == 0
    assert loaded.metrics.slot_binding_method == "not_instrumented"
    assert loaded.metrics.semantic_qualification == "not_instrumented"
