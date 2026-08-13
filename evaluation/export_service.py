"""Campaign Export Schema v2 composition and content-policy projection."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from evaluation.analytics import EvaluationAnalyticsService
from evaluation.db import CampaignRepository, CampaignResultRepository
from evaluation.export_schemas import (
    ExportAvailability,
    ExportCampaignIdentityV2,
    ExportCampaignRequest,
    ExportCampaignResponse,
    ExportClaimV2,
    ExportContextPackV2,
    ExportDiagnosticsDataV2,
    ExportEvidenceCoverageV2,
    ExportEvidenceReferenceV2,
    ExportGraphEventV2,
    ExportGraphEvidenceItemV2,
    ExportHumanEvaluationDataV2,
    ExportHumanEvalQueueV2,
    ExportHumanRatingV2,
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
    ExportV9ExecutionObservabilityV2,
    resolve_export_content_policy,
)
from evaluation.observability_storage import redact_sensitive_value
from evaluation.release_metrics import ReleaseMetricsService
from evaluation.research_analytics import CanonicalRunObservability, ResearchAnalyticsService


def _availability(status: str = "complete", *reasons: str) -> ExportAvailability:
    return ExportAvailability(status=status, reasons=list(reasons))


def _references(rows: list[dict[str, Any]]) -> list[ExportEvidenceReferenceV2]:
    fields = set(ExportEvidenceReferenceV2.model_fields)
    return [
        ExportEvidenceReferenceV2.model_validate(
            {key: value for key, value in row.items() if key in fields}
        )
        for row in rows
        if isinstance(row, dict)
    ]


def _project_agentic_v9(
    canonical: CanonicalRunObservability, request: ExportCampaignRequest
) -> ExportV9ExecutionObservabilityV2 | None:
    if canonical.agentic_v9 is None:
        return None
    row = canonical.agentic_v9.model_dump(mode="python")
    if not request.include_retrieved_excerpts:
        for item in row.get("evidence_packets", []):
            item["packet"]["statement"] = None
    if not request.include_answers:
        for item in row.get("final_claims", []):
            item["statement"] = None
    row["comparison"] = redact_sensitive_value(row.get("comparison"))
    return ExportV9ExecutionObservabilityV2.model_validate(row)


def _project_export_run_observability(
    *, canonical: CanonicalRunObservability, request: ExportCampaignRequest
) -> ExportRunObservabilityDataV2:
    """Project one canonical run through the sole detailed content boundary."""
    result = canonical.result
    tokens = canonical.token_breakdown
    trace_events = []
    for item in canonical.trace_events:
        row = item.model_dump(mode="python", exclude={"error"})
        row["payload"] = (
            redact_sensitive_value(row["payload"])
            if request.include_raw_trace_payloads
            else {}
        )
        trace_events.append(ExportTraceEventV2.model_validate(row))
    llm_calls = []
    for item in canonical.llm_calls:
        row = item.model_dump(mode="python", exclude={"payload", "error"})
        preview_policy = resolve_export_content_policy(
            request, captured_at_execution=item.prompt_capture_status == "captured"
        )
        full_policy = resolve_export_content_policy(
            request,
            captured_at_execution=item.full_prompt_capture_status == "captured",
        )
        row["prompt_preview"] = item.prompt_preview if preview_policy.prompt_preview_allowed else None
        row["full_prompt"] = (
            redact_sensitive_value(item.payload.get("full_prompt"))
            if full_policy.full_prompt_allowed
            else None
        )
        llm_calls.append(ExportLlmCallV2.model_validate(row))
    chunks = [
        ExportRetrievalChunkV2.model_validate(
            {
                **item.model_dump(mode="python", exclude={"payload"}),
                "excerpt": item.excerpt if request.include_retrieved_excerpts else None,
                "provenance": "persisted",
                "availability": _availability(),
            }
        )
        for item in canonical.retrieval_chunks
    ]
    claims = [
        ExportClaimV2(
            claim_id=item.claim_id,
            run_id=item.run_id,
            campaign_id=item.campaign_id,
            attempt_id=item.attempt_id,
            condition_id=item.condition_id,
            schema_version=item.schema_version,
            span_id=item.span_id,
            claim_text=item.claim_text if request.include_answers else None,
            claim_type=item.claim_type,
            support_status=item.support_status,
            evidence_refs=_references(item.evidence),
            unsupported_reason=item.unsupported_reason,
            repair_action=item.payload.get("repair_action"),
            post_repair_status=item.payload.get("post_repair_status"),
            extraction_status=canonical.claim_extraction_status,
            created_at=item.created_at,
        )
        for item in canonical.claims
    ]
    coverage = (
        [
            ExportEvidenceCoverageV2.model_validate(
                {
                    **item,
                    "fact_text": item.get("fact_text") if request.include_answers else None,
                }
            )
            for item in canonical.evidence_coverage
        ]
        if canonical.evidence_coverage is not None
        else None
    )
    return ExportRunObservabilityDataV2(
        run_id=result.id,
        campaign_id=result.campaign_id,
        run_summary=ExportRunSummaryV2(
            run_id=result.id,
            campaign_id=result.campaign_id,
            question_id=result.question_id,
            mode=result.mode,
            repeat_number=result.repeat_number,
            answer_preview=result.answer[:500] if request.include_answers else None,
            latency_ms=result.total_latency_ms if result.total_latency_ms is not None else result.latency_ms,
            total_tokens=tokens.total_tokens,
            accounting_status=tokens.accounting_status if tokens.accounting_status != "incomplete_legacy" else "not_available",
            created_at=result.created_at,
        ),
        accounting_diagnostics=tokens,
        trace_events=trace_events,
        llm_calls=llm_calls,
        retrieval_events=[ExportRetrievalEventV2.model_validate(item.model_dump(mode="python", exclude={"payload"})) for item in canonical.retrieval_events],
        retrieval_chunks=chunks,
        context_packs=[
            ExportContextPackV2.model_validate(
                {
                    **item.model_dump(mode="python", exclude={"payload", "retrieved_but_not_packed_evidence"}),
                    "retrieved_but_not_packed_evidence": _references(item.retrieved_but_not_packed_evidence),
                }
            )
            for item in canonical.context_packs
        ],
        tool_calls=[ExportToolCallV2.model_validate(item.model_dump(mode="python", exclude={"payload"})) for item in canonical.tool_calls],
        routing_decisions=[ExportRoutingDecisionV2.model_validate(item.model_dump(mode="python", exclude={"payload"})) for item in canonical.routing_decisions],
        graph_events=[ExportGraphEventV2.model_validate(item.model_dump(mode="python", exclude={"graph_feature_flags"})) for item in canonical.graph_events],
        graph_evidence_items=[ExportGraphEvidenceItemV2.model_validate(item.model_dump(mode="python")) for item in canonical.graph_evidence_items],
        graph_observability_status=canonical.graph_observability_status,
        claims=claims,
        claim_extraction_status=canonical.claim_extraction_status,
        human_ratings=[ExportHumanRatingV2.model_validate(item.model_dump(mode="python", exclude={"payload"})) for item in canonical.human_ratings],
        evidence_coverage=coverage,
        evidence_coverage_status=canonical.evidence_coverage_status,
        agentic_v9=_project_agentic_v9(canonical, request),
    )


class EvaluationExportService:
    """Compose the single authenticated campaign export contract."""

    def __init__(
        self,
        *,
        campaigns: Any | None = None,
        results: Any | None = None,
        analytics: Any | None = None,
        research: Any | None = None,
        release: Any | None = None,
    ) -> None:
        self._campaigns = campaigns or CampaignRepository()
        self._results = results or CampaignResultRepository()
        self._analytics = analytics or EvaluationAnalyticsService()
        self._research = research or ResearchAnalyticsService()
        self._release = release or ReleaseMetricsService()

    async def export_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        request: ExportCampaignRequest,
    ) -> ExportCampaignResponse:
        """Return one complete v2 export or propagate any required failure."""
        campaign = await self._campaigns.get(user_id=user_id, campaign_id=campaign_id)
        results = await self._results.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        responses = await asyncio.gather(
            self._research.get_summary(user_id=user_id, campaign_id=campaign_id),
            self._release.get_report(user_id=user_id, campaign_id=campaign_id),
            self._research.get_question_comparison(user_id=user_id, campaign_id=campaign_id),
            self._research.get_agent_behavior(user_id=user_id, campaign_id=campaign_id),
            self._analytics.router_analysis(user_id=user_id, campaign_id=campaign_id),
            self._analytics.ablation(user_id=user_id, campaign_id=campaign_id),
            self._analytics.human_vs_auto(user_id=user_id, campaign_id=campaign_id),
            self._analytics.human_eval_queue(user_id=user_id, campaign_id=campaign_id),
            self._analytics.campaign_errors(user_id=user_id, campaign_id=campaign_id),
            self._analytics.campaign_stage_warnings(user_id=user_id, campaign_id=campaign_id),
            self._research.get_official_ragas_by_run(
                user_id=user_id, campaign_id=campaign_id, results=results
            ),
        )
        summary, release, question, behavior, router, ablation, human, queue, errors, warnings, ragas = responses
        canonical = (
            await self._research.get_campaign_run_observability(
                user_id=user_id, campaign_id=campaign_id, results=results
            )
            if request.include_run_observability
            else {}
        )
        accounting_by_run = (
            {run_id: item.token_breakdown for run_id, item in canonical.items()}
            if request.include_run_observability
            else await self._research.get_campaign_run_accounting(
                user_id=user_id,
                campaign_id=campaign_id,
                results=results,
            )
        )
        expected_ids = {result.id for result in results}
        if request.include_run_observability and set(canonical) != expected_ids:
            raise ValueError("canonical observability result IDs do not match campaign results")
        release_data = release.model_dump(mode="python") if hasattr(release, "model_dump") else vars(release)
        if release_data.get("availability") == "not_applicable":
            release_data.update(manifest=None, arms=[], statistics=None)
        release_projection = ExportReleaseMetricsV2.model_validate(release_data)
        complete = _availability()
        release_availability = (
            _availability(
                "not_applicable",
                release_projection.not_applicable_reason or "not_applicable",
            )
            if release_projection.availability == "not_applicable"
            else complete
        )
        runs: list[ExportRunV2] = []
        for result in results:
            detail = (
                _project_export_run_observability(
                    canonical=canonical[result.id], request=request
                )
                if request.include_run_observability
                else None
            )
            accounting = accounting_by_run[result.id]
            runs.append(
                ExportRunV2(
                    result=_project_result(result, request),
                    ragas_metrics=ragas.get(result.id, {}),
                    accounting=accounting,
                    latency=ExportRunLatencyV2(
                        latency_ms=result.latency_ms,
                        total_latency_ms=result.total_latency_ms,
                        started_at=result.started_at,
                        completed_at=result.completed_at,
                    ),
                    observability=ExportRunObservabilityV2(
                        included=request.include_run_observability,
                        availability=(
                            complete
                            if detail
                            else _availability("not_applicable", "not_requested")
                        ),
                        data=detail,
                    ),
                )
            )
        return ExportCampaignResponse(
            export_metadata=ExportMetadataV2(
                exported_at=datetime.now(timezone.utc),
                options=request,
                redaction=ExportRedactionMetadata(),
                availability_warnings=[],
            ),
            campaign=ExportCampaignIdentityV2(
                id=campaign.id,
                name=campaign.name,
                status=campaign.status,
                benchmark_id=campaign.config.benchmark_id,
                modes=list(campaign.config.modes),
                repeat_count=campaign.config.repeat_count,
                created_at=campaign.created_at,
                updated_at=campaign.updated_at,
            ),
            sections=ExportSectionsV2(
                overview=ExportSection(
                    availability=complete,
                    data=ExportOverviewDataV2(
                        research_summary=summary,
                        release_metrics=ExportSection(
                            availability=release_availability,
                            data=release_projection,
                        ),
                    ),
                ),
                question_analysis=ExportSection(availability=complete, data=question),
                agent_behavior=ExportSection(availability=complete, data=behavior),
                router_analysis=ExportSection(availability=complete, data=router),
                ablation=ExportSection(availability=complete, data=ablation),
                human_evaluation=ExportSection(
                    availability=complete,
                    data=ExportHumanEvaluationDataV2(
                        comparison=human,
                        queue=ExportHumanEvalQueueV2(
                            campaign_id=queue.campaign_id,
                            rows=[
                                {
                                    **item.model_dump(mode="python"),
                                    "answer_preview": (
                                        item.answer_preview
                                        if request.include_answers
                                        else None
                                    ),
                                }
                                for item in queue.rows
                            ],
                        ),
                    ),
                ),
                diagnostics=ExportSection(
                    availability=complete,
                    data=ExportDiagnosticsDataV2(errors=errors, stage_warnings=warnings),
                ),
            ),
            runs=runs,
        )


def _project_result(result: Any, request: ExportCampaignRequest) -> ExportResultV2:
    return ExportResultV2(
        run_id=result.id,
        campaign_id=result.campaign_id,
        question_id=result.question_id,
        question=result.question,
        mode=result.mode,
        run_number=result.run_number,
        repeat_number=result.repeat_number,
        condition_id=result.condition_id,
        execution_profile=result.execution_profile,
        context_policy_version=result.context_policy_version,
        agentic_execution_version=result.agentic_execution_version,
        execution_identity=result.execution_identity,
        response_status=result.response_status,
        status=result.status,
        answer=result.answer if request.include_answers else None,
        ground_truth=result.ground_truth if request.include_answers else None,
        ground_truth_short=result.ground_truth_short if request.include_answers else None,
        contexts=result.contexts if request.include_retrieved_excerpts else None,
        source_doc_ids=list(result.source_doc_ids),
        latency_ms=result.latency_ms,
        total_latency_ms=result.total_latency_ms,
        total_tokens=result.total_tokens,
        created_at=result.created_at,
    )
