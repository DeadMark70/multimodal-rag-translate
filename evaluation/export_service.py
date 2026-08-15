"""Campaign Export Schema v2 composition and content-policy projection."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from core.sensitive_data import is_sensitive_credential_key
from evaluation.analytics import EvaluationAnalyticsService
from evaluation.campaign_schemas import CampaignResult, CampaignResultStatus
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
from evaluation.observability_storage import (
    redact_sensitive_text,
    safe_comparison_projection,
)
from evaluation.release_metrics import ReleaseMetricsService
from evaluation.research_analytics import (
    CanonicalRunObservability,
    ResearchAnalyticsService,
    _project_interactive_run_observability,
)


_MAX_EXPORT_TEXT_CHARS = 64 * 1024
_ANSWER_CONTENT_KEYS = {
    "answer",
    "answer_preview",
    "answer_text",
    "claim",
    "claim_text",
    "completion",
    "final_answer",
    "final_claim",
}
_EXCERPT_CONTENT_KEYS = {
    "context",
    "contexts",
    "excerpt",
    "fact_text",
    "retrieved_excerpt",
    "statement",
}
_PERMANENTLY_EXCLUDED_KEYS = {
    "error",
    "errors",
    "provider_body",
    "provider_error",
    "provider_response",
    "raw_provider_response",
    "stack",
    "stack_trace",
    "traceback",
}


def _safe_export_text(value: object) -> str:
    text = redact_sensitive_text(value)
    if len(text) > _MAX_EXPORT_TEXT_CHARS:
        return f"{text[: _MAX_EXPORT_TEXT_CHARS - 3]}..."
    return text


def _safe_optional_export_text(value: object | None) -> str | None:
    return _safe_export_text(value) if value is not None else None


def _sanitize_export_value(value: Any) -> Any:
    """Credential-redact and bound every free-text value in the artifact."""
    if isinstance(value, dict):
        return {
            str(key): (
                item
                if is_sensitive_credential_key(str(key))
                and isinstance(item, str)
                and item in {"redacted", "[redacted]"}
                else "[redacted]"
                if is_sensitive_credential_key(str(key))
                else _sanitize_export_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_export_value(item) for item in value]
    return _safe_export_text(value) if isinstance(value, str) else value


def _project_trace_payload(
    value: Any, *, request: ExportCampaignRequest, field_name: str | None = None
) -> Any:
    """Allow-list raw trace structure while enforcing answer/excerpt policy."""
    if field_name in _ANSWER_CONTENT_KEYS and not request.include_answers:
        return None
    if field_name in _EXCERPT_CONTENT_KEYS and not request.include_retrieved_excerpts:
        return None
    if isinstance(value, dict):
        projected: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _PERMANENTLY_EXCLUDED_KEYS:
                continue
            projected[key_text] = (
                "[redacted]"
                if is_sensitive_credential_key(key_text)
                else _project_trace_payload(
                    item, request=request, field_name=key_text.lower()
                )
            )
        return projected
    if isinstance(value, (list, tuple)):
        return [
            _project_trace_payload(item, request=request, field_name=field_name)
            for item in value
        ]
    return _safe_export_text(value) if isinstance(value, str) else value


def _availability(status: str = "complete", *reasons: str) -> ExportAvailability:
    return ExportAvailability(status=status, reasons=list(reasons))


def _run_observability_availability(
    *, result: CampaignResult, requested: bool, has_detail: bool
) -> ExportAvailability:
    if not requested:
        return _availability("not_applicable", "not_requested")
    if result.status == CampaignResultStatus.FAILED:
        return _availability("partial", "run_failed_before_observability")
    if has_detail:
        return _availability()
    return _availability("not_available", "observability_missing")


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
    comparison = row.get("comparison")
    row["comparison"] = (
        _sanitize_export_value(safe_comparison_projection(comparison))
        if isinstance(comparison, dict)
        else None
    )
    return ExportV9ExecutionObservabilityV2.model_validate(
        _sanitize_export_value(row)
    )


def _project_export_run_observability(
    *, canonical: CanonicalRunObservability, request: ExportCampaignRequest
) -> ExportRunObservabilityDataV2:
    """Project one canonical run through the sole detailed content boundary."""
    interactive = _project_interactive_run_observability(canonical)
    trace_events = []
    for source, projected in zip(
        canonical.trace_events, interactive.trace_events, strict=True
    ):
        row = projected.model_dump(mode="python")
        row["payload"] = (
            _project_trace_payload(source.payload, request=request)
            if request.include_raw_trace_payloads
            else {}
        )
        trace_events.append(ExportTraceEventV2.model_validate(row))
    llm_calls = []
    for source, projected in zip(
        canonical.llm_calls, interactive.llm_calls, strict=True
    ):
        row = projected.model_dump(mode="python")
        preview_policy = resolve_export_content_policy(
            request, captured_at_execution=source.prompt_capture_status == "captured"
        )
        full_policy = resolve_export_content_policy(
            request,
            captured_at_execution=source.full_prompt_capture_status == "captured",
        )
        row["prompt_preview"] = (
            projected.prompt_preview
            if preview_policy.prompt_preview_allowed
            else None
        )
        row["full_prompt"] = (
            _safe_optional_export_text(source.payload.get("full_prompt"))
            if full_policy.full_prompt_allowed
            else None
        )
        llm_calls.append(ExportLlmCallV2.model_validate(row))
    chunks = [
        ExportRetrievalChunkV2.model_validate(
            {
                **item.model_dump(mode="python"),
                "excerpt": (
                    item.excerpt
                    if request.include_retrieved_excerpts
                    else None
                ),
            }
        )
        for item in interactive.retrieval_chunks
    ]
    claims = [
        ExportClaimV2.model_validate(
            {
                **item.model_dump(mode="python"),
                "claim_text": item.claim_text if request.include_answers else None,
            }
        )
        for item in interactive.claims
    ]
    coverage = (
        [
            ExportEvidenceCoverageV2.model_validate(
                {
                    **item,
                    "fact_text": (
                        _safe_optional_export_text(item.get("fact_text"))
                        if request.include_answers
                        else None
                    ),
                }
            )
            for item in canonical.evidence_coverage
        ]
        if canonical.evidence_coverage is not None
        else None
    )
    return ExportRunObservabilityDataV2(
        run_id=interactive.run_id,
        campaign_id=interactive.campaign_id,
        run_summary=ExportRunSummaryV2.model_validate(
            {
                **interactive.run_summary.model_dump(mode="python"),
                "answer_preview": (
                    interactive.run_summary.answer_preview
                    if request.include_answers
                    else None
                ),
            }
        ),
        accounting_diagnostics=interactive.accounting_diagnostics,
        trace_events=trace_events,
        llm_calls=llm_calls,
        retrieval_events=[
            ExportRetrievalEventV2.model_validate(item.model_dump(mode="python"))
            for item in interactive.retrieval_events
        ],
        retrieval_chunks=chunks,
        context_packs=[
            ExportContextPackV2.model_validate(item.model_dump(mode="python"))
            for item in interactive.context_packs
        ],
        tool_calls=[
            ExportToolCallV2.model_validate(item.model_dump(mode="python"))
            for item in interactive.tool_calls
        ],
        routing_decisions=[
            ExportRoutingDecisionV2.model_validate(item.model_dump(mode="python"))
            for item in interactive.routing_decisions
        ],
        graph_events=[
            ExportGraphEventV2.model_validate(item.model_dump(mode="python"))
            for item in interactive.graph_events
        ],
        graph_evidence_items=[
            ExportGraphEvidenceItemV2.model_validate(item.model_dump(mode="python"))
            for item in interactive.graph_evidence_items
        ],
        graph_observability_status=interactive.graph_observability_status,
        claims=claims,
        claim_extraction_status=interactive.claim_extraction_status,
        human_ratings=[
            ExportHumanRatingV2.model_validate(item.model_dump(mode="python"))
            for item in interactive.human_ratings
        ],
        evidence_coverage=coverage,
        evidence_coverage_status=interactive.evidence_coverage_status,
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
                        availability=_run_observability_availability(
                            result=result,
                            requested=request.include_run_observability,
                            has_detail=detail is not None,
                        ),
                        data=detail,
                    ),
                )
            )
        response = ExportCampaignResponse(
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
                        research_summary=summary.model_dump(mode="python"),
                        release_metrics=ExportSection(
                            availability=release_availability,
                            data=release_projection,
                        ),
                    ),
                ),
                question_analysis=ExportSection(
                    availability=complete, data=question.model_dump(mode="python")
                ),
                agent_behavior=ExportSection(
                    availability=complete, data=behavior.model_dump(mode="python")
                ),
                router_analysis=ExportSection(
                    availability=complete, data=router.model_dump(mode="python")
                ),
                ablation=ExportSection(
                    availability=complete, data=ablation.model_dump(mode="python")
                ),
                human_evaluation=ExportSection(
                    availability=complete,
                    data=ExportHumanEvaluationDataV2(
                        comparison=human.model_dump(mode="python"),
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
                    data=ExportDiagnosticsDataV2(
                        errors=errors.model_dump(mode="python"),
                        stage_warnings=warnings.model_dump(mode="python"),
                    ),
                ),
            ),
            runs=runs,
        )
        return ExportCampaignResponse.model_validate(
            _sanitize_export_value(response.model_dump(mode="json"))
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
