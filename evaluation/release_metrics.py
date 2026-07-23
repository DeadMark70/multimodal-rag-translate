"""Fail-closed, token-only Agentic v9 release metric derivation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, Field

from evaluation.benchmark_release import (
    BenchmarkRun,
    build_manifest,
    clustered_paired_bootstrap,
    ratio_of_sums,
    successful_p95,
    validate_benchmark_runs,
)
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.analytics import EvaluationAnalyticsService
from evaluation.db import CampaignRepository, CampaignResultRepository, RagasScoreRepository
from evaluation.observability_storage import EvaluationObservabilityRepository


class ReleaseMetric(BaseModel):
    value: float | int | None = None
    reason: str | None = None


class ReleaseArmSummary(BaseModel):
    mode: str
    condition_id: str
    execution_profile: str
    agentic_execution_version: str
    shadow_evaluation_policy: str | None = None
    response_status_counts: dict[str, int] = Field(default_factory=dict)
    run_count: int = 0
    complete_run_count: int = 0
    accounting_complete_run_count: int = 0


class ReleaseMetricsReport(BaseModel):
    benchmark_id: str
    benchmark_kind: str
    comparable: bool
    availability: Literal["available", "not_applicable"] = "available"
    not_applicable_reason: str | None = None
    gate_reasons: list[str] = Field(default_factory=list)
    manifest: dict[str, object] = Field(default_factory=dict)
    arms: list[ReleaseArmSummary] = Field(default_factory=list)
    required_slot_coverage: ReleaseMetric = Field(default_factory=ReleaseMetric)
    important_unsupported_claim_rate: ReleaseMetric = Field(default_factory=ReleaseMetric)
    provenance_failure_rate: ReleaseMetric = Field(default_factory=ReleaseMetric)
    pack_efficiency: ReleaseMetric = Field(default_factory=ReleaseMetric)
    graph_locator_success: ReleaseMetric = Field(default_factory=ReleaseMetric)
    graph_locator_fallback: ReleaseMetric = Field(default_factory=ReleaseMetric)
    final_generation_count: ReleaseMetric = Field(default_factory=ReleaseMetric)
    latency_p95_ms: ReleaseMetric = Field(default_factory=ReleaseMetric)
    token_ratio: ReleaseMetric = Field(default_factory=ReleaseMetric)
    paired_quality_delta: ReleaseMetric = Field(default_factory=ReleaseMetric)
    paired_quality_ci_lower: ReleaseMetric = Field(default_factory=ReleaseMetric)
    paired_quality_ci_upper: ReleaseMetric = Field(default_factory=ReleaseMetric)
    category_quality_deltas: dict[str, ReleaseMetric] = Field(default_factory=dict)
    per_question_quality_deltas: dict[str, ReleaseMetric] = Field(default_factory=dict)
    statistics: dict[str, object] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReleaseRun:
    run_id: str
    campaign_id: str
    question_id: str
    repeat_number: int
    mode: str
    condition_id: str
    execution_profile: str
    agentic_execution_version: str
    shadow_evaluation_policy: str | None
    completed: bool
    timed_out: bool
    accounting_status: str
    phase_attribution_status: str
    required_ragas_complete: bool
    golden_available: bool
    used_evidence_mapped: bool
    golden_question_fingerprint: str | None
    environment_fingerprint: str | None
    evaluator_fingerprint: str | None
    response_status: str | None
    required_slot_count: int | None
    supported_slot_count: int | None
    important_claim_count: int | None
    unsupported_important_claim_count: int | None
    provenance_failure_count: int | None
    packed_evidence_count: int | None
    available_evidence_count: int | None
    graph_locator_success_count: int | None
    graph_locator_fallback_count: int | None
    final_generation_count: int | None
    runtime_tokens: int | None
    latency_ms: float | None
    quality_score: float | None
    category: str | None

    def benchmark_run(self) -> BenchmarkRun:
        return BenchmarkRun(
            run_id=self.run_id,
            campaign_id=self.campaign_id,
            question_id=self.question_id,
            repeat_number=self.repeat_number,
            mode=self.mode,
            condition_id=self.condition_id,
            execution_profile=self.execution_profile,
            agentic_execution_version=self.agentic_execution_version,
            shadow_evaluation_policy=self.shadow_evaluation_policy,
            completed=self.completed,
            timed_out=self.timed_out,
            accounting_complete=(
                self.accounting_status == "complete"
                and self.phase_attribution_status == "complete"
            ),
            golden_question_fingerprint=self.golden_question_fingerprint,
            environment_fingerprint=self.environment_fingerprint,
            evaluator_fingerprint=self.evaluator_fingerprint,
            quality_score=self.quality_score,
            runtime_tokens=self.runtime_tokens,
            latency_ms=self.latency_ms,
            category=self.category,
        )


def derive_release_metrics(*, benchmark_id: str, runs: list[ReleaseRun]) -> ReleaseMetricsReport:
    """Derive release data once; gate failure blanks every derived number."""
    benchmark_runs = [run.benchmark_run() for run in runs]
    validation = validate_benchmark_runs(benchmark_runs)
    manifest = build_manifest(benchmark_id=benchmark_id, runs=benchmark_runs)
    reasons = set(validation.reasons)
    official_runs = validation.official_runs
    for run in runs:
        if not run.golden_available:
            reasons.add("missing_golden_data")
        if run.agentic_execution_version == "v9" and not run.shadow_evaluation_policy and not run.used_evidence_mapped:
            reasons.add("missing_used_evidence_mapping")
        if not run.required_ragas_complete:
            reasons.add("required_ragas_incomplete")
        if run.accounting_status != "complete" or run.phase_attribution_status != "complete":
            reasons.add("partial_accounting")
    if any(run.runtime_tokens is None for run in official_runs):
        reasons.add("runtime_token_instrumentation_missing")
    if any(run.latency_ms is None for run in official_runs):
        reasons.add("latency_instrumentation_missing")
    naive_tokens = [run.runtime_tokens for run in official_runs if run.identity.official_label == "naive"]
    if naive_tokens and all(token is not None for token in naive_tokens) and sum(int(token) for token in naive_tokens) <= 0:
        reasons.add("runtime_token_denominator_nonpositive")
    reasons = sorted(reasons)
    report = ReleaseMetricsReport(
        benchmark_id=benchmark_id,
        benchmark_kind=manifest.kind,
        comparable=not reasons,
        gate_reasons=reasons,
        manifest={
            "benchmark_id": manifest.benchmark_id,
            "kind": manifest.kind,
            "arm_order_seed": manifest.arm_order_seed,
            "ordered_blocks": [asdict(block) for block in manifest.ordered_blocks],
            "evaluator_blinding": manifest.evaluator_blinding,
            "environment_fingerprint": manifest.environment_fingerprint,
            "evaluator_fingerprint": manifest.evaluator_fingerprint,
            "non_blocking_ablations": list(manifest.non_blocking_ablations),
        },
        arms=_arm_summaries(runs),
    )
    if reasons:
        return _blocked_report(report)
    return _measured_report(report, runs, benchmark_runs)


def _blocked_report(report: ReleaseMetricsReport) -> ReleaseMetricsReport:
    unavailable = ReleaseMetric(reason=_release_gate_reason(report.gate_reasons))
    report.required_slot_coverage = unavailable
    report.important_unsupported_claim_rate = unavailable
    report.provenance_failure_rate = unavailable
    report.pack_efficiency = unavailable
    report.graph_locator_success = unavailable
    report.graph_locator_fallback = unavailable
    report.final_generation_count = unavailable
    report.latency_p95_ms = unavailable
    report.token_ratio = unavailable
    report.paired_quality_delta = unavailable
    report.paired_quality_ci_lower = unavailable
    report.paired_quality_ci_upper = unavailable
    report.statistics = {
        "method": "paired_question_cluster_bootstrap",
        "availability": "release_gate_blocked",
    }
    return report


def _release_gate_reason(reasons: list[str]) -> str:
    """Expose the concrete gate causes on every unavailable derived metric."""
    return "release_gate_blocked" if not reasons else f"release_gate_blocked:{','.join(sorted(reasons))}"


def _measured_report(
    report: ReleaseMetricsReport,
    runs: list[ReleaseRun],
    benchmark_runs: list[BenchmarkRun],
) -> ReleaseMetricsReport:
    evidence_runs = [run for run in runs if run.agentic_execution_version == "v9" and not run.shadow_evaluation_policy]
    report.required_slot_coverage = _ratio_metric(
        sum(run.supported_slot_count or 0 for run in evidence_runs),
        sum(run.required_slot_count or 0 for run in evidence_runs),
        "required_slot_instrumentation_missing",
        bool(evidence_runs) and all(run.required_slot_count is not None and run.supported_slot_count is not None for run in evidence_runs),
    )
    report.important_unsupported_claim_rate = _ratio_metric(
        sum(run.unsupported_important_claim_count or 0 for run in evidence_runs),
        sum(run.important_claim_count or 0 for run in evidence_runs),
        "important_claim_instrumentation_missing",
        bool(evidence_runs) and all(run.important_claim_count is not None and run.unsupported_important_claim_count is not None for run in evidence_runs),
    )
    report.provenance_failure_rate = _ratio_metric(
        sum(run.provenance_failure_count or 0 for run in evidence_runs),
        len(evidence_runs),
        "provenance_instrumentation_missing",
        bool(evidence_runs) and all(run.provenance_failure_count is not None for run in evidence_runs),
    )
    report.pack_efficiency = _ratio_metric(
        sum(run.packed_evidence_count or 0 for run in evidence_runs),
        sum(run.available_evidence_count or 0 for run in evidence_runs),
        "context_pack_instrumentation_missing",
        bool(evidence_runs) and all(run.packed_evidence_count is not None and run.available_evidence_count is not None for run in evidence_runs),
    )
    graph_instrumented = all(
        run.graph_locator_success_count is not None and run.graph_locator_fallback_count is not None
        for run in evidence_runs
    )
    report.graph_locator_success = ReleaseMetric(
        value=sum(run.graph_locator_success_count or 0 for run in evidence_runs) if graph_instrumented else None,
        reason=None if graph_instrumented else "graph_not_instrumented",
    )
    report.graph_locator_fallback = ReleaseMetric(
        value=sum(run.graph_locator_fallback_count or 0 for run in evidence_runs) if graph_instrumented else None,
        reason=None if graph_instrumented else "graph_not_instrumented",
    )
    final_instrumented = bool(evidence_runs) and all(run.final_generation_count is not None for run in evidence_runs)
    report.final_generation_count = ReleaseMetric(
        value=max((run.final_generation_count or 0) for run in evidence_runs) if final_instrumented else None,
        reason=None if final_instrumented else "final_generation_instrumentation_missing",
    )
    v9 = [run for run in benchmark_runs if run.identity.official_label == "agentic-v9"]
    naive = [run for run in benchmark_runs if run.identity.official_label == "naive"]
    token_ratio = ratio_of_sums(v9, naive)
    report.token_ratio = ReleaseMetric(
        value=token_ratio,
        reason=None if token_ratio is not None else "runtime_token_ratio_unavailable",
    )
    latency_p95 = successful_p95(v9)
    report.latency_p95_ms = ReleaseMetric(
        value=latency_p95,
        reason=None if latency_p95 is not None else "latency_p95_unavailable",
    )
    bootstrap = clustered_paired_bootstrap(benchmark_runs, seed=20260722)
    report.paired_quality_delta = ReleaseMetric(value=bootstrap.mean_delta, reason=None if bootstrap.mean_delta is not None else "quality_score_missing")
    report.paired_quality_ci_lower = ReleaseMetric(value=bootstrap.ci_lower, reason=None if bootstrap.ci_lower is not None else "quality_score_missing")
    report.paired_quality_ci_upper = ReleaseMetric(value=bootstrap.ci_upper, reason=None if bootstrap.ci_upper is not None else "quality_score_missing")
    report.statistics = {
        "method": bootstrap.method,
        "seed": bootstrap.seed,
        "resamples": bootstrap.resamples,
        "cluster_count": bootstrap.cluster_count,
        "repeat_aggregation": bootstrap.repeat_aggregation,
        "token_ratio_method": "ratio_of_summed_official_runtime_tokens",
        "final_generation_count_aggregation": "maximum_across_official_v9_runs",
    }
    report.category_quality_deltas = _quality_deltas(runs, group=lambda item: item.category or "uncategorized")
    report.per_question_quality_deltas = _quality_deltas(runs, group=lambda item: item.question_id)
    return report


def _ratio_metric(numerator: int, denominator: int, missing_reason: str, instrumented: bool) -> ReleaseMetric:
    if not instrumented:
        return ReleaseMetric(reason=missing_reason)
    if denominator == 0:
        return ReleaseMetric(reason="zero_denominator")
    return ReleaseMetric(value=numerator / denominator)


def _arm_summaries(runs: list[ReleaseRun]) -> list[ReleaseArmSummary]:
    grouped: dict[tuple[str, str, str, str, str | None], list[ReleaseRun]] = {}
    for run in runs:
        key = (run.mode, run.condition_id, run.execution_profile, run.agentic_execution_version, run.shadow_evaluation_policy)
        grouped.setdefault(key, []).append(run)
    summaries: list[ReleaseArmSummary] = []
    for key, members in sorted(grouped.items()):
        statuses: dict[str, int] = {}
        for item in members:
            status = item.response_status or "not_instrumented"
            statuses[status] = statuses.get(status, 0) + 1
        summaries.append(
            ReleaseArmSummary(
                mode=key[0],
                condition_id=key[1],
                execution_profile=key[2],
                agentic_execution_version=key[3],
                shadow_evaluation_policy=key[4],
                response_status_counts=statuses,
                run_count=len(members),
                complete_run_count=sum(item.completed for item in members),
                accounting_complete_run_count=sum(
                    item.accounting_status == "complete" and item.phase_attribution_status == "complete"
                    for item in members
                ),
            )
        )
    return summaries


def _quality_deltas(runs: list[ReleaseRun], *, group) -> dict[str, ReleaseMetric]:
    by_group: dict[str, dict[tuple[str, int], dict[str, ReleaseRun]]] = {}
    for run in runs:
        if run.quality_score is None:
            continue
        benchmark = run.benchmark_run()
        arm = benchmark.identity.official_label
        if arm not in {"naive", "agentic-v9"}:
            continue
        by_group.setdefault(group(run), {}).setdefault(benchmark.pair_key, {})[arm] = run
    result: dict[str, ReleaseMetric] = {}
    for name, pairs in sorted(by_group.items()):
        deltas = [arms["agentic-v9"].quality_score - arms["naive"].quality_score for arms in pairs.values() if "naive" in arms and "agentic-v9" in arms and arms["agentic-v9"].quality_score is not None and arms["naive"].quality_score is not None]
        result[name] = ReleaseMetric(value=mean(deltas), reason=None) if deltas else ReleaseMetric(reason="unpaired_quality_data")
    return result


class ReleaseMetricsService:
    """Own release derivation over durable campaign, accounting, and trace rows."""

    def __init__(
        self,
        *,
        campaigns: CampaignRepository | None = None,
        results: CampaignResultRepository | None = None,
        ragas_scores: RagasScoreRepository | None = None,
        accounting: EvaluationAccountingStore | None = None,
        observability: EvaluationObservabilityRepository | None = None,
        analytics: EvaluationAnalyticsService | None = None,
    ) -> None:
        self._campaigns = campaigns or CampaignRepository()
        self._results = results or CampaignResultRepository()
        self._scores = ragas_scores or RagasScoreRepository()
        self._accounting = accounting or EvaluationAccountingStore()
        self._observability = observability or EvaluationObservabilityRepository()
        self._analytics = analytics or EvaluationAnalyticsService(
            campaign_repository=self._campaigns,
            result_repository=self._results,
            observability_repository=self._observability,
        )

    async def get_report(self, *, user_id: str, campaign_id: str) -> ReleaseMetricsReport:
        anchor = await self._campaigns.get(user_id=user_id, campaign_id=campaign_id)
        if not anchor.config.benchmark_id:
            return ReleaseMetricsReport(
                benchmark_id="",
                benchmark_kind="not_applicable",
                comparable=False,
                availability="not_applicable",
                not_applicable_reason="benchmark_not_configured",
                gate_reasons=["benchmark_not_configured"],
            )

        benchmark_id = anchor.config.benchmark_id
        campaigns = await self._campaigns.list_by_user(user_id=user_id)
        selected = [
            campaign
            for campaign in campaigns
            if campaign.id == campaign_id
            or (anchor.config.benchmark_id and campaign.config.benchmark_id == anchor.config.benchmark_id)
        ]
        runs: list[ReleaseRun] = []
        for campaign in selected:
            scores = await self._scores.list_for_campaign(user_id=user_id, campaign_id=campaign.id)
            evaluator_fingerprints = evaluator_fingerprints_from_work_metadata(
                await self._scores.list_work_metadata_for_campaign(
                    user_id=user_id,
                    campaign_id=campaign.id,
                )
            )
            score_by_result: dict[str, dict[str, float]] = {}
            for row in scores:
                value = row.get("metric_value")
                if isinstance(value, (int, float)):
                    score_by_result.setdefault(str(row["campaign_result_id"]), {})[str(row["metric_name"])] = float(value)
            for result in await self._results.list_for_campaign(user_id=user_id, campaign_id=campaign.id):
                runs.append(
                    await self._release_run(
                        user_id=user_id,
                        result=result,
                        score_map=score_by_result.get(result.id, {}),
                        evaluator_fingerprint=evaluator_fingerprints.get(result.id),
                    )
                )
        report = derive_release_metrics(benchmark_id=benchmark_id, runs=runs)
        return report

    async def _release_run(
        self,
        *,
        user_id: str,
        result,
        score_map: dict[str, float],
        evaluator_fingerprint: str | None,
    ) -> ReleaseRun:

        breakdown = await self._run_tokens(campaign_id=result.campaign_id, run_id=result.id)
        detail = await self._analytics.run_detail(user_id=user_id, run_id=result.id)
        v9 = detail.agentic_v9
        final_claims = list(v9.final_claims) if v9 else []
        required_metrics = {"answer_correctness", "faithfulness", "answer_relevancy"}
        contract = v9.contract if v9 else None
        context_pack = v9.context_pack if v9 else None
        graph_events = await self._observability.list_graph_events_for_run(result.id)
        graph_requested = bool(contract and contract.graph_policy != "never")
        graph_success = sum(1 for event in graph_events if event.graph_to_chunk_success_rate and event.graph_to_chunk_success_rate > 0)
        graph_fallback = sum(1 for event in graph_events if event.graph_route.lower().startswith("fallback"))
        environment_snapshot = {
            "model_thinking": result.model_config_snapshot,
            "system": {
                key: value
                for key, value in result.system_version_snapshot.items()
                if key not in _ARM_BOOKKEEPING_SNAPSHOT_KEYS
            },
        }
        return ReleaseRun(
            run_id=result.id,
            campaign_id=result.campaign_id,
            question_id=result.question_id,
            repeat_number=result.repeat_number,
            mode=result.mode,
            condition_id=result.condition_id or f"{result.mode}-{result.agentic_execution_version}",
            execution_profile=result.execution_profile or "not_recorded",
            agentic_execution_version=result.agentic_execution_version,
            shadow_evaluation_policy=result.shadow_evaluation_policy,
            completed=result.status.value == "completed",
            timed_out="timeout" in (result.error_message or "").lower(),
            accounting_status=breakdown.accounting_status,
            phase_attribution_status=breakdown.phase_attribution_status,
            required_ragas_complete=required_metrics.issubset(score_map),
            golden_available=bool(result.question_snapshot),
            used_evidence_mapped=(
                bool(final_claims)
                and all(claim.evidence_ids for claim in final_claims)
                if result.agentic_execution_version == "v9" and not result.shadow_evaluation_policy
                else True
            ),
            golden_question_fingerprint=(
                golden_question_fingerprint(result.question_snapshot)
                if result.question_snapshot
                else None
            ),
            environment_fingerprint=(
                environment_fingerprint(environment_snapshot)
                if result.model_config_snapshot
                else None
            ),
            evaluator_fingerprint=evaluator_fingerprint,
            response_status=result.response_status,
            required_slot_count=len(contract.required_slots) if contract else None,
            supported_slot_count=len(v9.sufficiency.supported_slot_ids) if v9 and v9.sufficiency else None,
            important_claim_count=len(final_claims) if v9 else None,
            unsupported_important_claim_count=sum(not claim.evidence_ids for claim in final_claims) if v9 else None,
            provenance_failure_count=sum(not claim.evidence_ids for claim in final_claims) if v9 else None,
            packed_evidence_count=len(context_pack.packed_evidence_ids) if context_pack else None,
            available_evidence_count=len(v9.evidence_packets) if v9 else None,
            graph_locator_success_count=graph_success if graph_requested and graph_events else None,
            graph_locator_fallback_count=graph_fallback if graph_requested and graph_events else None,
            final_generation_count=v9.metrics.final_generation_count if v9 else None,
            runtime_tokens=breakdown.total_tokens,
            latency_ms=result.total_latency_ms,
            quality_score=score_map.get("answer_correctness"),
            category=result.category,
        )

    async def _run_tokens(self, *, campaign_id: str, run_id: str):
        scopes = await self._accounting.list_campaign_scopes(campaign_id)
        matching = [
            scope
            for scope in scopes
            if scope.scope_type == "execution_run" and scope.run_id == run_id and scope.accounting_schema_version == "2"
        ]
        if not matching:
            from evaluation.accounting_schemas import TokenBreakdown

            return TokenBreakdown(accounting_status="incomplete_legacy", phase_attribution_status="not_available")
        events = await self._accounting.list_campaign_events(campaign_id)
        from evaluation.research_analytics import _tokens

        return _tokens(matching, [event for event in events if event.scope_id in {scope.scope_id for scope in matching}])


_ARM_BOOKKEEPING_SNAPSHOT_KEYS = frozenset(
    {
        "mode",
        "run_number",
        "repeat_number",
        "condition_id",
        "condition_label",
        "execution_profile",
        "execution_identity",
        "agentic_execution_version",
        "shadow_evaluation_policy",
        "ablation_flags",
    }
)
_REQUIRED_RAGAS_METRICS = frozenset(
    {"answer_correctness", "faithfulness", "answer_relevancy"}
)


_GOLDEN_QUESTION_SNAPSHOT_KEYS = (
    "id",
    "question",
    "ground_truth",
    "ground_truth_short",
    "key_points",
    "expected_evidence",
)
_ENVIRONMENT_EXCLUDED_SYSTEM_KEYS = _ARM_BOOKKEEPING_SNAPSHOT_KEYS | frozenset(
    {"question", "golden", "evaluator"}
)


def golden_question_fingerprint(question_snapshot: dict[str, Any]) -> str:
    """Fingerprint the immutable golden material for one question/repeat pair."""
    from evaluation.benchmark_release import stable_snapshot_fingerprint

    return stable_snapshot_fingerprint(
        {
            key: question_snapshot.get(key)
            for key in _GOLDEN_QUESTION_SNAPSHOT_KEYS
        }
    )


def environment_fingerprint(snapshot: dict[str, Any]) -> str:
    """Fingerprint only question-independent frozen runtime environment inputs."""
    from evaluation.benchmark_release import stable_snapshot_fingerprint

    system = snapshot.get("system")
    if isinstance(system, dict):
        normalized_system = {
            key: value
            for key, value in system.items()
            if key not in _ENVIRONMENT_EXCLUDED_SYSTEM_KEYS
        }
    else:
        normalized_system = None
    return stable_snapshot_fingerprint(
        {
            "model_thinking": snapshot.get("model_thinking"),
            "system_environment": normalized_system,
        }
    )


def evaluator_fingerprints_from_work_metadata(
    metadata_rows: list[dict[str, Any]],
) -> dict[str, str | None]:
    """Build per-result evaluator fingerprints from immutable RAGAS work metadata.

    A result is eligible only when every required RAGAS metric has one complete,
    deterministic evaluator descriptor. Missing or conflicting descriptors remain
    ``None`` so benchmark validation fails closed.
    """
    from evaluation.benchmark_release import stable_snapshot_fingerprint

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in metadata_rows:
        result_id = row.get("campaign_result_id")
        metric_name = row.get("metric_name")
        if not isinstance(result_id, str) or not result_id or metric_name not in _REQUIRED_RAGAS_METRICS:
            continue
        grouped.setdefault(result_id, {}).setdefault(str(metric_name), []).append(row)

    fingerprints: dict[str, str | None] = {}
    required_fields = (
        "evaluation_signature",
        "metric_version",
        "compatibility_signature",
        "compatibility_signature_version",
        "evaluator_model",
        "evaluator_config",
    )
    for result_id, by_metric in grouped.items():
        descriptor: dict[str, dict[str, Any]] = {}
        valid = set(by_metric) == _REQUIRED_RAGAS_METRICS
        for metric_name in _REQUIRED_RAGAS_METRICS:
            rows = by_metric.get(metric_name, [])
            if len(rows) != 1 or any(rows[0].get(field) is None for field in required_fields):
                valid = False
                continue
            descriptor[metric_name] = {
                field: rows[0][field]
                for field in required_fields
            }
        fingerprints[result_id] = (
            stable_snapshot_fingerprint({"required_ragas": descriptor}) if valid else None
        )
    return fingerprints
