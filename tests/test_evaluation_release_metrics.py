from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest

from core.auth import get_current_user_id
from evaluation.release_metrics import (
    ReleaseRun,
    ReleaseMetricsService,
    derive_release_metrics,
    environment_fingerprint,
    evaluator_fingerprints_from_work_metadata,
    golden_question_fingerprint,
)
from evaluation.campaign_schemas import CampaignLifecycleStatus
from evaluation.router import get_release_metrics_service
from evaluation import db as evaluation_db
from main import app


def _run(*, run_id: str, mode: str, version: str, complete: bool = True, golden: bool = True, used_evidence: bool = True, accounting: str = "complete") -> ReleaseRun:
    return ReleaseRun(
        run_id=run_id,
        campaign_id="campaign",
        question_id="Q9",
        repeat_number=1,
        mode=mode,
        condition_id=f"{mode}-{version}",
        execution_profile=f"{mode}-{version}",
        agentic_execution_version=version,
        shadow_evaluation_policy=None,
        completed=complete,
        timed_out=False,
        accounting_status=accounting,
        phase_attribution_status=accounting,
        required_ragas_complete=True,
        golden_available=golden,
        used_evidence_mapped=used_evidence,
        golden_question_fingerprint="golden-Q9",
        environment_fingerprint="environment-v1",
        evaluator_fingerprint="evaluator-v1",
        response_status="complete",
        required_slot_count=2,
        supported_slot_count=2,
        important_claim_count=2,
        unsupported_important_claim_count=0,
        provenance_failure_count=0,
        packed_evidence_count=3,
        available_evidence_count=4,
        graph_locator_success_count=1,
        graph_locator_fallback_count=0,
        final_generation_count=1,
        runtime_tokens=100 if version == "v9" else 50,
        latency_ms=1000.0,
        quality_score=0.8,
        category="retrieval",
    )


def test_release_metrics_fail_closed_without_used_evidence_mapping() -> None:
    report = derive_release_metrics(
        benchmark_id="bench-1",
        runs=[
            _run(run_id="naive", mode="naive", version="v8"),
            _run(run_id="v8", mode="agentic", version="v8"),
            _run(run_id="v9", mode="agentic", version="v9", used_evidence=False),
        ],
    )

    assert report.comparable is False
    assert "missing_used_evidence_mapping" in report.gate_reasons
    assert report.token_ratio.value is None
    assert report.token_ratio.reason == "release_gate_blocked:missing_used_evidence_mapping"


def test_release_metrics_emit_measured_v9_evidence_metrics_when_complete() -> None:
    report = derive_release_metrics(
        benchmark_id="bench-1",
        runs=[
            _run(run_id="naive", mode="naive", version="v8"),
            _run(run_id="v8", mode="agentic", version="v8"),
            _run(run_id="v9", mode="agentic", version="v9"),
        ],
    )

    assert report.comparable is True
    assert report.required_slot_coverage.value == 1.0
    assert report.important_unsupported_claim_rate.value == 0.0
    assert report.pack_efficiency.value == 0.75
    assert report.final_generation_count.value == 1
    assert report.token_ratio.value == 2.0
    assert report.token_ratio.reason is None
    assert report.statistics["final_generation_count_aggregation"] == "maximum_across_official_v9_runs"


def test_release_metrics_never_substitute_zero_for_partial_accounting() -> None:
    report = derive_release_metrics(
        benchmark_id="bench-1",
        runs=[
            _run(run_id="naive", mode="naive", version="v8"),
            _run(run_id="v8", mode="agentic", version="v8"),
            _run(run_id="v9", mode="agentic", version="v9", accounting="partial"),
        ],
    )

    assert "partial_accounting" in report.gate_reasons
    assert report.required_slot_coverage.value is None
    assert report.required_slot_coverage.reason == "release_gate_blocked:partial_accounting"


@pytest.fixture
def release_metrics_160_run_fixture() -> list[SimpleNamespace]:
    """A formal-sized result set used to protect release-report read bounds."""

    rows = []
    for index in range(160):
        mode, version, quality = (
            ("naive", "v8", 0.4) if index % 2 == 0 else ("agentic", "v9", 0.7)
        )
        run_number = index // 2 + 1
        rows.append(
            SimpleNamespace(
                id=f"{mode}-{run_number}",
                campaign_id="campaign-1",
                question_id=f"Q{index // 2}",
                repeat_number=1,
                mode=mode,
                condition_id=f"{mode}-{run_number}",
                execution_profile=f"{mode}-{version}",
                agentic_execution_version=version,
                shadow_evaluation_policy=None,
                status=SimpleNamespace(value="completed"),
                error_message=None,
                question_snapshot={"id": f"Q{index // 2}"},
                model_config_snapshot={"model": "test"},
                system_version_snapshot={},
                response_status="complete",
                total_latency_ms=100.0,
                category="retrieval",
                source_attempt_id=None,
                quality=quality,
            )
        )
    return rows


@pytest.mark.asyncio
async def test_release_metrics_no_benchmark_skips_all_160_run_bulk_loaders(
    release_metrics_160_run_fixture: list[SimpleNamespace],
) -> None:
    assert len(release_metrics_160_run_fixture) == 160

    class _Campaigns:
        async def get(self, *, user_id: str, campaign_id: str):
            assert user_id == "user-1"
            assert campaign_id == "campaign-1"
            return SimpleNamespace(config=SimpleNamespace(benchmark_id=None))

        async def list_by_user(self, *, user_id: str):
            raise AssertionError("campaign discovery must not run without a benchmark")

    class _MustNotLoad:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self.rows = rows
            self.calls: list[str] = []

        def __getattr__(self, name: str):
            self.calls.append(name)
            raise AssertionError(f"{name} must not run without a benchmark")

    bulk_loaders = _MustNotLoad(release_metrics_160_run_fixture)

    report = await ReleaseMetricsService(
        campaigns=_Campaigns(),
        results=bulk_loaders,
        ragas_scores=bulk_loaders,
        accounting=bulk_loaders,
        observability=bulk_loaders,
        analytics=bulk_loaders,
    ).get_report(user_id="user-1", campaign_id="campaign-1")

    assert report.availability == "not_applicable"
    assert report.not_applicable_reason == "benchmark_not_configured"
    assert report.benchmark_id == ""
    assert report.benchmark_kind == "not_applicable"
    assert report.comparable is False
    assert report.gate_reasons == ["benchmark_not_configured"]
    assert bulk_loaders.calls == []


class _ReleaseMetricsCacheCampaigns:
    def __init__(self, campaigns):
        self.campaigns = campaigns

    async def get(self, *, user_id: str, campaign_id: str):
        return next(campaign for campaign in self.campaigns if campaign.id == campaign_id)

    async def list_by_user(self, *, user_id: str):
        return self.campaigns


class _ReleaseMetricsCacheResults:
    def __init__(self) -> None:
        self.calls = 0

    async def list_for_campaign_release(self, *, user_id: str, campaign_id: str):
        self.calls += 1
        return []


class _ReleaseMetricsCacheScores:
    def __init__(self) -> None:
        self.score_calls = 0
        self.metadata_calls = 0

    async def list_for_campaign(self, *, user_id: str, campaign_id: str):
        self.score_calls += 1
        return []

    async def list_work_metadata_for_campaign(self, *, user_id: str, campaign_id: str):
        self.metadata_calls += 1
        return []


class _ReleaseMetricsCacheAccounting:
    def __init__(self) -> None:
        self.calls = 0

    async def load_campaign_snapshot(self, campaign_id: str):
        self.calls += 1
        return SimpleNamespace(scopes_by_run_id={}, events_by_scope_id={})


class _ReleaseMetricsCacheObservability:
    def __init__(self) -> None:
        self.calls = 0

    async def load_campaign_release_snapshot(self, campaign_id: str):
        self.calls += 1
        return SimpleNamespace(
            materializations_by_run_id={},
            evidence_packets_by_run_id={},
            slot_resolutions_by_run_id={},
            claims_by_run_id={},
            context_packs_by_run_id={},
            graph_events_by_run_id={},
        )


def _release_metrics_cache_campaign(
    campaign_id: str,
    *,
    status: CampaignLifecycleStatus = CampaignLifecycleStatus.COMPLETED,
    updated_at: datetime | None = None,
):
    return SimpleNamespace(
        id=campaign_id,
        config=SimpleNamespace(benchmark_id="bench-1"),
        status=status,
        updated_at=updated_at or datetime(2026, 7, 24, tzinfo=timezone.utc),
    )


def _release_metrics_cache_service(campaigns):
    results = _ReleaseMetricsCacheResults()
    scores = _ReleaseMetricsCacheScores()
    accounting = _ReleaseMetricsCacheAccounting()
    observability = _ReleaseMetricsCacheObservability()
    return (
        ReleaseMetricsService(
            campaigns=_ReleaseMetricsCacheCampaigns(campaigns),
            results=results,
            ragas_scores=scores,
            accounting=accounting,
            observability=observability,
        ),
        results,
        scores,
        accounting,
        observability,
    )


@pytest.mark.asyncio
async def test_release_metrics_caches_unchanged_terminal_benchmark_loads() -> None:
    service, results, scores, accounting, observability = _release_metrics_cache_service(
        [_release_metrics_cache_campaign("campaign-1")]
    )

    await service.get_report(user_id="user-1", campaign_id="campaign-1")
    await service.get_report(user_id="user-1", campaign_id="campaign-1")

    assert (results.calls, scores.score_calls, scores.metadata_calls, accounting.calls, observability.calls) == (1, 1, 1, 1, 1)


@pytest.mark.asyncio
async def test_release_metrics_reloads_when_any_selected_campaign_marker_changes() -> None:
    campaigns = [
        _release_metrics_cache_campaign("campaign-1"),
        _release_metrics_cache_campaign("campaign-2"),
    ]
    service, results, scores, accounting, observability = _release_metrics_cache_service(campaigns)

    await service.get_report(user_id="user-1", campaign_id="campaign-1")
    campaigns[1].updated_at += timedelta(seconds=1)
    await service.get_report(user_id="user-1", campaign_id="campaign-1")

    assert (results.calls, scores.score_calls, scores.metadata_calls, accounting.calls, observability.calls) == (4, 4, 4, 4, 4)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [CampaignLifecycleStatus.RUNNING, CampaignLifecycleStatus.EVALUATING],
)
async def test_release_metrics_does_not_cache_nonterminal_benchmark_loads(
    status: CampaignLifecycleStatus,
) -> None:
    service, results, scores, accounting, observability = _release_metrics_cache_service(
        [_release_metrics_cache_campaign("campaign-1", status=status)]
    )

    await service.get_report(user_id="user-1", campaign_id="campaign-1")
    await service.get_report(user_id="user-1", campaign_id="campaign-1")

    assert (results.calls, scores.score_calls, scores.metadata_calls, accounting.calls, observability.calls) == (2, 2, 2, 2, 2)


@pytest.mark.asyncio
async def test_release_metrics_160_run_benchmark_keeps_repository_call_count_constant(
    release_metrics_160_run_fixture: list[SimpleNamespace],
) -> None:
    """Release projection reads are constant per selected campaign, not per run."""
    assert len(release_metrics_160_run_fixture) == 160

    class _Campaigns:
        async def get(self, *, user_id: str, campaign_id: str):
            return SimpleNamespace(
                id=campaign_id,
                config=SimpleNamespace(benchmark_id="bench-1"),
                status=CampaignLifecycleStatus.COMPLETED,
                updated_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
            )

        async def list_by_user(self, *, user_id: str):
            return [
                SimpleNamespace(
                    id="campaign-1",
                    config=SimpleNamespace(benchmark_id="bench-1"),
                    status=CampaignLifecycleStatus.COMPLETED,
                    updated_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
                )
            ]

    class _Results:
        calls = 0

        async def list_for_campaign_release(self, *, user_id: str, campaign_id: str):
            self.calls += 1
            return release_metrics_160_run_fixture

    class _Scores:
        score_calls = 0
        metadata_calls = 0

        async def list_for_campaign(self, *, user_id: str, campaign_id: str):
            self.score_calls += 1
            return [
                {
                    "campaign_result_id": result.id,
                    "metric_name": "answer_correctness",
                    "metric_value": result.quality,
                }
                for result in release_metrics_160_run_fixture
            ]

        async def list_work_metadata_for_campaign(self, *, user_id: str, campaign_id: str):
            self.metadata_calls += 1
            return []

    class _Accounting:
        calls = 0

        async def load_campaign_snapshot(self, campaign_id: str):
            self.calls += 1
            return SimpleNamespace(scopes_by_run_id={}, events_by_scope_id={})

    class _Observability:
        calls = 0

        async def load_campaign_release_snapshot(self, campaign_id: str):
            self.calls += 1
            return SimpleNamespace(
                materializations_by_run_id={},
                evidence_packets_by_run_id={},
                slot_resolutions_by_run_id={},
                claims_by_run_id={},
                context_packs_by_run_id={},
                graph_events_by_run_id={},
            )

    results = _Results()
    scores = _Scores()
    accounting = _Accounting()
    observability = _Observability()
    await ReleaseMetricsService(
        campaigns=_Campaigns(),
        results=results,
        ragas_scores=scores,
        accounting=accounting,
        observability=observability,
    ).get_report(user_id="user-1", campaign_id="campaign-1")

    assert (results.calls, scores.score_calls, scores.metadata_calls, accounting.calls, observability.calls) == (1, 1, 1, 1, 1)


def test_fingerprint_layers_keep_distinct_goldens_out_of_shared_environment() -> None:
    question_one = {
        "id": "Q1",
        "question": "question one",
        "ground_truth": "answer one",
        "key_points": ["one"],
        "expected_evidence": ["evidence-one"],
    }
    question_two = {
        "id": "Q2",
        "question": "question two",
        "ground_truth": "answer two",
        "key_points": ["two"],
        "expected_evidence": ["evidence-two"],
    }
    base_environment = {
        "model_thinking": {"model": "model-v1", "thinking_mode": False},
        "system": {
            "corpus": {"index": "corpus-v1"},
            "prompt": {"template": "prompt-v1"},
            "phase": {"policy": "phase-v1"},
            "code": {"commit": "code-v1"},
        },
        "mode": "naive",
        "run_number": 1,
        "repeat_number": 1,
        "condition_id": "naive-official",
    }
    other_arm = {
        **base_environment,
        "mode": "agentic",
        "run_number": 2,
        "repeat_number": 8,
        "condition_id": "agentic-v9-official",
    }

    assert golden_question_fingerprint(question_one) != golden_question_fingerprint(question_two)
    assert environment_fingerprint(base_environment) == environment_fingerprint(other_arm)
    assert environment_fingerprint(base_environment) != environment_fingerprint(
        {**other_arm, "system": {"prompt": {"template": "prompt-v2"}}}
    )


def test_release_metrics_accepts_production_shaped_distinct_goldens_in_one_environment() -> None:
    question_snapshots = {
        "Q1": {
            "id": "Q1",
            "question": "question one",
            "ground_truth": "answer one",
            "key_points": ["one"],
            "expected_evidence": [{"doc_id": "doc-one"}],
        },
        "Q2": {
            "id": "Q2",
            "question": "question two",
            "ground_truth": "answer two",
            "key_points": ["two"],
            "expected_evidence": [{"doc_id": "doc-two"}],
        },
    }
    frozen_environment = {
        "model_thinking": {"model_name": "judgeable-model", "thinking_mode": False},
        "system": {
            "corpus": {"knowledge_base_id": "kb-1", "index_version": "index-1"},
            "prompt": {"pack_version": "prompt-1"},
            "phase": {"policy_version": "phase-1"},
            "code": {"commit": "commit-1"},
        },
    }
    runs = [
        replace(
            _run(run_id=f"{question_id}-{version}", mode=mode, version=version),
            question_id=question_id,
            golden_question_fingerprint=golden_question_fingerprint(snapshot),
            environment_fingerprint=environment_fingerprint(frozen_environment),
        )
        for question_id, snapshot in question_snapshots.items()
        for mode, version in (("naive", "v8"), ("agentic", "v8"), ("agentic", "v9"))
    ]

    report = derive_release_metrics(benchmark_id="bench-1", runs=runs)

    assert report.comparable is True
    assert report.manifest["environment_fingerprint"] == environment_fingerprint(frozen_environment)
    assert "snapshot_fingerprint" not in report.manifest


def test_release_metrics_fail_closed_on_evaluator_or_runtime_instrumentation_mismatch() -> None:
    evaluator_mismatch = derive_release_metrics(
        benchmark_id="bench-1",
        runs=[
            _run(run_id="naive", mode="naive", version="v8"),
            _run(run_id="v8", mode="agentic", version="v8"),
            replace(_run(run_id="v9", mode="agentic", version="v9"), evaluator_fingerprint="evaluator-v2"),
        ],
    )
    assert evaluator_mismatch.comparable is False
    assert "incompatible_evaluator_metadata" in evaluator_mismatch.gate_reasons

    missing_tokens = derive_release_metrics(
        benchmark_id="bench-1",
        runs=[
            _run(run_id="naive", mode="naive", version="v8"),
            _run(run_id="v8", mode="agentic", version="v8"),
            replace(_run(run_id="v9", mode="agentic", version="v9"), runtime_tokens=None),
        ],
    )
    assert missing_tokens.comparable is False
    assert "runtime_token_instrumentation_missing" in missing_tokens.gate_reasons
    assert missing_tokens.token_ratio.reason == "release_gate_blocked:runtime_token_instrumentation_missing"


def test_evaluator_work_metadata_requires_a_complete_deterministic_signature_per_result() -> None:
    complete = [
        {
            "campaign_result_id": "run-1",
            "metric_name": metric,
            "evaluation_signature": "eval-sig",
            "metric_version": "1",
            "compatibility_signature": "compatible",
            "compatibility_signature_version": "1",
            "evaluator_model": "judge-v1",
            "evaluator_config": {"temperature": 0},
        }
        for metric in ("answer_correctness", "faithfulness", "answer_relevancy")
    ]
    assert evaluator_fingerprints_from_work_metadata(complete) == {"run-1": evaluator_fingerprints_from_work_metadata(complete)["run-1"]}
    assert evaluator_fingerprints_from_work_metadata(complete)["run-1"] is not None
    assert evaluator_fingerprints_from_work_metadata(complete[:-1])["run-1"] is None


def test_release_metrics_api_serializes_fail_closed_values(tmp_path) -> None:
    class _Service:
        async def get_report(self, *, user_id: str, campaign_id: str):
            assert user_id == "user-1"
            assert campaign_id == "campaign-1"
            return derive_release_metrics(
                benchmark_id="bench-1",
                runs=[
                    _run(run_id="naive", mode="naive", version="v8"),
                    _run(run_id="v8", mode="agentic", version="v8"),
                    _run(run_id="v9", mode="agentic", version="v9", used_evidence=False),
                ],
            )

    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_release_metrics_service] = lambda: _Service()
    try:
        with (
            patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
            patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
            patch.object(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db"),
            TestClient(app) as client,
        ):
            response = client.get("/api/evaluation/campaigns/campaign-1/release-metrics")
        assert response.status_code == 200
        assert response.json()["token_ratio"] == {
            "value": None,
            "reason": "release_gate_blocked:missing_used_evidence_mapping",
        }
    finally:
        app.dependency_overrides = {}


def test_runtime_openapi_exposes_release_metrics_and_benchmark_identity() -> None:
    schema = app.openapi()

    assert "/api/evaluation/campaigns/{campaign_id}/release-metrics" in schema["paths"]
    assert "ReleaseMetricsReport" in schema["components"]["schemas"]
    benchmark_schema = schema["components"]["schemas"]["CampaignCreateRequest"]["properties"]["benchmark_id"]
    assert {item["type"] for item in benchmark_schema["anyOf"]} == {"string", "null"}
