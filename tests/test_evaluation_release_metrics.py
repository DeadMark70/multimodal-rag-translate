from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from evaluation.release_metrics import (
    ReleaseRun,
    derive_release_metrics,
    environment_fingerprint,
    evaluator_fingerprints_from_work_metadata,
    golden_question_fingerprint,
)
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
