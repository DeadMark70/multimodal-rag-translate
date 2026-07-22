"""Tests for evaluation research analytics API endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from evaluation.campaign_engine import CampaignEngine
from evaluation import db as evaluation_db
from evaluation.rag_modes import BenchmarkExecutionResult
from main import app


class FakeRagasEvaluator:
    async def evaluate_campaign(self, *, on_progress=None, **kwargs) -> str:
        if on_progress:
            await on_progress(1, 1, "Q-ANALYTICS", "agentic")
        return "fake-ragas"


def _make_upload_root() -> Path:
    root = Path(os.environ["EVALUATION_TEST_TMPDIR"]) / f"analytics_api_{uuid4().hex}" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_db_path() -> Path:
    root = Path(os.environ["EVALUATION_TEST_TMPDIR"]) / f"analytics_db_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root / "evaluation.db"


@contextmanager
def _build_client(
    user_id: str,
    upload_root: Path,
    db_path: Path,
    engine: CampaignEngine,
    *,
    with_auth: bool = True,
):
    process_worker = Mock(is_configured=False)
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("evaluation.storage.BASE_UPLOAD_FOLDER", str(upload_root)),
        patch("evaluation.db.EVALUATION_DB_PATH", db_path),
        patch("evaluation.campaign_engine.get_campaign_engine", return_value=engine),
        patch("evaluation.job_worker.get_evaluation_job_worker", return_value=process_worker),
        patch("evaluation.router.get_campaign_engine", return_value=engine),
    ):
        if with_auth:
            app.dependency_overrides[get_current_user_id] = lambda: user_id
        with TestClient(app) as client:
            yield client
    app.dependency_overrides = {}


def _wait_for_completed(client: TestClient, campaign_id: str) -> dict:
    deadline = time.time() + 5
    latest = {}
    while time.time() < deadline:
        response = client.get("/api/evaluation/campaigns")
        assert response.status_code == 200
        latest = next(item for item in response.json() if item["id"] == campaign_id)
        if latest["status"] == "completed":
            return latest
        time.sleep(0.05)
    raise AssertionError(f"Campaign did not complete: {latest}")


def _create_test_case(client: TestClient) -> None:
    response = client.post(
        "/api/evaluation/test-cases",
        json={
            "id": "Q-ANALYTICS",
            "question": "Where is Fact A?",
            "ground_truth": "Fact A is in paper A",
            "ground_truth_short": "paper A",
            "key_points": ["Fact A"],
            "ragas_focus": ["faithfulness"],
            "category": "analytics",
            "difficulty": "medium",
            "source_docs": ["paper-a.pdf"],
            "atomic_facts": [{"atomic_fact_id": "F1", "fact_text": "Fact A"}],
            "expected_evidence": [
                {"evidence_id": "E1", "doc_id": "paper-a.pdf", "atomic_fact_id": "F1"}
            ],
        },
    )
    assert response.status_code == 200


def _campaign_payload() -> dict:
    return {
        "name": "Analytics",
        "test_case_ids": ["Q-ANALYTICS"],
        "modes": ["agentic"],
        "model_config": {
            "id": "cfg-1",
            "name": "Balanced",
            "model_name": "gemini-2.5-flash",
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "thinking_mode": False,
            "thinking_budget": 8192,
        },
        "model_config_id": "cfg-1",
        "repeat_count": 1,
        "batch_size": 1,
        "rpm_limit": 60,
    }


async def _seed_legacy_campaign_result(*, db_path: Path, user_id: str) -> tuple[str, str]:
    now = datetime.now(timezone.utc).isoformat()
    campaign_id = f"legacy-campaign-{uuid4().hex}"
    run_id = f"legacy-run-{uuid4().hex}"
    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        await evaluation_db.init_db()
        async with evaluation_db.connect_db() as connection:
            await connection.execute(
                """
                INSERT INTO campaigns (
                    id, user_id, name, status, phase, config_json, completed_units, total_units,
                    evaluation_completed_units, evaluation_total_units, current_question_id,
                    current_mode, error_message, cancel_requested, created_at, started_at,
                    completed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, 1, 0, 0, NULL, NULL, NULL, 0, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    user_id,
                    "Legacy",
                    "completed",
                    "execution",
                    json.dumps(
                        {
                            "test_case_ids": ["Q-LEGACY"],
                            "modes": ["agentic"],
                            "model_config": {
                                "id": "cfg-legacy",
                                "name": "Legacy",
                                "model_name": "gemini-2.5-flash",
                                "temperature": 0.7,
                                "top_p": 0.95,
                                "top_k": 40,
                                "max_input_tokens": 8192,
                                "max_output_tokens": 2048,
                                "thinking_mode": False,
                                "thinking_budget": 8192,
                                "thinking_level": None,
                                "thinking_include_thoughts": False,
                            },
                            "repeat_count": 1,
                            "batch_size": 1,
                            "rpm_limit": 60,
                            "ragas_batch_size": 8,
                            "ragas_parallel_batches": 8,
                            "ragas_rpm_limit": 1000,
                        },
                        ensure_ascii=False,
                    ),
                    now,
                    now,
                    now,
                    now,
                ),
            )
            await connection.execute(
                """
                INSERT INTO campaign_results (
                    id, campaign_id, user_id, question_id, question, ground_truth,
                    mode, run_number, answer, contexts_json, source_doc_ids_json,
                    expected_sources_json, latency_ms, token_usage_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, '[]', '[]', '[]', 0, '{}', 'completed', ?)
                """,
                (
                    run_id,
                    campaign_id,
                    user_id,
                    "Q-LEGACY",
                    "Legacy question?",
                    "Legacy answer",
                    "agentic",
                    "Legacy answer",
                    now,
                ),
            )
            await connection.commit()
    return campaign_id, run_id


def test_research_analytics_endpoints_return_owned_run_details() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            ground_truth_short=test_case.ground_truth_short,
            key_points=list(test_case.key_points),
            ragas_focus=list(test_case.ragas_focus),
            mode=kwargs["mode"],
            answer="Fact A is in paper A.",
            contexts=["Fact A is in paper A."],
            source_doc_ids=["paper-a.pdf"],
            expected_sources=["paper-a.pdf"],
            latency_ms=17,
            token_usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            category=test_case.category,
            difficulty=test_case.difficulty,
            agent_trace={
                "classifier_decision": {"router_version": "semantic-v1"},
                "steps": [
                    {
                        "step_id": "s1",
                        "phase": "execution",
                        "step_type": "graph",
                        "title": "Graph lookup",
                        "tool_calls": [
                            {
                                "tool_name": "graph_search",
                                "tool_type": "graph",
                                "action": "SEARCH_GRAPH",
                                "status": "completed",
                            }
                        ],
                    }
                ],
                "claims": [
                    {
                        "claim_text": "Fact A is in paper A.",
                        "support_status": "supported",
                        "evidence": [{"chunk_id": "paper-a.pdf:1"}],
                    }
                ],
            },
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root = _make_upload_root()
    db_path = _make_db_path()
    with _build_client("user-a", upload_root, db_path, engine) as client:
        _create_test_case(client)
        created = client.post("/api/evaluation/campaigns", json=_campaign_payload())
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]
        _wait_for_completed(client, campaign_id)

        results = client.get(f"/api/evaluation/campaigns/{campaign_id}/results").json()["results"]
        run_id = results[0]["id"]

        overview = client.get(f"/api/evaluation/campaigns/{campaign_id}/overview")
        assert overview.status_code == 200
        assert overview.json()["analysis_unit"] == "execution"
        assert overview.json()["sample_count"] == 1
        assert overview.json()["independent_question_count"] == 1
        assert overview.json()["cost_status"] == "unknown"
        assert overview.json()["total_cost_usd"] is None
        assert overview.json()["unpriced_call_count"] >= 1

        runs = client.get(f"/api/evaluation/campaigns/{campaign_id}/runs")
        assert runs.status_code == 200
        run_list_item = runs.json()["runs"][0]
        assert run_list_item["run_id"] == run_id
        # Ordinary pre-v9 campaigns retain the v8 default when read through
        # the analytics projection rather than the full result payload.
        assert run_list_item["condition_id"] is None
        assert run_list_item["agentic_execution_version"] == "v8"

        aggregate_endpoint_expectations = {
            "mode-comparison": ("execution", 1),
            "question-comparison": ("question", 1),
            "cost-latency": ("execution", 1),
            "router-analysis": ("execution", 1),
            "ablation": ("execution", 1),
            "human-vs-auto": ("execution", 0),
            "repeat-stability": ("execution", 1),
        }
        for endpoint, (analysis_unit, sample_count) in aggregate_endpoint_expectations.items():
            response = client.get(f"/api/evaluation/campaigns/{campaign_id}/{endpoint}")
            assert response.status_code == 200
            assert response.json()["campaign_id"] == campaign_id
            assert response.json()["analysis_unit"] == analysis_unit
            assert response.json()["sample_count"] == sample_count

        dashboard = client.get(f"/api/evaluation/campaigns/{campaign_id}/analytics-dashboard")
        assert dashboard.status_code == 200
        dashboard_body = dashboard.json()
        assert dashboard_body["campaign_id"] == campaign_id
        assert dashboard_body["overview"]["sample_count"] == 1
        assert dashboard_body["mode_comparison"]["sample_count"] == 1
        assert dashboard_body["question_comparison"]["sample_count"] == 1
        assert dashboard_body["cost_latency"]["sample_count"] == 1
        assert dashboard_body["router_analysis"]["sample_count"] == 1
        assert dashboard_body["ablation"]["sample_count"] == 1
        assert dashboard_body["human_vs_auto"]["sample_count"] == 0
        assert dashboard_body["human_queue"]["campaign_id"] == campaign_id
        assert dashboard_body["errors"]["campaign_id"] == campaign_id
        assert dashboard_body["runs"]["runs"][0]["run_id"] == run_id

        endpoint_expectations = {
            "trace": "trace_events",
            "retrieval": "retrieval_events",
            "context": "context_packs",
            "llm-calls": "llm_calls",
            "tools": "tool_calls",
            "visual": "tool_calls",
            "graph": "tool_calls",
            "claims": "claims",
            "metrics": "derived_metrics",
        }
        for endpoint, key in endpoint_expectations.items():
            response = client.get(f"/api/evaluation/runs/{run_id}/{endpoint}")
            assert response.status_code == 200
            assert response.json()["run_id"] == run_id
            assert key in response.json()

        detail = client.get(f"/api/evaluation/runs/{run_id}/detail")
        assert detail.status_code == 200
        assert detail.json()["run_id"] == run_id
        assert detail.json()["agentic_v9"] is None

        assert client.get(f"/api/evaluation/runs/{run_id}/graph").json()["tool_calls"][0]["tool_name"] == "graph_search"
        assert client.get(f"/api/evaluation/runs/{run_id}/visual").json()["tool_calls"] == []

        observability = client.get(
            f"/api/evaluation/campaigns/{campaign_id}/runs/{run_id}/observability"
        )
        assert observability.status_code == 200
        assert observability.json()["run_summary"]["run_id"] == run_id
        assert observability.json()["run_summary"]["question_id"] == "Q-ANALYTICS"
        assert observability.json()["run_summary"]["answer_preview"] == "Fact A is in paper A."
        assert observability.json()["run_summary"]["accounting_status"] in {
            "complete",
            "partial",
            "not_available",
        }
        assert observability.json()["evidence_coverage"][0]["atomic_fact_id"] == "F1"
        assert observability.json()["evidence_coverage"][0]["retrieved"] is True
        assert observability.json()["evidence_coverage_status"] == "complete"

        behavior = client.get(f"/api/evaluation/campaigns/{campaign_id}/agent-behavior")
        assert behavior.status_code == 200
        behavior_rows = behavior.json()["rows"]
        assert behavior_rows[0]["run_id"] == run_id
        assert behavior_rows[0]["question_id"] == "Q-ANALYTICS"
        assert behavior_rows[0]["trace_status"] in {"completed", "not_instrumented"}
        assert behavior_rows[0]["accounting_status"] in {"complete", "partial", "not_available"}
        assert "unsupported_claim_ratio" in behavior_rows[0]
        assert "supported_claim_ratio" in behavior_rows[0]

        diff = client.get(f"/api/evaluation/runs/{run_id}/diff?baseline_run_id={run_id}")
        assert diff.status_code == 200
        assert diff.json()["run_id"] == run_id
        assert diff.json()["baseline_run_id"] == run_id
        assert diff.json()["token_delta"] == 0
        assert diff.json()["comparison_scope"] == "same_run"
        assert diff.json()["answer_change_status"] == "unchanged"

        legacy_campaign_id, legacy_run_id = asyncio.run(
            _seed_legacy_campaign_result(db_path=db_path, user_id="user-a")
        )
        legacy_trace = client.get(f"/api/evaluation/runs/{legacy_run_id}/trace")
        assert legacy_trace.status_code == 200
        assert legacy_trace.json()["trace_events"] == []
        legacy_retrieval = client.get(f"/api/evaluation/runs/{legacy_run_id}/retrieval")
        assert legacy_retrieval.status_code == 200
        assert legacy_retrieval.json()["retrieval_events"] == []
        assert legacy_retrieval.json()["retrieval_chunks"] == []
        incompatible_diff = client.get(f"/api/evaluation/runs/{run_id}/diff?baseline_run_id={legacy_run_id}")
        assert incompatible_diff.status_code == 400
        legacy_overview = client.get(f"/api/evaluation/campaigns/{legacy_campaign_id}/overview")
        assert legacy_overview.status_code == 200

    with _build_client("user-b", upload_root, db_path, engine) as other_client:
        denied = other_client.get(f"/api/evaluation/runs/{run_id}/trace")
        assert denied.status_code == 404
        denied_tools = other_client.get(f"/api/evaluation/runs/{run_id}/tools")
        assert denied_tools.status_code == 404
        denied_overview = other_client.get(f"/api/evaluation/campaigns/{campaign_id}/overview")
        assert denied_overview.status_code == 404
        denied_diff = other_client.get(f"/api/evaluation/runs/{run_id}/diff?baseline_run_id={run_id}")
        assert denied_diff.status_code == 404

    with _build_client("user-a", upload_root, db_path, engine, with_auth=False) as unauthenticated_client:
        unauthenticated = unauthenticated_client.get(f"/api/evaluation/runs/{run_id}/trace")
        assert unauthenticated.status_code == 401
        unauthenticated_overview = unauthenticated_client.get(
            f"/api/evaluation/campaigns/{campaign_id}/overview"
        )
        assert unauthenticated_overview.status_code == 401


def test_v9_campaign_preflight_uses_golden_routes_and_reports_incompatible_setup() -> None:
    """The admission check is deterministic, per question, and token-only."""
    engine = CampaignEngine(runner=Mock(), ragas_evaluator=FakeRagasEvaluator())
    temp_root = Path(tempfile.mkdtemp(prefix="analytics_v9_preflight_"))
    upload_root = temp_root / "uploads"
    upload_root.mkdir()
    db_path = temp_root / "evaluation.db"
    with _build_client("user-a", upload_root, db_path, engine) as client:
        created_case = client.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q10",
                "question": "Golden route preflight question",
                "ground_truth": "Golden route preflight answer",
            },
        )
        assert created_case.status_code == 200

        compatible = client.post(
            "/api/evaluation/campaigns/preflight",
            json={
                "test_case_ids": ["Q10"],
                "model_config": {
                    "id": "cfg-v9", "name": "v9", "model_name": "gemini-2.5-flash",
                    "temperature": 0.7, "top_p": 0.95, "top_k": 40,
                    "max_input_tokens": 8192, "max_output_tokens": 2048,
                    "thinking_mode": False, "thinking_budget": 8192,
                },
                "runtime_token_budget": 10000, "max_llm_calls": 4,
            },
        )
        assert compatible.status_code == 200
        compatible_question = compatible.json()["questions"][0]
        assert compatible_question["question_id"] == "Q10"
        assert compatible_question["expected_route"] == "single_lookup"
        assert compatible_question["status"] == "feasible"
        assert compatible_question["issues"] == []
        assert "cost" not in compatible.json()

        incompatible = client.post(
            "/api/evaluation/campaigns/preflight",
            json={
                "test_case_ids": ["Q10"],
                "model_config": {
                    "id": "cfg-v9-unknown-thinking", "name": "v9", "model_name": "gemini-2.5-flash",
                    "temperature": 0.7, "top_p": 0.95, "top_k": 40,
                    "max_input_tokens": 8192, "max_output_tokens": 2048,
                    "thinking_mode": True, "thinking_budget": -1,
                },
                "runtime_token_budget": 10000, "max_llm_calls": 4,
            },
        )
        assert incompatible.status_code == 200
        issue = incompatible.json()["questions"][0]["issues"][0]
        assert issue["status"] == "configuration_incompatible"
        assert issue["reason"] == "thinking_reserve_unknown"
