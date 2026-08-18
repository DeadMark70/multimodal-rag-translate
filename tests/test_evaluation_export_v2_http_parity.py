from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from evaluation.accounting_schemas import AccountingScopeStart, UsageEventCreate
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.db import RagasScoreRepository
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.rag_modes import BenchmarkExecutionResult
from evaluation.trace_schemas import EvaluationEvidencePacket
from evaluation.campaign_engine import CampaignEngine
from tests.test_evaluation_export_redaction import (
    FakeRagasEvaluator,
    _build_client,
    _campaign_payload,
    _make_workspace_paths,
    _seed_export_rows,
    _seed_stage_warning,
    _wait_for_completed,
)


def _json(client, path: str) -> dict:
    response = client.get(path)
    assert response.status_code == 200, response.text
    return response.json()


def _differences(left: object, right: object, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [f"{path}: {left!r} != {right!r}"]
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append(f"{path}.{key}: missing from one side")
            else:
                differences.extend(_differences(left[key], right[key], f"{path}.{key}"))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            differences.append(f"{path}: lengths {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            differences.extend(
                _differences(left_item, right_item, f"{path}[{index}]")
            )
        return differences
    return [] if left == right else [f"{path}: {left!r} != {right!r}"]


async def _seed_official_scores_and_evidence(
    *, campaign_id: str, results: list[dict]
) -> None:
    score_rows: list[dict] = []
    for index, result in enumerate(results, start=1):
        for metric_name, value in (
            ("answer_correctness", 0.7 + index / 100),
            ("faithfulness", 0.8 + index / 100),
            ("answer_relevancy", 0.6 + index / 100),
        ):
            score_rows.append(
                {
                    "campaign_result_id": result["id"],
                    "metric_name": metric_name,
                    "metric_value": value,
                    "source_attempt_id": result["source_attempt_id"],
                    "evaluation_signature": f"{result['id']}-{metric_name}",
                    "details": {
                        "evaluator_model": "parity-judge",
                        "metric_version": "v1",
                        "compatibility_signature": f"parity-{metric_name}",
                        "compatibility_signature_version": "v2",
                    },
                }
            )
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-a", campaign_id=campaign_id, score_rows=score_rows
    )
    accounting = EvaluationAccountingStore()
    for result in results:
        scope_id = f"parity-scope-{result['id']}"
        await accounting.start_scope(
            AccountingScopeStart(
                scope_id=scope_id,
                campaign_id=campaign_id,
                scope_type="execution_run",
                scope_key=result["source_attempt_id"],
                run_id=result["id"],
                targets=[
                    {
                        "job_id": "parity-job",
                        "work_item_id": f"parity-work-{result['id']}",
                        "attempt_id": result["source_attempt_id"],
                        "campaign_result_id": result["id"],
                        "is_official": True,
                    }
                ],
            )
        )
        await accounting.record_event(
            UsageEventCreate(
                usage_event_id=f"parity-usage-{result['id']}",
                scope_id=scope_id,
                campaign_id=campaign_id,
                scope_type="execution_run",
                scope_key=result["source_attempt_id"],
                run_id=result["id"],
                phase="answer_generation",
                purpose="generation",
                input_tokens=10,
                output_text_tokens=6,
                reported_total_tokens=16,
                usage_status="measured",
                reconciliation_status="balanced",
                pricing_status="unknown_model",
                created_at=datetime.now(timezone.utc),
            )
        )
        await accounting.finalize_scope(scope_id, "completed")

    agentic = next(result for result in results if result["mode"] == "agentic")
    await EvaluationObservabilityRepository().record_evidence_packet(
        EvaluationEvidencePacket(
            attempt_id=agentic["source_attempt_id"],
            run_id=agentic["id"],
            campaign_id=campaign_id,
            evidence_id="parity-evidence-1",
            packet={
                "schema_version": "1",
                "evidence_id": "parity-evidence-1",
                "task_id": "task-1",
                "round_id": "round-1",
                "query_id": "query-1",
                "slot_ids": ["slot-1"],
                "statement": "Durable evidence.",
                "support_type": "direct",
                "source": {"doc_id": "doc-1", "chunk_id": "chunk-1"},
                "scope": {"dataset": "parity"},
                "locator": {"pdf_page_index": 0},
            },
            created_at=datetime.now(timezone.utc),
        )
    )


def test_authenticated_http_panel_and_export_v2_objects_are_identical() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        mode = kwargs["mode"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=mode,
            answer=f"{mode} grounded answer",
            contexts=["Durable retrieved context"],
            source_doc_ids=["doc-1"],
            expected_sources=[],
            latency_ms=10,
            token_usage={"input_tokens": 10, "output_tokens": 6, "total_tokens": 16},
            category=test_case.category,
            difficulty=test_case.difficulty,
            agent_trace={"agentic_v9": {"retrieval_diagnostics": []}}
            if mode == "agentic"
            else {},
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root, db_path = _make_workspace_paths("export_http_parity")

    with _build_client("user-a", upload_root, db_path, engine) as client:
        created_case = client.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q-EXPORT",
                "question": "What failed?",
                "ground_truth": "A safe answer",
                "source_docs": [],
                "requires_multi_doc_reasoning": False,
            },
        )
        assert created_case.status_code == 200
        campaign_payload = _campaign_payload()
        campaign_payload["modes"] = ["naive", "agentic"]
        # This parity fixture deliberately exercises the legacy v9 envelope.
        campaign_payload["agentic_execution_version"] = "v9"
        created = client.post("/api/evaluation/campaigns", json=campaign_payload)
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]
        _wait_for_completed(client, campaign_id)

        results = _json(
            client, f"/api/evaluation/campaigns/{campaign_id}/results"
        )["results"]
        assert {result["mode"] for result in results} == {"naive", "agentic"}
        agentic = next(result for result in results if result["mode"] == "agentic")
        asyncio.run(
            _seed_export_rows(
                run_id=agentic["id"],
                campaign_id=campaign_id,
                attempt_id=agentic["source_attempt_id"],
            )
        )
        asyncio.run(_seed_stage_warning(run_id=agentic["id"], campaign_id=campaign_id))
        asyncio.run(
            _seed_official_scores_and_evidence(
                campaign_id=campaign_id, results=results
            )
        )
        rating = client.post(
            f"/api/evaluation/runs/{agentic['id']}/human-ratings",
            json={
                "rubric_version": "v1",
                "correctness_score": 0.9,
                "faithfulness_score": 0.8,
                "completeness_score": 0.75,
                "citation_quality_score": 0.85,
                "usefulness_score": 0.8,
                "comments": "grounded",
            },
        )
        assert rating.status_code == 200

        research = _json(
            client, f"/api/evaluation/campaigns/{campaign_id}/research-summary"
        )
        release = _json(
            client, f"/api/evaluation/campaigns/{campaign_id}/release-metrics"
        )
        question = _json(
            client,
            f"/api/evaluation/campaigns/{campaign_id}/research-question-comparison",
        )
        behavior = _json(
            client, f"/api/evaluation/campaigns/{campaign_id}/agent-behavior"
        )
        router = _json(
            client, f"/api/evaluation/campaigns/{campaign_id}/router-analysis"
        )
        ablation = _json(
            client, f"/api/evaluation/campaigns/{campaign_id}/ablation"
        )
        human = _json(
            client, f"/api/evaluation/campaigns/{campaign_id}/human-vs-auto"
        )
        queue = _json(
            client, f"/api/evaluation/campaigns/{campaign_id}/human-eval-queue"
        )
        errors = _json(client, f"/api/evaluation/campaigns/{campaign_id}/errors")
        warnings = _json(
            client, f"/api/evaluation/campaigns/{campaign_id}/stage-warnings"
        )
        interactive = _json(
            client,
            f"/api/evaluation/campaigns/{campaign_id}/runs/{agentic['id']}/observability",
        )

        summary_response = client.post(
            f"/api/evaluation/campaigns/{campaign_id}/export", json={}
        )
        full_response = client.post(
            f"/api/evaluation/campaigns/{campaign_id}/export",
            json={"include_run_observability": True},
        )
        assert summary_response.status_code == 200, summary_response.text
        assert full_response.status_code == 200, full_response.text
        exported = summary_response.json()
        exported_full = full_response.json()

        research_differences = _differences(
            exported["sections"]["overview"]["data"]["research_summary"],
            research,
        )
        assert not research_differences, "\n".join(research_differences)
        assert exported["sections"]["overview"]["data"]["release_metrics"]["data"] == release
        assert exported["sections"]["question_analysis"]["data"] == question
        assert exported["sections"]["agent_behavior"]["data"] == behavior
        assert exported["sections"]["router_analysis"]["data"] == router
        assert exported["sections"]["ablation"]["data"] == ablation
        assert exported["sections"]["human_evaluation"]["data"]["comparison"] == human
        assert exported["sections"]["human_evaluation"]["data"]["queue"] == queue
        assert exported["sections"]["diagnostics"]["data"]["errors"] == errors
        assert exported["sections"]["diagnostics"]["data"]["stage_warnings"] == warnings
        full_run = next(
            run
            for run in exported_full["runs"]
            if run["result"]["run_id"] == agentic["id"]
        )
        observability_differences = _differences(
            full_run["observability"]["data"], interactive
        )
        assert not observability_differences, "\n".join(observability_differences)
        assert {run["result"]["run_id"] for run in exported["runs"]} == {
            result["id"] for result in results
        }
        assert all(run["ragas_metrics"] for run in exported["runs"])
        accounting_totals = {
            run["result"]["run_id"]: run["accounting"]["total_tokens"]
            for run in exported["runs"]
        }
        assert accounting_totals == {
            result["id"]: 16 for result in results
        }, accounting_totals
        assert interactive["agentic_v9"]["evidence_packets"][0]["evidence_id"] == (
            "parity-evidence-1"
        )
        assert router["rows"]
        assert human["rows"]
        assert errors["rows"]
        assert warnings["rows"]
