"""Fail-closed checks for the Agentic v9 smoke-plan and offline verifier."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from evaluation.smoke_verification import (
    DEFAULT_SMOKE_QUESTION_IDS,
    build_release_manifest,
    build_smoke_plan,
    execute_smoke_plan,
    verify_campaign_export,
)


def _complete_export() -> dict[str, object]:
    return {
        "campaign": {
            "config": {
                "prompt_capture_policy": {
                    "hash": True,
                    "preview": True,
                    "full_prompt": False,
                }
            }
        },
        "runs": [
            {
                "id": "run-v9",
                "mode": "agentic",
                "total_tokens": 12,
                "agent_trace": {
                    "agentic_v9": {
                        "query_contract": {
                            "contract_version": "2",
                            "route": "multi_hop",
                            "required_slots": [{"slot_id": "fact-a"}],
                            "route_decision": {
                                "selected_route": "multi_hop",
                                "route_reason": "The question requires two sources.",
                            },
                        },
                        "slot_resolutions": [
                            {
                                "slot_id": "fact-a",
                                "status": "supported",
                                "evidence_ids": ["evidence-a"],
                            }
                        ],
                        "final_claims": [
                            {
                                "slot_id": "fact-a",
                                "support_type": "direct",
                                "evidence_ids": ["evidence-a"],
                            }
                        ],
                        "metrics": {"reconciled_tokens": 12},
                    }
                },
            }
        ],
        "llm_calls": [
            {
                "run_id": "run-v9",
                "phase": "final_answer",
                "reservation_id": "reservation-a",
                "provider_attempt": 1,
                "prompt_hash": "hash-a",
                "prompt_capture_status": "captured",
                "full_prompt_capture_status": "not_captured_at_execution",
                "total_tokens": 12,
                "payload": {"usage_status": "measured", "official_total_tokens": 12},
            }
        ],
    }


def test_dry_run_has_fixed_v9_questions_and_no_paired_naive() -> None:
    plan = build_smoke_plan()

    assert plan["execution"] == "dry_run"
    assert plan["question_ids"] == list(DEFAULT_SMOKE_QUESTION_IDS)
    assert plan["modes"] == ["agentic"]
    assert plan["agentic_execution_version"] == "v9"
    assert plan["repeat_count"] == 1


def test_paired_naive_is_explicit_opt_in() -> None:
    assert build_smoke_plan(include_naive=True)["modes"] == ["agentic", "naive"]


def test_cli_dry_run_executes_from_repository_root_without_network() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/run_agentic_v9_smoke.py"],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["smoke_plan"]["execution"] == "dry_run"
    assert payload["execution"]["status"] == "not_executed"


def test_execute_rejects_missing_authorization_guards_before_transport() -> None:
    called = False

    def transport(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("transport must not be reached")

    with pytest.raises(ValueError, match="base URL"):
        execute_smoke_plan(
            build_smoke_plan(),
            base_url="",
            preset_name="",
            auth_header="",
            confirmation="",
            transport=transport,
        )

    assert not called


def test_execute_uses_injected_fake_transport_only_after_all_guards() -> None:
    calls: list[tuple[str, str]] = []

    def transport(
        method: str, url: str, _headers: dict[str, str], payload: object | None
    ) -> object:
        calls.append((method, url))
        if url.endswith("/test-cases"):
            return [{"id": question_id, "question_id": question_id} for question_id in DEFAULT_SMOKE_QUESTION_IDS]
        if url.endswith("/model-configs"):
            return [{"id": "preset-id", "name": "release-safe", "model_name": "fake"}]
        assert method == "POST"
        assert isinstance(payload, dict)
        assert payload["model_config_id"] == "preset-id"
        return {"campaign_id": "created-campaign"}

    result = execute_smoke_plan(
        build_smoke_plan(),
        base_url="https://evaluation.example",
        preset_name="release-safe",
        auth_header="Authorization: Bearer test",
        confirmation="I_UNDERSTAND_EXECUTE",
        transport=transport,
    )

    assert result["campaign_id"] == "created-campaign"
    assert [method for method, _url in calls] == ["GET", "GET", "POST"]


def test_missing_artifact_is_not_executed() -> None:
    report = verify_campaign_export(None)

    assert report.status == "not_executed"
    assert report.requirements["campaign_export"].status == "not_executed"


def test_missing_required_v9_fields_is_never_pass() -> None:
    report = verify_campaign_export({"runs": [{"id": "run-v9", "mode": "agentic"}]})

    assert report.status in {"partial", "fail"}
    assert report.status != "pass"


def test_complete_v9_export_passes_and_writes_a_reproducible_manifest(tmp_path: Path) -> None:
    artifact = _complete_export()
    artifact_path = tmp_path / "campaign-redacted.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    report = verify_campaign_export(artifact)
    manifest = build_release_manifest(
        report,
        backend_commit="backend-commit",
        frontend_commit="frontend-commit",
        setup_snapshot={"preset": "release-safe"},
        dataset_identity="golden-v2",
        input_paths={"campaign_export": artifact_path},
    )

    assert report.status == "pass"
    assert manifest["repository_commits"] == {
        "backend": "backend-commit",
        "frontend": "frontend-commit",
    }
    assert manifest["input_hashes"]["campaign_export"].startswith("sha256:")
    assert manifest["release_gate_results"]["overall_status"] == "pass"


def test_unsupported_slot_cannot_be_exported_as_a_supported_final_claim() -> None:
    artifact = _complete_export()
    v9 = artifact["runs"][0]["agent_trace"]["agentic_v9"]
    v9["slot_resolutions"][0] = {"slot_id": "fact-a", "status": "not_found"}
    v9["repairs"] = [{"target_slot_ids": ["fact-a"]}]

    report = verify_campaign_export(artifact)

    assert report.status == "fail"
    assert report.requirements["supported_claims"].status == "fail"
