"""Fail-closed checks for the Agentic v9 smoke-plan and offline verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from data_base.agentic_v9.repair import RepairPlan
from data_base.agentic_v9.schemas import ResolvedSourceScope, RetrievalTask
from evaluation.smoke_verification import (
    DEFAULT_SMOKE_QUESTION_IDS,
    build_release_manifest,
    build_smoke_plan,
    execute_smoke_plan,
    verify_campaign_export,
)


def _complete_export() -> dict[str, Any]:
    runs = [_complete_run(question_id) for question_id in DEFAULT_SMOKE_QUESTION_IDS]
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
        "runs": runs,
        "llm_calls": [
            {
                "run_id": run["id"],
                "phase": "final_answer",
                "reservation_id": "reservation-a",
                "provider_attempt": 1,
                "status": "success",
                "prompt_hash": "hash-a",
                "prompt_capture_status": "captured",
                "full_prompt_capture_status": "not_captured_at_execution",
                "total_tokens": 12,
                "payload": {"usage_status": "measured", "official_total_tokens": 12},
            }
            for run in runs
        ],
    }


def _complete_run(question_id: str) -> dict[str, Any]:
    return {
        "id": f"run-{question_id}",
        "question_id": question_id,
        "repeat_number": 1,
        "status": "completed",
        "mode": "agentic",
        "total_tokens": 12,
        "agent_trace": {
            "agentic_execution_version": "v9",
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
                "budget_reservations": [
                    {
                        "reservation_id": "reservation-a",
                        "phase": "final_answer",
                        "provider_attempt": 1,
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
            },
        },
    }


def _recovery_diagnostics_export() -> dict[str, Any]:
    """Represent the redacted Task 1--3 Agentic v9 trace projection."""
    artifact = _complete_export()
    for run in artifact["runs"]:
        question_id = run["question_id"]
        v9 = run["agent_trace"]["agentic_v9"]
        v9["query_contract"]["required_slots"][0]["authorized_source_doc_ids"] = [
            "doc-authorized"
        ]
        v9["query_contract"]["resolved_source_scope"] = {
            "authorized_doc_ids": ["doc-authorized"],
            "source_name_to_doc_ids": {},
        }
        if question_id in {"Q14", "Q16"}:
            v9["query_contract"]["required_slots"][0]["locator_hints"] = [
                "Table 3"
            ]
        v9["evidence_packets"] = [
            {
                "evidence_id": "evidence-a",
                "slot_ids": ["fact-a"],
                "source": {"doc_id": "doc-authorized"},
            }
        ]
        v9["context_pack"] = {"packed_evidence_ids": ["evidence-a"], "token_count": 4}
        v9["retrieval_diagnostics"] = [
            {
                "source_filter": {"authorized_doc_ids": ["doc-authorized"], "post_filter_count": 1},
                "reranking": {"fallback_reason": None, "selected_count": 1},
            }
        ]
        v9["locator_diagnostics"] = (
            [{"slot_id": "fact-a", "state": "matched"}]
            if question_id in {"Q14", "Q16"}
            else []
        )
        v9["budget_reservations"].append(
            {
                "reservation_id": "evidence-reservation",
                "phase": "evidence_extract",
                "provider_attempt": 1,
            }
        )
        run["total_tokens"] = 24
        v9["metrics"]["reconciled_tokens"] = 24
        artifact["llm_calls"].append(
            {
                "run_id": run["id"],
                "phase": "evidence_extract",
                "reservation_id": "evidence-reservation",
                "provider_attempt": 1,
                "status": "success",
                "prompt_hash": "hash-evidence",
                "prompt_capture_status": "captured",
                "full_prompt_capture_status": "not_captured_at_execution",
                "total_tokens": 12,
                "payload": {"usage_status": "measured", "official_total_tokens": 12},
            }
        )
    return artifact


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


def test_cli_refuses_to_overwrite_an_existing_manifest(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    artifact = tmp_path / "campaign.json"
    artifact.write_text(json.dumps(_complete_export()), encoding="utf-8")
    manifest = tmp_path / "release.json"
    manifest.write_text("do-not-overwrite", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_agentic_v9_smoke.py",
            "--artifact",
            str(artifact),
            "--manifest",
            str(manifest),
        ],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "already exists" in completed.stderr
    assert manifest.read_text(encoding="utf-8") == "do-not-overwrite"


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


@pytest.mark.parametrize(
    ("base_url", "preset_name", "auth_header", "confirmation", "message"),
    [
        ("https://evaluation.example", "", "Authorization: Bearer test", "I_UNDERSTAND_EXECUTE", "preset"),
        ("https://evaluation.example", "release-safe", "", "I_UNDERSTAND_EXECUTE", "authentication"),
        ("https://evaluation.example", "release-safe", "not-a-header", "I_UNDERSTAND_EXECUTE", "Header-Name"),
        ("https://evaluation.example", "release-safe", "Authorization: Bearer test", "no", "confirm-execute"),
    ],
)
def test_each_execute_guard_rejects_before_transport(
    base_url: str,
    preset_name: str,
    auth_header: str,
    confirmation: str,
    message: str,
) -> None:
    def transport(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("transport must not be reached")

    with pytest.raises(ValueError, match=message):
        execute_smoke_plan(
            build_smoke_plan(),
            base_url=base_url,
            preset_name=preset_name,
            auth_header=auth_header,
            confirmation=confirmation,
            transport=transport,
        )


def test_execute_uses_injected_fake_transport_only_after_all_guards() -> None:
    calls: list[tuple[str, str, object | None]] = []

    def transport(
        method: str, url: str, _headers: dict[str, str], payload: object | None
    ) -> object:
        calls.append((method, url, payload))
        if url.endswith("/test-cases"):
            return [{"id": question_id, "question_id": question_id} for question_id in DEFAULT_SMOKE_QUESTION_IDS]
        if url.endswith("/model-configs"):
            return [{"id": "preset-id", "name": "release-safe", "model_name": "fake"}]
        assert method == "POST"
        assert isinstance(payload, dict)
        assert payload["model_config_id"] == "preset-id"
        return {"campaign_id": "created-campaign", "Authorization": "secret-sentinel"}

    result = execute_smoke_plan(
        build_smoke_plan(),
        base_url="https://evaluation.example",
        preset_name="release-safe",
        auth_header="Authorization: Bearer test",
        confirmation="I_UNDERSTAND_EXECUTE",
        transport=transport,
    )

    assert result["campaign_id"] == "created-campaign"
    assert "secret-sentinel" not in json.dumps(result)
    assert [method for method, _url, _payload in calls] == ["GET", "GET", "POST"]
    payload = calls[-1][2]
    assert payload["test_case_ids"] == list(DEFAULT_SMOKE_QUESTION_IDS)
    assert payload["modes"] == ["agentic"]
    assert payload["agentic_execution_version"] == "v9"
    assert payload["repeat_count"] == 1


def test_missing_artifact_is_not_executed() -> None:
    report = verify_campaign_export(None)

    assert report.status == "not_executed"
    assert report.requirements["campaign_export"].status == "not_executed"


def test_missing_required_v9_fields_is_never_pass() -> None:
    report = verify_campaign_export({"runs": [{"id": "run-v9", "mode": "agentic"}]})

    assert report.status in {"partial", "fail"}
    assert report.status != "pass"


def test_generic_agentic_without_explicit_v9_trace_is_not_certified() -> None:
    report = verify_campaign_export(
        {"runs": [{"id": "run-v8", "question_id": "Q5", "mode": "agentic", "status": "completed"}]}
    )

    assert report.status == "partial"


def test_failed_or_wrong_question_v9_run_fails_fixed_plan_coverage() -> None:
    artifact = _complete_export()
    artifact["runs"][0]["status"] = "failed"
    artifact["runs"][1]["question_id"] = "Q999"

    report = verify_campaign_export(artifact)

    assert report.status == "fail"
    assert report.requirements["plan_coverage"].status == "fail"


def test_complete_v9_export_passes_and_writes_a_reproducible_manifest(tmp_path: Path) -> None:
    artifact = _recovery_diagnostics_export()
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


def test_recovery_diagnostics_fail_closed_on_task_1_to_3_regressions() -> None:
    artifact = _recovery_diagnostics_export()

    assert verify_campaign_export(artifact).requirements["retrieval_evidence_recovery"].status == "pass"

    all_zero_contexts = copy.deepcopy(artifact)
    for run in all_zero_contexts["runs"]:
        run["agent_trace"]["agentic_v9"]["context_pack"]["packed_evidence_ids"] = []
    assert verify_campaign_export(all_zero_contexts).status == "fail"

    outside_scope = copy.deepcopy(artifact)
    outside_scope["runs"][0]["agent_trace"]["agentic_v9"]["evidence_packets"][0]["source"]["doc_id"] = "doc-blocked"
    assert verify_campaign_export(outside_scope).status == "fail"

    locator_only_empty_evidence = copy.deepcopy(artifact)
    locator_only_empty_evidence["runs"][0]["agent_trace"]["agentic_v9"]["evidence_packets"] = []
    locator_only_empty_evidence["runs"][0]["agent_trace"]["agentic_v9"]["locator_diagnostics"] = [
        {"slot_id": "fact-a", "state": "unavailable"}
    ]
    assert verify_campaign_export(locator_only_empty_evidence).status == "fail"

    configuration_incompatible = copy.deepcopy(artifact)
    configuration_incompatible["runs"][0]["agent_trace"]["response_status"] = "configuration_incompatible"
    assert verify_campaign_export(configuration_incompatible).status == "fail"

    absent_locator_state = copy.deepcopy(artifact)
    absent_locator_state["runs"][3]["agent_trace"]["agentic_v9"]["locator_diagnostics"] = []
    assert verify_campaign_export(absent_locator_state).status == "partial"

    excessive_evidence_calls = copy.deepcopy(artifact)
    extra_call = copy.deepcopy(excessive_evidence_calls["llm_calls"][-1])
    extra_call["provider_attempt"] = 2
    extra_call["reservation_id"] = "evidence-reservation-2"
    excessive_evidence_calls["runs"][0]["agent_trace"]["agentic_v9"]["budget_reservations"].append(
        {
            "reservation_id": "evidence-reservation-2",
            "phase": "evidence_extract",
            "provider_attempt": 2,
        }
    )
    excessive_evidence_calls["llm_calls"].append(extra_call)
    assert verify_campaign_export(excessive_evidence_calls).status == "fail"

    lost_rerank_candidates = copy.deepcopy(artifact)
    reranking = lost_rerank_candidates["runs"][0]["agent_trace"]["agentic_v9"]["retrieval_diagnostics"][0]["reranking"]
    reranking.update({"fallback_reason": "exception", "selected_count": 0})
    assert verify_campaign_export(lost_rerank_candidates).status == "fail"


def test_locator_diagnostics_must_cover_each_structured_slot() -> None:
    artifact = _recovery_diagnostics_export()
    q14 = artifact["runs"][3]["agent_trace"]["agentic_v9"]
    q14["query_contract"]["required_slots"].append(
        {
            "slot_id": "ordinary-slot",
            "description": "State the ordinary result.",
            "authorized_source_doc_ids": ["doc-authorized"],
        }
    )
    q14["slot_resolutions"].append(
        {
            "slot_id": "ordinary-slot",
            "status": "supported",
            "evidence_ids": ["evidence-a"],
        }
    )
    q14["final_claims"].append(
        {
            "slot_id": "ordinary-slot",
            "support_type": "direct",
            "evidence_ids": ["evidence-a"],
        }
    )
    q14["locator_diagnostics"] = [{"slot_id": "ordinary-slot", "state": "matched"}]

    report = verify_campaign_export(artifact)

    assert report.status == "fail"
    assert report.requirements["retrieval_evidence_recovery"].status == "fail"


def test_packet_source_scope_uses_source_name_slot_authorization() -> None:
    artifact = _recovery_diagnostics_export()
    v9 = artifact["runs"][0]["agent_trace"]["agentic_v9"]
    contract = v9["query_contract"]
    contract["resolved_source_scope"] = {
        "authorized_doc_ids": ["doc-alpha", "doc-beta"],
        "source_name_to_doc_ids": {"Alpha.pdf": ["doc-alpha"]},
    }
    slot = contract["required_slots"][0]
    slot["authorized_source_doc_ids"] = []
    slot["source_name_hints"] = ["Alpha.pdf"]
    v9["evidence_packets"][0]["source"]["doc_id"] = "doc-alpha"

    report = verify_campaign_export(artifact)

    assert report.status == "pass"


def test_packet_source_scope_rejects_unknown_slot_ids() -> None:
    artifact = _recovery_diagnostics_export()
    packet = artifact["runs"][0]["agent_trace"]["agentic_v9"]["evidence_packets"][0]
    packet["slot_ids"] = ["unknown-slot"]

    report = verify_campaign_export(artifact)

    assert report.status == "fail"
    assert report.requirements["retrieval_evidence_recovery"].status == "fail"


def test_manifest_recursively_redacts_setup_secrets(tmp_path: Path) -> None:
    artifact_path = tmp_path / "campaign.json"
    artifact_path.write_text(json.dumps(_complete_export()), encoding="utf-8")

    manifest = build_release_manifest(
        verify_campaign_export(_complete_export()),
        backend_commit="backend",
        frontend_commit="frontend",
        setup_snapshot={
            "nested": {
                "Authorization": "secret-sentinel",
                "api_key": "secret-sentinel",
                "Cookie": "secret-sentinel",
            }
        },
        dataset_identity="golden-v2",
        input_paths={"campaign_export": artifact_path},
    )

    assert "secret-sentinel" not in json.dumps(manifest)


def test_persisted_route_decision_and_resolution_status_are_required() -> None:
    artifact = _complete_export()
    v9 = artifact["runs"][0]["agent_trace"]["agentic_v9"]
    v9["query_contract"].pop("route_decision")
    v9["query_contract"]["route_reason"] = "invented fallback is not a decision"
    v9["slot_resolutions"][0] = {"slot_id": "fact-a"}
    v9.pop("final_claims")

    report = verify_campaign_export(artifact)

    assert report.status == "fail"
    assert report.requirements["contract_and_route"].status == "fail"
    assert report.requirements["slots_and_resolutions"].status == "fail"


def test_real_nested_repair_plan_targets_not_found_slot() -> None:
    artifact = _complete_export()
    run = artifact["runs"][0]
    v9 = run["agent_trace"]["agentic_v9"]
    v9["slot_resolutions"][0] = {"slot_id": "fact-a", "status": "not_found"}
    v9["final_claims"] = []
    repair = RepairPlan(
        repair_round_index=1,
        tasks=[
            RetrievalTask(
                task_id="repair-task",
                round_id="repair-1",
                query_id="Q5",
                query="source locator target",
                target_slot_ids=["fact-a"],
                source_scope=ResolvedSourceScope(authorized_doc_ids=["doc-a"]),
            )
        ],
    )
    v9["repairs"] = [repair.model_dump(mode="json")]

    report = verify_campaign_export(artifact)

    assert report.requirements["repair_for_missing_slots"].status == "pass"


def test_conflicted_terminal_slot_does_not_require_a_retrieval_repair() -> None:
    artifact = _complete_export()
    v9 = artifact["runs"][0]["agent_trace"]["agentic_v9"]
    v9["slot_resolutions"][0] = {
        "slot_id": "fact-a",
        "status": "conflicted",
        "reason": "scope conflict recorded",
    }
    v9["final_claims"] = []

    report = verify_campaign_export(artifact)

    assert report.requirements["repair_for_missing_slots"].status == "pass"


def test_unsupported_slot_cannot_be_exported_as_a_supported_final_claim() -> None:
    artifact = _complete_export()
    v9 = artifact["runs"][0]["agent_trace"]["agentic_v9"]
    v9["slot_resolutions"][0] = {"slot_id": "fact-a", "status": "not_found"}
    v9["repairs"] = [{"tasks": [{"target_slot_ids": ["fact-a"]}]}]

    report = verify_campaign_export(artifact)

    assert report.status == "fail"
    assert report.requirements["supported_claims"].status == "fail"


@pytest.mark.parametrize(
    "claim",
    [
        {"support_type": "direct", "evidence_ids": []},
        {"slot_id": "unknown-slot", "support_type": "calculated", "evidence_ids": ["evidence-a"]},
        {"slot_id": "fact-a", "support_type": "comparative_inference", "evidence_ids": []},
    ],
)
def test_positive_final_claims_require_slot_and_evidence_provenance(claim: dict[str, Any]) -> None:
    artifact = _complete_export()
    artifact["runs"][0]["agent_trace"]["agentic_v9"]["final_claims"] = [claim]

    report = verify_campaign_export(artifact)

    assert report.status == "fail"
    assert report.requirements["supported_claims"].status == "fail"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda artifact: artifact["runs"][0]["agent_trace"]["agentic_v9"].pop("budget_reservations"),
        lambda artifact: artifact["llm_calls"].__setitem__(0, {**artifact["llm_calls"][0], "reservation_id": "orphan"}),
    ],
)
def test_provider_attempts_require_bidirectional_budget_linkage(mutate: Any) -> None:
    artifact = _complete_export()
    mutate(artifact)

    report = verify_campaign_export(artifact)

    assert report.status in {"partial", "fail"}
    assert report.requirements["phase_linked_provider_attempts"].status in {"partial", "fail"}
