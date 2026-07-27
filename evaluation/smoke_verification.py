"""Dry-run Agentic v9 smoke planning and fail-closed export verification.

This module deliberately consumes JSON-shaped exports instead of ORM objects so a
release artifact can be checked on a disconnected machine.  Unknown legacy or
incomplete fields are represented as ``partial`` rather than being inferred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from core.sensitive_data import sanitize_credential_value


DEFAULT_SMOKE_QUESTION_IDS = ("Q5", "Q7", "Q11", "Q14", "Q16")
EXECUTION_CONFIRMATION = "I_UNDERSTAND_EXECUTE"
VerificationStatus = Literal["pass", "fail", "partial", "not_executed"]
Transport = Callable[[str, str, dict[str, str], object | None], object]


@dataclass(frozen=True)
class RequirementResult:
    status: VerificationStatus
    reason: str | None = None


@dataclass(frozen=True)
class VerificationReport:
    status: VerificationStatus
    requirements: dict[str, RequirementResult]
    residual_failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "requirements": {
                name: asdict(result) for name, result in self.requirements.items()
            },
            "residual_failures": list(self.residual_failures),
        }


def build_smoke_plan(*, include_naive: bool = False) -> dict[str, Any]:
    """Build the fixed, non-mutating five-question Agentic v9 smoke plan."""
    return {
        "execution": "dry_run",
        "question_ids": list(DEFAULT_SMOKE_QUESTION_IDS),
        "modes": ["agentic", *( ["naive"] if include_naive else [])],
        "agentic_execution_version": "v9",
        "repeat_count": 1,
        "paired_naive": include_naive,
        "live_provider_calls_authorized": False,
    }


def execute_smoke_plan(
    plan: Mapping[str, Any],
    *,
    base_url: str,
    preset_name: str,
    auth_header: str,
    confirmation: str,
    transport: Transport,
) -> dict[str, Any]:
    """Submit a campaign only after all explicit execution guardrails pass.

    ``transport`` is injected to keep this function testable and to ensure that
    the default CLI path never imports or instantiates a network client.
    """
    _validate_execute_inputs(
        base_url=base_url,
        preset_name=preset_name,
        auth_header=auth_header,
        confirmation=confirmation,
    )
    normalized_base_url = base_url.rstrip("/")
    headers = _auth_headers(auth_header)
    test_cases = _as_list(
        transport("GET", f"{normalized_base_url}/api/evaluation/test-cases", headers, None)
    )
    model_configs = _as_list(
        transport("GET", f"{normalized_base_url}/api/evaluation/model-configs", headers, None)
    )
    test_case_ids = _resolve_test_case_ids(plan, test_cases)
    preset = _find_named_preset(preset_name, model_configs)
    payload = {
        "name": "Agentic v9 guarded smoke",
        "test_case_ids": test_case_ids,
        "modes": list(plan.get("modes") or ["agentic"]),
        "model_config": preset,
        "model_config_id": preset.get("id"),
        "repeat_count": int(plan.get("repeat_count") or 1),
        "agentic_execution_version": "v9",
        "prompt_capture_policy": {
            "hash": True,
            "preview": True,
            "full_prompt": False,
        },
    }
    response = transport(
        "POST", f"{normalized_base_url}/api/evaluation/campaigns", headers, payload
    )
    if not isinstance(response, dict):
        raise ValueError("campaign creation response must be a JSON object")
    return _as_mapping(_sanitize_public_value(response))


def verify_campaign_export(artifact: Mapping[str, Any] | None) -> VerificationReport:
    """Verify a redacted campaign export without assuming a database model.

    A release result only passes when all v9 contract and observability evidence
    is present.  This is intentionally conservative: unknown fields are never
    treated as zero, successful, or complete.
    """
    if not isinstance(artifact, Mapping):
        return _not_executed_report("campaign export artifact was not supplied")
    runs = _as_list(artifact.get("runs"))
    if not runs:
        return _not_executed_report("campaign export contains no runs")

    v9_runs = [run for run in runs if _is_v9_candidate(run)]
    if not v9_runs:
        return _report_from_requirements(
            {
                "campaign_export": RequirementResult("pass"),
                "plan_coverage": RequirementResult(
                    "partial", "no explicit Agentic v9 runs found in campaign export"
                ),
            }
        )

    llm_calls = _as_list(artifact.get("llm_calls"))
    campaign = _as_mapping(artifact.get("campaign"))
    setup = _as_mapping(campaign.get("config")) or campaign
    requirements: dict[str, RequirementResult] = {
        "campaign_export": RequirementResult("pass"),
        "plan_coverage": _verify_plan_coverage(v9_runs),
        "contract_and_route": _verify_contracts(v9_runs),
        "slots_and_resolutions": _verify_slots(v9_runs),
        "repair_for_missing_slots": _verify_repairs(v9_runs),
        "phase_linked_provider_attempts": _verify_provider_attempts(v9_runs, llm_calls),
        "token_reconciliation": _verify_tokens(v9_runs, llm_calls),
        "capture_availability": _verify_capture(v9_runs, llm_calls, setup),
        "supported_claims": _verify_supported_claims(v9_runs),
        "retrieval_evidence_recovery": _verify_retrieval_evidence_recovery(
            v9_runs, llm_calls
        ),
    }
    return _report_from_requirements(requirements)


def build_release_manifest(
    report: VerificationReport,
    *,
    backend_commit: str | None,
    frontend_commit: str | None,
    setup_snapshot: Mapping[str, Any] | None,
    dataset_identity: str | None,
    input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a portable, JSON-safe release-verification manifest."""
    snapshot = _as_mapping(_sanitize_public_value(dict(setup_snapshot or {})))
    input_hashes = {
        str(name): _hash_path(Path(path)) for name, path in (input_paths or {}).items()
    }
    return {
        "manifest_version": "1",
        "repository_commits": {
            "backend": backend_commit or "not_supplied",
            "frontend": frontend_commit or "not_supplied",
        },
        "evaluation_setup": snapshot or None,
        "evaluation_setup_hash": _stable_hash(snapshot) if snapshot else None,
        "dataset_identity": dataset_identity or "not_supplied",
        "input_hashes": input_hashes,
        "release_gate_results": {
            "overall_status": report.status,
            "requirements": {
                name: asdict(result) for name, result in report.requirements.items()
            },
        },
        "residual_failures": list(report.residual_failures),
        "runtime_smoke": (
            "not_executed"
            if report.requirements.get("campaign_export", RequirementResult("partial")).status
            == "not_executed"
            else "externally_supplied_artifact_verified"
        ),
    }


def load_campaign_export(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON export, returning ``None`` only when no artifact exists."""
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid campaign export: {candidate}") from exc
    if not isinstance(payload, dict):
        raise ValueError("campaign export root must be a JSON object")
    return payload


def _validate_execute_inputs(
    *, base_url: str, preset_name: str, auth_header: str, confirmation: str
) -> None:
    if not base_url.strip():
        raise ValueError("--execute requires a base URL")
    if not preset_name.strip():
        raise ValueError("--execute requires a named Evaluation Setup preset")
    if not auth_header.strip():
        raise ValueError("--execute requires authentication input")
    if confirmation != EXECUTION_CONFIRMATION:
        raise ValueError(
            "--execute requires --confirm-execute I_UNDERSTAND_EXECUTE"
        )


def _auth_headers(auth_header: str) -> dict[str, str]:
    name, separator, value = auth_header.partition(":")
    if not separator or not name.strip() or not value.strip():
        raise ValueError("authentication input must use 'Header-Name: value'")
    return {name.strip(): value.strip(), "Content-Type": "application/json"}


def _resolve_test_case_ids(plan: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    identifiers: dict[str, str] = {}
    for row in rows:
        question_id = str(row.get("question_id") or row.get("id") or "")
        row_id = str(row.get("id") or "")
        if question_id and row_id:
            identifiers[question_id] = row_id
    missing = [
        question_id
        for question_id in _as_strings(plan.get("question_ids"))
        if question_id not in identifiers
    ]
    if missing:
        raise ValueError(f"required smoke test cases unavailable: {', '.join(missing)}")
    return [identifiers[question_id] for question_id in _as_strings(plan.get("question_ids"))]


def _find_named_preset(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    preset = next((row for row in rows if row.get("name") == name), None)
    if not isinstance(preset, dict):
        raise ValueError(f"named Evaluation Setup preset not found: {name}")
    return preset


def _is_v9_candidate(run: dict[str, Any]) -> bool:
    trace = _as_mapping(run.get("agent_trace"))
    v9 = _as_mapping(trace.get("agentic_v9")) or _as_mapping(run.get("agentic_v9"))
    version = str(
        trace.get("agentic_execution_version")
        or run.get("agentic_execution_version")
        or ""
    )
    return bool(v9) and version == "v9"


def _v9_payload(run: dict[str, Any]) -> dict[str, Any]:
    trace = _as_mapping(run.get("agent_trace"))
    return _as_mapping(trace.get("agentic_v9")) or _as_mapping(run.get("agentic_v9"))


def _verify_plan_coverage(runs: list[dict[str, Any]]) -> RequirementResult:
    question_ids: list[str] = []
    for run in runs:
        run_id = str(run.get("id") or run.get("run_id") or "").strip()
        question_id = str(run.get("question_id") or "").strip()
        status = run.get("status")
        repeat = run.get("repeat_number", run.get("repeat"))
        if not run_id or not question_id or status is None or repeat is None:
            return RequirementResult("partial", "v9 run identity, completion status, or repeat is missing")
        if str(status) != "completed" or repeat != 1:
            return RequirementResult("fail", "v9 smoke runs must be completed with repeat 1")
        question_ids.append(question_id)
    expected = list(DEFAULT_SMOKE_QUESTION_IDS)
    if sorted(question_ids) != sorted(expected):
        return RequirementResult("fail", "v9 smoke question coverage is not exactly Q5/Q7/Q11/Q14/Q16")
    return RequirementResult("pass")


def _verify_contracts(runs: list[dict[str, Any]]) -> RequirementResult:
    for run in runs:
        v9 = _v9_payload(run)
        contract = _as_mapping(v9.get("query_contract")) or _as_mapping(v9.get("contract"))
        decision = _as_mapping(contract.get("route_decision"))
        if not v9 or not contract:
            return RequirementResult("partial", "v9 contract observability missing")
        if str(contract.get("contract_version") or "") != "2":
            return RequirementResult("fail", "v9 contract version is not 2")
        route_reason = str(decision.get("route_reason") or "").strip()
        route = str(decision.get("selected_route") or "").strip()
        if not route or not route_reason:
            return RequirementResult("fail", "actual route rationale missing")
    return RequirementResult("pass")


def _resolution_rows(v9: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _as_list(v9.get("slot_resolutions")):
        resolution = _as_mapping(raw.get("resolution")) if isinstance(raw, dict) else {}
        row = {**(_as_mapping(raw)), **resolution}
        if row:
            rows.append(row)
    return rows


def _verify_slots(runs: list[dict[str, Any]]) -> RequirementResult:
    for run in runs:
        v9 = _v9_payload(run)
        contract = _as_mapping(v9.get("query_contract")) or _as_mapping(v9.get("contract"))
        slots = _as_list(contract.get("required_slots"))
        resolutions = _resolution_rows(v9)
        if not slots or not resolutions or "final_claims" not in v9:
            return RequirementResult("fail", "atomic slots or final slot resolutions missing")
        expected_ids = {str(slot.get("slot_id") or "").strip() for slot in slots}
        valid_statuses = {"supported", "conflicted", "explicitly_unavailable", "not_found"}
        if not all(expected_ids) or len(expected_ids) != len(slots):
            return RequirementResult("fail", "required atomic slots must have unique non-empty IDs")
        if any(str(row.get("status") or "") not in valid_statuses for row in resolutions):
            return RequirementResult("fail", "final slot resolution status is missing or invalid")
        resolved_ids = {str(row.get("slot_id") or "") for row in resolutions if row.get("slot_id")}
        if not expected_ids.issubset(resolved_ids):
            return RequirementResult("fail", "not every required atomic slot has a final resolution")
    return RequirementResult("pass")


def _verify_repairs(runs: list[dict[str, Any]]) -> RequirementResult:
    for run in runs:
        v9 = _v9_payload(run)
        not_found = {
            str(row.get("slot_id"))
            for row in _resolution_rows(v9)
            if str(row.get("status") or "") == "not_found" and row.get("slot_id")
        }
        repair_slots = {
            str(slot_id)
            for repair in _as_list(v9.get("repairs"))
            for task in _as_list(_as_mapping(repair).get("tasks"))
            for slot_id in _as_strings(task.get("target_slot_ids"))
        }
        if not_found and not_found.difference(repair_slots):
            return RequirementResult("fail", "missing slots lack slot-targeted repair traces")
        for row in _resolution_rows(v9):
            status = str(row.get("status") or "")
            if status in {"conflicted", "explicitly_unavailable"} and not _has_terminal_resolution_state(v9, row):
                return RequirementResult("partial", "terminal slot state lacks stop or arbitration evidence")
    return RequirementResult("pass")


def _has_terminal_resolution_state(v9: Mapping[str, Any], resolution: Mapping[str, Any]) -> bool:
    if str(resolution.get("reason") or "").strip() or str(resolution.get("resolution_stage") or "").strip():
        return True
    sufficiency = _as_mapping(v9.get("sufficiency"))
    if str(sufficiency.get("stop_reason") or "").strip():
        return True
    slot_id = str(resolution.get("slot_id") or "")
    return any(
        str(item.get("slot_id") or "") == slot_id and str(item.get("reason") or "").strip()
        for item in _as_list(v9.get("conflicts"))
    )


def _calls_for_run(llm_calls: list[dict[str, Any]], run: Mapping[str, Any]) -> list[dict[str, Any]]:
    run_id = str(run.get("id") or run.get("run_id") or "")
    return [call for call in llm_calls if str(call.get("run_id") or "") == run_id]


def _verify_provider_attempts(runs: list[dict[str, Any]], llm_calls: list[dict[str, Any]]) -> RequirementResult:
    for run in runs:
        v9 = _v9_payload(run)
        reservations = _as_list(v9.get("budget_reservations"))
        if not reservations:
            return RequirementResult("partial", "budget reservation observability missing")
        reservation_by_id: dict[str, dict[str, Any]] = {}
        for reservation in reservations:
            reservation_id = str(reservation.get("reservation_id") or "").strip()
            if not reservation_id or not str(reservation.get("phase") or "").strip() or not isinstance(reservation.get("provider_attempt"), int):
                return RequirementResult("partial", "budget reservation identity is incomplete")
            if reservation_id in reservation_by_id:
                return RequirementResult("fail", "duplicate budget reservation ID")
            reservation_by_id[reservation_id] = reservation
        calls = _calls_for_run(llm_calls, run)
        if not calls:
            return RequirementResult("partial", "phase-linked provider attempts missing")
        seen_identities: set[tuple[str, int]] = set()
        terminal_by_reservation: set[str] = set()
        for call in calls:
            if not str(call.get("phase") or "") or not str(call.get("reservation_id") or "") or not isinstance(call.get("provider_attempt"), int):
                return RequirementResult("partial", "provider attempt identity is incomplete")
            reservation_id = str(call["reservation_id"])
            attempt = int(call["provider_attempt"])
            identity = (reservation_id, attempt)
            if identity in seen_identities:
                return RequirementResult("fail", "duplicate terminal provider attempt identity")
            seen_identities.add(identity)
            reservation = reservation_by_id.get(reservation_id)
            if reservation is None:
                return RequirementResult("fail", "provider attempt references an unknown reservation")
            if str(reservation.get("phase")) != str(call.get("phase")) or reservation.get("provider_attempt") != attempt:
                return RequirementResult("fail", "provider attempt does not match its reservation identity")
            if str(call.get("status") or "") not in {"success", "error", "cancelled"}:
                return RequirementResult("partial", "provider attempt lacks a terminal status")
            terminal_by_reservation.add(reservation_id)
        if set(reservation_by_id) != terminal_by_reservation:
            return RequirementResult("fail", "a budget reservation has no terminal provider attempt")
    return RequirementResult("pass")


def _run_total_tokens(run: Mapping[str, Any]) -> int | None:
    for candidate in (run.get("total_tokens"), _as_mapping(run.get("usage")).get("total_tokens")):
        if isinstance(candidate, int):
            return candidate
    return None


def _verify_tokens(runs: list[dict[str, Any]], llm_calls: list[dict[str, Any]]) -> RequirementResult:
    for run in runs:
        v9 = _v9_payload(run)
        metrics = _as_mapping(v9.get("metrics"))
        explicit_status = str(
            _as_mapping(v9.get("token_reconciliation")).get("status")
            or metrics.get("token_reconciliation_status")
            or run.get("token_reconciliation_status")
            or ""
        )
        if explicit_status in {"partial", "not_available"}:
            return RequirementResult("partial", "token reconciliation explicitly partial")
        runtime_total = _run_total_tokens(run)
        reconciled = metrics.get("reconciled_tokens")
        calls = _calls_for_run(llm_calls, run)
        totals = [
            _as_mapping(call.get("payload")).get("official_total_tokens")
            for call in calls
        ]
        if not isinstance(runtime_total, int) or not isinstance(reconciled, int) or not totals or not all(isinstance(value, int) for value in totals):
            return RequirementResult("partial", "exact token reconciliation evidence missing")
        if runtime_total != reconciled or runtime_total != sum(int(value) for value in totals):
            return RequirementResult("fail", "runtime, reconciled, and provider token totals disagree")
    return RequirementResult("pass")


def _verify_capture(
    runs: list[dict[str, Any]], llm_calls: list[dict[str, Any]], setup: Mapping[str, Any]
) -> RequirementResult:
    policy = _as_mapping(setup.get("prompt_capture_policy"))
    if not policy:
        return RequirementResult("partial", "Evaluation Setup prompt capture policy missing")
    for run in runs:
        calls = _calls_for_run(llm_calls, run)
        if not calls:
            return RequirementResult("partial", "prompt capture availability rows missing")
        for call in calls:
            if policy.get("hash") is True and not call.get("prompt_hash"):
                return RequirementResult("fail", "setup requires prompt hashes but one is absent")
            if policy.get("preview") is True:
                preview_status = str(call.get("prompt_capture_status") or "")
                if not preview_status:
                    return RequirementResult("partial", "prompt preview availability missing")
                if preview_status in {"not_captured_at_execution", "capture_failed"}:
                    return RequirementResult("fail", "setup requires previews but one was unavailable")
            if policy.get("full_prompt") is True:
                full_status = str(call.get("full_prompt_capture_status") or "")
                if not full_status:
                    return RequirementResult("partial", "full prompt availability missing")
                if full_status not in {"captured", "redacted"}:
                    return RequirementResult("fail", "setup requires full prompts but one was unavailable")
    return RequirementResult("pass")


def _verify_supported_claims(runs: list[dict[str, Any]]) -> RequirementResult:
    positive_support = {"direct", "calculated", "comparative_inference", "supported"}
    for run in runs:
        v9 = _v9_payload(run)
        resolutions = {
            str(row.get("slot_id")): str(row.get("status") or "")
            for row in _resolution_rows(v9)
            if row.get("slot_id")
        }
        for claim in _as_list(v9.get("final_claims")):
            item = _as_mapping(claim)
            slot_id = str(item.get("slot_id") or "")
            claim_status = str(item.get("status") or item.get("support_type") or "")
            if claim_status not in positive_support:
                continue
            if not slot_id:
                return RequirementResult("fail", "supported final claim is missing a slot binding")
            if resolutions.get(slot_id) != "supported":
                return RequirementResult("fail", "unsupported slot emitted as a supported final claim")
            if not _as_strings(item.get("evidence_ids")):
                return RequirementResult("fail", "supported final claim is missing evidence provenance")
    return RequirementResult("pass")


def _verify_retrieval_evidence_recovery(
    runs: list[dict[str, Any]], llm_calls: list[dict[str, Any]]
) -> RequirementResult:
    """Fail closed on the bounded retrieval diagnostics projected by v9."""
    empty_context_runs = 0
    for run in runs:
        v9 = _v9_payload(run)
        trace = _as_mapping(run.get("agent_trace"))
        completion = _as_mapping(v9.get("completion"))
        if (
            trace.get("response_status") == "configuration_incompatible"
            or completion.get("status") == "configuration_incompatible"
        ):
            return RequirementResult("fail", "run reports configuration incompatibility")

        packets = _as_list(v9.get("evidence_packets"))
        if "evidence_packets" not in v9:
            return RequirementResult("partial", "retrieval evidence packets are missing")
        question_id = str(run.get("question_id") or "")
        if question_id in {"Q5", "Q7", "Q11"} and not packets:
            return RequirementResult("fail", "required recovery question has no evidence packets")
        source_error = _verify_packet_source_scope(v9, packets)
        if source_error is not None:
            return source_error

        context_pack = v9.get("context_pack")
        if not isinstance(context_pack, Mapping):
            return RequirementResult("partial", "context-pack diagnostics are missing")
        packed_ids = context_pack.get("packed_evidence_ids")
        if not isinstance(packed_ids, list):
            return RequirementResult("partial", "packed evidence IDs are missing")
        empty_context_runs += int(not packed_ids)

        diagnostics = _as_list(v9.get("retrieval_diagnostics"))
        if not diagnostics:
            return RequirementResult("partial", "retrieval diagnostics are missing")
        for diagnostic in diagnostics:
            source_filter = _as_mapping(diagnostic.get("source_filter"))
            reranking = _as_mapping(diagnostic.get("reranking"))
            post_filter_count = source_filter.get("post_filter_count")
            selected_count = reranking.get("selected_count")
            if not isinstance(post_filter_count, int) or not isinstance(selected_count, int):
                return RequirementResult("partial", "reranking recovery counts are missing")
            if reranking.get("fallback_reason") and post_filter_count > 0 and selected_count == 0:
                return RequirementResult(
                    "fail", "reranker fallback removed authorized retrieval candidates"
                )

        if question_id in {"Q14", "Q16"}:
            locator_diagnostics = _as_list(v9.get("locator_diagnostics"))
            if not locator_diagnostics:
                return RequirementResult("partial", "locator state diagnostics are missing")
            states = {str(item.get("state") or "") for item in locator_diagnostics}
            if not states.issubset({"matched", "mismatched", "unavailable"}):
                return RequirementResult("fail", "locator diagnostics contain an invalid state")

        evidence_calls = [
            call for call in _calls_for_run(llm_calls, run) if call.get("phase") == "evidence_extract"
        ]
        if len(evidence_calls) > 1:
            return RequirementResult("fail", "more than one evidence_extract call was recorded")

    if empty_context_runs == len(runs):
        return RequirementResult("fail", "all smoke runs have zero packed evidence contexts")
    return RequirementResult("pass")


def _verify_packet_source_scope(
    v9: Mapping[str, Any], packets: list[dict[str, Any]]
) -> RequirementResult | None:
    contract = _as_mapping(v9.get("query_contract")) or _as_mapping(v9.get("contract"))
    scope = _as_mapping(contract.get("resolved_source_scope"))
    default_ids = set(_as_strings(scope.get("authorized_doc_ids")))
    slots = {
        str(slot.get("slot_id") or ""): set(
            _as_strings(slot.get("authorized_source_doc_ids"))
        )
        for slot in _as_list(contract.get("required_slots"))
        if str(slot.get("slot_id") or "")
    }
    for packet in packets:
        doc_id = str(_as_mapping(packet.get("source")).get("doc_id") or "")
        slot_ids = _as_strings(packet.get("slot_ids"))
        if not doc_id or not slot_ids:
            return RequirementResult("partial", "evidence packet source scope is incomplete")
        for slot_id in slot_ids:
            authorized = slots.get(slot_id, default_ids)
            if not authorized:
                return RequirementResult("partial", "slot authorization diagnostics are missing")
            if doc_id not in authorized:
                return RequirementResult("fail", "out-of-scope document entered evidence")
    return None


def _not_executed_report(reason: str) -> VerificationReport:
    return VerificationReport(
        status="not_executed",
        requirements={"campaign_export": RequirementResult("not_executed", reason)},
        residual_failures=(reason,),
    )


def _partial_report(reason: str) -> VerificationReport:
    return VerificationReport(
        status="partial",
        requirements={"campaign_export": RequirementResult("partial", reason)},
        residual_failures=(reason,),
    )


def _report_from_requirements(requirements: dict[str, RequirementResult]) -> VerificationReport:
    statuses = [result.status for result in requirements.values()]
    overall: VerificationStatus = (
        "fail" if "fail" in statuses else "partial" if "partial" in statuses else "pass"
    )
    residual = tuple(
        f"{name}:{result.reason or result.status}"
        for name, result in requirements.items()
        if result.status != "pass"
    )
    return VerificationReport(overall, requirements, residual)


def _as_mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _as_strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list | tuple) else []


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def _hash_path(path: Path) -> str | None:
    if not path.is_file():
        return None
    return f"sha256:{sha256(path.read_bytes()).hexdigest()}"


def _sanitize_public_value(value: Any) -> Any:
    sanitized, _ = sanitize_credential_value(value)
    return _redact_cookie_values(sanitized)


def _redact_cookie_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if str(key).replace("-", "").replace("_", "").lower() in {"cookie", "setcookie"}
            else _redact_cookie_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_cookie_values(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_cookie_values(item) for item in value]
    return value
