"""Tests for the production-equivalent Agentic v9 planner canary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from core.llm_factory import current_llm_runtime_overrides
from data_base.agentic_v9.contract_planner import (
    atomic_contract_planner_response_schema,
)


_CURRENT_RESPONSE = {
    "evidence_requirements": [
        {
            "description": "Identify the reported score.",
            "source_name_hints": [],
            "locator_hints": ["Table 1"],
            "expected_answer_type": "text",
            "depends_on_requirement_indexes": [],
            "visual_policy": "never",
        }
    ],
    "synthesis_obligations": [],
    "response_constraints": [],
    "comparison": None,
    "confidence": 1.0,
}
_MINIMAL_SCHEMA = {
    "type": "object",
    "properties": {"canary": {"type": "string", "enum": ["ok"]}},
    "required": ["canary"],
    "additionalProperties": False,
}


@pytest.fixture
def canary() -> ModuleType:
    return importlib.import_module("scripts.agentic_v9_contract_planner_canary")


@pytest.fixture
def model_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "model-config.json"
    path.write_text(
        json.dumps(
            {
                "id": "server-config",
                "name": "Real server config",
                "model_name": "gemini-2.5-flash-lite",
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 40,
                "max_input_tokens": 777,
                "max_output_tokens": 1000,
                "thinking_mode": False,
            }
        ),
        encoding="utf-8",
    )
    return path


class _RecordingProvider:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[object] = []
        self.runtime_overrides: list[dict[str, Any]] = []

    async def ainvoke(self, messages: object) -> object:
        self.calls.append(messages)
        self.runtime_overrides.append(current_llm_runtime_overrides())
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _version(_package_name: str) -> str:
    return "test-version"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema_name", "content", "expected_schema"),
    [
        ("current", json.dumps(_CURRENT_RESPONSE), atomic_contract_planner_response_schema()),
        ("minimal", '{"canary":"ok"}', _MINIMAL_SCHEMA),
    ],
)
async def test_schema_modes_use_shared_boundary_and_one_production_configured_attempt(
    canary: ModuleType,
    model_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_name: str,
    content: str,
    expected_schema: dict[str, Any],
) -> None:
    provider = _RecordingProvider(SimpleNamespace(content=content))
    observed_schemas: list[object] = []
    builder_overrides: list[dict[str, Any]] = []

    def fake_shared_builder(*, response_schema: object) -> object:
        observed_schemas.append(response_schema)
        builder_overrides.append(current_llm_runtime_overrides())
        return provider

    monkeypatch.setattr(
        canary, "build_contract_planning_provider", fake_shared_builder
    )

    exit_code, payload = await canary.run_canary(
        schema_name,
        model_config_path=model_config_path,
        version_reader=_version,
    )

    assert exit_code == 0
    assert observed_schemas == [expected_schema]
    assert len(provider.calls) == 1
    assert builder_overrides == provider.runtime_overrides
    assert builder_overrides == [
        {
            "model_name": "gemini-2.5-flash-lite",
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
            "max_input_tokens": 777,
            "max_output_tokens": 768,
            "setup_max_input_tokens": 777,
            "setup_max_output_tokens": 1000,
            "thinking_enabled": False,
        }
    ]
    assert payload == {
        "success": True,
        "schema": schema_name,
        "failure_stage": None,
        "failure_code": None,
        "package_versions": {
            "google-genai": "test-version",
            "langchain-google-genai": "test-version",
            "pydantic": "test-version",
        },
        "model_identifier": "gemini-2.5-flash-lite",
        "response_received": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "body", "exit_code", "failure_stage", "failure_code"),
    [
        (
            "missing.json",
            None,
            10,
            "model_config_load",
            "file_unavailable",
        ),
        (
            "invalid-json.json",
            '{"api_key":"config-secret"',
            11,
            "model_config_decode",
            "invalid_json",
        ),
        (
            "invalid-model.json",
            '{"id":"x","name":"x","model_name":"","api_key":"config-secret"}',
            12,
            "model_config_validation",
            "pydantic_validation_failed",
        ),
    ],
)
async def test_invalid_model_config_fails_before_provider_and_is_sanitized(
    canary: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    body: str | None,
    exit_code: int,
    failure_stage: str,
    failure_code: str,
) -> None:
    config_path = tmp_path / filename
    if body is not None:
        config_path.write_text(body, encoding="utf-8")
    builder_calls: list[object] = []
    monkeypatch.setattr(
        canary,
        "build_contract_planning_provider",
        lambda **kwargs: builder_calls.append(kwargs),
    )

    actual_exit, payload = await canary.run_canary(
        "current",
        model_config_path=config_path,
        version_reader=_version,
    )

    assert actual_exit == exit_code
    assert builder_calls == []
    assert payload["failure_stage"] == failure_stage
    assert payload["failure_code"] == failure_code
    assert payload["response_received"] is False
    assert "config-secret" not in json.dumps(payload)
    assert "api_key" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_non_utf8_model_config_is_sanitized_before_provider_setup(
    canary: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "invalid-encoding.json"
    config_path.write_bytes(b"\xffapi_key=config-secret")
    builder_calls: list[object] = []
    monkeypatch.setattr(
        canary,
        "build_contract_planning_provider",
        lambda **kwargs: builder_calls.append(kwargs),
    )

    exit_code, payload = await canary.run_canary(
        "current",
        model_config_path=config_path,
        version_reader=_version,
    )

    assert exit_code == 11
    assert builder_calls == []
    assert payload["failure_stage"] == "model_config_decode"
    assert payload["failure_code"] == "invalid_json"
    assert "config-secret" not in json.dumps(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "exit_code", "failure_stage", "failure_code", "received"),
    [
        (
            RuntimeError("api_key=provider-secret prompt=private-prompt"),
            30,
            "provider_invocation",
            "provider_attempt_failed",
            False,
        ),
        (
            SimpleNamespace(content=""),
            31,
            "provider_empty_response",
            "empty_response",
            True,
        ),
        (
            SimpleNamespace(content="private-response-body"),
            32,
            "response_decode",
            "invalid_json",
            True,
        ),
        (
            SimpleNamespace(content='{"confidence":1.0,"secret":"response-secret"}'),
            33,
            "schema_validation",
            "pydantic_validation_failed",
            True,
        ),
    ],
)
async def test_provider_failures_are_single_attempt_nonzero_and_sanitized(
    canary: ModuleType,
    model_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: object,
    exit_code: int,
    failure_stage: str,
    failure_code: str,
    received: bool,
) -> None:
    provider = _RecordingProvider(outcome)
    monkeypatch.setattr(
        canary,
        "build_contract_planning_provider",
        lambda *, response_schema: provider,
    )

    actual_exit, payload = await canary.run_canary(
        "current",
        model_config_path=model_config_path,
        version_reader=_version,
    )
    rendered = json.dumps(payload)

    assert actual_exit == exit_code
    assert len(provider.calls) == 1
    assert payload["success"] is False
    assert payload["failure_stage"] == failure_stage
    assert payload["failure_code"] == failure_code
    assert payload["response_received"] is received
    assert set(payload) == {
        "success",
        "schema",
        "failure_stage",
        "failure_code",
        "package_versions",
        "model_identifier",
        "response_received",
    }
    for forbidden in (
        "provider-secret",
        "private-prompt",
        "private-response-body",
        "response-secret",
        "api_key",
    ):
        assert forbidden not in rendered


@pytest.mark.asyncio
async def test_provider_setup_failure_is_sanitized_and_nonzero(
    canary: ModuleType,
    model_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_build(*, response_schema: object) -> object:
        del response_schema
        raise RuntimeError("api_key=setup-secret")

    monkeypatch.setattr(canary, "build_contract_planning_provider", fail_to_build)

    exit_code, payload = await canary.run_canary(
        "minimal",
        model_config_path=model_config_path,
        version_reader=_version,
    )

    assert exit_code == 20
    assert payload["failure_stage"] == "provider_setup"
    assert payload["failure_code"] == "provider_binding_failed"
    assert payload["response_received"] is False
    assert "setup-secret" not in json.dumps(payload)


def test_missing_model_config_argument_prints_one_sanitized_json_document(
    canary: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = canary.main(
        ["--schema", "minimal"],
        version_reader=_version,
    )

    captured = capsys.readouterr()
    assert exit_code == 10
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    payload = json.loads(captured.out)
    assert payload["failure_stage"] == "model_config_load"
    assert payload["failure_code"] == "file_unavailable"
    assert payload["response_received"] is False
