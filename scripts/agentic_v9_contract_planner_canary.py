#!/usr/bin/env python3
"""Probe the production Agentic v9 contract-planner boundary exactly once."""

from __future__ import annotations

import argparse
import asyncio
from importlib import metadata
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Literal, Protocol

from pydantic import ValidationError

# Direct ``python scripts/...`` execution sets ``sys.path[0]`` to ``scripts``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_factory import llm_runtime_override  # noqa: E402
from data_base.agentic_v9 import contract_planner as planner_module  # noqa: E402
from data_base.agentic_v9.phase_policy import (  # noqa: E402
    agentic_phase_policy_scope,
    resolve_phase_policy,
)
from data_base.agentic_v9.provider_boundary import (  # noqa: E402
    build_contract_planning_provider,
)
from evaluation.model_capabilities import (  # noqa: E402
    normalize_model_config_for_runtime,
)
from evaluation.schemas import ModelConfig  # noqa: E402


SchemaName = Literal["current", "minimal"]
VersionReader = Callable[[str], str]

_PACKAGE_NAMES = ("google-genai", "langchain-google-genai", "pydantic")
_MINIMAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"canary": {"type": "string", "enum": ["ok"]}},
    "required": ["canary"],
    "additionalProperties": False,
}
_PROMPT_BY_SCHEMA: dict[SchemaName, str] = {
    "current": (
        "Return one answer-free planning decision with one evidence requirement, "
        "no comparison, and confidence 1.0."
    ),
    "minimal": 'Return the canary object with canary set to "ok".',
}
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}\Z")


class AsyncProvider(Protocol):
    """Minimal asynchronous provider surface used by the canary."""

    async def ainvoke(self, messages: object) -> object:
        """Make one provider attempt."""


def _safe_identifier(value: object) -> str:
    candidate = value if isinstance(value, str) else ""
    return candidate if _SAFE_IDENTIFIER.fullmatch(candidate) else "unknown"


def _package_versions(version_reader: VersionReader) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in _PACKAGE_NAMES:
        try:
            versions[package_name] = _safe_identifier(version_reader(package_name))
        except metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def _payload(
    *,
    success: bool,
    schema_name: SchemaName,
    package_versions: dict[str, str],
    model_identifier: str,
    response_received: bool,
    failure_stage: str | None = None,
    failure_code: str | None = None,
) -> dict[str, Any]:
    return {
        "success": success,
        "schema": schema_name,
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        "package_versions": package_versions,
        "model_identifier": model_identifier,
        "response_received": response_received,
    }


def _failure(
    exit_code: int,
    *,
    schema_name: SchemaName,
    package_versions: dict[str, str],
    model_identifier: str,
    response_received: bool,
    failure_stage: str,
    failure_code: str,
) -> tuple[int, dict[str, Any]]:
    return exit_code, _payload(
        success=False,
        schema_name=schema_name,
        package_versions=package_versions,
        model_identifier=model_identifier,
        response_received=response_received,
        failure_stage=failure_stage,
        failure_code=failure_code,
    )


def _load_model_config(
    model_config_path: Path | None,
) -> tuple[ModelConfig | None, tuple[int, str, str] | None]:
    if model_config_path is None:
        return None, (10, "model_config_load", "file_unavailable")
    try:
        raw_text = model_config_path.read_text(encoding="utf-8")
    except OSError:
        return None, (10, "model_config_load", "file_unavailable")
    except UnicodeError:
        return None, (11, "model_config_decode", "invalid_json")
    try:
        decoded = json.loads(raw_text)
    except json.JSONDecodeError:
        return None, (11, "model_config_decode", "invalid_json")
    try:
        return ModelConfig.model_validate(decoded), None
    except ValidationError:
        return None, (
            12,
            "model_config_validation",
            "pydantic_validation_failed",
        )


def _schema_for(schema_name: SchemaName) -> dict[str, Any]:
    if schema_name == "current":
        return planner_module.atomic_contract_planner_response_schema()
    return _MINIMAL_SCHEMA


def _response_content(response: object) -> str:
    content: object = None
    if isinstance(response, dict):
        content = response.get("content")
    else:
        content = getattr(response, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise planner_module.PlannerProviderEmptyResponseError
    return content


def _validate_response(schema_name: SchemaName, response: object) -> None:
    content = _response_content(response)
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as error:
        raise planner_module.PlannerResponseDecodeError from error

    if schema_name == "current":
        planner_module._parse_decision(response)
        return
    if decoded != {"canary": "ok"}:
        raise planner_module.PlannerSchemaValidationError


async def run_canary(
    schema_name: SchemaName,
    *,
    model_config_path: Path | None,
    version_reader: VersionReader = metadata.version,
) -> tuple[int, dict[str, Any]]:
    """Run one configured provider attempt and return only sanitized metadata."""
    package_versions = _package_versions(version_reader)
    model_config, config_failure = _load_model_config(model_config_path)
    if config_failure is not None:
        exit_code, failure_stage, failure_code = config_failure
        return _failure(
            exit_code,
            schema_name=schema_name,
            package_versions=package_versions,
            model_identifier="unknown",
            response_received=False,
            failure_stage=failure_stage,
            failure_code=failure_code,
        )
    assert model_config is not None

    model_identifier = _safe_identifier(model_config.model_name)
    normalized_config = normalize_model_config_for_runtime(
        model_config.model_dump(mode="json")
    )
    phase_policy = resolve_phase_policy(
        "contract_planning",
        setup_output_ceiling=model_config.max_output_tokens,
        setup_input_ceiling=model_config.max_input_tokens,
        remaining_input_budget=model_config.max_input_tokens,
    )

    with (
        llm_runtime_override(**normalized_config),
        agentic_phase_policy_scope(phase_policy),
    ):
        try:
            provider: AsyncProvider = build_contract_planning_provider(
                response_schema=_schema_for(schema_name)
            )
        except Exception:
            return _failure(
                20,
                schema_name=schema_name,
                package_versions=package_versions,
                model_identifier=model_identifier,
                response_received=False,
                failure_stage="provider_setup",
                failure_code="provider_binding_failed",
            )
        try:
            response = await provider.ainvoke(
                [{"role": "user", "content": _PROMPT_BY_SCHEMA[schema_name]}]
            )
        except Exception:
            return _failure(
                30,
                schema_name=schema_name,
                package_versions=package_versions,
                model_identifier=model_identifier,
                response_received=False,
                failure_stage="provider_invocation",
                failure_code="provider_attempt_failed",
            )

    try:
        _validate_response(schema_name, response)
    except planner_module.PlannerProviderEmptyResponseError:
        return _failure(
            31,
            schema_name=schema_name,
            package_versions=package_versions,
            model_identifier=model_identifier,
            response_received=True,
            failure_stage="provider_empty_response",
            failure_code="empty_response",
        )
    except planner_module.PlannerResponseDecodeError:
        return _failure(
            32,
            schema_name=schema_name,
            package_versions=package_versions,
            model_identifier=model_identifier,
            response_received=True,
            failure_stage="response_decode",
            failure_code="invalid_json",
        )
    except planner_module.PlannerSchemaValidationError:
        return _failure(
            33,
            schema_name=schema_name,
            package_versions=package_versions,
            model_identifier=model_identifier,
            response_received=True,
            failure_stage="schema_validation",
            failure_code="pydantic_validation_failed",
        )

    return 0, _payload(
        success=True,
        schema_name=schema_name,
        package_versions=package_versions,
        model_identifier=model_identifier,
        response_received=True,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", choices=("current", "minimal"), required=True)
    parser.add_argument("--model-config-json", type=Path)
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    version_reader: VersionReader = metadata.version,
) -> int:
    """Run the CLI and write exactly one JSON document to stdout."""
    args = _parse_args(argv)
    exit_code, payload = asyncio.run(
        run_canary(
            args.schema,
            model_config_path=args.model_config_json,
            version_reader=version_reader,
        )
    )
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
