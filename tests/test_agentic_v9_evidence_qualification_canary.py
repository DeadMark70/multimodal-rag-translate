"""Tests for the production evidence-qualification canary."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


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


@pytest.mark.asyncio
async def test_canary_uses_shared_schema_bound_provider_once(
    model_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert importlib.util.find_spec(
        "scripts.agentic_v9_evidence_qualification_canary"
    ) is not None
    canary = importlib.import_module(
        "scripts.agentic_v9_evidence_qualification_canary"
    )
    calls: list[object] = []
    schemas: list[object] = []

    class Provider:
        async def ainvoke(self, messages: object) -> object:
            calls.append(messages)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "packets": [
                                {
                                    "source_evidence_id": "E1",
                                    "slot_ids": ["S1"],
                                    "statement": (
                                        "The method uses a two-stage decoder for small lesions."
                                    ),
                                }
                        ]
                    }
                )
            )

    monkeypatch.setattr(
        canary,
        "_build_provider",
        lambda *, response_schema, model_config: (
            schemas.append(response_schema) or Provider()
        ),
    )

    exit_code, payload = await canary.run_canary(
        model_config_path=model_config_path,
        invoke=True,
        version_reader=lambda _name: "test-version",
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert schemas[0]["required"] == ["packets"]
    assert calls[0][0]["role"] == "user"
    assert payload["success"] is True
    assert payload["qualified_packet_count"] == 1
    assert payload["semantic_qualification"] == "provider_qualified"
    assert payload["failure_code"] is None


@pytest.mark.asyncio
async def test_canary_defaults_to_construction_only_without_invoking_provider(
    model_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = importlib.import_module(
        "scripts.agentic_v9_evidence_qualification_canary"
    )
    constructed: list[object] = []
    overrides: list[dict[str, object]] = []

    @contextmanager
    def runtime_override(**kwargs: object):
        overrides.append(kwargs)
        yield

    llm_factory = importlib.import_module("core.llm_factory")
    monkeypatch.setattr(llm_factory, "llm_runtime_override", runtime_override)

    class Provider:
        async def ainvoke(self, _messages: object) -> object:
            raise AssertionError("construction mode must not invoke the provider")

    monkeypatch.setattr(
        canary,
        "_build_provider",
        lambda *, response_schema, model_config: (
            constructed.append(response_schema) or Provider()
        ),
    )

    exit_code, payload = await canary.run_canary(
        model_config_path=model_config_path,
        invoke=False,
        version_reader=lambda _name: "test-version",
    )

    assert exit_code == 0
    assert len(constructed) == 1
    assert overrides[0]["max_retries"] == 0
    assert payload["mode"] == "construction"
    assert payload["response_received"] is False
    assert payload["qualified_packet_count"] == 0


@pytest.mark.asyncio
async def test_invalid_config_fails_before_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = importlib.import_module(
        "scripts.agentic_v9_evidence_qualification_canary"
    )
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not-json", encoding="utf-8")
    monkeypatch.setattr(
        canary,
        "_build_provider",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider must not be constructed")
        ),
    )

    exit_code, payload = await canary.run_canary(
        model_config_path=invalid_path,
        invoke=False,
        version_reader=lambda _name: "test-version",
    )

    assert exit_code != 0
    assert payload["failure_code"] == "model_config_invalid"


@pytest.mark.asyncio
async def test_canary_sanitizes_provider_failure(
    model_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = importlib.import_module(
        "scripts.agentic_v9_evidence_qualification_canary"
    )

    class Provider:
        async def ainvoke(self, _messages: object) -> object:
            raise RuntimeError("api_key=secret prompt=private")

    monkeypatch.setattr(
        canary,
        "_build_provider",
        lambda *, response_schema, model_config: Provider(),
    )

    exit_code, payload = await canary.run_canary(
        model_config_path=model_config_path,
        invoke=True,
        version_reader=lambda _name: "test-version",
    )

    rendered = json.dumps(payload)
    assert exit_code != 0
    assert payload["failure_code"] == "provider_attempt_failed"
    assert "secret" not in rendered
    assert "private" not in rendered
