#!/usr/bin/env python3
"""Probe the production Agentic v9 evidence provider exactly once."""

from __future__ import annotations

import argparse
import asyncio
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Any, Callable, Literal, Mapping, Protocol

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.schemas import ModelConfig  # noqa: E402


VersionReader = Callable[[str], str]
CanaryMode = Literal["construction", "invoke"]
_PACKAGES = ("google-genai", "langchain-google-genai", "pydantic")


class AsyncProvider(Protocol):
    async def ainvoke(self, messages: object) -> object:
        """Make one provider attempt."""


def _versions(reader: VersionReader) -> dict[str, str]:
    result: dict[str, str] = {}
    for package in _PACKAGES:
        try:
            result[package] = str(reader(package))[:200]
        except Exception:
            result[package] = "unknown"
    return result


def _payload(
    *,
    success: bool,
    model_identifier: str,
    package_versions: Mapping[str, str],
    response_received: bool,
    mode: CanaryMode,
    qualified_packet_count: int = 0,
    semantic_qualification: str = "not_attempted",
    failure_code: str | None = None,
) -> dict[str, Any]:
    return {
        "success": success,
        "failure_code": failure_code,
        "model_identifier": model_identifier[:200],
        "package_versions": dict(package_versions),
        "response_received": response_received,
        "mode": mode,
        "qualified_packet_count": qualified_packet_count,
        "semantic_qualification": semantic_qualification,
    }


def _load_config(path: Path | None) -> ModelConfig | None:
    if path is None:
        return None
    try:
        return ModelConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError):
        return None


def _build_provider(
    *, response_schema: Mapping[str, Any], model_config: ModelConfig
) -> AsyncProvider:
    del model_config
    from data_base.agentic_v9.provider_boundary import (
        build_evidence_qualification_provider,
    )

    return build_evidence_qualification_provider(response_schema=response_schema)


def _canary_inputs() -> tuple[object, list[object], str]:
    from data_base.agentic_v9.evidence_pool import EvidencePoolItem
    from data_base.agentic_v9.schemas import (
        EvidencePacket,
        EvidenceScope,
        EvidenceSource,
        QueryContract,
        RequiredSlot,
        SourceLocator,
    )

    statement = "The method uses a two-stage decoder for small lesions."
    contract = QueryContract(
        route="exact_structured",
        intent="Identify the decoder architecture.",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Describe the decoder architecture.")
        ],
    )
    item = EvidencePoolItem(
        EvidencePacket(
            schema_version="1",
            evidence_id="E1",
            task_id="task-canary",
            round_id="round-1",
            query_id="query-1",
            slot_ids=["S1"],
            statement=statement,
            support_type="direct",
            source=EvidenceSource(doc_id="doc-canary", chunk_id="chunk-1"),
            scope=EvidenceScope(),
            locator=SourceLocator(section="Results"),
        ),
        metadata={"text": statement},
        retrieval_scores={"reranker": 1.0},
    )
    return contract, [item], "What decoder architecture does the source describe?"


async def run_canary(
    *,
    model_config_path: Path | None,
    invoke: bool = False,
    version_reader: VersionReader = metadata.version,
) -> tuple[int, dict[str, Any]]:
    """Run one configured evidence provider attempt with sanitized output."""
    package_versions = _versions(version_reader)
    model_config = _load_config(model_config_path)
    if model_config is None:
        return 10, _payload(
            success=False,
            model_identifier="unknown",
            package_versions=package_versions,
            response_received=False,
            mode="invoke" if invoke else "construction",
            failure_code="model_config_invalid",
        )

    model_identifier = model_config.model_name
    try:
        from core.llm_factory import llm_runtime_override
        from data_base.agentic_v9.phase_policy import (
            agentic_phase_policy_scope,
            resolve_phase_policy,
        )
        from data_base.agentic_v9.provider_boundary import (
            evidence_qualification_response_schema,
        )
        from evaluation.model_capabilities import normalize_model_config_for_runtime

        normalized = normalize_model_config_for_runtime(
            model_config.model_dump(mode="json")
        )
        policy = resolve_phase_policy(
            "evidence_extract",
            setup_output_ceiling=model_config.max_output_tokens,
            setup_input_ceiling=model_config.max_input_tokens,
            remaining_input_budget=model_config.max_input_tokens,
        )
        schema = evidence_qualification_response_schema()
        with (
            llm_runtime_override(**normalized, max_retries=0),
            agentic_phase_policy_scope(policy),
        ):
            if not invoke:
                _build_provider(
                    response_schema=schema,
                    model_config=model_config,
                )
                return 0, _payload(
                    success=True,
                    model_identifier=model_identifier,
                    package_versions=package_versions,
                    response_received=False,
                    mode="construction",
                )

            from data_base.agentic_v9.budget_controller import RunBudgetController
            from data_base.agentic_v9.budgeted_llm import BudgetedLlmInvoker
            from data_base.agentic_v9.evidence_extractor import EvidenceExtractor

            controller = RunBudgetController(
                max_llm_calls=2,
                runtime_token_budget=max(
                    10_000,
                    model_config.max_input_tokens * 2
                    + model_config.max_output_tokens * 4,
                ),
                setup_snapshot=model_config.model_dump(mode="json"),
                final_input_tokens=1,
            )
            contract, pool, question = _canary_inputs()
            outcome = await EvidenceExtractor(
                BudgetedLlmInvoker(
                    controller=controller,
                    provider_factory=lambda _purpose: _build_provider(
                        response_schema=schema,
                        model_config=model_config,
                    ),
                    provider_name="canary",
                    model_name=model_identifier,
                )
            ).extract_with_outcome(
                contract,
                pool,
                repairs_complete=True,
                question=question,
            )
    except Exception:
        return 30, _payload(
            success=False,
            model_identifier=model_identifier,
            package_versions=package_versions,
            response_received=False,
            mode="invoke" if invoke else "construction",
            semantic_qualification="provider_failed" if invoke else "not_attempted",
            failure_code="provider_attempt_failed",
        )

    count = len(outcome.packets)
    if outcome.status != "provider_qualified" or count < 1:
        return 31, _payload(
            success=False,
            model_identifier=model_identifier,
            package_versions=package_versions,
            response_received=outcome.provider_response_received,
            mode="invoke",
            semantic_qualification=outcome.status,
            failure_code=outcome.failure_code or "no_qualified_packets",
        )
    return 0, _payload(
        success=True,
        model_identifier=model_identifier,
        package_versions=package_versions,
        response_received=True,
        mode="invoke",
        qualified_packet_count=count,
        semantic_qualification=outcome.status,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config-json", type=Path)
    parser.add_argument(
        "--invoke",
        action="store_true",
        help="Perform the single real provider attempt; default only constructs the boundary.",
    )
    args = parser.parse_args(argv)
    exit_code, payload = asyncio.run(
        run_canary(model_config_path=args.model_config_json, invoke=args.invoke)
    )
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
