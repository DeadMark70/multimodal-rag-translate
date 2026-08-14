"""Regression coverage for the production Agentic v9 behavior rollback."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from data_base.agentic_v9.schemas import (
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
)
from data_base.agentic_v9.visual_evidence_extractor import (
    VisualEvidenceExtractionResult,
)
from evaluation.agentic_v9_admission import V9AdmissionContract
from evaluation.agentic_v9_campaign_runtime import AgenticV9CampaignRuntime


class _StructuredFinalProvider:
    def __init__(self) -> None:
        self.ainvoke = AsyncMock(side_effect=self._respond)

    @staticmethod
    async def _respond(messages: list[dict[str, object]]) -> object:
        payload = json.loads(str(messages[-1]["content"]))
        packet = payload["packed_evidence_packets"][0]
        return SimpleNamespace(
            content=json.dumps(
                {
                    "supported_findings": [
                        {
                            "slot_id": "S1",
                            "statement": (
                                "The authorized source reports a score of 0.91."
                            ),
                            "support_type": "direct",
                            "evidence_ids": [packet["evidence_id"]],
                            "premise_evidence_ids": [],
                        }
                    ],
                    "unresolved_requirements": [],
                }
            ),
            usage_metadata={"input_tokens": 12, "output_tokens": 9},
        )


@pytest.mark.asyncio
async def test_missing_visual_capability_preserves_authorized_text_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        contract_version="1",
        route="exact_structured",
        intent="read the reported table score",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="reported table score",
                locator_hints=["Table 1"],
            )
        ],
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=5,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs: object) -> V9AdmissionContract:
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract",
        admission,
    )
    provider = _StructuredFinalProvider()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="Table 1 reports a score of 0.91 points.",
                    metadata={
                        "doc_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "page_number": 2,
                        "table_id": "Table 1",
                    },
                )
            ]
        ),
        visual_extractor=AsyncMock(
            return_value=VisualEvidenceExtractionResult()
        ),
        provider_factory=lambda _purpose: provider,
    )

    result = await runtime.execute(
        question="What score is reported in Table 1?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={
            "max_input_tokens": 4096,
            "max_output_tokens": 8192,
            "max_llm_calls": 5,
            "runtime_token_budget": 50_000,
            "thinking_mode": False,
        },
        trace_id="legacy-v9-visual-fail-soft",
    )

    assert result.agent_trace["response_status"] == "complete"
    assert "The authorized source reports a score of 0.91." in result.answer
    assert "[v1:doc-1 p.3, Table 1;" in result.answer
    assert result.documents
    assert result.source_doc_ids == ["doc-1"]
    assert (
        result.agent_trace["agentic_v9"]["visual_execution"]["state"]
        == "required_but_not_satisfied"
    )
