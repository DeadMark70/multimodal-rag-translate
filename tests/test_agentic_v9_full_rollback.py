"""Regression coverage for the production Agentic v9 behavior rollback."""

from __future__ import annotations

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


class _LegacyTextProvider:
    def __init__(self, purpose: str) -> None:
        async def invoke(messages: list[dict[str, object]]) -> SimpleNamespace:
            content: object = "The authorized source reports a score of 0.91."
            if purpose == "evidence_extraction":
                content = {
                    "packets": [
                        {
                            "source_evidence_id": "E1",
                            "slot_ids": ["S1"],
                        }
                    ]
                }
            return SimpleNamespace(
                content=content,
                usage_metadata={"input_tokens": 12, "output_tokens": 9},
            )

        self.ainvoke = AsyncMock(side_effect=invoke)


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
            RequiredSlot(slot_id="S1", description="reported table score")
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
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="Table 1 reports a score of 0.91.",
                    metadata={
                        "doc_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "page_number": 2,
                    },
                )
            ]
        ),
        visual_extractor=AsyncMock(
            return_value=VisualEvidenceExtractionResult()
        ),
        provider_factory=_LegacyTextProvider,
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
    assert result.answer == "The authorized source reports a score of 0.91."
    assert result.documents
    assert result.source_doc_ids == ["doc-1"]
    v9 = result.agent_trace["agentic_v9"]
    contract = v9["query_contract"]
    assert contract["contract_version"] == "2"
    assert [slot["slot_id"] for slot in contract["required_slots"]] == ["S1"]
    assert v9["metrics"]["atomic_planner_call_count"] <= 1
    assert v9["metrics"]["comparison_planner_call_count"] == 0
    assert v9["metrics"]["slot_binding_method"] == "task_target_inherited"
    assert v9["metrics"]["semantic_qualification"] == "provider_qualified"
    assert (
        v9["visual_execution"]["state"]
        == "required_but_not_satisfied"
    )

