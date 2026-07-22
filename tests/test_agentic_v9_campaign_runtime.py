"""Production-adapter coverage for the Agentic v9 campaign path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from evaluation.agentic_v9_campaign_runtime import AgenticV9CampaignRuntime


class _Provider:
    def __init__(self) -> None:
        self.ainvoke = AsyncMock(
            return_value=SimpleNamespace(
                content="The reported score is 0.91.",
                usage_metadata={"input_tokens": 12, "output_tokens": 7},
            )
        )


def _setup() -> dict[str, object]:
    return {
        "max_input_tokens": 4096,
        "max_output_tokens": 256,
        "thinking_mode": False,
    }


async def _identity_reference_resolver(
    _user_id: str, references: list[str]
) -> dict[str, str]:
    """Keep unit tests independent of the production document repository."""
    return {reference: reference for reference in references}


@pytest.mark.asyncio
async def test_v9_campaign_runtime_runs_core_and_emits_real_evidence_trace() -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="The source reports a score of 0.91.",
                metadata={"doc_id": "doc-1", "page_number": 2, "chunk_id": "chunk-1"},
            )
        ]
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="attempt-trace-1",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert v9["query_contract"]["resolved_source_scope"]["authorized_doc_ids"] == [
        "doc-1"
    ]
    assert v9["evidence_packets"]
    assert v9["slot_resolutions"]
    assert v9["sufficiency"]["response_status"] == "complete"
    assert result.documents
    retrieve_documents.assert_awaited()
    provider.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_v9_campaign_runtime_resolves_filename_scope_to_canonical_document_id() -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="The source reports a score of 0.91.",
                metadata={"doc_id": "doc-1", "page_number": 2, "chunk_id": "chunk-1"},
            )
        ]
    )

    async def resolve_references(_user_id: str, references: list[str]) -> dict[str, str]:
        assert references == ["paper.pdf"]
        return {"paper.pdf": "doc-1"}

    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=resolve_references,
    )

    result = await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=["paper.pdf"],
        setup_snapshot=_setup(),
        trace_id="attempt-trace-filename-scope",
    )

    assert result.agent_trace["agentic_v9"]["query_contract"]["resolved_source_scope"]["authorized_doc_ids"] == ["doc-1"]
    assert result.agent_trace["agentic_v9"]["query_contract"]["resolved_source_scope"]["requested_doc_ids"] == ["doc-1"]
    assert result.agent_trace["response_status"] == "complete"
    retrieve_documents.assert_awaited()


@pytest.mark.asyncio
async def test_v9_runtime_rejects_incompatible_setup_before_provider_or_retrieval() -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={"thinking_mode": False},
        trace_id="attempt-trace-incompatible",
    )

    assert result.agent_trace["response_status"] == "configuration_incompatible"
    assert result.agent_trace["agentic_v9"]["configuration_incompatible"]["stage"] == "pre_route"
    assert result.documents == []
    retrieve_documents.assert_not_awaited()
    provider.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_v9_runtime_repeats_feasibility_after_contract_before_retrieval() -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    # A table request requires visual extraction in the deterministic contract.
    # Its route budget cannot admit visual + evidence + final provider work.
    result = await runtime.execute(
        question="What is the table score?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="attempt-trace-post-contract",
    )

    incompatible = result.agent_trace["agentic_v9"]["configuration_incompatible"]
    assert incompatible["stage"] == "post_contract"
    assert result.agent_trace["response_status"] == "configuration_incompatible"
    retrieve_documents.assert_not_awaited()
    provider.ainvoke.assert_not_awaited()
