"""Campaign-to-v9 adapter contracts for Wave 5 task 14."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    FinalAnswerResult,
    SourceLocator,
)
from evaluation.agentic_campaign_adapter import (
    campaign_execution_identity,
    used_evidence_documents,
)
from evaluation.campaign_schemas import CampaignConfig
from evaluation.schemas import ModelConfig


def _model_config() -> ModelConfig:
    return ModelConfig(
        id="model-1", name="test-model", provider="openai", model_name="test-model"
    )


def _packet(evidence_id: str, statement: str, doc_id: str) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["slot-1"],
        statement=statement,
        support_type="direct",
        source=EvidenceSource(doc_id=doc_id, chunk_id=f"chunk-{doc_id}"),
        scope=EvidenceScope(),
        locator=SourceLocator(section="Results"),
    )


def test_campaign_version_and_shadow_policy_are_explicit_and_validated() -> None:
    config = CampaignConfig(
        test_case_ids=["q-1"],
        modes=["agentic-v9-shadow"],
        model_config=_model_config(),
        agentic_execution_version="v9",
        shadow_evaluation_policy="research",
    )

    assert config.agentic_execution_version == "v9"
    assert config.shadow_evaluation_policy == "research"
    assert campaign_execution_identity("agentic-v9-shadow", "v9") == (
        "agentic-v9-shadow",
        "agentic",
        "v9",
    )

    with pytest.raises(ValidationError, match="agentic-v9-shadow requires"):
        CampaignConfig(
            test_case_ids=["q-1"],
            modes=["agentic-v9-shadow"],
            model_config=_model_config(),
            agentic_execution_version="v8",
        )


def test_campaign_identity_aliases_keep_core_mode_and_version_separate() -> None:
    assert campaign_execution_identity("naive-baseline", "v8") == (
        "naive-baseline",
        "naive",
        "v8",
    )
    assert campaign_execution_identity("agentic-v8", "v8") == (
        "agentic-v8",
        "agentic",
        "v8",
    )
    assert campaign_execution_identity("v9", "v9") == ("v9", "agentic", "v9")


def test_ragas_documents_include_only_cited_used_evidence_and_dedupe() -> None:
    packets = [
        _packet("used-1", "supported value", "doc-1"),
        _packet("unused-2", "unpacked candidate", "doc-2"),
        _packet("used-3", "supported value", "doc-1"),
    ]
    final = FinalAnswerResult(
        response_status="complete",
        answer="supported value",
        used_evidence_ids=["used-1", "used-3"],
    )

    documents = used_evidence_documents(packets, final)

    assert [item.page_content for item in documents] == ["supported value"]
    assert documents[0].metadata["evidence_id"] == "used-1"
    assert documents[0].metadata["doc_id"] == "doc-1"
    assert all(isinstance(item, Document) for item in documents)
