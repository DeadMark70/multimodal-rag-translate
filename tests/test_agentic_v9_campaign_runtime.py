"""Production-adapter coverage for the Agentic v9 campaign path."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

import evaluation.agentic_v9_campaign_runtime as runtime_module
from data_base.rag_pipeline_schemas import RagRetrievalResult as PipelineRetrievalResult
from data_base.reranker import DocumentReranker
from evaluation.agentic_v9_campaign_runtime import AgenticV9CampaignRuntime
from evaluation.agentic_v9_admission import V9AdmissionContract
from evaluation.campaign_schemas import V9ContextPack
from evaluation.retrieval_profiles import AGENTIC_V9_OPEN_CORPUS_PROFILE
from data_base.agentic_v9.schemas import (
    ComparisonPlannerOutcome,
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
    SourceLocator,
    TaskRetrievalResult,
)
from data_base.agentic_v9.visual_evidence_extractor import (
    VisualEvidenceExtractionResult,
)


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


def _retrieved_documents() -> list[Document]:
    return [
        Document(
            page_content=f"chunk-{index}",
            metadata={"doc_id": "doc-1", "chunk_id": f"chunk-{index}"},
        )
        for index in range(8)
    ]


def _patch_v9_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    documents: list[Document],
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "get_user_retriever_async",
        AsyncMock(return_value=object()),
    )
    monkeypatch.setattr(
        runtime_module,
        "retrieve_hybrid_documents",
        AsyncMock(return_value=PipelineRetrievalResult(documents=documents)),
    )


@pytest.mark.asyncio
async def test_v9_retrieval_reranks_eight_to_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _retrieved_documents()
    _patch_v9_retrieval(monkeypatch, documents)
    reranker = SimpleNamespace(
        rerank_with_scores_strict=lambda _query, docs, _top_k: [
            (docs[index], float(index + 1)) for index in reversed(range(8))
        ]
    )
    monkeypatch.setattr(
        DocumentReranker, "is_initialized", classmethod(lambda _cls: True)
    )
    monkeypatch.setattr(
        DocumentReranker, "get_instance", classmethod(lambda _cls: reranker)
    )

    selected = await runtime_module._retrieve_documents("user-a", "question", ["doc-1"])

    assert [document.page_content for document in selected] == [
        document.page_content for document in documents[7:3:-1]
    ]
    assert all(
        document.metadata["agentic_v9_reranking"]["status"] == "executed"
        for document in selected
    )
    assert all(
        document.metadata["agentic_v9_reranking"]["rerank_score"] is not None
        for document in selected
    )


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("single_lookup", False),
        ("exact_structured", False),
        ("bounded_compare", True),
        ("multi_hop", True),
        ("multi_document_exact", True),
        ("graph_relational", True),
    ],
)
def test_v9_candidate_diversification_is_limited_to_multi_source_routes(
    route: str, expected: bool
) -> None:
    assert runtime_module._requires_diverse_rerank_candidates(route) is expected


@pytest.mark.asyncio
async def test_v9_retrieval_falls_back_to_hybrid_top_four_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _retrieved_documents()
    _patch_v9_retrieval(monkeypatch, documents)
    monkeypatch.setattr(
        DocumentReranker, "is_initialized", classmethod(lambda _cls: False)
    )

    selected = await runtime_module._retrieve_documents("user-a", "question", ["doc-1"])

    assert [document.page_content for document in selected] == [
        document.page_content for document in documents[:4]
    ]
    assert all(
        document.metadata["agentic_v9_reranking"]
        == {
            "status": "fallback",
            "fallback_reason": "reranker_unavailable",
            "candidate_count": 8,
            "selected_count": 4,
            "pre_rerank_rank": index,
            "post_rerank_rank": index,
            "rerank_score": None,
            "candidate_diversification": {
                "policy": "tail_source_diversity_r1",
                "enabled": False,
                "applied": False,
                "retrieved_doc_ids": ["doc-1"],
                "candidate_doc_ids": ["doc-1"],
                "represented_doc_ids_before_tail": [],
                "admitted_doc_ids": [],
            },
        }
        for index, document in enumerate(selected, start=1)
    )


@pytest.mark.asyncio
async def test_v9_retrieval_falls_back_to_hybrid_top_four_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _retrieved_documents()
    _patch_v9_retrieval(monkeypatch, documents)

    def fail_reranking(*_args):
        raise RuntimeError("private provider detail")

    reranker = SimpleNamespace(rerank_with_scores_strict=fail_reranking)
    monkeypatch.setattr(
        DocumentReranker, "is_initialized", classmethod(lambda _cls: True)
    )
    monkeypatch.setattr(
        DocumentReranker, "get_instance", classmethod(lambda _cls: reranker)
    )

    selected = await runtime_module._retrieve_documents("user-a", "question", ["doc-1"])

    assert [document.page_content for document in selected] == [
        document.page_content for document in documents[:4]
    ]
    assert all(
        document.metadata["agentic_v9_reranking"]["fallback_reason"] == "reranker_error"
        for document in selected
    )
    assert all(
        document.metadata["agentic_v9_reranking"]["rerank_score"] is None
        for document in selected
    )


@pytest.mark.asyncio
async def test_v9_retrieval_falls_back_to_hybrid_top_four_on_empty_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _retrieved_documents()
    _patch_v9_retrieval(monkeypatch, documents)
    reranker = SimpleNamespace(
        rerank_with_scores_strict=lambda _query, _docs, _top_k: []
    )
    monkeypatch.setattr(
        DocumentReranker, "is_initialized", classmethod(lambda _cls: True)
    )
    monkeypatch.setattr(
        DocumentReranker, "get_instance", classmethod(lambda _cls: reranker)
    )

    selected = await runtime_module._retrieve_documents("user-a", "question", ["doc-1"])

    assert [document.page_content for document in selected] == [
        document.page_content for document in documents[:4]
    ]
    assert all(
        document.metadata["agentic_v9_reranking"]["fallback_reason"]
        == "reranker_empty_result"
        for document in selected
    )
    assert all(
        document.metadata["agentic_v9_reranking"]["rerank_score"] is None
        for document in selected
    )


@pytest.mark.asyncio
async def test_v9_campaign_runtime_runs_core_and_emits_real_evidence_trace() -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="The source reports a score of 0.91.",
                metadata={
                    "doc_id": "doc-1",
                    "page_number": 2,
                    "chunk_id": "chunk-1",
                    "agentic_v9_reranking": {
                        "status": "executed",
                        "fallback_reason": None,
                        "candidate_count": 8,
                        "selected_count": 4,
                        "pre_rerank_rank": 2,
                        "post_rerank_rank": 1,
                        "rerank_score": 0.93,
                    },
                },
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
    assert result.agent_trace["execution_profile"] == (
        "agentic_eval_v9_explicit_scope_hybrid8_rerank8_diverse_tail2_top4_finalpack_r1"
    )
    assert v9["query_contract"]["resolved_source_scope"]["authorized_doc_ids"] == [
        "doc-1"
    ]
    assert v9["evidence_packets"]
    assert v9["slot_resolutions"]
    assert v9["sufficiency"]["response_status"] == "complete"
    assert v9["retrieval_diagnostics"] == [
        {
            "task_id": "attempt-trace-1:round-1:source-group-1",
            "status": "executed",
            "fallback_reason": None,
            "candidate_count": 8,
            "selected_count": 1,
            "selected": [
                {
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "content_hash": hashlib.sha256(
                        "The source reports a score of 0.91.".encode("utf-8")
                    ).hexdigest(),
                    "pre_rerank_rank": 2,
                    "post_rerank_rank": 1,
                    "rerank_score": 0.93,
                }
            ],
        }
    ]
    assert result.documents
    retrieve_documents.assert_awaited()
    provider.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_v9_comparison_planner_overlays_subject_tasks_and_caps_each_at_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        SimpleNamespace(
            content=json.dumps(
                {
                    "is_comparison": True,
                    "subjects": [
                        {
                            "subject_id": "nnmamba",
                            "display_name": "nnMamba",
                            "aliases": [],
                            "retrieval_query": "nnMamba parameters FLOPs",
                        },
                        {
                            "subject_id": "efficientmednext_l",
                            "display_name": "EfficientMedNeXt-L",
                            "aliases": [],
                            "retrieval_query": "EfficientMedNeXt-L parameters FLOPs",
                        },
                    ],
                    "dimensions": ["parameters", "FLOPs"],
                    "qualification": None,
                }
            ),
            usage_metadata={"input_tokens": 20, "output_tokens": 10},
        ),
        SimpleNamespace(
            content="The evidence supports a bounded comparison.",
            usage_metadata={"input_tokens": 12, "output_tokens": 7},
        ),
    ]
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare two models.",
        required_slots=[RequiredSlot(slot_id="base", description="comparison")],
        max_retrieval_rounds=1,
        max_llm_calls=1,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        runtime_module, "build_v9_admission_contract", admission
    )
    async def retrieve_subject_documents(
        _user_id: str, retrieval_query: str, _authorized_doc_ids: list[str]
    ) -> list[Document]:
        subject = (
            "nnmamba"
            if "nnMamba" in retrieval_query
            else "efficientmednext_l"
        )
        return [
            Document(
                page_content=f"{subject} evidence chunk {index}.",
                metadata={
                    "doc_id": "doc-1",
                    "chunk_id": f"{subject}-chunk-{index}",
                },
            )
            for index in range(4)
        ]

    retrieve_documents = AsyncMock(side_effect=retrieve_subject_documents)
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="nnMamba vs. EfficientMedNeXt-L: which is more efficient?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="comparison-runtime",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert [
        subject["subject_id"]
        for subject in v9["query_contract"]["comparison_plan"]["subjects"]
    ] == ["nnmamba", "efficientmednext_l"]
    assert retrieve_documents.await_count == 2
    assert [row["selected_count"] for row in v9["retrieval_diagnostics"]] == [2, 2]
    assert v9["comparison_planner"]["status"] == "planned"
    assert {
        tuple(packet["slot_ids"]) for packet in v9["evidence_packets"]
    } == {
        ("comparison-subject:nnmamba",),
        ("comparison-subject:efficientmednext_l",),
    }
    packed_ids = set(v9["context_pack"]["packed_evidence_ids"])
    packed_packets = [
        packet
        for packet in v9["evidence_packets"]
        if packet["evidence_id"] in packed_ids
    ]
    assert len(packed_packets) == 4
    assert {
        tuple(packet["slot_ids"]) for packet in packed_packets
    } == {
        ("comparison-subject:nnmamba",),
        ("comparison-subject:efficientmednext_l",),
    }
    assert provider.ainvoke.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("repair_succeeds", "expected_status"),
    [(True, "complete"), (False, "qualified_partial")],
)
async def test_v9_comparison_repairs_a_missing_subject_once_and_caps_status(
    monkeypatch: pytest.MonkeyPatch,
    repair_succeeds: bool,
    expected_status: str,
) -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        SimpleNamespace(
            content=json.dumps(
                {
                    "is_comparison": True,
                    "subjects": [
                        {
                            "subject_id": "model_a",
                            "display_name": "Model A",
                            "aliases": ["A"],
                            "retrieval_query": "Model A accuracy",
                        },
                        {
                            "subject_id": "model_b",
                            "display_name": "Model B",
                            "aliases": ["B"],
                            "retrieval_query": "Model B accuracy",
                        },
                    ],
                    "dimensions": ["accuracy"],
                    "qualification": None,
                }
            ),
            usage_metadata={"input_tokens": 20, "output_tokens": 10},
        ),
        SimpleNamespace(
            content="The evidence supports the available comparison.",
            usage_metadata={"input_tokens": 12, "output_tokens": 7},
        ),
    ]
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-a", "doc-b"],
        resolved_doc_ids=["doc-a", "doc-b"],
        authorized_doc_ids=["doc-a", "doc-b"],
    )
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare two models.",
        required_slots=[RequiredSlot(slot_id="base", description="comparison")],
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=1,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        runtime_module, "build_v9_admission_contract", admission
    )
    model_b_attempts = 0

    async def retrieve_subject_documents(
        _user_id: str, retrieval_query: str, _authorized_doc_ids: list[str]
    ) -> list[Document]:
        nonlocal model_b_attempts
        if "Model B" in retrieval_query:
            model_b_attempts += 1
            if model_b_attempts == 1 or not repair_succeeds:
                return []
            subject = "model_b"
            doc_id = "doc-b"
        else:
            subject = "model_a"
            doc_id = "doc-a"
        return [
            Document(
                page_content=f"{subject} accuracy evidence {index}.",
                metadata={
                    "doc_id": doc_id,
                    "chunk_id": f"{subject}-chunk-{index}",
                },
            )
            for index in range(2)
        ]

    retrieve_documents = AsyncMock(side_effect=retrieve_subject_documents)
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Model A vs. Model B: which has better accuracy?",
        user_id="user-a",
        authorized_doc_ids=["doc-a", "doc-b"],
        setup_snapshot=_setup(),
        trace_id=f"comparison-repair-{repair_succeeds}",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert result.agent_trace["response_status"] == expected_status
    assert model_b_attempts == 2
    assert retrieve_documents.await_count == 3
    assert len(v9["repairs"]) == 1
    assert len(v9["repairs"][0]["tasks"]) == 1
    assert v9["repairs"][0]["tasks"][0]["subject_id"] == "model_b"
    if repair_succeeds:
        packed_ids = set(v9["context_pack"]["packed_evidence_ids"])
        packed_packets = [
            packet
            for packet in v9["evidence_packets"]
            if packet["evidence_id"] in packed_ids
        ]
        assert {
            tuple(packet["slot_ids"]) for packet in packed_packets
        } == {
            ("comparison-subject:model_a",),
            ("comparison-subject:model_b",),
        }


@pytest.mark.asyncio
async def test_v9_comparison_status_uses_final_balanced_packet_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        SimpleNamespace(
            content=json.dumps(
                {
                    "is_comparison": True,
                    "subjects": [
                        {
                            "subject_id": "model_a",
                            "display_name": "Model A",
                            "aliases": [],
                            "retrieval_query": "Model A accuracy",
                        },
                        {
                            "subject_id": "model_b",
                            "display_name": "Model B",
                            "aliases": [],
                            "retrieval_query": "Model B accuracy",
                        },
                    ],
                    "dimensions": ["accuracy"],
                    "qualification": None,
                }
            ),
            usage_metadata={"input_tokens": 20, "output_tokens": 10},
        ),
        SimpleNamespace(
            content="Only the packed evidence may be used.",
            usage_metadata={"input_tokens": 12, "output_tokens": 7},
        ),
    ]
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare models.",
        required_slots=[RequiredSlot(slot_id="base", description="comparison")],
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=1,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        runtime_module, "build_v9_admission_contract", admission
    )
    shared_document = Document(
        page_content="One shared chunk was returned for both subject queries.",
        metadata={"doc_id": "doc-1", "chunk_id": "shared-chunk"},
    )
    retrieve_documents = AsyncMock(return_value=[shared_document])
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Model A vs. Model B: which has better accuracy?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="comparison-shared-source",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert result.agent_trace["response_status"] == "qualified_partial"
    packed_ids = set(v9["context_pack"]["packed_evidence_ids"])
    packed_packets = [
        packet
        for packet in v9["evidence_packets"]
        if packet["evidence_id"] in packed_ids
    ]
    assert len(packed_packets) == 1
    assert len(packed_packets[0]["slot_ids"]) == 1
    assert retrieve_documents.await_count == 3
    assert len(v9["repairs"]) == 1
    assert len(v9["repairs"][0]["tasks"]) == 1
    assert v9["repairs"][0]["tasks"][0]["subject_id"] == "model_b"


@pytest.mark.asyncio
async def test_v9_comparison_planner_failure_preserves_base_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        RuntimeError("planner unavailable"),
        SimpleNamespace(
            content="Fallback answer from retrieved evidence.",
            usage_metadata={"input_tokens": 12, "output_tokens": 7},
        ),
    ]
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="The source contains usable comparison evidence.",
                metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
            )
        ]
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Model A vs. Model B: which performs better?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="comparison-fallback",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert "comparison_plan" not in v9["query_contract"]
    assert v9["comparison_planner"] == {
        "requested": True,
        "status": "fallback",
        "fallback_reason": "provider_error",
        "latency_ms": v9["comparison_planner"]["latency_ms"],
    }
    assert result.documents
    retrieve_documents.assert_awaited()
    assert provider.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_v9_comparison_specialization_flag_restores_existing_path() -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="The source contains comparison evidence.",
                metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
            )
        ]
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
        comparison_specialization_enabled=False,
    )

    result = await runtime.execute(
        question="Model A vs. Model B: which performs better?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="comparison-disabled",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert "comparison_plan" not in v9["query_contract"]
    assert v9["comparison_planner"]["requested"] is False
    assert provider.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_v9_forced_comparison_timeout_never_clears_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def timed_out_plan(*_args, **_kwargs):
        return ComparisonPlannerOutcome(
            status="fallback",
            fallback_reason="timeout",
            latency_ms=64_000,
        )

    monkeypatch.setattr(
        runtime_module.ComparisonPlanner,
        "plan",
        timed_out_plan,
    )
    provider = _Provider()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="Fallback evidence remains available.",
                metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
            )
        ]
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Model A vs. Model B: which performs better?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="comparison-timeout",
    )

    assert result.agent_trace["agentic_v9"]["comparison_planner"][
        "fallback_reason"
    ] == "timeout"
    assert result.documents
    retrieve_documents.assert_awaited()


@pytest.mark.asyncio
async def test_v9_campaign_runtime_activates_soft_final_context_policy_with_rerank_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    observed: dict[str, object] = {}
    original_pack = runtime_module.EvidenceContextPacker.pack

    def recording_pack(self, packets, **kwargs):
        observed["quality_by_evidence_id"] = dict(
            kwargs["quality_by_evidence_id"]
        )
        observed["selection_policy"] = kwargs["selection_policy"]
        return original_pack(self, packets, **kwargs)

    monkeypatch.setattr(
        runtime_module.EvidenceContextPacker, "pack", recording_pack
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="The primary source reports 0.91.",
                    metadata={
                        "doc_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "agentic_v9_reranking": {
                            "status": "executed",
                            "post_rerank_rank": 1,
                            "rerank_score": 0.93,
                        },
                    },
                ),
                Document(
                    page_content="A secondary source reports 0.89.",
                    metadata={
                        "doc_id": "doc-2",
                        "chunk_id": "chunk-4",
                        "agentic_v9_reranking": {
                            "status": "executed",
                            "post_rerank_rank": 4,
                            "rerank_score": 0.71,
                        },
                    },
                ),
            ]
        ),
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What score is reported?",
        user_id="user-a",
        authorized_doc_ids=["doc-1", "doc-2"],
        setup_snapshot=_setup(),
        trace_id="soft-final-pack-trace",
    )

    assert observed["quality_by_evidence_id"]
    assert set(observed["quality_by_evidence_id"].values()) == {0.25, 1.0}
    assert observed["selection_policy"].version == "soft_final_pack_r1"
    context_pack = result.agent_trace["agentic_v9"]["context_pack"]
    assert context_pack["selection_policy_version"] == "soft_final_pack_r1"
    assert context_pack["candidate_count"] >= len(
        context_pack["packed_evidence_ids"]
    )
    assert any(
        row["base_quality"] > 0 for row in context_pack["selection_decisions"]
    )


def test_v9_context_pack_schema_accepts_historical_trace_payload() -> None:
    context_pack = V9ContextPack.model_validate(
        {
            "packed_evidence_ids": ["evidence:legacy"],
            "dropped_evidence_ids": ["evidence:excluded"],
            "token_count": 123,
        }
    )

    assert context_pack.packed_evidence_ids == ["evidence:legacy"]
    assert context_pack.dropped_evidence_ids == ["evidence:excluded"]
    assert context_pack.token_count == 123
    assert context_pack.selection_policy_version is None
    assert context_pack.candidate_count is None
    assert context_pack.selection_decisions == []


def test_retrieval_diagnostic_projection_retains_fallback_details() -> None:
    diagnostics = runtime_module._retrieval_diagnostic_projection(
        "task:source-group-1",
        [
            Document(
                page_content="  exact selected content  ",
                metadata={
                    "original_doc_uid": "legacy-doc-1",
                    "chunk_id": "chunk-1",
                    "agentic_v9_reranking": {
                        "status": "fallback",
                        "fallback_reason": "reranker_unavailable",
                        "candidate_count": 8,
                        "selected_count": 4,
                        "pre_rerank_rank": 1,
                        "post_rerank_rank": 1,
                        "rerank_score": None,
                    },
                },
            )
        ],
    )

    assert diagnostics == {
        "task_id": "task:source-group-1",
        "status": "fallback",
        "fallback_reason": "reranker_unavailable",
        "candidate_count": 8,
        "selected_count": 1,
        "selected": [
            {
                "doc_id": "legacy-doc-1",
                "chunk_id": "chunk-1",
                "content_hash": hashlib.sha256(
                    "  exact selected content  ".encode("utf-8")
                ).hexdigest(),
                "pre_rerank_rank": 1,
                "post_rerank_rank": 1,
                "rerank_score": None,
            }
        ],
    }


def test_retrieval_diagnostic_projection_retains_candidate_diversification() -> None:
    diagnostics = runtime_module._retrieval_diagnostic_projection(
        "task:source-group-1",
        [
            Document(
                page_content="selected content",
                metadata={
                    "doc_id": "primary",
                    "agentic_v9_reranking": {
                        "status": "executed",
                        "fallback_reason": None,
                        "candidate_count": 8,
                        "selected_count": 4,
                        "pre_rerank_rank": 1,
                        "post_rerank_rank": 1,
                        "rerank_score": 0.93,
                        "candidate_diversification": {
                            "policy": "tail_source_diversity_r1",
                            "enabled": True,
                            "applied": True,
                            "retrieved_doc_ids": [
                                "primary",
                                "secondary",
                                "tertiary",
                            ],
                            "candidate_doc_ids": [
                                "primary",
                                "secondary",
                                "tertiary",
                            ],
                            "represented_doc_ids_before_tail": ["primary"],
                            "admitted_doc_ids": ["secondary", "tertiary"],
                        },
                    },
                },
            )
        ],
    )

    assert diagnostics["candidate_diversification"] == {
        "policy": "tail_source_diversity_r1",
        "enabled": True,
        "applied": True,
        "retrieved_doc_ids": ["primary", "secondary", "tertiary"],
        "candidate_doc_ids": ["primary", "secondary", "tertiary"],
        "represented_doc_ids_before_tail": ["primary"],
        "admitted_doc_ids": ["secondary", "tertiary"],
    }


def test_annotate_rerank_selection_copies_candidate_diversification() -> None:
    selection = PipelineRetrievalResult(
        documents=[Document(page_content="selected content", metadata={"doc_id": "primary"})],
        metadata={
            "reranking": {
                "candidate_count": 8,
                "post_rerank_ranks": [
                    {"pre_rerank_rank": 1, "score": 0.93},
                ],
                "candidate_diversification": {
                    "policy": "tail_source_diversity_r1",
                    "enabled": True,
                    "applied": True,
                    "retrieved_doc_ids": ["primary", "secondary"],
                    "candidate_doc_ids": ["primary", "secondary"],
                    "represented_doc_ids_before_tail": ["primary"],
                    "admitted_doc_ids": ["secondary"],
                },
            }
        },
    )

    annotated = runtime_module._annotate_rerank_selection(
        selection, status="executed", fallback_reason=None
    )

    assert annotated[0].metadata["agentic_v9_reranking"][
        "candidate_diversification"
    ] == {
        "policy": "tail_source_diversity_r1",
        "enabled": True,
        "applied": True,
        "retrieved_doc_ids": ["primary", "secondary"],
        "candidate_doc_ids": ["primary", "secondary"],
        "represented_doc_ids_before_tail": ["primary"],
        "admitted_doc_ids": ["secondary"],
    }


def test_retrieval_diagnostic_projection_uses_chunk_projection_fallback_id() -> None:
    document = Document(
        page_content="selected content",
        metadata={
            "doc_id": "doc-1",
            "agentic_v9_reranking": {
                "status": "executed",
                "fallback_reason": None,
                "candidate_count": 8,
                "selected_count": 4,
                "pre_rerank_rank": 2,
                "post_rerank_rank": 1,
                "rerank_score": 0.93,
            },
        },
    )

    diagnostics = runtime_module._retrieval_diagnostic_projection(
        "task:source-group-1", [document]
    )

    assert diagnostics["selected"][0]["chunk_id"] == "task:source-group-1:chunk-1"
    assert (
        runtime_module._chunk_projection(
            document, 0, task_id="task:source-group-1"
        )["chunk_id"]
        == "task:source-group-1:chunk-1"
    )


def test_chunk_projection_preserves_reranking_and_typed_provenance() -> None:
    projection = runtime_module._chunk_projection(
        Document(
            page_content="selected content",
            metadata={
                "doc_id": "doc-1",
                "chunk_id": "chunk-7",
                "asset_id": "asset-1",
                "figure_id": "Figure 2",
                "agentic_v9_reranking": {
                    "status": "executed",
                    "post_rerank_rank": 2,
                    "rerank_score": 0.42,
                },
            },
        ),
        0,
    )

    assert projection["reranking"] == {
        "status": "executed",
        "post_rerank_rank": 2,
        "rerank_score": 0.42,
    }
    assert projection["asset_id"] == "asset-1"
    assert projection["figure_id"] == "Figure 2"


def test_chunk_projection_without_reranking_does_not_fabricate_a_score() -> None:
    projection = runtime_module._chunk_projection(
        Document(
            page_content="selected content",
            metadata={"doc_id": "doc-1", "chunk_id": "chunk-7"},
        ),
        0,
    )

    assert "reranking" not in projection


def test_rerank_quality_is_keyed_by_emitted_evidence_id() -> None:
    contract = QueryContract(
        route="exact_structured",
        intent="extract a value",
        required_slots=[RequiredSlot(slot_id="S1", description="value")],
        resolved_source_scope=ResolvedSourceScope(
            requested_doc_ids=["doc-1"],
            resolved_doc_ids=["doc-1"],
            authorized_doc_ids=["doc-1"],
        ),
    )
    projected = runtime_module._chunk_projection(
        Document(
            page_content="The result is 0.42.",
            metadata={
                "doc_id": "doc-1",
                "chunk_id": "chunk-7",
                "agentic_v9_reranking": {
                    "status": "executed",
                    "post_rerank_rank": 2,
                    "rerank_score": 0.42,
                },
            },
        ),
        0,
    )
    result = runtime_module._evidence_packets_for_results(
        results=(
            TaskRetrievalResult(
                task_id="task:source-group-1",
                retrieval=runtime_module.RagRetrievalResult(
                    retrieval_id="trace:task:source-group-1", chunks=[projected]
                ),
            ),
        ),
        contract=contract,
        trace_id="trace",
        task_slot_ids={},
    )

    assert len(result.packets) == 1
    assert result.quality_by_evidence_id == {result.packets[0].evidence_id: 0.5}


@pytest.mark.asyncio
async def test_v9_campaign_runtime_resolves_open_corpus_from_user_acl() -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="The source reports a score of 0.91.",
                metadata={"doc_id": "doc-2", "page_number": 2, "chunk_id": "chunk-1"},
            )
        ]
    )
    reference_resolver = AsyncMock()
    owned_document_ids_resolver = AsyncMock(return_value=["doc-2", "doc-1"])
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=reference_resolver,
        owned_document_ids_resolver=owned_document_ids_resolver,
    )

    result = await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=None,
        setup_snapshot=_setup(),
        trace_id="attempt-trace-open-corpus",
    )

    owned_document_ids_resolver.assert_awaited_once_with("user-a")
    reference_resolver.assert_not_awaited()
    retrieved_scope = retrieve_documents.await_args.args[2]
    assert retrieved_scope == ["doc-1", "doc-2"]
    assert result.agent_trace["execution_profile"] == AGENTIC_V9_OPEN_CORPUS_PROFILE
    assert result.agent_trace["execution_profile"] == (
        "agentic_eval_v9_open_corpus_hybrid8_rerank8_diverse_tail2_top4_finalpack_r1"
    )
    assert result.agent_trace["agentic_v9"]["retrieval_scope"] == {
        "policy": "open_user_corpus",
        "expected_sources_used_at_runtime": False,
    }
    assert result.agent_trace["agentic_v9"]["query_contract"]["resolved_source_scope"][
        "authorized_doc_ids"
    ] == ["doc-1", "doc-2"]
    assert result.documents


@pytest.mark.asyncio
async def test_v9_campaign_runtime_resolves_filename_scope_to_canonical_document_id() -> (
    None
):
    provider = _Provider()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="The source reports a score of 0.91.",
                metadata={"doc_id": "doc-1", "page_number": 2, "chunk_id": "chunk-1"},
            )
        ]
    )

    async def resolve_references(
        _user_id: str, references: list[str]
    ) -> dict[str, str]:
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

    assert result.agent_trace["agentic_v9"]["query_contract"]["resolved_source_scope"][
        "authorized_doc_ids"
    ] == ["doc-1"]
    assert result.agent_trace["agentic_v9"]["query_contract"]["resolved_source_scope"][
        "requested_doc_ids"
    ] == ["doc-1"]
    assert result.agent_trace["response_status"] == "complete"
    retrieve_documents.assert_awaited()


@pytest.mark.asyncio
async def test_v9_runtime_rejects_incompatible_setup_before_provider_or_retrieval() -> (
    None
):
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
    assert result.agent_trace["execution_profile"] == (
        "agentic_eval_v9_explicit_scope_hybrid8_rerank8_diverse_tail2_top4_finalpack_r1"
    )
    assert (
        result.agent_trace["agentic_v9"]["configuration_incompatible"]["stage"]
        == "pre_route"
    )
    assert result.documents == []
    retrieve_documents.assert_not_awaited()
    provider.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_v9_runtime_repeats_feasibility_after_contract_before_retrieval(
    monkeypatch,
) -> None:
    provider = _Provider()
    retrieve_documents = AsyncMock()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="exact_structured",
        intent="extract a table value",
        required_slots=[RequiredSlot(slot_id="S1", description="table value")],
        visual_required=True,
        evidence_extraction_required=True,
        max_llm_calls=2,
        runtime_token_budget=40_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )

    # The contract requires visual + evidence + final provider work but permits
    # only two calls.  It must be rejected before retrieval starts.
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


@pytest.mark.asyncio
async def test_required_graph_locator_is_executed_and_recorded_before_complete_answer(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="The relation is source-bound.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    graph_locator = AsyncMock(
        return_value=SimpleNamespace(
            documents=[document],
            resolved_source_documents=[document],
            resolved_source_doc_ids=["doc-1"],
            resolved_source_chunk_ids=["chunk-1"],
            candidate_item_ids=["graph-item-1"],
            resolved_item_ids=["graph-item-1"],
            scope_approved_item_ids=["graph-item-1"],
            scored_item_ids=["graph-item-1"],
            packed_item_ids=["graph-item-1"],
            route="local-first",
            path="source_expand",
            fallback=None,
            graph_latency_ms=7,
        )
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="graph_relational",
        intent="relation",
        required_slots=[RequiredSlot(slot_id="S1", description="relation")],
        graph_policy="required_locator",
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=4,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        graph_locator=graph_locator,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What relation is recorded?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="required-graph-trace",
    )

    assert result.agent_trace["response_status"] == "complete"
    assert result.agent_trace["agentic_v9"]["graph_execution"]["state"] == "executed"
    graph_locator.assert_awaited_once()


@pytest.mark.asyncio
async def test_required_graph_locator_without_source_evidence_keeps_text_complete(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="The vector result is not graph evidence.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    graph_locator = AsyncMock(
        return_value=SimpleNamespace(
            documents=[document],
            resolved_source_documents=[],
            resolved_source_doc_ids=[],
            resolved_source_chunk_ids=[],
            candidate_item_ids=["graph-item-1"],
            resolved_item_ids=[],
            scope_approved_item_ids=[],
            scored_item_ids=[],
            packed_item_ids=[],
            route="local-first",
            path="source_expand",
            fallback="no_source_bound_graph_evidence",
            graph_latency_ms=7,
        )
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="graph_relational",
        intent="relation",
        required_slots=[RequiredSlot(slot_id="S1", description="relation")],
        graph_policy="required_locator",
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=4,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        graph_locator=graph_locator,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What relation is recorded?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="missing-required-graph-trace",
    )

    graph_execution = result.agent_trace["agentic_v9"]["graph_execution"]
    assert result.agent_trace["response_status"] == "complete"
    assert graph_execution["state"] == "required_but_not_satisfied"
    assert graph_execution["failure_reason"] == "no_source_bound_graph_evidence"


@pytest.mark.asyncio
async def test_required_visual_evidence_is_recorded_before_complete_answer(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="Table 1 reports the result.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="exact_structured",
        intent="table value",
        required_slots=[RequiredSlot(slot_id="S1", description="table value")],
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    async def extract_visual(task, _documents, _question, _controller):
        return VisualEvidenceExtractionResult(
            packets=(
                EvidencePacket(
                    schema_version="1",
                    evidence_id="visual-evidence-1",
                    task_id=task.task_id,
                    round_id=task.round_id,
                    query_id=task.query_id,
                    slot_ids=list(task.target_slot_ids),
                    statement="The table reports 0.91.",
                    support_type="direct",
                    source=EvidenceSource(
                        doc_id="doc-1", chunk_id="chunk-1", asset_id="asset-1"
                    ),
                    scope=EvidenceScope(),
                    locator=SourceLocator(pdf_page_index=1, table_id="table-1"),
                    validation_status="deterministic_valid",
                ),
            )
        )

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        visual_extractor=extract_visual,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is in the table?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="required-visual-trace",
    )

    assert result.agent_trace["response_status"] == "complete"
    assert result.agent_trace["agentic_v9"]["visual_execution"]["state"] == "executed"


@pytest.mark.asyncio
async def test_missing_required_visual_evidence_keeps_text_complete(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="Table 1 reports the result.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="exact_structured",
        intent="table value",
        required_slots=[RequiredSlot(slot_id="S1", description="table value")],
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        visual_extractor=AsyncMock(return_value=VisualEvidenceExtractionResult()),
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is in the table?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="missing-required-visual-trace",
    )

    visual = result.agent_trace["agentic_v9"]["visual_execution"]
    assert result.agent_trace["response_status"] == "complete"
    assert visual["state"] == "required_but_not_satisfied"
    assert visual["failure_reason"] == "no_eligible_visual_evidence"


@pytest.mark.asyncio
async def test_required_visual_execution_error_remains_qualified_partial(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="Table 1 reports the result.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="exact_structured",
        intent="table value",
        required_slots=[RequiredSlot(slot_id="S1", description="table value")],
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    async def failing_visual_extractor(*_args):
        raise RuntimeError("visual provider unavailable")

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract", admission
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        visual_extractor=failing_visual_extractor,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is in the table?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="failed-required-visual-trace",
    )

    visual = result.agent_trace["agentic_v9"]["visual_execution"]
    assert result.agent_trace["response_status"] == "qualified_partial"
    assert visual["state"] == "required_but_not_satisfied"
    assert visual["failure_reason"] == "RuntimeError:stage_execution_failed"
