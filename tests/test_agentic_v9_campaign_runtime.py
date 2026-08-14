"""Production-adapter coverage for the Agentic v9 campaign path."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document

import evaluation.agentic_v9_campaign_runtime as runtime_module
from data_base.agentic_v9.contract_planner import (
    atomic_contract_planner_response_schema,
)
from data_base.rag_pipeline_schemas import RagRetrievalResult as PipelineRetrievalResult
from data_base.reranker import DocumentReranker
from evaluation.agentic_v9_campaign_runtime import AgenticV9CampaignRuntime
from evaluation.agentic_v9_admission import V9AdmissionContract
from evaluation.campaign_schemas import V9ContextPack
from evaluation.retrieval_profiles import AGENTIC_V9_OPEN_CORPUS_PROFILE
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    LlmInvoker,
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
    SourceLocator,
    TaskRetrievalResult,
)
from data_base.agentic_v9.visual_evidence_extractor import (
    VisualEvidenceExtractionResult,
)
from data_base.rag_graph_locator import GraphSourceLocatorResult


class _Provider:
    def __init__(self) -> None:
        self.ainvoke = AsyncMock(
            return_value=SimpleNamespace(
                content="The reported score is 0.91.",
                usage_metadata={"input_tokens": 12, "output_tokens": 7},
            )
        )


@pytest.mark.asyncio
async def test_atomic_contract_planner_provider_binds_schema_without_replacing_raw_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = SimpleNamespace(
        content='{"evidence_requirements":[],"synthesis_obligations":[],"response_constraints":[],"comparison":null,"confidence":1.0}',
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )
    captured: dict[str, object] = {}

    class _BindableProvider:
        def bind(self, **kwargs: object) -> "_BindableProvider":
            captured.update(kwargs)
            return self

        async def ainvoke(self, messages: object) -> object:
            del messages
            return raw

    monkeypatch.setattr(runtime_module, "get_llm", lambda purpose: _BindableProvider())

    provider = runtime_module._provider_for_purpose("atomic_contract_planning")
    response = await provider.ainvoke([])

    assert captured["response_mime_type"] == "application/json"
    assert captured["response_schema"] == atomic_contract_planner_response_schema()
    assert response is raw
    assert response.usage_metadata["total_tokens"] == 10


def test_noncomparison_provider_is_not_schema_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnboundProvider:
        def bind(self, **kwargs: object) -> object:
            raise AssertionError(f"unexpected schema binding: {kwargs}")

    provider = _UnboundProvider()
    monkeypatch.setattr(runtime_module, "get_llm", lambda purpose: provider)

    assert runtime_module._provider_for_purpose("final_answer") is provider


class _RecordingObserver:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.partial_reasons: list[str] = []

    async def on_terminal_attempt(self, observation: object) -> bool:
        self.calls.append(observation)
        return True

    def mark_partial(self, reason: str) -> None:
        self.partial_reasons.append(reason)


def _setup() -> dict[str, object]:
    return {
        "max_input_tokens": 4096,
        "max_output_tokens": 256,
        "thinking_mode": False,
    }


def test_requirement_guided_runtime_setup_flag_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTIC_V9_REQUIREMENT_GUIDED_RUNTIME", "on")

    assert runtime_module._resolve_requirement_guided_runtime(
        {"requirement_guided_runtime": False}
    ) == (False, "setup_snapshot", None)


def test_requirement_guided_runtime_reads_environment_when_setup_omits_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTIC_V9_REQUIREMENT_GUIDED_RUNTIME", "on")

    assert runtime_module._resolve_requirement_guided_runtime({}) == (
        True,
        "environment",
        None,
    )


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
async def test_v9_graph_route_usage_is_budgeted_observed_and_reconciled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        SimpleNamespace(
            content='{"query_kind":"relation","path":"local-first"}',
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
            },
        ),
        SimpleNamespace(
            content="Graph-aware evidence answer.",
            usage_metadata={
                "input_tokens": 12,
                "output_tokens": 7,
                "total_tokens": 19,
            },
        ),
    ]
    observer = _RecordingObserver()
    source_document = Document(
        page_content="Source-backed relationship evidence.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    retrieve_documents = AsyncMock(return_value=[source_document])
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="graph_relational",
        intent="Trace a relationship.",
        required_slots=[RequiredSlot(slot_id="base", description="relationship")],
        graph_policy="required_locator",
        max_retrieval_rounds=1,
        max_llm_calls=2,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    async def observed_graph_locator(
        question: str,
        user_id: str,
        vector_documents: list[Document],
        authorized_doc_ids: list[str],
        runtime_contract: QueryContract,
        *,
        llm_invoker: LlmInvoker | None = None,
    ) -> GraphSourceLocatorResult:
        assert question
        assert user_id == "user-a"
        assert authorized_doc_ids == ["doc-1"]
        assert runtime_contract.route == "graph_relational"
        assert llm_invoker is not None
        await llm_invoker.invoke(
            phase="graph_route",
            purpose="graph_extraction",
            messages=[{"role": "user", "content": question}],
        )
        return GraphSourceLocatorResult(
            documents=vector_documents,
            resolved_source_documents=vector_documents,
            resolved_source_doc_ids=["doc-1"],
            resolved_source_chunk_ids=["chunk-1"],
            candidate_item_ids=[],
            resolved_item_ids=[],
            scope_approved_item_ids=[],
            scored_item_ids=[],
            packed_item_ids=[],
            route="local-first",
            path="source_expand",
            fallback=None,
            graph_latency_ms=1,
            bundle=None,
            chunk_lookup=SimpleNamespace(),
            resolved_chunks=[],
            scoped_chunks=[],
            graph_documents=[],
        )

    monkeypatch.setattr(runtime_module, "build_v9_admission_contract", admission)
    monkeypatch.setattr(
        runtime_module,
        "_locate_graph_documents",
        observed_graph_locator,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        llm_call_observer=observer,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Trace the relationship path from ModelA to ModelB.",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="observed-graph-route",
    )

    assert [call.phase for call in observer.calls] == [
        "graph_route",
        "final_answer",
    ]
    assert sum(call.usage["total_tokens"] for call in observer.calls) == 26
    assert result.usage["total_tokens"] == 26
    assert observer.partial_reasons == []
    assert result.agent_trace["agentic_v9"]["retrieval_diagnostics"]
    assert result.agent_trace["execution_profile"] == (
        runtime_module.agentic_v9_execution_profile(open_user_corpus=False)
    )
    assert result.agent_trace["context_policy_version"] == (
        runtime_module.AGENTIC_V9_CONTEXT_POLICY_VERSION
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
        "agentic_eval_v9_explicit_scope_hybrid8_rerank8_diverse_tail2_top4_"
        "finalpack_r1_comparison_structured_v2"
    )
    assert v9["query_contract"]["contract_version"] == "2"
    assert v9["metrics"]["atomic_planner_call_count"] <= 1
    assert v9["metrics"]["comparison_planner_call_count"] == 0
    assert v9["metrics"]["slot_binding_method"] == "task_target_inherited"
    assert v9["metrics"]["semantic_qualification"] == "not_enabled"
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
    assert provider.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_v9_comparison_planner_overlays_subject_tasks_and_caps_each_at_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        SimpleNamespace(
            content=json.dumps(
                {
                    "evidence_requirements": [
                        {
                            "description": "nnMamba parameters and FLOPs",
                            "source_name_hints": [],
                            "locator_hints": [],
                            "expected_answer_type": "text",
                            "depends_on_requirement_indexes": [],
                            "visual_policy": "never",
                        },
                        {
                            "description": "EfficientMedNeXt-L parameters and FLOPs",
                            "source_name_hints": [],
                            "locator_hints": [],
                            "expected_answer_type": "text",
                            "depends_on_requirement_indexes": [],
                            "visual_policy": "never",
                        },
                    ],
                    "synthesis_obligations": [
                        {
                            "kind": "comparison",
                            "description": "Compare nnMamba and EfficientMedNeXt-L",
                            "depends_on_requirement_indexes": [0, 1],
                        }
                    ],
                    "response_constraints": [],
                    "comparison": {
                        "subjects": [
                            {
                                "subject_id": "nnmamba",
                                "display_name": "nnMamba",
                                "aliases": [],
                                "retrieval_query": "nnMamba parameters FLOPs",
                                "evidence_requirement_indexes": [0],
                            },
                            {
                                "subject_id": "efficientmednext-l",
                                "display_name": "EfficientMedNeXt-L",
                                "aliases": [],
                                "retrieval_query": "EfficientMedNeXt-L parameters FLOPs",
                                "evidence_requirement_indexes": [1],
                            },
                        ],
                        "dimensions": ["parameters", "FLOPs"],
                        "qualification": None,
                    },
                    "confidence": 0.95,
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
        max_llm_calls=2,
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
            else "efficientmednext-l"
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
    assert v9["query_contract"]["contract_version"] == "2"
    assert [
        subject["subject_id"]
        for subject in v9["query_contract"]["comparison_plan"]["subjects"]
    ] == ["nnmamba", "efficientmednext-l"]
    assert retrieve_documents.await_count == 2
    assert [row["selected_count"] for row in v9["retrieval_diagnostics"]] == [2, 2]
    assert v9["metrics"]["atomic_planner_call_count"] == 1
    assert v9["metrics"]["comparison_planner_call_count"] == 0
    assert v9["metrics"]["slot_binding_method"] == "task_target_inherited"
    assert v9["metrics"]["semantic_qualification"] == "not_enabled"
    assert v9["comparison"]["planner_status"] == "planned"
    assert v9["comparison"]["coverage_before_repair"] == ["nnmamba", "efficientmednext-l"]
    assert v9["comparison"]["coverage_after_repair"] == ["nnmamba", "efficientmednext-l"]
    assert v9["comparison"]["final_status"] == "complete"
    assert v9["comparison"]["final_evidence_subjects"] == ["nnmamba", "efficientmednext-l"]
    assert v9["comparison"]["final_evidence_count"] == 4
    assert provider.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_invalid_comparison_subjects_preserve_base_contract_and_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        SimpleNamespace(
            content=json.dumps(
                {
                    "evidence_requirements": [
                        {
                            "description": "MedSAM-2 claim evidence",
                            "source_name_hints": [],
                            "locator_hints": [],
                            "expected_answer_type": "text",
                            "depends_on_requirement_indexes": [],
                            "visual_policy": "never",
                        }
                    ],
                    "synthesis_obligations": [],
                    "response_constraints": [],
                    "comparison": None,
                    "confidence": 0.9,
                }
            ),
            usage_metadata={"input_tokens": 20, "output_tokens": 10},
        ),
        SimpleNamespace(
            content="The evidence supports a qualified answer.",
            usage_metadata={"input_tokens": 12, "output_tokens": 7},
        ),
    ]
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="MedSAM-2 evidence.",
                metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
            )
        ]
    )
    scope = ResolvedSourceScope(
        requested_doc_ids=["doc-1"],
        resolved_doc_ids=["doc-1"],
        authorized_doc_ids=["doc-1"],
    )
    contract = QueryContract(
        route="single_lookup",
        intent="Check claims about one model.",
        required_slots=[RequiredSlot(slot_id="base", description="claim evidence")],
        max_retrieval_rounds=1,
        max_llm_calls=2,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(runtime_module, "build_v9_admission_contract", admission)
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question=(
            "Compare these two claims about MedSAM-2: support for single-prompt "
            "segmentation and sensitivity to initial bounding box prompt quality."
        ),
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="invalid-comparison-subjects",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert v9["query_contract"]["contract_version"] == "2"
    assert v9["query_contract"].get("comparison_plan") is None
    assert v9["metrics"]["atomic_planner_call_count"] == 1
    assert v9["metrics"]["comparison_planner_call_count"] == 0
    assert v9["metrics"]["slot_binding_method"] == "task_target_inherited"
    assert v9["metrics"]["semantic_qualification"] == "not_enabled"
    assert retrieve_documents.await_count == 1
    assert result.documents
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
                    "evidence_requirements": [
                        {
                            "description": "Model A accuracy",
                            "source_name_hints": [],
                            "locator_hints": [],
                            "expected_answer_type": "text",
                            "depends_on_requirement_indexes": [],
                            "visual_policy": "never",
                        },
                        {
                            "description": "Model B accuracy",
                            "source_name_hints": [],
                            "locator_hints": [],
                            "expected_answer_type": "text",
                            "depends_on_requirement_indexes": [],
                            "visual_policy": "never",
                        },
                    ],
                    "synthesis_obligations": [
                        {
                            "kind": "comparison",
                            "description": "Compare Model A and Model B accuracy",
                            "depends_on_requirement_indexes": [0, 1],
                        }
                    ],
                    "response_constraints": [],
                    "comparison": {
                        "subjects": [
                            {
                                "subject_id": "model-a",
                                "display_name": "Model A",
                                "aliases": [],
                                "retrieval_query": "Model A accuracy",
                                "evidence_requirement_indexes": [0],
                            },
                            {
                                "subject_id": "model-b",
                                "display_name": "Model B",
                                "aliases": [],
                                "retrieval_query": "Model B accuracy",
                                "evidence_requirement_indexes": [1],
                            },
                        ],
                        "dimensions": ["accuracy"],
                        "qualification": None,
                    },
                    "confidence": 0.95,
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
        max_repair_rounds=1,
        max_llm_calls=2,
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
            subject = "model-b"
            doc_id = "doc-b"
        else:
            subject = "model-a"
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
    assert v9["repairs"][0]["tasks"][0]["subject_id"] == "model-b"
    assert v9["comparison"]["coverage_before_repair"] == ["model-a"]
    assert v9["comparison"]["missing_before_repair"] == ["model-b"]
    assert v9["comparison"]["repair_executed"] is True
    assert v9["comparison"]["final_status"] == expected_status
    if repair_succeeds:
        assert v9["comparison"]["coverage_after_repair"] == [
            "model-a",
            "model-b",
        ]
        assert v9["comparison"]["missing_after_repair"] == []
    else:
        assert v9["comparison"]["coverage_after_repair"] == ["model-a"]
        assert v9["comparison"]["missing_after_repair"] == ["model-b"]


@pytest.mark.asyncio
async def test_v9_comparison_status_uses_final_balanced_packet_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        SimpleNamespace(
            content=json.dumps(
                {
                    "evidence_requirements": [
                        {
                            "description": "Model A accuracy",
                            "source_name_hints": [],
                            "locator_hints": [],
                            "expected_answer_type": "text",
                            "depends_on_requirement_indexes": [],
                            "visual_policy": "never",
                        },
                        {
                            "description": "Model B accuracy",
                            "source_name_hints": [],
                            "locator_hints": [],
                            "expected_answer_type": "text",
                            "depends_on_requirement_indexes": [],
                            "visual_policy": "never",
                        },
                    ],
                    "synthesis_obligations": [
                        {
                            "kind": "comparison",
                            "description": "Compare Model A and Model B accuracy",
                            "depends_on_requirement_indexes": [0, 1],
                        }
                    ],
                    "response_constraints": [],
                    "comparison": {
                        "subjects": [
                            {
                                "subject_id": "model-a",
                                "display_name": "Model A",
                                "aliases": [],
                                "retrieval_query": "Model A accuracy",
                                "evidence_requirement_indexes": [0],
                            },
                            {
                                "subject_id": "model-b",
                                "display_name": "Model B",
                                "aliases": [],
                                "retrieval_query": "Model B accuracy",
                                "evidence_requirement_indexes": [1],
                            },
                        ],
                        "dimensions": ["accuracy"],
                        "qualification": None,
                    },
                    "confidence": 0.95,
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
        max_repair_rounds=1,
        max_llm_calls=2,
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
    assert retrieve_documents.await_count == 3
    assert len(v9["repairs"]) == 1
    assert len(v9["repairs"][0]["tasks"]) == 1
    assert v9["repairs"][0]["tasks"][0]["subject_id"] == "model-b"


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
    assert v9["query_contract"]["contract_version"] == "2"
    assert v9["metrics"]["atomic_planner_call_count"] == 1
    assert v9["metrics"]["comparison_planner_call_count"] == 0
    assert v9["metrics"]["slot_binding_method"] == "task_target_inherited"
    assert v9["metrics"]["semantic_qualification"] == "not_enabled"
    assert result.documents
    retrieve_documents.assert_awaited()
    assert provider.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_v9_comparison_transport_diagnostics_reach_agent_trace() -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        SimpleNamespace(
            content="not valid json",
            usage_metadata={"input_tokens": 10, "output_tokens": 5},
        ),
        SimpleNamespace(
            content="Fallback answer from retrieved evidence.",
            usage_metadata={"input_tokens": 12, "output_tokens": 7},
        ),
    ]
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="The source contains usable comparison evidence.",
                    metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
                )
            ]
        ),
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Compare Model A vs. Model B for accuracy.",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="comparison-transport-diagnostics",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert v9["query_contract"]["contract_version"] == "2"
    assert v9["metrics"]["atomic_planner_call_count"] == 1
    assert v9["metrics"]["comparison_planner_call_count"] == 0
    assert provider.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_atomic_contract_planning_high_confidence_deterministic_zero_planner_calls() -> None:
    observed_calls: list[dict[str, Any]] = []

    class _RecordingObsProvider:
        async def ainvoke(self, messages: Any) -> Any:
            observed_calls.append({"messages": messages})
            return SimpleNamespace(
                content="The reported score is 0.91 and the method is described.",
                usage_metadata={"input_tokens": 12, "output_tokens": 7},
            )

    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="Table 1 reports a score of 0.91.",
                    metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
                )
            ]
        ),
        provider_factory=lambda _purpose: _RecordingObsProvider(),
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="1. What is the reported score in Table 1? 2. What is the method?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="atomic-case-1-deterministic",
    )

    v9 = result.agent_trace["agentic_v9"]
    contract = v9["query_contract"]
    assert contract["contract_version"] == "2"
    assert [slot["slot_id"] for slot in contract["required_slots"]] == ["S1", "S2"]
    assert v9["metrics"]["atomic_planner_call_count"] == 0
    assert v9["metrics"]["comparison_planner_call_count"] == 0
    assert v9["metrics"]["slot_binding_method"] == "task_target_inherited"
    assert v9["metrics"]["semantic_qualification"] == "not_enabled"
    assert len(observed_calls) == 1
    # Assert final prompt message is strictly Question: ...\n\nEvidence:\n... unchanged
    final_messages = observed_calls[0]["messages"]
    assert final_messages[0]["role"] == "system"
    assert final_messages[1]["role"] == "user"
    assert final_messages[1]["content"].startswith(
        "Question: 1. What is the reported score in Table 1? 2. What is the method?\n\nEvidence:\n"
    )


@pytest.mark.asyncio
async def test_atomic_contract_planning_low_confidence_comparison_one_planner_call() -> None:
    recorded_calls: list[dict[str, Any]] = []

    class _PlannerAndAnswerProvider:
        async def ainvoke(self, messages: Any) -> Any:
            recorded_calls.append({"messages": messages})
            if len(recorded_calls) == 1:
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "evidence_requirements": [
                                {
                                    "description": "Model A efficiency and speed",
                                    "source_name_hints": [],
                                    "locator_hints": [],
                                    "expected_answer_type": "text",
                                    "depends_on_requirement_indexes": [],
                                    "visual_policy": "never",
                                },
                                {
                                    "description": "Model B efficiency and speed",
                                    "source_name_hints": [],
                                    "locator_hints": [],
                                    "expected_answer_type": "text",
                                    "depends_on_requirement_indexes": [],
                                    "visual_policy": "never",
                                },
                            ],
                            "synthesis_obligations": [
                                {
                                    "kind": "comparison",
                                    "description": "Compare Model A and Model B",
                                    "depends_on_requirement_indexes": [0, 1],
                                }
                            ],
                            "response_constraints": [],
                            "comparison": {
                                "subjects": [
                                    {
                                        "subject_id": "model_a",
                                        "display_name": "Model A",
                                        "aliases": [],
                                        "retrieval_query": "Model A efficiency speed",
                                        "evidence_requirement_indexes": [0],
                                    },
                                    {
                                        "subject_id": "model_b",
                                        "display_name": "Model B",
                                        "aliases": [],
                                        "retrieval_query": "Model B efficiency speed",
                                        "evidence_requirement_indexes": [1],
                                    },
                                ],
                                "dimensions": ["efficiency", "speed"],
                                "qualification": None,
                            },
                            "confidence": 0.9,
                        }
                    ),
                    usage_metadata={"input_tokens": 20, "output_tokens": 15},
                )
            return SimpleNamespace(
                content="Model A is more efficient while Model B is faster.",
                usage_metadata={"input_tokens": 30, "output_tokens": 10},
            )

    async def retrieve_subject_docs(
        _user_id: str, query: str, _authorized_doc_ids: list[str]
    ) -> list[Document]:
        subject = "model-a" if "Model A" in query else "model-b"
        return [
            Document(
                page_content=f"{subject} comparison evidence.",
                metadata={"doc_id": "doc-1", "chunk_id": f"{subject}-chunk-1"},
            )
        ]

    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(side_effect=retrieve_subject_docs),
        provider_factory=lambda _purpose: _PlannerAndAnswerProvider(),
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Compare the efficiency and speed of Model A versus Model B in depth.",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={
            "max_input_tokens": 4096,
            "max_output_tokens": 8192,
            "max_llm_calls": 5,
            "runtime_token_budget": 50_000,
            "thinking_mode": False,
        },
        trace_id="atomic-case-2-comparison",
    )

    v9 = result.agent_trace["agentic_v9"]
    contract = v9["query_contract"]
    assert contract["contract_version"] == "2"
    assert [slot["slot_id"] for slot in contract["required_slots"]] == ["S1", "S2"]
    assert v9["metrics"]["atomic_planner_call_count"] == 1
    assert v9["metrics"]["comparison_planner_call_count"] == 0
    assert v9["metrics"]["slot_binding_method"] == "task_target_inherited"
    assert v9["metrics"]["semantic_qualification"] == "not_enabled"
    assert len(recorded_calls) == 2


@pytest.mark.asyncio
async def test_atomic_contract_planning_budget_rejection_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data_base.agentic_v9.budget_feasibility import (
        FeasibilityResult,
        FeasibilityStatus,
    )

    recorded_calls: list[dict[str, Any]] = []

    class _SingleCallProvider:
        async def ainvoke(self, messages: Any) -> Any:
            recorded_calls.append({"messages": messages})
            return SimpleNamespace(
                content="Degraded final answer from retrieved evidence.",
                usage_metadata={"input_tokens": 15, "output_tokens": 8},
            )

    orig_post = runtime_module.validate_post_contract_feasibility

    def mock_post(*args: Any, **kwargs: Any) -> FeasibilityResult:
        if kwargs.get("contract_plan_requested"):
            return FeasibilityResult(
                status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
                reason="budget_exhausted",
                required_provider_calls={},
                max_provider_calls_by_phase={},
                max_tool_operations=0,
                reserved_tokens=0,
            )
        return orig_post(*args, **kwargs)

    monkeypatch.setattr(
        runtime_module, "validate_post_contract_feasibility", mock_post
    )

    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="Reported data in source.",
                    metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
                )
            ]
        ),
        provider_factory=lambda _purpose: _SingleCallProvider(),
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Compare the performance of system X versus system Y across multiple benchmarks.",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="atomic-case-3-budget-rejection",
    )

    assert result.agent_trace["response_status"] == "complete"
    v9 = result.agent_trace["agentic_v9"]
    contract = v9["query_contract"]
    assert contract["contract_version"] == "2"
    assert v9["metrics"]["atomic_planner_call_count"] == 0
    assert v9["metrics"]["comparison_planner_call_count"] == 0
    assert v9["metrics"]["slot_binding_method"] == "task_target_inherited"
    assert v9["metrics"]["semantic_qualification"] == "not_enabled"
    assert len(recorded_calls) == 1


@pytest.mark.asyncio
async def test_atomic_contract_planning_malformed_response_degrades_gracefully() -> None:
    recorded_calls: list[dict[str, Any]] = []

    class _MalformedThenAnswerProvider:
        async def ainvoke(self, messages: Any) -> Any:
            recorded_calls.append({"messages": messages})
            if len(recorded_calls) == 1:
                return SimpleNamespace(
                    content="{invalid json syntax",
                    usage_metadata={"input_tokens": 10, "output_tokens": 5},
                )
            return SimpleNamespace(
                content="Recovered answer from baseline retrieved evidence.",
                usage_metadata={"input_tokens": 20, "output_tokens": 8},
            )

    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="Recovered source evidence.",
                    metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
                )
            ]
        ),
        provider_factory=lambda _purpose: _MalformedThenAnswerProvider(),
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Compare Model Alpha and Model Beta on throughput and accuracy.",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={
            "max_input_tokens": 4096,
            "max_output_tokens": 8192,
            "max_llm_calls": 5,
            "runtime_token_budget": 50_000,
            "thinking_mode": False,
        },
        trace_id="atomic-case-4-malformed",
    )

    assert result.agent_trace["response_status"] == "complete"
    v9 = result.agent_trace["agentic_v9"]
    contract = v9["query_contract"]
    assert contract["contract_version"] == "2"
    assert v9["metrics"]["atomic_planner_call_count"] == 1
    assert v9["metrics"]["comparison_planner_call_count"] == 0
    assert v9["metrics"]["slot_binding_method"] == "task_target_inherited"
    assert v9["metrics"]["semantic_qualification"] == "not_enabled"
    assert len(recorded_calls) == 2



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
async def test_v9_runtime_persists_requirement_shadow_without_influencing_behavior() -> (
    None
):
    provider = _Provider()
    document = Document(
        page_content="The source reports a score of 0.91.",
        metadata={"doc_id": "doc-1", "page_number": 2, "chunk_id": "chunk-1"},
    )
    retrieve_documents = AsyncMock(return_value=[document])
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
        trace_id="attempt-trace-requirement-shadow",
    )

    v9 = result.agent_trace["agentic_v9"]
    shadow = v9["requirement_shadow"]
    assert shadow["schema_version"] == "shadow_requirements_v2"
    assert shadow["behavior_influence"] is False
    assert shadow["support_assessment"] == "candidate_only"
    assert shadow["summary"]["requirement_count"] == 1
    assert shadow["requirements"][0]["candidate_evidence_refs"] == [
        "doc-1:chunk-1"
    ]
    assert v9["visual_execution"]["state"] == "not_requested"
    assert result.agent_trace["response_status"] == "complete"
    assert result.documents
    assert provider.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_v9_requirement_guided_runtime_defaults_off_and_keeps_baseline_query() -> None:
    provider = _Provider()
    document = Document(
        page_content="The source reports a score of 0.91.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    retrieve_documents = AsyncMock(return_value=[document])
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="attempt-trace-requirement-guidance-off",
    )

    assert retrieve_documents.await_count == 1
    assert "Advisory answer obligations" not in retrieve_documents.await_args.args[1]
    assert provider.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_v9_requirement_guided_runtime_on_adds_advisory_without_extra_llm_call() -> None:
    provider = _Provider()
    document = Document(
        page_content="The source reports a score of 0.91.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    retrieve_documents = AsyncMock(return_value=[document])
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    await runtime.execute(
        question="1. What is the reported score? 2. What is the method?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "requirement_guided_runtime": True},
        trace_id="attempt-trace-requirement-guidance-on",
    )

    assert retrieve_documents.await_count >= 1
    for call in retrieve_documents.await_args_list:
        assert "Advisory answer obligations" not in call.args[1]
    assert provider.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_requirement_shadow_failure_cannot_fail_or_downgrade_the_run(
    monkeypatch,
) -> None:
    provider = _Provider()
    document = Document(
        page_content="The source reports a score of 0.91.",
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(return_value=[document]),
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )
    monkeypatch.setattr(
        runtime_module,
        "build_requirement_shadow",
        Mock(side_effect=RuntimeError("shadow analyzer failed")),
        raising=False,
    )

    result = await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="attempt-trace-requirement-shadow-failure",
    )

    shadow = result.agent_trace["agentic_v9"]["requirement_shadow"]
    assert result.agent_trace["response_status"] == "complete"
    assert shadow == {
        "schema_version": "shadow_requirements_v2",
        "behavior_influence": False,
        "status": "unavailable",
        "reason": "diagnostic_projection_failed",
    }


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
        "agentic_eval_v9_open_corpus_hybrid8_rerank8_diverse_tail2_top4_"
        "finalpack_r1_comparison_structured_v2"
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
        "agentic_eval_v9_explicit_scope_hybrid8_rerank8_diverse_tail2_top4_"
        "finalpack_r1_comparison_structured_v2"
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
