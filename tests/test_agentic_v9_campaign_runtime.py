"""Production-adapter coverage for the Agentic v9 campaign path."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document

import evaluation.agentic_v9_campaign_runtime as runtime_module
from data_base.agentic_v9.comparison_planner import (
    comparison_planner_response_schema,
)
from data_base.agentic_v9.execution_policy import V9ExecutionPolicyRuntime
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
    ExecutionPolicy,
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
        self.ainvoke = AsyncMock(side_effect=self._respond)

    @staticmethod
    def _respond(messages: list[dict[str, Any]]) -> object:
        system = str(messages[0].get("content", ""))
        if "curate only the remaining prose evidence slots" in system:
            purpose = "evidence_extraction"
        elif "Return JSON with exactly supported_findings" in system:
            purpose = "final_answer"
        elif "Verify only the listed claims" in system:
            purpose = "claim_verifier"
        else:
            return SimpleNamespace(
                content="The reported score is 0.91.",
                usage_metadata={"input_tokens": 12, "output_tokens": 7},
            )
        return _StructuredProviderFactory().respond(purpose, messages)


def _set_nonstructured_responses(
    provider: _Provider, *responses: object
) -> None:
    pending = iter(responses)

    async def respond(messages: list[dict[str, Any]]) -> object:
        system = str(messages[0].get("content", ""))
        if any(
            marker in system
            for marker in (
                "curate only the remaining prose evidence slots",
                "Return JSON with exactly supported_findings",
                "Verify only the listed claims",
            )
        ):
            return _Provider._respond(messages)
        response = next(pending)
        if isinstance(response, BaseException):
            raise response
        return response

    provider.ainvoke.side_effect = respond


class _PurposeProvider:
    def __init__(self, owner: "_StructuredProviderFactory", purpose: str) -> None:
        self._owner = owner
        self._purpose = purpose

    async def ainvoke(self, messages: list[dict[str, Any]]) -> object:
        return self._owner.respond(self._purpose, messages)


class _StructuredProviderFactory:
    """Production-shaped provider double for v9 qualification and synthesis."""

    _SOURCE_LINE = re.compile(
        r"^(evidence:[^ ]+) \[eligible slots: ([^\]]+)\]: (.+)$"
    )

    def __init__(
        self,
        *,
        qualification_slots_by_round: list[set[str] | None] | None = None,
        final_slots: set[str] | None = None,
        final_statement_by_slot: dict[str, str] | None = None,
        verifier_supported: bool = True,
    ) -> None:
        self.qualification_slots_by_round = qualification_slots_by_round or []
        self.final_slots = final_slots
        self.final_statement_by_slot = final_statement_by_slot or {}
        self.verifier_supported = verifier_supported
        self.purposes: list[str] = []
        self.qualification_outputs: list[dict[str, object]] = []
        self.final_payloads: list[dict[str, Any]] = []
        self._qualification_round = 0

    def __call__(self, purpose: str) -> _PurposeProvider:
        self.purposes.append(purpose)
        return _PurposeProvider(self, purpose)

    def respond(self, purpose: str, messages: list[dict[str, Any]]) -> object:
        if purpose == "evidence_extraction":
            content = self._qualification_response(messages)
        elif purpose in {"final_answer", "agentic_v9_final_answer"}:
            content = self._final_response(messages)
        elif purpose == "claim_verifier":
            content = self._verifier_response(messages)
        else:
            raise AssertionError(f"unexpected provider purpose: {purpose}")
        return SimpleNamespace(
            content=json.dumps(content),
            usage_metadata={
                "input_tokens": 2,
                "output_tokens": 1,
                "total_tokens": 3,
            },
        )

    def _qualification_response(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, object]:
        selected_slots = (
            self.qualification_slots_by_round[self._qualification_round]
            if self._qualification_round < len(self.qualification_slots_by_round)
            else None
        )
        self._qualification_round += 1
        packets: list[dict[str, object]] = []
        for line in str(messages[-1]["content"]).splitlines():
            match = self._SOURCE_LINE.fullmatch(line)
            if match is None:
                continue
            evidence_id, raw_slots, statement = match.groups()
            slot_ids = [
                slot_id
                for slot_id in raw_slots.split(",")
                if selected_slots is None or slot_id in selected_slots
            ]
            if slot_ids:
                packets.append(
                    {
                        "source_evidence_id": evidence_id,
                        "slot_ids": slot_ids,
                        "statement": statement,
                    }
                )
        result: dict[str, object] = {"packets": packets}
        self.qualification_outputs.append(result)
        return result

    def _final_response(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, object]:
        payload = json.loads(str(messages[-1]["content"]))
        self.final_payloads.append(payload)
        packets = payload["packed_evidence_packets"]
        findings: list[dict[str, object]] = []
        unresolved: list[dict[str, str]] = []
        for slot in payload["contract"]["required_slots"]:
            slot_id = slot["slot_id"]
            packet = next(
                (
                    item
                    for item in packets
                    if slot_id in item["slot_ids"]
                ),
                None,
            )
            if (
                packet is None
                or self.final_slots is not None
                and slot_id not in self.final_slots
            ):
                unresolved.append(
                    {"slot_id": slot_id, "reason": "qualified evidence unavailable"}
                )
                continue
            findings.append(
                {
                    "slot_id": slot_id,
                    "statement": self.final_statement_by_slot.get(
                        slot_id, packet["statement"]
                    ),
                    "support_type": "direct",
                    "evidence_ids": [packet["evidence_id"]],
                    "premise_evidence_ids": [],
                }
            )
        return {
            "supported_findings": findings,
            "unresolved_requirements": unresolved,
        }

    def _verifier_response(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, object]:
        payload = json.loads(str(messages[-1]["content"]))
        return {
            "verdicts": [
                {
                    "claim_id": claim["claim_id"],
                    "supported": self.verifier_supported,
                    "reason": None if self.verifier_supported else "not supported",
                }
                for claim in payload["claims"]
            ]
        }


@pytest.mark.asyncio
async def test_comparison_provider_binds_schema_without_replacing_raw_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = SimpleNamespace(
        content='{"is_comparison":false,"subjects":[],"dimensions":[]}',
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

    provider = runtime_module._provider_for_purpose("agentic_v9_comparison_plan")
    response = await provider.ainvoke([])

    assert captured["response_mime_type"] == "application/json"
    assert captured["response_schema"] == comparison_planner_response_schema()
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
    _set_nonstructured_responses(
        provider,
        SimpleNamespace(
            content='{"query_kind":"relation","path":"local-first"}',
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
            },
        ),
    )
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
        max_llm_calls=3,
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
        "evidence_extract",
        "final_answer",
    ]
    assert sum(call.usage["total_tokens"] for call in observer.calls) == 13
    assert result.usage["total_tokens"] == 13
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
    _set_nonstructured_responses(
        provider,
        SimpleNamespace(
            content=json.dumps(
                {
                    "is_comparison": True,
                        "subjects": [
                            {
                                "name": "nnMamba",
                                "query": "nnMamba parameters FLOPs",
                            },
                            {
                                "name": "EfficientMedNeXt-L",
                                "query": "EfficientMedNeXt-L parameters FLOPs",
                        },
                    ],
                    "dimensions": ["parameters", "FLOPs"],
                    "qualification": None,
                }
            ),
            usage_metadata={"input_tokens": 20, "output_tokens": 10},
        ),
    )
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
    assert [
        subject["subject_id"]
        for subject in v9["query_contract"]["comparison_plan"]["subjects"]
    ] == ["nnmamba", "efficientmednext-l"]
    assert retrieve_documents.await_count == 2
    assert [row["selected_count"] for row in v9["retrieval_diagnostics"]] == [2, 2]
    assert v9["comparison_planner"]["status"] == "planned"
    assert v9["comparison"] == {
        "planner_status": "planned",
        "planner_latency_ms": v9["comparison"]["planner_latency_ms"],
        "planner_fallback_reason": None,
        "fallback_stage": None,
        "validation_issues": [],
        "is_comparison": True,
        "subjects": [
            {
                "subject_id": "nnmamba",
                "display_name": "nnMamba",
                "aliases": [],
                "retrieval_query": "nnMamba parameters FLOPs",
            },
            {
                "subject_id": "efficientmednext-l",
                "display_name": "EfficientMedNeXt-L",
                "aliases": [],
                "retrieval_query": "EfficientMedNeXt-L parameters FLOPs",
            },
        ],
        "dimensions": ["parameters", "FLOPs"],
        "task_diagnostics": v9["comparison"]["task_diagnostics"],
        "coverage_before_repair": ["nnmamba", "efficientmednext-l"],
        "missing_before_repair": [],
        "repair_executed": False,
        "coverage_after_repair": ["nnmamba", "efficientmednext-l"],
        "missing_after_repair": [],
        "final_status": "complete",
        "final_evidence_subjects": [
            "nnmamba",
        "efficientmednext-l",
        ],
        "final_evidence_count": 4,
        "final_evidence": v9["comparison"]["final_evidence"],
    }
    assert {
        (
            item["doc_id"],
            item["chunk_id"],
            tuple(item["subject_ids"]),
        )
        for item in v9["comparison"]["final_evidence"]
    } == {
        ("doc-1", "nnmamba-chunk-0", ("nnmamba",)),
        ("doc-1", "nnmamba-chunk-1", ("nnmamba",)),
        (
            "doc-1",
            "efficientmednext-l-chunk-0",
            ("efficientmednext-l",),
        ),
        (
            "doc-1",
            "efficientmednext-l-chunk-1",
            ("efficientmednext-l",),
        ),
    }
    assert {
        row["subject_id"] for row in v9["comparison"]["task_diagnostics"]
    } == {"nnmamba", "efficientmednext-l"}
    assert all(
        row["query_hash"].startswith("sha256:")
        and len(row["query_preview"]) <= 160
        for row in v9["comparison"]["task_diagnostics"]
    )
    assert {
        tuple(packet["slot_ids"]) for packet in v9["evidence_packets"]
    } == {
        ("comparison-subject:nnmamba",),
        ("comparison-subject:efficientmednext-l",),
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
        ("comparison-subject:efficientmednext-l",),
    }
    assert provider.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_invalid_comparison_subjects_preserve_base_contract_and_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    _set_nonstructured_responses(
        provider,
        SimpleNamespace(
            content=json.dumps(
                    {
                        "is_comparison": False,
                        "subjects": [],
                        "dimensions": [],
                    "qualification": None,
                }
            ),
            usage_metadata={"input_tokens": 20, "output_tokens": 10},
        ),
    )
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
    assert v9["comparison_planner"]["status"] == "fallback"
    assert v9["comparison_planner"]["fallback_reason"] == "not_comparison"
    assert "comparison_plan" not in v9["query_contract"]
    assert retrieve_documents.await_count == 1
    assert result.documents
    assert provider.ainvoke.await_count == 3


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
    _set_nonstructured_responses(
        provider,
        SimpleNamespace(
            content=json.dumps(
                {
                    "is_comparison": True,
                        "subjects": [
                                {
                                    "name": "Model A",
                                    "query": "Model A accuracy",
                                },
                                {
                                    "name": "Model B",
                                    "query": "Model B accuracy",
                            },
                    ],
                    "dimensions": ["accuracy"],
                    "qualification": None,
                }
            ),
            usage_metadata={"input_tokens": 20, "output_tokens": 10},
        ),
    )
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
        max_llm_calls=3,
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
        packed_ids = set(v9["context_pack"]["packed_evidence_ids"])
        packed_packets = [
            packet
            for packet in v9["evidence_packets"]
            if packet["evidence_id"] in packed_ids
        ]
        assert {
            tuple(packet["slot_ids"]) for packet in packed_packets
        } == {
            ("comparison-subject:model-a",),
            ("comparison-subject:model-b",),
        }
    else:
        assert v9["comparison"]["coverage_after_repair"] == ["model-a"]
        assert v9["comparison"]["missing_after_repair"] == ["model-b"]


@pytest.mark.asyncio
async def test_v9_comparison_status_uses_final_balanced_packet_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    _set_nonstructured_responses(
        provider,
        SimpleNamespace(
            content=json.dumps(
                {
                    "is_comparison": True,
                        "subjects": [
                                {
                                    "name": "Model A",
                                    "query": "Model A accuracy",
                                },
                                {
                                    "name": "Model B",
                                    "query": "Model B accuracy",
                            },
                    ],
                    "dimensions": ["accuracy"],
                    "qualification": None,
                }
            ),
            usage_metadata={"input_tokens": 20, "output_tokens": 10},
        ),
    )
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
        max_llm_calls=3,
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
    assert v9["repairs"][0]["tasks"][0]["subject_id"] == "model-b"


@pytest.mark.asyncio
async def test_v9_comparison_planner_failure_preserves_base_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    _set_nonstructured_responses(provider, RuntimeError("planner unavailable"))
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
        "fallback_stage": None,
        "validation_issues": [],
        "latency_ms": v9["comparison_planner"]["latency_ms"],
    }
    assert result.documents
    retrieve_documents.assert_awaited()
    assert provider.ainvoke.await_count == 3
    assert v9["comparison"] == {
        "planner_status": "fallback",
        "planner_latency_ms": v9["comparison"]["planner_latency_ms"],
        "planner_fallback_reason": "provider_error",
        "fallback_stage": None,
        "validation_issues": [],
        "is_comparison": False,
        "subjects": [],
        "dimensions": [],
        "task_diagnostics": [],
        "coverage_before_repair": [],
        "missing_before_repair": [],
        "repair_executed": False,
        "coverage_after_repair": [],
        "missing_after_repair": [],
        "final_status": "complete",
        "final_evidence_subjects": [],
        "final_evidence_count": 1,
        "final_evidence": v9["comparison"]["final_evidence"],
    }
    assert v9["comparison"]["final_evidence"][0]["doc_id"] == "doc-1"
    assert v9["comparison"]["final_evidence"][0]["chunk_id"] == "chunk-1"
    assert v9["comparison"]["final_evidence"][0]["subject_ids"] == []


@pytest.mark.asyncio
async def test_v9_comparison_transport_diagnostics_reach_agent_trace() -> None:
    provider = _Provider()
    _set_nonstructured_responses(
        provider,
        SimpleNamespace(
            content=json.dumps(
                {
                    "is_comparison": True,
                        "subjects": [
                            {
                                "name": "Model A",
                            },
                            {
                                "name": "Model B",
                                "query": "Model B accuracy",
                        },
                    ],
                    "dimensions": ["accuracy"],
                }
            ),
            usage_metadata={"input_tokens": 10, "output_tokens": 5},
        ),
    )
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

    expected_issues = [{"path": "subjects.0.query", "type": "missing"}]
    v9 = result.agent_trace["agentic_v9"]
    assert v9["comparison_planner"]["fallback_stage"] == "transport_schema"
    assert v9["comparison_planner"]["validation_issues"] == expected_issues
    assert v9["comparison"]["fallback_stage"] == "transport_schema"
    assert v9["comparison"]["validation_issues"] == expected_issues
    assert provider.ainvoke.await_count == 3


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
    assert "comparison" not in v9
    assert provider.ainvoke.await_count == 2


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
async def test_v9_comparison_planner_uses_its_own_outer_phase_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def delayed_fallback(*_args, **_kwargs):
        await asyncio.sleep(0.04)
        return ComparisonPlannerOutcome(
            status="fallback",
            fallback_reason="timeout",
            latency_ms=40,
        )

    monkeypatch.setattr(
        runtime_module.ComparisonPlanner,
        "plan",
        delayed_fallback,
    )
    policy = ExecutionPolicy(
        total_deadline_s=1.0,
        phase_timeouts_s={
            "route_plan": 0.02,
            "comparison_plan": 0.10,
            "retrieval_judge": 0.10,
            "evidence_extract": 0.10,
            "visual_extract": 0.10,
            "final_answer": 0.10,
        },
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="Fallback evidence remains available.",
                    metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
                )
            ]
        ),
        provider_factory=lambda _purpose: _Provider(),
        document_reference_resolver=_identity_reference_resolver,
        policy_runtime=V9ExecutionPolicyRuntime(policy),
    )

    result = await runtime.execute(
        question="Model A vs. Model B: which performs better?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="comparison-outer-timeout",
    )

    assert result.agent_trace["agentic_v9"]["comparison"][
        "planner_fallback_reason"
    ] == "timeout"
    assert result.documents


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

    result = await runtime.execute(
        question="What is the reported score?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="attempt-trace-requirement-guidance-off",
    )

    assert retrieve_documents.await_args.args[1] == "What is the reported score?"
    guidance = result.agent_trace["agentic_v9"]["requirement_guidance"]
    assert guidance["enabled"] is False
    assert guidance["mode"] == "off"
    assert guidance["applied_task_count"] == 0
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

    result = await runtime.execute(
        question="1. What is the reported score? 2. What is the method?",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "requirement_guided_runtime": True},
        trace_id="attempt-trace-requirement-guidance-on",
    )

    retrieval_query = retrieve_documents.await_args.args[1]
    assert "Advisory answer obligations" in retrieval_query
    assert "reported score" in retrieval_query
    guidance = result.agent_trace["agentic_v9"]["requirement_guidance"]
    assert guidance["enabled"] is True
    assert guidance["mode"] == "advisory"
    assert guidance["applied_task_count"] >= 1
    assert provider.ainvoke.await_count == 2


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


def _wave2_scope(*doc_ids: str) -> ResolvedSourceScope:
    return ResolvedSourceScope(
        requested_doc_ids=list(doc_ids),
        resolved_doc_ids=list(doc_ids),
        authorized_doc_ids=list(doc_ids),
    )


def _wave2_document(doc_id: str, statement: str) -> Document:
    return Document(
        page_content=statement,
        metadata={"doc_id": doc_id, "chunk_id": f"chunk-{doc_id}"},
    )


def _patch_wave2_admission(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scope: ResolvedSourceScope,
    contract: QueryContract,
) -> None:
    async def admission(**_kwargs: object) -> V9AdmissionContract:
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(runtime_module, "build_v9_admission_contract", admission)


@pytest.mark.asyncio
async def test_campaign_qualifies_two_slots_and_persists_slot_bound_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _wave2_scope("doc-1", "doc-2")
    contract = QueryContract(
        route="exact_structured",
        intent="Report the requested protocol facts.",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="first protocol",
                authorized_source_doc_ids=["doc-1"],
            ),
            RequiredSlot(
                slot_id="S2",
                description="second protocol",
                authorized_source_doc_ids=["doc-2"],
            ),
        ],
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )
    documents = {
        "doc-1": _wave2_document("doc-1", "Protocol Alpha uses frozen layers."),
        "doc-2": _wave2_document("doc-2", "Protocol Beta tunes all layers."),
    }

    async def retrieve(
        _user_id: str,
        _query: str,
        authorized_doc_ids: list[str],
    ) -> list[Document]:
        return [documents[doc_id] for doc_id in authorized_doc_ids]

    _patch_wave2_admission(monkeypatch, scope=scope, contract=contract)
    provider = _StructuredProviderFactory()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve,
        provider_factory=provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Report the two requested facts.",
        user_id="user-a",
        authorized_doc_ids=list(scope.authorized_doc_ids),
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="wave2-two-slot-trace",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert set(v9["sufficiency"]["supported_slot_ids"]) == {"S1", "S2"}
    packed_ids = set(v9["context_pack"]["packed_evidence_ids"])
    assert {
        slot_id
        for packet in v9["evidence_packets"]
        if packet["evidence_id"] in packed_ids
        for slot_id in packet["slot_ids"]
    } == {"S1", "S2"}
    assert {claim["slot_id"] for claim in v9["final_claims"]} == {"S1", "S2"}
    assert len(v9["final_claims"]) == 2
    assert result.agent_trace["response_status"] == "complete"
    assert {packet["validation_status"] for packet in v9["evidence_packets"]} == {
        "quote_bound"
    }
    assert len(provider.qualification_outputs[0]["packets"]) == 2
    assert len(
        {
            packet["source_evidence_id"]
            for packet in provider.qualification_outputs[0]["packets"]
        }
    ) == 2
    assert {packet["source"]["doc_id"] for packet in v9["evidence_packets"]} == {
        "doc-1",
        "doc-2",
    }, {
        "retrieval_diagnostics": v9["retrieval_diagnostics"],
        "evidence_packets": v9["evidence_packets"],
        "qualification_outputs": provider.qualification_outputs,
        "purposes": provider.purposes,
    }
    packet_docs = {
        packet["evidence_id"]: packet["source"]["doc_id"]
        for packet in v9["evidence_packets"]
    }
    claimed_docs = {
        packet_docs[evidence_id]
        for claim in v9["final_claims"]
        for evidence_id in claim["evidence_ids"]
    }
    assert {document.metadata["doc_id"] for document in result.documents} == claimed_docs
    assert provider.final_payloads[0]["contract"]["required_slots"]
    assert provider.final_payloads[0]["slot_resolutions"]


@pytest.mark.asyncio
async def test_campaign_rejects_raw_candidate_when_qualification_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _wave2_scope("doc-1")
    contract = QueryContract(
        route="exact_structured",
        intent="Report the protocol.",
        required_slots=[RequiredSlot(slot_id="S1", description="protocol")],
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=2,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )
    _patch_wave2_admission(monkeypatch, scope=scope, contract=contract)
    provider = _StructuredProviderFactory(qualification_slots_by_round=[set()])
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[_wave2_document("doc-1", "The protocol uses frozen layers.")]
        ),
        provider_factory=provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Report the protocol.",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="wave2-invalid-qualification",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert result.agent_trace["response_status"] == "insufficient"
    assert v9["evidence_packets"] == []
    assert v9["final_claims"] == []
    assert v9["metrics"]["final_generation_count"] == 0


@pytest.mark.asyncio
async def test_campaign_qualifies_repair_evidence_before_marking_slot_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _wave2_scope("doc-1")
    contract = QueryContract(
        route="exact_structured",
        intent="Report the recovered protocol.",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="recovered protocol",
                authorized_source_doc_ids=["doc-1"],
            )
        ],
        max_retrieval_rounds=2,
        max_repair_rounds=1,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )
    retrieval_count = 0

    async def retrieve(
        _user_id: str,
        _query: str,
        _authorized_doc_ids: list[str],
    ) -> list[Document]:
        nonlocal retrieval_count
        retrieval_count += 1
        if retrieval_count == 1:
            return []
        return [_wave2_document("doc-1", "The recovered protocol freezes layers.")]

    _patch_wave2_admission(monkeypatch, scope=scope, contract=contract)
    provider = _StructuredProviderFactory()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve,
        provider_factory=provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Report the recovered protocol.",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="wave2-repair-qualification",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert retrieval_count == 2
    assert result.agent_trace["response_status"] == "complete"
    assert v9["repairs"][0]["repair_round_index"] == 1
    assert v9["slot_resolutions"][0]["status"] == "supported"
    assert v9["evidence_packets"][0]["validation_status"] == "quote_bound"


@pytest.mark.asyncio
async def test_campaign_denied_repair_qualification_preserves_prior_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _wave2_scope("doc-1", "doc-2")
    contract = QueryContract(
        route="multi_document_exact",
        intent="Report both protocols.",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="known protocol",
                authorized_source_doc_ids=["doc-1"],
            ),
            RequiredSlot(
                slot_id="S2",
                description="missing protocol",
                authorized_source_doc_ids=["doc-2"],
            ),
        ],
        max_retrieval_rounds=2,
        max_repair_rounds=1,
        max_llm_calls=2,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )
    doc_2_retrievals = 0

    async def retrieve(
        _user_id: str,
        _query: str,
        authorized_doc_ids: list[str],
    ) -> list[Document]:
        nonlocal doc_2_retrievals
        if authorized_doc_ids == ["doc-1"]:
            return [_wave2_document("doc-1", "The known protocol freezes layers.")]
        doc_2_retrievals += 1
        if doc_2_retrievals <= 2:
            return []
        return [_wave2_document("doc-2", "The missing protocol tunes layers.")]

    _patch_wave2_admission(monkeypatch, scope=scope, contract=contract)
    provider = _StructuredProviderFactory(final_slots={"S1"})
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve,
        provider_factory=provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Report both protocols.",
        user_id="user-a",
        authorized_doc_ids=list(scope.authorized_doc_ids),
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="wave2-denied-repair-qualification",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert result.agent_trace["response_status"] == "qualified_partial"
    assert {packet["source"]["doc_id"] for packet in v9["evidence_packets"]} == {
        "doc-1"
    }
    assert v9["slot_resolutions"] == [
        {
            "slot_id": "S1",
            "status": "supported",
            "evidence_ids": [v9["evidence_packets"][0]["evidence_id"]],
            "reason": None,
            "resolution_stage": "sufficiency_gate",
        },
        {
            "slot_id": "S2",
            "status": "not_found",
            "evidence_ids": [],
            "reason": "No valid evidence or persisted resolution is available.",
            "resolution_stage": "sufficiency_gate",
        },
    ]
    assert [document.metadata["doc_id"] for document in result.documents] == [
        "doc-1"
    ]


@pytest.mark.asyncio
async def test_campaign_rejected_high_risk_claim_is_not_used_as_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _wave2_scope("doc-1")
    statement = "Model A is best with a score of 91 points."
    contract = QueryContract(
        route="exact_structured",
        intent="Report the recorded score.",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="recorded score",
                locator_hints=["Section Results"],
            )
        ],
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )
    _patch_wave2_admission(monkeypatch, scope=scope, contract=contract)
    provider = _StructuredProviderFactory(
        final_statement_by_slot={"S1": statement},
        verifier_supported=False,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content=statement,
                    metadata={
                        "doc_id": "doc-1",
                        "chunk_id": "chunk-doc-1",
                        "section": "Results",
                    },
                )
            ]
        ),
        provider_factory=provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Report the recorded score.",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="wave2-rejected-high-risk-claim",
    )

    final_claims = result.agent_trace["agentic_v9"]["final_claims"]
    assert final_claims, {
        "status": result.agent_trace["response_status"],
        "purposes": provider.purposes,
        "v9": result.agent_trace["agentic_v9"],
    }
    claim = final_claims[0]
    assert result.agent_trace["response_status"] == "insufficient"
    assert claim["support_type"] == "qualified"
    assert claim["qualified_reason"] == "not supported"
    assert result.documents == []
    assert result.source_doc_ids == []


def test_deterministic_partial_does_not_accept_unverified_high_risk_claim() -> None:
    scope = _wave2_scope("doc-1")
    contract = QueryContract(
        route="exact_structured",
        intent="Report the recorded score.",
        required_slots=[RequiredSlot(slot_id="S1", description="recorded score")],
        max_retrieval_rounds=1,
        max_llm_calls=1,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )
    packet = EvidencePacket(
        schema_version="1",
        evidence_id="det:high-risk",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["S1"],
        statement="Model A is best with a score of 91 points.",
        support_type="direct",
        source=EvidenceSource(
            doc_id="doc-1",
            chunk_id="chunk-1",
            source_span_hash="sha256:source-span",
        ),
        scope=EvidenceScope(),
        locator=SourceLocator(section="Results"),
        extractor_version="v9-deterministic-1",
        validation_status="deterministic_valid",
    )
    evaluation = runtime_module.evaluate_sufficiency(contract, (packet,))

    result = runtime_module._deterministic_partial_answer(
        contract=contract,
        evaluation=evaluation,
        packets=(packet,),
    )

    assert result.response_status == "insufficient"
    assert result.used_evidence_ids == []
    assert result.claims[0].support_type == "qualified"
    assert result.claims[0].qualified_reason == (
        "claim_verification_required_but_unavailable"
    )
    assert "Unable to confirm" in result.answer


@pytest.mark.asyncio
async def test_campaign_preserves_validated_visual_packet_through_final_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _wave2_scope("doc-1")
    contract = QueryContract(
        route="exact_structured",
        intent="Report the visible table fact.",
        required_slots=[
            RequiredSlot(
                slot_id="S1",
                description="visible table fact",
                visual_policy="required",
            )
        ],
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
    )

    async def extract_visual(
        task: object,
        _documents: list[Document],
        _question: str,
        _controller: object,
    ) -> VisualEvidenceExtractionResult:
        return VisualEvidenceExtractionResult(
            packets=(
                EvidencePacket(
                    schema_version="1",
                    evidence_id="visual-evidence-1",
                    task_id=task.task_id,
                    round_id=task.round_id,
                    query_id=task.query_id,
                    slot_ids=list(task.target_slot_ids),
                    statement="The table shows the visible result.",
                    support_type="direct",
                    source=EvidenceSource(
                        doc_id="doc-1", chunk_id="chunk-doc-1", asset_id="asset-1"
                    ),
                    scope=EvidenceScope(),
                    locator=SourceLocator(pdf_page_index=1, table_id="table-1"),
                    validation_status="deterministic_valid",
                ),
            )
        )

    _patch_wave2_admission(monkeypatch, scope=scope, contract=contract)
    provider = _StructuredProviderFactory()
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[_wave2_document("doc-1", "")]
        ),
        visual_extractor=extract_visual,
        provider_factory=provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Report the visible table fact.",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot={**_setup(), "max_output_tokens": 8192},
        trace_id="wave2-visual-preservation",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert result.agent_trace["response_status"] == "complete"
    assert [packet["evidence_id"] for packet in v9["evidence_packets"]] == [
        "visual-evidence-1"
    ]
    assert v9["final_claims"][0]["slot_id"] == "S1"
    assert v9["final_claims"][0]["evidence_ids"] == ["visual-evidence-1"]
