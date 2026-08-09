"""Functional orchestration stages for the legacy RAG answer path."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document

from data_base import (
    rag_crag,
    rag_filtering,
    rag_graph_locator,
    rag_graph_runtime,
)
from data_base.document_metadata import get_document_id
from data_base.query_transformer import (
    transform_query_multi,
    transform_query_with_hyde,
)
from data_base.rag_filtering import (
    RERANK_CANDIDATE_LIMIT,
    RERANK_TARGET_K,
    filter_and_rerank_retrieval,
)
from data_base.rag_pipeline_schemas import ProgressCallback, RAGResult
from data_base.rag_retrieval import retrieve_hybrid_documents
from data_base.rag_crag import CragRewriteMode
from data_base.reranker import DocumentReranker
from data_base.vector_store_manager import (
    get_user_retriever_async,
    invoke_retriever_queries_async,
)
from graph_rag.feature_flags import get_graph_feature_flags

logger = logging.getLogger(__name__)

LegacyRagResponse = Union[tuple[str, List[str]], RAGResult]


@dataclass(slots=True)
class RetrievalStageOutcome:
    """Evidence and runtime dependencies produced by retrieval selection."""

    documents: List[Document]
    retriever: Any = None
    reranker_available: bool = False
    target_k: int = 0
    terminal_result: Optional[LegacyRagResponse] = None


@dataclass(slots=True)
class CragStageOutcome:
    """Documents produced by optional corrective retrieval."""

    documents: List[Document]
    terminal_result: Optional[LegacyRagResponse] = None


@dataclass(slots=True)
class GraphStageOutcome:
    """Documents and legacy prompt evidence produced by Graph routing."""

    documents: List[Document]
    graph_context: str = ""
    graph_evidence_documents: List[Document] = field(default_factory=list)


def _terminal_result(
    message: str,
    source_doc_ids: List[str],
    return_docs: bool,
) -> LegacyRagResponse:
    """Project a terminal pipeline state into the legacy public response."""
    if return_docs:
        return RAGResult(message, source_doc_ids, [])
    return (message, source_doc_ids)


async def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    stage: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a best-effort progress callback when provided."""
    if progress_callback is None:
        return
    await progress_callback(stage, details)


def _resolve_retrieval_policy(
    mode_hints: Optional[Dict[str, Any]],
) -> Dict[str, int]:
    """Resolve bounded retrieval policy overrides from mode hints."""
    policy = (mode_hints or {}).get("retrieval_policy")
    if not isinstance(policy, dict):
        return {}

    resolved: Dict[str, int] = {}
    retrieval_k_raw = policy.get("retrieval_k")
    target_k_raw = policy.get("target_k")
    if retrieval_k_raw is not None:
        try:
            resolved["retrieval_k"] = max(2, min(40, int(retrieval_k_raw)))
        except (TypeError, ValueError):
            pass
    if target_k_raw is not None:
        try:
            resolved["target_k"] = max(2, min(20, int(target_k_raw)))
        except (TypeError, ValueError):
            pass
    return resolved


async def _run_retrieval_stage(
    *,
    question: str,
    user_id: str,
    doc_ids: Optional[List[str]],
    enable_reranking: bool,
    enable_hyde: bool,
    enable_multi_query: bool,
    plain_mode: bool,
    mode_hints: Optional[Dict[str, Any]],
    progress_callback: Optional[ProgressCallback],
    return_docs: bool,
) -> RetrievalStageOutcome:
    """Retrieve, filter, and optionally rerank evidence for generation."""
    retrieval_policy = _resolve_retrieval_policy(mode_hints)
    retrieval_k = int(
        retrieval_policy.get(
            "retrieval_k",
            RERANK_CANDIDATE_LIMIT if enable_reranking else (18 if doc_ids else 6),
        )
    )
    retriever = await get_user_retriever_async(
        user_id,
        retrieval_k,
        plain_mode=plain_mode,
    )
    if retriever is None:
        return RetrievalStageOutcome(
            documents=[],
            terminal_result=_terminal_result(
                "抱歉，您還沒有建立任何知識庫文件，請先上傳 PDF。",
                [],
                return_docs,
            ),
        )

    try:
        retrieval_result = await retrieve_hybrid_documents(
            question,
            retriever,
            enable_hyde=enable_hyde,
            enable_multi_query=enable_multi_query,
            progress_callback=progress_callback,
            hyde_transformer=transform_query_with_hyde,
            multi_query_transformer=transform_query_multi,
            query_executor=invoke_retriever_queries_async,
        )
    except (RuntimeError, ValueError) as error:
        logger.error("Retrieval error: %s", error, exc_info=True)
        return RetrievalStageOutcome(
            documents=[],
            terminal_result=_terminal_result(
                "抱歉，檢索知識庫時發生錯誤。",
                [],
                return_docs,
            ),
        )

    if not retrieval_result.documents:
        return RetrievalStageOutcome(
            documents=[],
            terminal_result=_terminal_result(
                "抱歉，在知識庫中找不到相關資訊。",
                [],
                return_docs,
            ),
        )

    target_k = int(retrieval_policy.get("target_k", RERANK_TARGET_K))
    reranker_available = DocumentReranker.is_initialized()
    selection_result = await run_in_threadpool(
        filter_and_rerank_retrieval,
        question,
        retrieval_result,
        doc_ids=doc_ids,
        enable_reranking=enable_reranking,
        reranker_available=reranker_available,
        target_k=target_k,
        max_candidates=RERANK_CANDIDATE_LIMIT,
    )
    documents = selection_result.documents
    reranking_metadata = selection_result.metadata["reranking"]

    if doc_ids and not documents:
        requested_ids = list(doc_ids)
        return RetrievalStageOutcome(
            documents=[],
            terminal_result=_terminal_result(
                "抱歉，在指定的文件中找不到相關資訊。",
                requested_ids,
                return_docs,
            ),
        )

    if doc_ids:
        document_chunk_count: Dict[str, int] = {}
        for document in documents:
            document_id = get_document_id(document.metadata) or "unknown"
            document_chunk_count[document_id] = (
                document_chunk_count.get(document_id, 0) + 1
            )
        logger.info("Multi-doc retrieval: %s", document_chunk_count)

    if enable_reranking:
        await _emit_progress(
            progress_callback,
            "reranking",
            {
                "reranker_available": reranker_available,
                "document_count": len(
                    selection_result.metadata["filtering"]["post_filter_ranks"]
                ),
                "candidate_count": reranking_metadata["candidate_count"],
            },
        )

    if enable_reranking and not reranker_available:
        logger.info(
            "Reranking requested but inactive: %s",
            DocumentReranker.runtime_metadata(reason="runtime_not_initialized"),
        )

    return RetrievalStageOutcome(
        documents=documents,
        retriever=retriever,
        reranker_available=reranker_available,
        target_k=target_k,
    )


async def _run_crag_stage(
    *,
    question: str,
    documents: List[Document],
    retriever: Any,
    enable_crag: bool,
    crag_rewrite_mode: CragRewriteMode,
    doc_ids: Optional[List[str]],
    enable_reranking: bool,
    reranker_available: bool,
    target_k: int,
    progress_callback: Optional[ProgressCallback],
    return_docs: bool,
) -> CragStageOutcome:
    """Apply the optional corrective retrieval guard."""
    if not enable_crag or not documents:
        return CragStageOutcome(documents=documents)

    try:
        crag_result = await rag_crag.run_corrective_retrieval(
            question=question,
            documents=documents,
            retriever=retriever,
            judge=rag_crag.judge_retrieved_documents,
            rewrite_mode=crag_rewrite_mode,
            doc_ids=doc_ids,
            enable_reranking=enable_reranking,
            reranker_available=reranker_available,
            target_k=target_k,
            progress_callback=progress_callback,
            hyde_transformer=transform_query_with_hyde,
            multi_query_transformer=transform_query_multi,
            query_executor=invoke_retriever_queries_async,
            rerank_documents=rag_filtering.rerank_documents_for_generation,
            limit_rerank_candidates=rag_filtering.limit_rerank_candidates,
        )
        if crag_result.status == "insufficient":
            await _emit_progress(
                progress_callback,
                "crag_correction",
                {"status": "insufficient_retrieval"},
            )
            return CragStageOutcome(
                documents=documents,
                terminal_result=_terminal_result(
                    "抱歉，檢索守衛判定目前檢索內容關聯性不足，請調整問題或補充文件後再試。",
                    list(doc_ids or []),
                    return_docs,
                ),
            )

        corrected_documents = crag_result.documents
        if crag_result.correction_applied:
            await _emit_progress(
                progress_callback,
                "crag_correction",
                {
                    "status": "rewrite_applied",
                    "document_count": len(corrected_documents),
                },
            )
        return CragStageOutcome(documents=corrected_documents)
    except Exception as crag_error:  # noqa: BLE001
        logger.warning(
            "CRAG guard failed; falling back to original retrieval: %s",
            crag_error,
        )
        return CragStageOutcome(documents=documents)


async def _run_graph_stage(
    *,
    question: str,
    user_id: str,
    documents: List[Document],
    doc_ids: Optional[List[str]],
    enable_graph_rag: bool,
    graph_search_mode: str,
    graph_execution_hints: Optional[Dict[str, Any]],
    mode_hints: Optional[Dict[str, Any]],
    return_docs: bool,
    progress_callback: Optional[ProgressCallback],
) -> GraphStageOutcome:
    """Resolve Graph strategy and dispatch to its focused execution helper."""
    graph_flags = get_graph_feature_flags(
        rag_graph_runtime._graph_feature_flag_config(graph_execution_hints)
    )
    if not enable_graph_rag:
        return GraphStageOutcome(documents=documents)

    asset_probe_result = (
        rag_graph_runtime._request_scoped_graph_asset_probe(
            user_id=user_id,
            question=question,
            documents=documents,
            requested_doc_ids=doc_ids,
        )
        if graph_flags.graph_asset_graph_enabled
        else False
    )
    manual_override, asset_registry_available = rag_graph_runtime._graph_gate_inputs(
        graph_execution_hints,
        mode_hints,
        graph_flags,
        asset_probe_result=asset_probe_result,
    )
    evidence_mode = rag_graph_runtime._graph_evidence_mode(
        mode_hints,
        graph_execution_hints,
        rag_graph_runtime._normalize_evaluation_metadata(
            mode_hints,
            graph_execution_hints,
        ),
    )
    strategy = rag_graph_runtime._graph_execution_strategy(
        question=question,
        flags=graph_flags,
        graph_evidence_mode=evidence_mode,
        manual_override=manual_override,
        asset_registry_available=asset_registry_available,
        oracle_graph_decision=rag_graph_runtime._oracle_graph_decision(
            graph_execution_hints,
            mode_hints,
        ),
    )

    if strategy.strategy == "skip":
        return await _run_graph_skip_strategy(
            question=question,
            documents=documents,
            graph_search_mode=graph_search_mode,
            graph_execution_hints=graph_execution_hints,
            mode_hints=mode_hints,
            progress_callback=progress_callback,
            strategy=strategy,
        )
    if strategy.strategy == "source_expand":
        return await _run_graph_source_expand_strategy(
            question=question,
            user_id=user_id,
            documents=documents,
            doc_ids=doc_ids,
            graph_search_mode=graph_search_mode,
            graph_execution_hints=graph_execution_hints,
            mode_hints=mode_hints,
            progress_callback=progress_callback,
            strategy=strategy,
            evidence_mode=evidence_mode,
        )
    return await _run_graph_raw_legacy_strategy(
        question=question,
        user_id=user_id,
        documents=documents,
        graph_search_mode=graph_search_mode,
        graph_execution_hints=graph_execution_hints,
        mode_hints=mode_hints,
        progress_callback=progress_callback,
        return_docs=return_docs,
        strategy=strategy,
    )


async def _run_graph_skip_strategy(
    *,
    question: str,
    documents: List[Document],
    graph_search_mode: str,
    graph_execution_hints: Optional[Dict[str, Any]],
    mode_hints: Optional[Dict[str, Any]],
    progress_callback: Optional[ProgressCallback],
    strategy: rag_graph_runtime.GraphExecutionStrategy,
) -> GraphStageOutcome:
    """Record an explicit Graph skip and return unchanged documents."""
    lifecycle = rag_graph_runtime.GraphEvidenceLifecycle([], [], [], [], [])
    details = rag_graph_runtime.GraphContextDetails(
        route_decision=rag_graph_runtime.GraphRouteDecision(
            query_kind="relation",
            path="skip",
            router_reason="; ".join(
                filter(
                    None,
                    (
                        (
                            f"gate={strategy.gate_decision.reason}"
                            if strategy.gate_decision
                            else None
                        ),
                        f"strategy={strategy.reason}",
                        lifecycle.to_router_reason(),
                    ),
                )
            ),
        ),
        matched_entity_ids=[],
        community_ids=[],
        candidate_evidence_count=0,
        graph_latency_ms=0,
    )
    await _emit_progress(
        progress_callback,
        "graph_context",
        {"search_mode": graph_search_mode, "gate_role": "skip"},
    )
    await rag_graph_runtime._record_graph_observability(
        question=question,
        graph_search_mode=graph_search_mode,
        graph_execution_hints=graph_execution_hints,
        mode_hints=mode_hints,
        graph_context_details=details,
        graph_evidence_units=[],
        lifecycle=lifecycle,
    )
    return GraphStageOutcome(documents=documents)


async def _run_graph_source_expand_strategy(
    *,
    question: str,
    user_id: str,
    documents: List[Document],
    doc_ids: Optional[List[str]],
    graph_search_mode: str,
    graph_execution_hints: Optional[Dict[str, Any]],
    mode_hints: Optional[Dict[str, Any]],
    progress_callback: Optional[ProgressCallback],
    strategy: rag_graph_runtime.GraphExecutionStrategy,
    evidence_mode: str,
) -> GraphStageOutcome:
    """Locate source chunks from Graph evidence and record observability."""
    await _emit_progress(
        progress_callback,
        "graph_context",
        {
            "search_mode": graph_search_mode,
            "gate_role": (
                strategy.gate_decision.role
                if strategy.gate_decision
                else strategy.strategy
            ),
        },
    )
    locator_result = await rag_graph_locator.locate_graph_sources(
        question=question,
        user_id=user_id,
        vector_documents=documents,
        requested_doc_ids=doc_ids,
        graph_execution_hints=graph_execution_hints,
        required_modalities=rag_graph_runtime._required_modalities_for_question(
            question
        ),
        evidence_mode=evidence_mode,
        bundle_locator=rag_graph_runtime._get_graph_evidence_bundle,
        search_mode=graph_search_mode,
        claim_scope_approver=rag_graph_runtime._claim_scope_approves_chunk,
    )
    lifecycle = rag_graph_runtime.GraphEvidenceLifecycle(
        candidate_item_ids=locator_result.candidate_item_ids,
        resolved_item_ids=locator_result.resolved_item_ids,
        scope_approved_item_ids=locator_result.scope_approved_item_ids,
        scored_item_ids=locator_result.scored_item_ids,
        packed_item_ids=locator_result.packed_item_ids,
        used_as_locator=True,
        graph_to_chunk_attempted=True,
    )
    evidence_units = []
    if locator_result.bundle is not None:
        evidence_units = rag_graph_runtime._graph_evidence_units_from_bundle(
            locator_result.bundle,
            items=list(locator_result.bundle.evidence_items),
        )
    if locator_result.fallback is not None:
        details = rag_graph_runtime._graph_fallback_context_details(
            reason=locator_result.fallback,
            graph_latency_ms=locator_result.graph_latency_ms,
            lifecycle=lifecycle,
        )
    else:
        details = rag_graph_runtime._graph_context_details_for_bundle(
            locator_result.bundle,
            strategy.gate_decision,
            lifecycle,
            locator_result.graph_latency_ms,
        )
    await rag_graph_runtime._record_graph_observability(
        question=question,
        graph_search_mode=graph_search_mode,
        graph_execution_hints=graph_execution_hints,
        mode_hints=mode_hints,
        graph_context_details=details,
        graph_evidence_units=evidence_units,
        lifecycle=lifecycle,
    )
    return GraphStageOutcome(documents=locator_result.documents)


async def _run_graph_raw_legacy_strategy(
    *,
    question: str,
    user_id: str,
    documents: List[Document],
    graph_search_mode: str,
    graph_execution_hints: Optional[Dict[str, Any]],
    mode_hints: Optional[Dict[str, Any]],
    progress_callback: Optional[ProgressCallback],
    return_docs: bool,
    strategy: rag_graph_runtime.GraphExecutionStrategy,
) -> GraphStageOutcome:
    """Load legacy raw Graph context and optional evaluation evidence."""
    await _emit_progress(
        progress_callback,
        "graph_context",
        {
            "search_mode": graph_search_mode,
            "gate_role": (
                strategy.gate_decision.role
                if strategy.gate_decision
                else strategy.strategy
            ),
        },
    )
    payload = await rag_graph_runtime._get_graph_context(
        question=question,
        user_id=user_id,
        search_mode=graph_search_mode,
        graph_execution_hints=graph_execution_hints,
        return_evidence=return_docs,
        return_details=return_docs,
    )
    if not return_docs:
        return GraphStageOutcome(documents=documents, graph_context=payload)

    graph_context, evidence_units, details = payload
    evidence_documents = rag_graph_runtime._to_graph_evidence_documents(evidence_units)
    await rag_graph_runtime._record_graph_observability(
        question=question,
        graph_search_mode=graph_search_mode,
        graph_execution_hints=graph_execution_hints,
        mode_hints=mode_hints,
        graph_context_details=details,
        graph_evidence_units=evidence_units,
    )
    return GraphStageOutcome(
        documents=documents,
        graph_context=graph_context,
        graph_evidence_documents=evidence_documents,
    )
