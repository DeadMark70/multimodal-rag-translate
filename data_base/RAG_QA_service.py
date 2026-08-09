"""
RAG Question Answering Service

Provides multimodal RAG-based question answering functionality
with enhanced reranking and query transformation.
"""

# Standard library
import base64
import logging
import os
from typing import (
    List,
    Any,
    Optional,
    Tuple,
    Union,
    Dict,
    TYPE_CHECKING,
)

# Type checking imports (avoid circular imports)
if TYPE_CHECKING:
    from data_base.schemas import ChatMessage

# Third-party
from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document

# Local application
from core.providers import get_llm
from core.llm_factory import get_llm_usage_metrics  # noqa: F401 - compatibility seam
from data_base.document_metadata import get_document_id
from data_base.rag_generation import (
    generate_legacy_answer_from_evidence,
    legacy_source_doc_ids,
    parse_legacy_visual_tool_request,
)
from data_base.rag_graph_runtime import (
    GraphContextDetails as GraphContextDetails,
    GraphNeedDecision as GraphNeedDecision,
    GraphExecutionStrategy as GraphExecutionStrategy,
    GraphEvidenceLifecycle as GraphEvidenceLifecycle,
    _graph_execution_strategy as _graph_execution_strategy,
    _get_graph_evidence_bundle as _get_graph_evidence_bundle,
    get_graph_evidence_bundle as get_graph_evidence_bundle,
    _get_graph_context as _get_graph_context,
    _to_graph_evidence_documents as _to_graph_evidence_documents,
    _normalize_evaluation_metadata as _normalize_evaluation_metadata,
    _graph_feature_flag_config as _graph_feature_flag_config,
    _graph_gate_inputs as _graph_gate_inputs,
    _oracle_graph_decision as _oracle_graph_decision,
    _claim_scope_approves_chunk as _claim_scope_approves_chunk,
    _required_modalities_for_question as _required_modalities_for_question,
    _request_scoped_graph_asset_probe as _request_scoped_graph_asset_probe,
    _graph_evidence_mode as _graph_evidence_mode,
    _graph_evidence_units_from_bundle as _graph_evidence_units_from_bundle,
    _graph_context_details_for_bundle as _graph_context_details_for_bundle,
    _graph_fallback_context_details as _graph_fallback_context_details,
    _record_graph_observability as _record_graph_observability,
)
from data_base.rag_graph_locator import locate_graph_sources
from data_base.vector_store_manager import (
    get_user_retriever_async,
    invoke_retriever_queries_async,
)
from data_base.reranker import DocumentReranker
from data_base.rag_filtering import (
    RERANK_CANDIDATE_LIMIT as _RERANK_CANDIDATE_LIMIT,
    RERANK_TARGET_K as _RERANK_TARGET_K,
    filter_and_rerank_retrieval,
    limit_rerank_candidates,
    rerank_documents_for_generation,
)
from data_base.rag_crag import (
    CragRewriteMode,
    build_crag_queries,
    judge_retrieved_documents,
    run_corrective_retrieval,
)
from data_base.rag_retrieval import retrieve_hybrid_documents
from data_base.rag_pipeline_schemas import (
    ProgressCallback as ProgressCallback,
    RAGResult as RAGResult,
)
from data_base.query_transformer import (
    transform_query_with_hyde,
    transform_query_multi,
)
from data_base.repository import fetch_document_filenames  # noqa: F401 - compatibility seam
from data_base.parent_child_store import ParentDocumentStore
from graph_rag.feature_flags import get_graph_feature_flags
from graph_rag.generic_mode import (
    GraphEvidence,
    GraphRouteDecision,
)

# Configure logging
logger = logging.getLogger(__name__)

# Flag to track initialization
_llm_initialized = False

async def _build_crag_queries(
    question: str,
    rewrite_mode: CragRewriteMode,
) -> List[str]:
    """Compatibility facade for callers of the extracted CRAG rewrite policy."""
    return await build_crag_queries(
        question,
        rewrite_mode,
        hyde_transformer=transform_query_with_hyde,
        multi_query_transformer=transform_query_multi,
    )


async def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    stage: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a best-effort progress callback when provided."""
    if progress_callback is None:
        return
    await progress_callback(stage, details)


async def get_user_retriever(user_id: str, k: int = 3, plain_mode: bool = False):
    """Backward-compatible async seam for tests and request handlers."""
    return await get_user_retriever_async(user_id, k, plain_mode=plain_mode)


def _parse_visual_tool_request(response: str) -> Optional[Dict[str, str]]:
    """Backward-compatible facade for the extracted legacy parser."""
    return parse_legacy_visual_tool_request(response)


async def initialize_llm_service() -> None:
    """
    Initializes the LLM service.

    This is now handled by the LLM factory with lazy initialization,
    but we keep this function for backward compatibility with startup events.
    """
    global _llm_initialized

    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not set")
        raise RuntimeError("GOOGLE_API_KEY not configured")

    # Pre-warm the LLM instance
    logger.info("Pre-warming RAG QA LLM...")
    get_llm("rag_qa")
    _llm_initialized = True
    logger.info("RAG QA LLM ready")


def _encode_image(image_path: str) -> Optional[str]:
    """
    Reads an image file and converts to Base64 string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 encoded string, or None if reading fails.
    """
    image_path = os.path.normpath(image_path)

    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return None

    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except IOError as e:
        logger.error(f"Error reading image {image_path}: {e}")
        return None


def _format_history_for_prompt(history: Optional[List["ChatMessage"]]) -> str:
    """
    Formats conversation history into a prompt-readable text block.

    Args:
        history: List of ChatMessage objects from conversation history.

    Returns:
        Formatted history string, or empty string if no history.
    """
    if not history:
        return ""

    lines = ["## 對話歷史"]
    for msg in history[-10:]:  # Limit to last 10 messages
        role_label = "使用者" if msg.role.value == "user" else "助手"
        lines.append(f"**{role_label}**: {msg.content}")

    return "\n".join(lines)


def _resolve_intent_hint(
    question: str, mode_hints: Optional[Dict[str, Any]]
) -> Optional[str]:
    hinted = str((mode_hints or {}).get("question_intent") or "").strip()
    if hinted:
        return hinted
    lowered = question.lower()
    if any(
        token in lowered
        for token in (
            "benchmark",
            "dice",
            "score",
            "metric",
            "flops",
            "param",
            "accuracy",
            "auc",
            "f1",
            "指標",
            "數值",
            "參數",
            "效能",
        )
    ):
        return "benchmark_data"
    if any(
        token in lowered
        for token in (
            "figure",
            "flow",
            "pipeline",
            "module",
            "流程",
            "架構",
            "順序",
            "重建",
        )
    ):
        return "figure_flow"
    return None


def _resolve_retrieval_policy(mode_hints: Optional[Dict[str, Any]]) -> dict[str, int]:
    """
    Resolve internal retrieval policy hints (additive, backward-compatible).

    Supported keys in `mode_hints.retrieval_policy`:
    - retrieval_k: retriever top-k candidate pull
    - target_k: final generation context cap
    """
    policy = (mode_hints or {}).get("retrieval_policy")
    if not isinstance(policy, dict):
        return {}

    resolved: dict[str, int] = {}
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


def _intent_constraints_for_prompt(
    question: str, mode_hints: Optional[Dict[str, Any]]
) -> str:
    intent = _resolve_intent_hint(question, mode_hints)
    if intent == "benchmark_data":
        return """
### 題型附加限制：Benchmark / 數據
13. 第一段必須先列出「模型-指標-數值-來源」清單；每列都要有來源標記。
14. 若缺少可驗證數值，必須明確寫「資料不足」，不得以外部常識補齊或猜測。
15. 禁止只給排名不給數值；若僅有相對描述，也要標記「資料不足」。
16. 不確定時請收斂回答範圍，只保留有直接證據的結論。
"""
    if intent == "figure_flow":
        return """
### 題型附加限制：Figure Flow / 架構重建
13. 第一段必須先輸出有序流程主鏈（格式建議：`A -> B -> C`）。
14. 第一段禁止重複題目或使用標題式開頭（例如：`### What is ...`）。
15. 必須保留題幹中的核心元件與機制名稱，且只能使用參考資料出現過的元件詞。
16. 若流程中某一步缺乏直接證據，請標記「資料不足」，不要擴寫背景敘述。
"""
    return ""


# Context Enricher constants
_MIN_CHUNK_LENGTH = 100  # Minimum characters to trigger expansion
_MAX_EXPANDED_CHUNKS = 5  # Maximum number of chunks to expand
_MAX_TOTAL_CHARS = 15000  # Maximum total characters after expansion


def _expand_short_chunks(
    documents: List[Document],
    user_id: str,
) -> List[Document]:
    """
    Expands short chunks using their parent documents for better context.

    When a retrieved chunk is too short (< 100 chars), this function
    replaces it with its parent chunk to avoid out-of-context answers.

    Args:
        documents: Retrieved documents from vector search.
        user_id: User's ID for accessing parent store.

    Returns:
        List of documents with short chunks expanded.

    Note:
        - Uses defensive programming for missing parent_id metadata
        - Implements token control to prevent prompt overflow
        - Wraps I/O operations in try-except for graceful degradation
    """
    if not documents:
        return documents

    # Check if any document needs expansion
    short_chunks = [
        (i, doc)
        for i, doc in enumerate(documents)
        if len(doc.page_content) < _MIN_CHUNK_LENGTH
    ]

    if not short_chunks:
        return documents

    # Load parent store with error handling
    try:
        parent_store = ParentDocumentStore(user_id)
    except (IOError, OSError, EOFError) as e:
        logger.warning(f"Failed to load parent store: {e}")
        return documents  # Return original documents on failure

    # Track expansion stats
    expanded_count = 0
    total_chars = sum(len(doc.page_content) for doc in documents)
    expanded_docs = list(documents)  # Create a copy

    for idx, doc in short_chunks:
        # Check expansion limits
        if expanded_count >= _MAX_EXPANDED_CHUNKS:
            logger.debug(f"Reached max expanded chunks limit ({_MAX_EXPANDED_CHUNKS})")
            break

        if total_chars >= _MAX_TOTAL_CHARS:
            logger.debug(f"Reached max total chars limit ({_MAX_TOTAL_CHARS})")
            break

        # Defensive programming: check for parent_id
        parent_id = doc.metadata.get("parent_id")
        if not parent_id:
            # No parent_id, skip this chunk
            continue

        # Try to get parent chunk
        try:
            parent_doc = parent_store.get_parent(parent_id)
            if parent_doc and len(parent_doc.page_content) > len(doc.page_content):
                # Calculate new total chars
                new_total = (
                    total_chars - len(doc.page_content) + len(parent_doc.page_content)
                )

                if new_total <= _MAX_TOTAL_CHARS:
                    # Create new document with parent content but preserve metadata
                    new_metadata = doc.metadata.copy()
                    new_metadata["expanded_from_parent"] = True
                    new_metadata["original_length"] = len(doc.page_content)

                    expanded_docs[idx] = Document(
                        page_content=parent_doc.page_content,
                        metadata=new_metadata,
                    )

                    total_chars = new_total
                    expanded_count += 1
                    logger.debug(
                        f"Expanded chunk {idx}: {len(doc.page_content)} -> "
                        f"{len(parent_doc.page_content)} chars"
                    )

        except (KeyError, AttributeError) as e:
            logger.warning(f"Failed to expand chunk {idx}: {e}")
            continue

    if expanded_count > 0:
        logger.info(f"Context Enricher: Expanded {expanded_count} short chunks")

    return expanded_docs


def _rerank_documents_for_generation(
    question: str,
    documents: List[Document],
    target_k: int = _RERANK_TARGET_K,
) -> List[Document]:
    """Compatibility facade for selection now owned by ``rag_filtering``."""
    return rerank_documents_for_generation(question, documents, target_k)


def _limit_rerank_candidates(
    documents: List[Document],
    max_candidates: int = _RERANK_CANDIDATE_LIMIT,
) -> List[Document]:
    """Compatibility facade for the retrieval-boundary candidate cap."""
    return limit_rerank_candidates(documents, max_candidates)


async def rag_answer_question(
    question: str,
    user_id: str,
    doc_ids: Optional[List[str]] = None,
    history: Optional[List["ChatMessage"]] = None,
    enable_reranking: bool = False,
    enable_hyde: bool = False,
    enable_multi_query: bool = False,
    enable_crag: bool = False,
    return_docs: bool = False,
    # GraphRAG parameters
    enable_graph_rag: bool = False,
    graph_search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
    mode_hints: Optional[Dict[str, Any]] = None,
    # Visual Verification (Phase 9)
    enable_visual_verification: bool = False,
    plain_mode: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    crag_rewrite_mode: CragRewriteMode = "hyde",
) -> Union[Tuple[str, List[str]], RAGResult]:
    """
    Performs multimodal RAG question answering for a specific user.

    Enhanced Pipeline:
    1. Get user's retriever
    2. (Optional) Query transformation (HyDE / Multi-Query)
    3. Execute retrieval (with optional doc_id filtering)
    4. (Optional) Rerank with local document reranker
    5. (Optional) CRAG retrieval guard + corrective rewrite
    6. (Optional) GraphRAG context enhancement
    7. Separate text and image data
    8. Build multimodal prompt (with optional conversation history)
    9. Call LLM
    10. (Optional) Visual Verification Re-Act loop (Phase 9)

    Args:
        question: The question to answer.
        user_id: The user's ID.
        doc_ids: Optional list of document IDs to filter results.
                 If None or empty, queries all documents.
        history: Optional conversation history for context-aware responses.
                 Limited to last 10 messages to control token usage.
        enable_reranking: If True, use local document reranking.
        enable_hyde: If True, use HyDE query transformation.
        enable_multi_query: If True, use multi-query with RRF fusion.
        enable_crag: If True, run retrieval guard and corrective rewrite before generation.
        return_docs: If True, returns RAGResult with documents for evaluation.
        enable_graph_rag: If True, enhance with knowledge graph context.
        graph_search_mode: Graph search mode (`generic` recommended; `auto/local/global/hybrid` are legacy compatibility values).
        graph_execution_hints: Internal generic-mode routing hints from execution layers.
        mode_hints: Optional intent/route hints for evaluation-time output constraints,
                    including additive `retrieval_policy` overrides.
        enable_visual_verification: If True, enable Re-Act loop for image details.
        plain_mode: If True, force plain retriever/prompt behavior for native baseline.
        crag_rewrite_mode: Corrective query policy used only when CRAG rejects
                           the initial retrieval. Defaults to legacy HyDE.

    Returns:
        Tuple of (answer, doc_ids) or RAGResult if return_docs=True.
    """

    # Step 1: Get LLM instance
    try:
        llm = get_llm("rag_qa")
    except (RuntimeError, KeyError, ValueError) as e:
        logger.error(f"Failed to get LLM: {e}")
        if return_docs:
            return RAGResult("抱歉，AI 模型尚未初始化 (API Key 可能有誤)。", [], [])
        return ("抱歉，AI 模型尚未初始化 (API Key 可能有誤)。", [])

    # Step 2: Get retriever (increase k for reranking)
    retrieval_policy = _resolve_retrieval_policy(mode_hints)
    retrieval_k = int(
        retrieval_policy.get(
            "retrieval_k",
            _RERANK_CANDIDATE_LIMIT if enable_reranking else (18 if doc_ids else 6),
        )
    )
    retriever = await get_user_retriever(user_id, k=retrieval_k, plain_mode=plain_mode)

    if retriever is None:
        if return_docs:
            return RAGResult("抱歉，您還沒有建立任何知識庫文件，請先上傳 PDF。", [], [])
        return ("抱歉，您還沒有建立任何知識庫文件，請先上傳 PDF。", [])

    # Steps 3-4: Query expansion, hybrid retrieval, and RRF fusion.
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
        docs = retrieval_result.documents
    except (RuntimeError, ValueError) as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        if return_docs:
            return RAGResult("抱歉，檢索知識庫時發生錯誤。", [], [])
        return ("抱歉，檢索知識庫時發生錯誤。", [])

    if not docs:
        if return_docs:
            return RAGResult("抱歉，在知識庫中找不到相關資訊。", [], [])
        return ("抱歉，在知識庫中找不到相關資訊。", [])

    # Step 4.5-5: Keep candidate filtering and reranking inside the retrieval
    # boundary.  CRAG and graph location intentionally remain later stages.
    target_k = int(retrieval_policy.get("target_k", _RERANK_TARGET_K))
    reranker_available = DocumentReranker.is_initialized()
    selection_result = await run_in_threadpool(
        filter_and_rerank_retrieval,
        question,
        retrieval_result,
        doc_ids=doc_ids,
        enable_reranking=enable_reranking,
        reranker_available=reranker_available,
        target_k=target_k,
        max_candidates=_RERANK_CANDIDATE_LIMIT,
    )
    docs = selection_result.documents
    reranking_metadata = selection_result.metadata["reranking"]

    if doc_ids and not docs:
        if return_docs:
            return RAGResult(
                "抱歉，在指定的文件中找不到相關資訊。", list(doc_ids), []
            )
        return ("抱歉，在指定的文件中找不到相關資訊。", list(doc_ids))

    if doc_ids:
        doc_chunk_count = {}
        for document in docs:
            document_id = get_document_id(document.metadata) or "unknown"
            doc_chunk_count[document_id] = doc_chunk_count.get(document_id, 0) + 1
        logger.info(f"Multi-doc retrieval: {doc_chunk_count}")

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

    # Step 5.4: Corrective Retrieval Guard (CRAG, opt-in only)
    if enable_crag and docs:
        try:
            crag_result = await run_corrective_retrieval(
                question=question,
                documents=docs,
                retriever=retriever,
                judge=judge_retrieved_documents,
                rewrite_mode=crag_rewrite_mode,
                doc_ids=doc_ids,
                enable_reranking=enable_reranking,
                reranker_available=reranker_available,
                target_k=target_k,
                progress_callback=progress_callback,
                hyde_transformer=transform_query_with_hyde,
                multi_query_transformer=transform_query_multi,
                query_executor=invoke_retriever_queries_async,
                rerank_documents=_rerank_documents_for_generation,
                limit_rerank_candidates=_limit_rerank_candidates,
            )
            if crag_result.status == "insufficient":
                await _emit_progress(
                    progress_callback,
                    "crag_correction",
                    {"status": "insufficient_retrieval"},
                )
                crag_message = "抱歉，檢索守衛判定目前檢索內容關聯性不足，請調整問題或補充文件後再試。"
                if return_docs:
                    return RAGResult(crag_message, list(doc_ids or []), [])
                return (crag_message, list(doc_ids or []))
            docs = crag_result.documents
            if crag_result.correction_applied:
                await _emit_progress(
                    progress_callback,
                    "crag_correction",
                    {"status": "rewrite_applied", "document_count": len(docs)},
                )
        except Exception as crag_exc:  # noqa: BLE001
            logger.warning(
                "CRAG guard failed; falling back to original retrieval: %s", crag_exc
            )

    # Step 5.5: GraphRAG context enhancement
    graph_context = ""
    graph_evidence_units: List[GraphEvidence] = []
    graph_context_details: Optional[GraphContextDetails] = None
    graph_evidence_documents_for_return: List[Document] = []
    graph_flags = get_graph_feature_flags(
        _graph_feature_flag_config(graph_execution_hints)
    )
    graph_execution_strategy: Optional[GraphExecutionStrategy] = None
    if enable_graph_rag:
        asset_probe_result = (
            _request_scoped_graph_asset_probe(
                user_id=user_id,
                question=question,
                documents=docs,
                requested_doc_ids=doc_ids,
            )
            if graph_flags.graph_asset_graph_enabled
            else False
        )
        manual_override, asset_registry_available = _graph_gate_inputs(
            graph_execution_hints,
            mode_hints,
            graph_flags,
            asset_probe_result=asset_probe_result,
        )
        graph_execution_strategy = _graph_execution_strategy(
            question=question,
            flags=graph_flags,
            graph_evidence_mode=_graph_evidence_mode(
                mode_hints,
                graph_execution_hints,
                _normalize_evaluation_metadata(mode_hints, graph_execution_hints),
            ),
            manual_override=manual_override,
            asset_registry_available=asset_registry_available,
            oracle_graph_decision=_oracle_graph_decision(
                graph_execution_hints,
                mode_hints,
            ),
        )

    if graph_execution_strategy and graph_execution_strategy.strategy == "skip":
        skipped_lifecycle = GraphEvidenceLifecycle([], [], [], [], [])
        graph_context_details = GraphContextDetails(
            route_decision=GraphRouteDecision(
                query_kind="relation",
                path="skip",
                router_reason="; ".join(
                    filter(
                        None,
                        (
                            (
                                f"gate={graph_execution_strategy.gate_decision.reason}"
                                if graph_execution_strategy.gate_decision
                                else None
                            ),
                            f"strategy={graph_execution_strategy.reason}",
                            skipped_lifecycle.to_router_reason(),
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
        await _record_graph_observability(
            question=question,
            graph_search_mode=graph_search_mode,
            graph_execution_hints=graph_execution_hints,
            mode_hints=mode_hints,
            graph_context_details=graph_context_details,
            graph_evidence_units=[],
            lifecycle=skipped_lifecycle,
        )
    elif graph_execution_strategy:
        await _emit_progress(
            progress_callback,
            "graph_context",
            {
                "search_mode": graph_search_mode,
                "gate_role": (
                    graph_execution_strategy.gate_decision.role
                    if graph_execution_strategy.gate_decision
                    else graph_execution_strategy.strategy
                ),
            },
        )
        if graph_execution_strategy.strategy == "source_expand":
            locator_result = await locate_graph_sources(
                question=question,
                user_id=user_id,
                vector_documents=docs,
                requested_doc_ids=doc_ids,
                graph_execution_hints=graph_execution_hints,
                required_modalities=_required_modalities_for_question(question),
                evidence_mode=_graph_evidence_mode(
                    mode_hints,
                    graph_execution_hints,
                    _normalize_evaluation_metadata(mode_hints, graph_execution_hints),
                ),
                bundle_locator=_get_graph_evidence_bundle,
                search_mode=graph_search_mode,
                claim_scope_approver=_claim_scope_approves_chunk,
            )
            docs = locator_result.documents
            lifecycle = GraphEvidenceLifecycle(
                candidate_item_ids=locator_result.candidate_item_ids,
                resolved_item_ids=locator_result.resolved_item_ids,
                scope_approved_item_ids=locator_result.scope_approved_item_ids,
                scored_item_ids=locator_result.scored_item_ids,
                packed_item_ids=locator_result.packed_item_ids,
                used_as_locator=True,
                graph_to_chunk_attempted=True,
            )
            if locator_result.bundle is not None:
                graph_evidence_units = _graph_evidence_units_from_bundle(
                    locator_result.bundle,
                    items=list(locator_result.bundle.evidence_items),
                )
            if locator_result.fallback is not None:
                graph_context_details = _graph_fallback_context_details(
                    reason=locator_result.fallback,
                    graph_latency_ms=locator_result.graph_latency_ms,
                    lifecycle=lifecycle,
                )
            else:
                graph_context_details = _graph_context_details_for_bundle(
                    locator_result.bundle,
                    graph_execution_strategy.gate_decision,
                    lifecycle,
                    locator_result.graph_latency_ms,
                )
            await _record_graph_observability(
                question=question,
                graph_search_mode=graph_search_mode,
                graph_execution_hints=graph_execution_hints,
                mode_hints=mode_hints,
                graph_context_details=graph_context_details,
                graph_evidence_units=graph_evidence_units,
                lifecycle=lifecycle,
            )
        elif graph_execution_strategy.strategy == "raw_legacy":
            graph_context_payload = await _get_graph_context(
                question=question,
                user_id=user_id,
                search_mode=graph_search_mode,
                graph_execution_hints=graph_execution_hints,
                return_evidence=return_docs,
                return_details=return_docs,
            )
            if return_docs:
                graph_context, graph_evidence_units, graph_context_details = (
                    graph_context_payload
                )
                graph_evidence_documents_for_return = _to_graph_evidence_documents(
                    graph_evidence_units
                )
                await _record_graph_observability(
                    question=question,
                    graph_search_mode=graph_search_mode,
                    graph_execution_hints=graph_execution_hints,
                    mode_hints=mode_hints,
                    graph_context_details=graph_context_details,
                    graph_evidence_units=graph_evidence_units,
                )
            else:
                graph_context = graph_context_payload

    # Step 5.6: Context Enricher - expand short chunks (advanced mode only)
    if not plain_mode:
        docs = await run_in_threadpool(_expand_short_chunks, docs, user_id)

    generated = await generate_legacy_answer_from_evidence(
        question=question,
        user_id=user_id,
        documents=docs,
        llm=llm,
        graph_context=graph_context,
        history_section=(
            f"\n{_format_history_for_prompt(history)}\n" if history else ""
        ),
        intent_constraints=_intent_constraints_for_prompt(question, mode_hints),
        plain_mode=plain_mode,
        enable_visual_verification=enable_visual_verification,
        progress_callback=progress_callback,
        image_encoder=_encode_image,
    )
    source_doc_ids = legacy_source_doc_ids(docs)
    if return_docs:
        returned_docs = (
            docs
            if generated.thought_process is None
            else [*docs, *graph_evidence_documents_for_return]
        )
        return RAGResult(
            generated.answer,
            source_doc_ids,
            returned_docs,
            generated.usage,
            thought_process=generated.thought_process,
            tool_calls=generated.tool_calls,
            visual_verification_meta=generated.visual_verification_meta,
        )
    return (generated.answer, source_doc_ids)
