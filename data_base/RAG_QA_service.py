"""
RAG Question Answering Service

Provides multimodal RAG-based question answering functionality
with enhanced reranking and query transformation.
"""

# Standard library
import base64
import json
import logging
import os
import re
from typing import (
    List,
    Any,
    Set,
    Optional,
    Tuple,
    NamedTuple,
    Union,
    Dict,
    TYPE_CHECKING,
    Callable,
    Awaitable,
)

# Type checking imports (avoid circular imports)
if TYPE_CHECKING:
    from data_base.schemas import ChatMessage

# Third-party
from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# Local application
from core.llm_factory import get_llm_usage_metrics
from core.providers import get_llm
from data_base.document_metadata import get_document_id
from data_base.vector_store_manager import get_user_retriever
from data_base.reranker import DocumentReranker
from data_base.query_transformer import (
    transform_query_with_hyde,
    transform_query_multi,
    reciprocal_rank_fusion,
)
from data_base.repository import fetch_document_filenames
from data_base.parent_child_store import ParentDocumentStore
from graph_rag.generic_mode import (
    GenericGraphRouter,
    GraphEvidence,
    GraphRouteDecision,
    GraphQueryHints,
    estimate_token_count,
    merge_graph_evidence,
)

# Configure logging
logger = logging.getLogger(__name__)

# GraphRAG-related keywords for auto mode detection
_GRAPH_KEYWORDS = [
    "關係", "連結", "趨勢", "比較", "對比",
    "這些論文", "這幾篇", "跨文件", "綜合",
    "relationship", "connection", "trend", "compare",
    "across", "these papers", "multi-document",
]

DEFAULT_GRAPH_LOCAL_HOPS = 2
DEFAULT_GRAPH_LOCAL_MAX_NODES = 20

# Flag to track initialization
_llm_initialized = False

# Visual verification Re-Act loop settings
MAX_VISUAL_ITERATIONS = 2  # Prevent infinite loops

# Prompt instruction for visual verification tool
VISUAL_TOOL_INSTRUCTION = """

## 視覺查證工具 (Visual Verification Tool)
如果上述圖片摘要資訊不足以回答問題，且你需要圖片中的**具體數據或細節**，
請不要猜測或回答「不知道」，而是輸出以下 JSON 指令（獨立一行）：

```json
{"action": "VERIFY_IMAGE", "path": "完整的圖片路徑", "question": "你想問的具體問題"}
```

**重要規則：**
1. `path` 必須完全複製上方 [圖片摘要] 中顯示的路徑，不可自行修改或編造
2. `question` 必須具體（例如包含年份、座標軸名稱、或特定物件）
3. 只有當現有摘要確實不足時才使用此工具

如果工具執行失敗，請誠實告知使用者無法獲取更多細節，並根據現有摘要回答。
"""

class RAGResult(NamedTuple):
    """Result from RAG question answering with optional documents."""
    answer: str
    source_doc_ids: List[str]
    documents: List[Document]
    usage: Dict[str, int] = {}
    thought_process: Optional[str] = None
    tool_calls: List[dict] = []
    agent_trace: Optional[dict] = None
    visual_verification_meta: Optional[Dict[str, Any]] = None


ProgressCallback = Callable[[str, Optional[Dict[str, Any]]], Awaitable[None]]


async def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    stage: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a best-effort progress callback when provided."""
    if progress_callback is None:
        return
    await progress_callback(stage, details)



def _parse_visual_tool_request(response: str) -> Optional[Dict[str, str]]:
    """
    Extracts VERIFY_IMAGE JSON from LLM response with tolerant parsing.

    Handles common LLM JSON formatting issues:
    - JSON wrapped in markdown code blocks
    - Extra whitespace/newlines
    - Minor formatting variations

    Args:
        response: LLM response text.

    Returns:
        Parsed dict with 'action', 'path', 'question' or None if not found.
    """
    # Pattern to match VERIFY_IMAGE JSON (tolerant of whitespace)
    patterns = [
        # Standard JSON format
        r'\{\s*"action"\s*:\s*"VERIFY_IMAGE"\s*,\s*"path"\s*:\s*"([^"]+)"\s*,\s*"question"\s*:\s*"([^"]+)"\s*\}',
        # Reordered fields
        r'\{\s*"action"\s*:\s*"VERIFY_IMAGE"[^}]*"path"\s*:\s*"([^"]+)"[^}]*"question"\s*:\s*"([^"]+)"[^}]*\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return {
                "action": "VERIFY_IMAGE",
                "path": match.group(1),
                "question": match.group(2),
            }
    
    # Fallback: try to find and parse any JSON with VERIFY_IMAGE
    json_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
    json_match = re.search(json_pattern, response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if data.get("action") == "VERIFY_IMAGE" and data.get("path") and data.get("question"):
                return data
        except json.JSONDecodeError:
            pass
    
    return None


async def _execute_visual_verification_loop(
    initial_response: str,
    context: str,
    question: str,
    user_id: str,
    llm: Any,
    source_doc_ids: List[str],
    image_paths: Optional[List[str]] = None,
    force_once_if_not_triggered: bool = False,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Re-Act loop for visual verification.

    If LLM requests VERIFY_IMAGE, execute the tool and re-prompt for synthesis.

    Args:
        initial_response: First LLM response (may contain tool request).
        context: Original context text.
        question: User's question.
        user_id: User ID for visual tool.
        llm: LLM instance for synthesis call.
        source_doc_ids: Source document IDs for logging.

    Returns:
        Tuple of (final answer, tool_results, visual_meta) after visual verification.
    """
    response = initial_response
    iteration = 0
    tool_results: List[Dict[str, Any]] = []
    attempted = False
    forced_fallback_used = False
    force_pending = bool(force_once_if_not_triggered and image_paths)

    while iteration < MAX_VISUAL_ITERATIONS:
        tool_request = _parse_visual_tool_request(response)
        tool_request_is_forced = False
        if not tool_request and force_pending:
            tool_request = {
                "action": "VERIFY_IMAGE",
                "path": image_paths[0],
                "question": question,
            }
            tool_request_is_forced = True
            forced_fallback_used = True
            force_pending = False
            logger.info("Visual verification forced fallback triggered for image-aware route")
        elif tool_request:
            force_pending = False

        if not tool_request:
            break  # No tool request, return as-is

        iteration += 1
        attempted = True
        logger.info(f"Visual verification iteration {iteration}: {tool_request.get('question', '')[:50]}")
        
        # Execute visual tool
        from data_base.visual_tools import verify_image_details
        result = await verify_image_details(
            image_path=tool_request.get("path", ""),
            question=tool_request.get("question", ""),
            user_id=user_id,
        )
        
        tool_results.append({
            "action": "VERIFY_IMAGE",
            "path": tool_request.get("path"),
            "question": tool_request.get("question"),
            "success": result.get("success"),
            "result": result.get("result") if result["success"] else result.get("error"),
            "forced_once": tool_request_is_forced,
        })
        
        # Build synthesis prompt with tool results
        results_json = json.dumps(tool_results, ensure_ascii=False, indent=2)
        synthesis_prompt = f"""你是學術研究助手。先前你請求了視覺查證工具來分析圖片。

## 原始背景資料
{context}

## 視覺查證結果
{results_json}

## 使用者問題
{question}

請根據以上所有資訊（包括視覺查證結果）生成完整的最終答案。
如果視覺查證失敗 (success: false)，請誠實告知使用者，並基於現有摘要盡可能回答。
請以繁體中文回答。"""

        from langchain_core.messages import HumanMessage
        synth_message = HumanMessage(content=synthesis_prompt)
        synth_response = await llm.ainvoke([synth_message])
        response = synth_response.content

        logger.info(f"Visual verification synthesis completed (iteration {iteration})")
        if tool_request_is_forced:
            break

    visual_meta = {
        "visual_verification_attempted": attempted,
        "visual_tool_call_count": len(tool_results),
        "visual_force_fallback_used": forced_fallback_used,
    }
    return response, tool_results, visual_meta



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


def _should_use_graph_search(question: str) -> bool:
    """
    Determine if question benefits from graph search (auto mode detection).
    
    Args:
        question: User's question.
        
    Returns:
        True if question contains graph-related keywords.
    """
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in _GRAPH_KEYWORDS)


def _legacy_graph_route_decision(
    search_mode: str,
    *,
    has_communities: bool,
) -> Optional[GraphRouteDecision]:
    """Map legacy public graph modes onto the generic execution core."""
    if search_mode == "local":
        return GraphRouteDecision(
            query_kind="fact",
            path="local-first",
            hops=DEFAULT_GRAPH_LOCAL_HOPS,
            max_nodes=DEFAULT_GRAPH_LOCAL_MAX_NODES,
            max_communities=1,
            token_budget=760,
        )

    if search_mode == "global":
        return GraphRouteDecision(
            query_kind="summary",
            path="global-first" if has_communities else "local-first",
            hops=1 if has_communities else DEFAULT_GRAPH_LOCAL_HOPS,
            max_nodes=8 if has_communities else DEFAULT_GRAPH_LOCAL_MAX_NODES,
            max_communities=3 if has_communities else 1,
            token_budget=1000 if has_communities else 760,
        )

    if search_mode == "hybrid":
        return GraphRouteDecision(
            query_kind="relation",
            path="blended" if has_communities else "local-first",
            hops=2,
            max_nodes=12 if has_communities else DEFAULT_GRAPH_LOCAL_MAX_NODES,
            max_communities=2 if has_communities else 1,
            token_budget=920 if has_communities else 760,
        )

    return None


async def _get_graph_context(
    question: str,
    user_id: str,
    search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Get context from knowledge graph.
    
    Args:
        question: User's question.
        user_id: User's ID.
        search_mode: Search mode (`generic` recommended; `auto/local/global/hybrid` are legacy compatibility values).
        graph_execution_hints: Optional internal routing hints for generic mode.
        
    Returns:
        Graph context string.
    """
    try:
        from graph_rag.store import GraphStore
        from graph_rag.global_search import global_search_evidence
        from graph_rag.local_search import local_search_evidence

        store = GraphStore(user_id)
        
        # Check if graph exists
        status = store.get_status()
        if not status.has_graph or status.node_count == 0:
            logger.debug(f"No graph data for user {user_id}")
            return ""

        if status.needs_optimization:
            logger.info(
                "Graph metadata for user %s is stale; skipping automatic chat-path optimization and waiting for explicit maintenance",
                user_id,
            )

        effective_mode = "generic" if search_mode == "auto" else search_mode
        hints = GraphQueryHints(**(graph_execution_hints or {}))
        has_hierarchy = bool(status.community_level_counts.get("1"))
        has_communities = status.community_count > 0

        decision = _legacy_graph_route_decision(
            effective_mode,
            has_communities=has_communities,
        )
        if decision is not None:
            logger.warning(
                "Legacy graph_search_mode '%s' requested; routing through generic graph core",
                effective_mode,
            )
        else:
            router = GenericGraphRouter()
            decision = await router.route(
                question,
                has_communities=has_communities,
                hints=hints,
            )

        local_evidence = []
        global_evidence = []

        if decision.path in ("local-first", "blended"):
            local_evidence, node_ids = await local_search_evidence(
                store,
                question,
                hops=decision.hops,
                max_nodes=decision.max_nodes,
            )
            if node_ids:
                logger.debug("Generic local search found %s nodes", len(node_ids))

        if decision.path in ("global-first", "blended") and status.community_count > 0:
            _, global_evidence, community_ids = await global_search_evidence(
                store,
                question,
                max_communities=decision.max_communities,
                level=1 if (has_hierarchy and decision.query_kind == "summary") else None,
            )
            if (
                decision.query_kind == "summary"
                and has_hierarchy
                and community_ids
            ):
                selected_leaf_ids = []
                for community_id in community_ids:
                    parent = next(
                        (community for community in store.get_communities(level=1) if community.id == community_id),
                        None,
                    )
                    if not parent or not parent.child_ids:
                        continue
                    selected_leaf_ids.extend(parent.child_ids[:2])
                if selected_leaf_ids:
                    leaf_communities = [
                        community for community in store.get_communities(level=0)
                        if community.id in selected_leaf_ids
                    ]
                    for leaf in leaf_communities:
                        text = f"{leaf.title or f'社群 {leaf.id}'}: {leaf.summary or ''}"
                        global_evidence.append(
                            GraphEvidence(
                                evidence_id=f"community-summary:{leaf.id}",
                                evidence_type="community_summary",
                                text=text,
                                score=0.6,
                                token_estimate=estimate_token_count(text),
                                metadata={"community_id": leaf.id, "level": leaf.level},
                            )
                        )
            if community_ids:
                logger.debug("Generic global search used %s communities", len(community_ids))

        merged_context, merged_units = merge_graph_evidence(
            local_evidence=local_evidence,
            global_evidence=global_evidence,
            token_budget=decision.token_budget,
        )
        logger.debug(
            "Generic graph route resolved to %s/%s with %s evidence units",
            decision.query_kind,
            decision.path,
            len(merged_units),
        )
        return merged_context
        
    except Exception as e:
        logger.warning(f"Graph context retrieval failed: {e}")
        return ""


# Context Enricher constants
_MIN_CHUNK_LENGTH = 100       # Minimum characters to trigger expansion
_MAX_EXPANDED_CHUNKS = 5      # Maximum number of chunks to expand
_MAX_TOTAL_CHARS = 15000      # Maximum total characters after expansion
_RERANK_TARGET_K = 8
_RERANK_CANDIDATE_LIMIT = 12
_RERANK_NOISE_KEYWORDS = ("SAM", "Segment Anything", "Interactive Segmentation", "SegVol")


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
        (i, doc) for i, doc in enumerate(documents)
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
        parent_id = doc.metadata.get('parent_id')
        if not parent_id:
            # No parent_id, skip this chunk
            continue
        
        # Try to get parent chunk
        try:
            parent_doc = parent_store.get_parent(parent_id)
            if parent_doc and len(parent_doc.page_content) > len(doc.page_content):
                # Calculate new total chars
                new_total = total_chars - len(doc.page_content) + len(parent_doc.page_content)
                
                if new_total <= _MAX_TOTAL_CHARS:
                    # Create new document with parent content but preserve metadata
                    new_metadata = doc.metadata.copy()
                    new_metadata['expanded_from_parent'] = True
                    new_metadata['original_length'] = len(doc.page_content)
                    
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


def _query_explicitly_requests_noise_topics(question: str) -> bool:
    """Return True when the query is explicitly about known noisy topics."""
    query_lower = question.lower()
    return any(keyword.lower() in query_lower for keyword in _RERANK_NOISE_KEYWORDS)


def _is_noise_document(doc: Document) -> bool:
    """Detect documents that match known noisy-topic heuristics."""
    content_sample = doc.page_content[:500]
    filename = doc.metadata.get("file_name") or doc.metadata.get("source_file") or ""
    return any(
        keyword in content_sample or keyword in filename
        for keyword in _RERANK_NOISE_KEYWORDS
    )


def _rerank_documents_for_generation(
    question: str,
    documents: List[Document],
    target_k: int = _RERANK_TARGET_K,
) -> List[Document]:
    """Rerank candidates and prefer non-noise documents before backfilling."""
    if not documents or not DocumentReranker.is_initialized():
        return documents[:target_k]

    reranker = DocumentReranker.get_instance()
    scored_docs = reranker.rerank_with_scores(question, documents, len(documents))
    if not scored_docs:
        return documents[:target_k]

    if _query_explicitly_requests_noise_topics(question):
        return [doc for doc, _ in scored_docs[:target_k]]

    non_noise_docs = [(doc, score) for doc, score in scored_docs if not _is_noise_document(doc)]
    noise_docs = [(doc, score) for doc, score in scored_docs if _is_noise_document(doc)]

    selected = list(non_noise_docs[:target_k])
    if len(selected) < target_k:
        selected.extend(noise_docs[: target_k - len(selected)])

    logger.debug(
        "Reranker selection complete: total=%s clean=%s noisy=%s selected=%s",
        len(scored_docs),
        len(non_noise_docs),
        len(noise_docs),
        len(selected),
    )
    return [doc for doc, _ in selected]


def _limit_rerank_candidates(
    documents: List[Document],
    max_candidates: int = _RERANK_CANDIDATE_LIMIT,
) -> List[Document]:
    """Cap reranker candidates to reduce peak inference memory usage."""
    if len(documents) <= max_candidates:
        return documents

    logger.info(
        "Capping reranker candidates from %s to %s for memory stability",
        len(documents),
        max_candidates,
    )
    return documents[:max_candidates]

async def rag_answer_question(
    question: str,
    user_id: str,
    doc_ids: Optional[List[str]] = None,
    history: Optional[List["ChatMessage"]] = None,
    enable_reranking: bool = True,
    enable_hyde: bool = False,
    enable_multi_query: bool = False,
    return_docs: bool = False,
    # GraphRAG parameters
    enable_graph_rag: bool = False,
    graph_search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
    # Visual Verification (Phase 9)
    enable_visual_verification: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Union[Tuple[str, List[str]], RAGResult]:
    """
    Performs multimodal RAG question answering for a specific user.

    Enhanced Pipeline:
    1. Get user's retriever
    2. (Optional) Query transformation (HyDE / Multi-Query)
    3. Execute retrieval (with optional doc_id filtering)
    4. (Optional) Rerank with local document reranker
    5. (Optional) GraphRAG context enhancement
    6. Separate text and image data
    7. Build multimodal prompt (with optional conversation history)
    8. Call LLM
    9. (Optional) Visual Verification Re-Act loop (Phase 9)

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
        return_docs: If True, returns RAGResult with documents for evaluation.
        enable_graph_rag: If True, enhance with knowledge graph context.
        graph_search_mode: Graph search mode (`generic` recommended; `auto/local/global/hybrid` are legacy compatibility values).
        graph_execution_hints: Internal generic-mode routing hints from execution layers.
        enable_visual_verification: If True, enable Re-Act loop for image details.

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
    retrieval_k = _RERANK_CANDIDATE_LIMIT if enable_reranking else (18 if doc_ids else 6)
    retriever = get_user_retriever(user_id, k=retrieval_k)
    
    if retriever is None:
        if return_docs:
            return RAGResult("抱歉，您還沒有建立任何知識庫文件，請先上傳 PDF。", [], [])
        return ("抱歉，您還沒有建立任何知識庫文件，請先上傳 PDF。", [])

    # Step 3: Query transformation
    search_queries = [question]
    
    if enable_hyde:
        await _emit_progress(progress_callback, "query_expansion", {"mode": "hyde"})
        hyde_doc = await transform_query_with_hyde(question, enabled=True)
        search_queries = [hyde_doc]
        logger.debug(f"HyDE transformed query: {hyde_doc[:100]}...")
    elif enable_multi_query:
        await _emit_progress(progress_callback, "query_expansion", {"mode": "multi_query"})
        search_queries = await transform_query_multi(question, enabled=True)
        logger.debug(f"Multi-query generated {len(search_queries)} queries")

    # Step 4: Execute retrieval
    try:
        await _emit_progress(
            progress_callback,
            "retrieval",
            {"query_count": len(search_queries)},
        )
        if len(search_queries) == 1:
            # Single query retrieval
            docs = retriever.invoke(search_queries[0])
        else:
            # Multi-query retrieval with RRF fusion
            all_results = []
            for q in search_queries:
                results = retriever.invoke(q)
                all_results.append(results)
            docs = reciprocal_rank_fusion(all_results)
            logger.debug(f"RRF fused {len(docs)} documents")
    except (RuntimeError, ValueError) as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        if return_docs:
            return RAGResult("抱歉，檢索知識庫時發生錯誤。", [], [])
        return ("抱歉，檢索知識庫時發生錯誤。", [])

    if not docs:
        if return_docs:
            return RAGResult("抱歉，在知識庫中找不到相關資訊。", [], [])
        return ("抱歉，在知識庫中找不到相關資訊。", [])

    # Step 4.5: Filter by doc_ids if specified (before reranking for efficiency)
    if doc_ids:
        doc_id_set = set(doc_ids)
        filtered_docs = [
            d for d in docs 
            if get_document_id(d.metadata) in doc_id_set
        ]
        docs = filtered_docs
        
        if not docs:
            if return_docs:
                return RAGResult("抱歉，在指定的文件中找不到相關資訊。", list(doc_ids), [])
            return ("抱歉，在指定的文件中找不到相關資訊。", list(doc_ids))
        
        # Debug: Count chunks per doc_id
        doc_chunk_count = {}
        for d in docs:
            did = get_document_id(d.metadata) or "unknown"
            doc_chunk_count[did] = doc_chunk_count.get(did, 0) + 1
        logger.info(f"Multi-doc retrieval: {doc_chunk_count}")

    # Step 5: Rerank with local document reranker (or fair multi-doc selection)
    target_k = _RERANK_TARGET_K
    reranker_available = DocumentReranker.is_initialized()
    rerank_candidates = _limit_rerank_candidates(docs) if enable_reranking else docs

    if enable_reranking:
        await _emit_progress(
            progress_callback,
            "reranking",
            {
                "reranker_available": reranker_available,
                "document_count": len(docs),
                "candidate_count": len(rerank_candidates),
            },
        )

    if enable_reranking and reranker_available and len(rerank_candidates) > 0:
        docs = await run_in_threadpool(
            _rerank_documents_for_generation,
            question,
            rerank_candidates,
            target_k,
        )
        logger.debug("Reranked to top %s documents", len(docs))
    elif enable_reranking and not reranker_available:
        logger.info(
            "Reranking requested but inactive: %s",
            DocumentReranker.runtime_metadata(reason="runtime_not_initialized"),
        )
    elif doc_ids and len(doc_ids) > 1:
        # Multi-doc fair selection (unchanged)
        docs_per_source = max(2, target_k // len(doc_ids))
        selected_docs = []
        docs_by_id = {}
        
        for d in docs:
            did = get_document_id(d.metadata) or "unknown"
            if did not in docs_by_id:
                docs_by_id[did] = []
            docs_by_id[did].append(d)
        
        # Take top N from each source
        for did in doc_ids:
            if did in docs_by_id:
                selected_docs.extend(docs_by_id[did][:docs_per_source])
        
        docs = selected_docs[:target_k]
        logger.info(f"Multi-doc fair selection: {len(docs)} docs from {len(docs_by_id)} sources")
    else:
        docs = docs[:target_k]

    # Step 5.5: GraphRAG context enhancement
    graph_context = ""
    if enable_graph_rag:
        await _emit_progress(
            progress_callback,
            "graph_context",
            {"search_mode": graph_search_mode},
        )
        graph_context = await _get_graph_context(
            question=question,
            user_id=user_id,
            search_mode=graph_search_mode,
            graph_execution_hints=graph_execution_hints,
        )

    # Step 5.6: Context Enricher - expand short chunks
    docs = _expand_short_chunks(docs, user_id)

    # Step 6: Separate data, label sources, and deduplicate
    text_context: List[str] = []
    image_paths: Set[str] = set()
    source_doc_ids: Set[str] = set()
    
    # Build doc_id to filename mapping for labeling
    doc_id_to_name = {}
    
    # Collect unique doc_ids first
    unique_doc_ids = set()
    for doc in docs:
        doc_id = get_document_id(doc.metadata)
        if doc_id:
            unique_doc_ids.add(doc_id)
    
    # Query database for actual filenames if we have doc_ids
    if unique_doc_ids:
        try:
            doc_id_to_name = await fetch_document_filenames(list(unique_doc_ids))
            logger.debug(f"Fetched filenames: {doc_id_to_name}")
        except Exception as e:
            logger.warning(f"Failed to fetch filenames from DB: {e}")
    
    # Fallback for any doc_ids not found in DB
    for doc in docs:
        doc_id = get_document_id(doc.metadata)
        if doc_id and doc_id not in doc_id_to_name:
            filename = doc.metadata.get("file_name") or doc.metadata.get("source_file") or f"文件-{doc_id[:8]}"
            doc_id_to_name[doc_id] = filename

    # Group chunks by source document (anti-hallucination strategy)
    chunks_by_doc: dict[str, List[str]] = {}
    
    for doc in docs:
        source = doc.metadata.get("source", "text")
        
        # Track source doc_ids
        doc_id = get_document_id(doc.metadata)
        if doc_id:
            source_doc_ids.add(doc_id)
        
        # Get source label for this chunk
        source_label = doc_id_to_name.get(doc_id, "未知來源") if doc_id else "未知來源"
        
        # Initialize list for this document if needed
        if source_label not in chunks_by_doc:
            chunks_by_doc[source_label] = []

        if source == "image":
            img_path = doc.metadata.get("image_path")
            if img_path and os.path.exists(img_path):
                image_paths.add(img_path)
                # Normalize path for Markdown compatibility (Windows backslash -> forward slash)
                normalized_path = img_path.replace("\\", "/")
                # Include image path in text context for LLM to reference
                if doc.page_content:
                    chunks_by_doc[source_label].append(
                        f"[圖片摘要] (路徑: {normalized_path})\n{doc.page_content}"
                    )
            elif doc.page_content:
                # No valid image path, just include summary
                chunks_by_doc[source_label].append(f"[圖片摘要] {doc.page_content}")
        else:
            if doc.page_content:
                chunks_by_doc[source_label].append(doc.page_content)
    
    # Build document-grouped context (each document's content is kept together)
    for filename, chunks in chunks_by_doc.items():
        file_section = f"=== 來源文件：{filename} ===\n（以下內容僅來自此文件，請勿與其他文件混淆）\n\n"
        file_section += "\n\n".join(chunks)
        text_context.append(file_section)

    # Step 7: Process images (limit count to avoid token explosion)
    MAX_IMAGES = 3
    image_list = list(image_paths)[:MAX_IMAGES]
    encoded_images: List[str] = []

    for img_path in image_list:
        b64 = _encode_image(img_path)
        if b64:
            encoded_images.append(b64)

    logger.debug(f"Text chunks: {len(text_context)}, Images: {len(encoded_images)}")

    # Step 8: Build interleaved multimodal message
    # Group text chunks with their associated images for clearer context
    context_text = "\n\n---\n\n".join(text_context) if text_context else "(無文字背景資訊)"

    # Format conversation history if provided
    history_text = _format_history_for_prompt(history)
    history_section = f"\n{history_text}\n" if history_text else ""
    
    # Format graph context if available
    graph_section = f"\n{graph_context}\n" if graph_context else ""

    # Enhanced prompt with anti-hallucination guidance (Phase 5: Conflict Arbitration)
    prompt_text = f"""你是一位學術研究助手，擅長分析文本與圖表。

## 參考資料
以下是從知識庫檢索到的相關內容，已按來源文件分組：

{context_text}
{graph_section}{history_section}
## 使用者問題
{question}

## 回答指引（請務必遵守）

### ⚠️ 重要：防止資訊混淆
1. **每份文件是獨立的研究**，請勿假設它們之間有關聯
2. **文件 A 中的聲明不適用於文件 B**：例如「文件 A 說 X 輸給 Y」不代表文件 B 中的技術也輸給 Y
3. 如果某個聲明只出現在一份文件中，請明確指出，不要套用到其他文件的內容
4. 回答比較問題時，請先分別總結每份文件的觀點，再進行對比

### 🔥 Phase 5 衝突處理守則（當文獻觀點衝突時）
5. **優先採信基準測試 (Benchmark)**：大規模系統性比較 > 單一實驗結果
6. **優先採信較新發表**：若可從內容推斷年份，較新的研究結論優先
7. **禁止和稀泥結論**：不可回答「兩者互有優劣」「效果因情況而異」等模糊表述
8. **衝突明確標註**：若發現文獻觀點衝突，必須使用以下格式：
   「一方面，[來源A] 主張...；另一方面，[來源B] 的 [Benchmark/實驗] 顯示...。
   根據證據權重，較可信的結論是...」

### 一般指引
9. 引用來源時請標註文件名稱
10. 仔細觀察圖表數據（如有提供）
11. 數學公式請使用 LaTeX 格式
12. 以繁體中文回答
{VISUAL_TOOL_INSTRUCTION if enable_visual_verification and image_paths else ''}
請根據以上資料回答問題："""

    # Build content list
    message_content: List[Any] = [{"type": "text", "text": prompt_text}]

    # Add images
    for b64_img in encoded_images:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })

    message = HumanMessage(content=message_content)

    # Step 9: Call LLM
    try:
        await _emit_progress(
            progress_callback,
            "answer_generation",
            {
                "image_count": len(encoded_images),
                "document_count": len(docs),
            },
        )
        response = await llm.ainvoke([message])
        answer = response.content
        usage_metadata = get_llm_usage_metrics(response)
        visual_verification_meta = {
            "visual_verification_attempted": False,
            "visual_tool_call_count": 0,
            "visual_force_fallback_used": False,
        }
        
        # Step 10: Visual Verification Re-Act Loop (Phase 9)
        tool_calls = []
        if enable_visual_verification and image_paths:
            answer, tool_calls, visual_verification_meta = await _execute_visual_verification_loop(
                initial_response=answer,
                context=context_text,
                question=question,
                user_id=user_id,
                llm=llm,
                source_doc_ids=list(source_doc_ids),
                image_paths=image_list,
                force_once_if_not_triggered=True,
            )
            # Note: Usage from visual verification loop is not currently aggregated into usage_metadata
        
        if return_docs:
            return RAGResult(
                answer, 
                list(source_doc_ids), 
                docs, 
                usage_metadata,
                thought_process=prompt_text, # Capture the prompt as thought trace
                tool_calls=tool_calls,
                visual_verification_meta=visual_verification_meta,
            )
        return (answer, list(source_doc_ids))

    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"LLM error for user {user_id}: {e}", exc_info=True)
        if return_docs:
            return RAGResult("抱歉，處理您的問題時發生錯誤。", list(source_doc_ids), docs, {})
        return ("抱歉，處理您的問題時發生錯誤。", list(source_doc_ids))




