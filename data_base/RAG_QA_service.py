"""
RAG Question Answering Service

Provides multimodal RAG-based question answering functionality
with enhanced reranking and query transformation.
"""

# Standard library
import base64
import logging
import os
from typing import List, Any, Set, Optional, Tuple, NamedTuple, Union, TYPE_CHECKING

# Type checking imports (avoid circular imports)
if TYPE_CHECKING:
    from data_base.schemas import ChatMessage

# Third-party
from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# Local application
from core.llm_factory import get_llm
from data_base.vector_store_manager import get_user_retriever
from data_base.reranker import rerank_documents, DocumentReranker
from data_base.query_transformer import (
    transform_query_with_hyde,
    transform_query_multi,
    reciprocal_rank_fusion,
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

# Flag to track initialization
_llm_initialized = False


class RAGResult(NamedTuple):
    """Result from RAG question answering with optional documents."""
    answer: str
    source_doc_ids: List[str]
    documents: List[Document]


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


async def _get_graph_context(
    question: str,
    user_id: str,
    search_mode: str = "auto",
) -> str:
    """
    Get context from knowledge graph.
    
    Args:
        question: User's question.
        user_id: User's ID.
        search_mode: Search mode (auto/local/global/hybrid).
        
    Returns:
        Graph context string.
    """
    try:
        from graph_rag.store import GraphStore
        from graph_rag.local_search import local_search
        from graph_rag.global_search import global_search, build_global_context
        
        store = GraphStore(user_id)
        
        # Check if graph exists
        status = store.get_status()
        if not status.has_graph or status.node_count == 0:
            logger.debug(f"No graph data for user {user_id}")
            return ""
        
        # Determine effective mode
        effective_mode = search_mode
        if search_mode == "auto":
            if _should_use_graph_search(question):
                effective_mode = "hybrid" if status.community_count > 0 else "local"
            else:
                effective_mode = "local"
        
        context_parts = []
        
        # Local search: entity expansion
        if effective_mode in ("local", "hybrid"):
            local_ctx, node_ids = await local_search(store, question, hops=2, max_nodes=20)
            if local_ctx:
                context_parts.append(local_ctx)
                logger.debug(f"Local search found {len(node_ids)} nodes")
        
        # Global search: community-based
        if effective_mode in ("global", "hybrid"):
            if status.community_count > 0:
                global_answer, community_ids = await global_search(store, question, max_communities=3)
                if global_answer:
                    context_parts.append(f"=== 社群分析 ===\n{global_answer}")
                    logger.debug(f"Global search used {len(community_ids)} communities")
            elif effective_mode == "global":
                # Fallback to local if global requested but no communities
                local_ctx, _ = await local_search(store, question, hops=2, max_nodes=20)
                if local_ctx:
                    context_parts.append(local_ctx)
        
        return "\n\n".join(context_parts)
        
    except Exception as e:
        logger.warning(f"Graph context retrieval failed: {e}")
        return ""


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
    graph_search_mode: str = "auto",
) -> Union[Tuple[str, List[str]], RAGResult]:
    """
    Performs multimodal RAG question answering for a specific user.

    Enhanced Pipeline:
    1. Get user's retriever
    2. (Optional) Query transformation (HyDE / Multi-Query)
    3. Execute retrieval (with optional doc_id filtering)
    4. (Optional) Rerank with Cross-Encoder
    5. (Optional) GraphRAG context enhancement
    6. Separate text and image data
    7. Build multimodal prompt (with optional conversation history)
    8. Call LLM

    Args:
        question: The question to answer.
        user_id: The user's ID.
        doc_ids: Optional list of document IDs to filter results.
                 If None or empty, queries all documents.
        history: Optional conversation history for context-aware responses.
                 Limited to last 10 messages to control token usage.
        enable_reranking: If True, use Cross-Encoder reranking.
        enable_hyde: If True, use HyDE query transformation.
        enable_multi_query: If True, use multi-query with RRF fusion.
        return_docs: If True, returns RAGResult with documents for evaluation.
        enable_graph_rag: If True, enhance with knowledge graph context.
        graph_search_mode: Graph search mode (local/global/hybrid/auto).

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
    retrieval_k = 50 if enable_reranking else (18 if doc_ids else 6)
    retriever = get_user_retriever(user_id, k=retrieval_k)
    
    if retriever is None:
        if return_docs:
            return RAGResult("抱歉，您還沒有建立任何知識庫文件，請先上傳 PDF。", [], [])
        return ("抱歉，您還沒有建立任何知識庫文件，請先上傳 PDF。", [])

    # Step 3: Query transformation
    search_queries = [question]
    
    if enable_hyde:
        hyde_doc = await transform_query_with_hyde(question, enabled=True)
        search_queries = [hyde_doc]
        logger.debug(f"HyDE transformed query: {hyde_doc[:100]}...")
    elif enable_multi_query:
        search_queries = await transform_query_multi(question, enabled=True)
        logger.debug(f"Multi-query generated {len(search_queries)} queries")

    # Step 4: Execute retrieval
    try:
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
            if d.metadata.get("doc_id") in doc_id_set or
               d.metadata.get("original_doc_uid") in doc_id_set
        ]
        docs = filtered_docs
        
        if not docs:
            if return_docs:
                return RAGResult("抱歉，在指定的文件中找不到相關資訊。", list(doc_ids), [])
            return ("抱歉，在指定的文件中找不到相關資訊。", list(doc_ids))
        
        logger.debug(f"Filtered to {len(docs)} docs from specified doc_ids")

    # Step 5: Rerank with Cross-Encoder
    if enable_reranking and len(docs) > 6:
        docs = await run_in_threadpool(
            rerank_documents,
            question,
            docs,
            top_k=6,
            enabled=True,
        )
        logger.debug(f"Reranked to top {len(docs)} documents")
    else:
        docs = docs[:6]

    # Step 5.5: GraphRAG context enhancement
    graph_context = ""
    if enable_graph_rag:
        graph_context = await _get_graph_context(
            question=question,
            user_id=user_id,
            search_mode=graph_search_mode,
        )

    # Step 6: Separate data and deduplicate
    text_context: List[str] = []
    image_paths: Set[str] = set()
    source_doc_ids: Set[str] = set()

    for doc in docs:
        source = doc.metadata.get("source", "text")
        
        # Track source doc_ids
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("original_doc_uid")
        if doc_id:
            source_doc_ids.add(doc_id)

        if source == "image":
            img_path = doc.metadata.get("image_path")
            if img_path and os.path.exists(img_path):
                image_paths.add(img_path)
            # Also add image summary to text context as supplementary info
            if doc.page_content:
                text_context.append(f"[圖片摘要] {doc.page_content}")
        else:
            if doc.page_content:
                text_context.append(doc.page_content)

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

    # Enhanced prompt with better multimodal guidance
    prompt_text = f"""你是一位學術研究助手，擅長分析文本與圖表。

## 參考資料
以下是從知識庫檢索到的相關內容：

{context_text}
{graph_section}{history_section}
## 使用者問題
{question}

## 回答指引
1. 仔細觀察圖表/圖片中的數據與趨勢（如有提供）
2. 結合文字內容與圖片資訊進行推理
3. 如有對話歷史，請延續先前的討論脈絡
4. 引用具體來源時，說明資訊出處
5. 數學公式請使用 LaTeX 格式 (例如 $\\frac{{a}}{{b}}$)
6. 以繁體中文回答
7. 如果圖片與問題無關，請忽略圖片

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
        response = await llm.ainvoke([message])
        if return_docs:
            return RAGResult(response.content, list(source_doc_ids), docs)
        return (response.content, list(source_doc_ids))
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"LLM error for user {user_id}: {e}", exc_info=True)
        if return_docs:
            return RAGResult("抱歉，處理您的問題時發生錯誤。", list(source_doc_ids), docs)
        return ("抱歉，處理您的問題時發生錯誤。", list(source_doc_ids))


# Backward compatible alias
async def rag_answer_question_simple(
    question: str,
    user_id: str,
    doc_ids: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Simple RAG QA without reranking or query transformation.
    
    Provided for backward compatibility and testing.
    
    Args:
        question: The question to answer.
        user_id: The user's ID.
        doc_ids: Optional document ID filter.
        
    Returns:
        Tuple of (answer, source_doc_ids).
    """
    return await rag_answer_question(
        question=question,
        user_id=user_id,
        doc_ids=doc_ids,
        enable_reranking=False,
        enable_hyde=False,
        enable_multi_query=False,
    )