"""Compatibility facade for legacy RAG question answering."""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from data_base.schemas import ChatMessage

from core.providers import get_llm
from data_base import rag_pipeline
from data_base.rag_crag import CragRewriteMode
from data_base.rag_graph_runtime import (
    GraphContextDetails as GraphContextDetails,
)
from data_base.rag_graph_runtime import (
    GraphEvidenceLifecycle as GraphEvidenceLifecycle,
)
from data_base.rag_graph_runtime import (
    GraphExecutionStrategy as GraphExecutionStrategy,
)
from data_base.rag_graph_runtime import (
    GraphNeedDecision as GraphNeedDecision,
)
from data_base.rag_graph_runtime import (
    get_graph_evidence_bundle as get_graph_evidence_bundle,
)
from data_base.rag_pipeline_schemas import (
    ProgressCallback as ProgressCallback,
)
from data_base.rag_pipeline_schemas import (
    RAGResult as RAGResult,
)

logger = logging.getLogger(__name__)
_llm_initialized = False


async def initialize_llm_service() -> None:
    """Pre-warm the legacy RAG model for startup compatibility."""
    global _llm_initialized

    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not set")
        raise RuntimeError("GOOGLE_API_KEY not configured")

    logger.info("Pre-warming RAG QA LLM...")
    get_llm("rag_qa")
    _llm_initialized = True
    logger.info("RAG QA LLM ready")


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
    enable_graph_rag: bool = False,
    graph_search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
    mode_hints: Optional[Dict[str, Any]] = None,
    enable_visual_verification: bool = False,
    plain_mode: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    crag_rewrite_mode: CragRewriteMode = "hyde",
) -> Union[Tuple[str, List[str]], RAGResult]:
    """Delegate legacy answer execution while preserving the public API."""
    return await rag_pipeline.run_rag_pipeline(
        question=question,
        user_id=user_id,
        doc_ids=doc_ids,
        history=history,
        enable_reranking=enable_reranking,
        enable_hyde=enable_hyde,
        enable_multi_query=enable_multi_query,
        enable_crag=enable_crag,
        return_docs=return_docs,
        enable_graph_rag=enable_graph_rag,
        graph_search_mode=graph_search_mode,
        graph_execution_hints=graph_execution_hints,
        mode_hints=mode_hints,
        enable_visual_verification=enable_visual_verification,
        plain_mode=plain_mode,
        progress_callback=progress_callback,
        crag_rewrite_mode=crag_rewrite_mode,
    )
