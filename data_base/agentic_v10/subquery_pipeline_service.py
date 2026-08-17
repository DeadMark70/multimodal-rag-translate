"""Agentic RAG v10 Pipeline Service: Sub-Query Decomposition, Sequential Rerank Top-2, and Grounded Synthesis."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Optional
from uuid import uuid4

from langchain_core.documents import Document

from core.prompt_loader import format_agentic_v10_prompt, get_agentic_v10_prompt_registry
from core.providers import get_llm
from data_base.agentic_v10.subquery_decomposer import SubQueryDecomposer, SubQueryItem
from data_base.document_metadata import get_document_id
from data_base.rag_pipeline_schemas import RAGResult
from data_base.rag_retrieval import retrieve_hybrid_documents
from data_base.reranker import DocumentReranker
from data_base.vector_store_manager import get_user_retriever_async

logger = logging.getLogger(__name__)

AGENTIC_V10_EXECUTION_PROFILE = "agentic_v10_subquery_sequential_rerank_top2"
AGENTIC_V10_CONTEXT_POLICY_VERSION = "v10_grounded_structured_pack"


def _extract_doc_title(doc: Document) -> str:
    """Extract human-readable document title or filename."""
    meta = doc.metadata or {}
    for key in ("title", "filename", "file_name", "source_name", "source", "doc_name"):
        val = meta.get(key)
        if val and isinstance(val, str) and val.strip():
            # return only basename if it's a filepath
            clean = val.replace("\\", "/").split("/")[-1]
            return clean
    return f"Doc_{str(get_document_id(meta) or '')[:8]}"


def _doc_hash(doc: Document) -> str:
    """Unique content fingerprint for deduplication."""
    doc_id = str(get_document_id(doc.metadata) or "")
    content = doc.page_content.strip()
    return hashlib.sha256(f"{doc_id}:{content}".encode("utf-8")).hexdigest()


class AgenticV10PipelineService:
    """End-to-end execution pipeline for Agentic RAG v10."""

    def __init__(
        self,
        decomposer: Optional[SubQueryDecomposer] = None,
        reranker: Optional[DocumentReranker] = None,
    ) -> None:
        self._decomposer = decomposer or SubQueryDecomposer()
        self._reranker = reranker

    async def execute(
        self,
        *,
        question: str,
        user_id: str,
        authorized_doc_ids: Optional[list[str]] = None,
        setup_snapshot: Optional[dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> RAGResult:
        """Execute the sub-query decomposition, sequential rerank, and synthesis flow."""
        start_time = time.perf_counter()
        trace_id = trace_id or str(uuid4())
        setup_snapshot = setup_snapshot or {}
        question_clean = question.strip()

        logger.info(
            "Starting Agentic v10 pipeline for query: %s (trace_id=%s)",
            question_clean[:80],
            trace_id,
        )

        # ------------------------------------------------------------------
        # Step 1: Sub-Query Decomposition (2~5 English Sub-Queries)
        # ------------------------------------------------------------------
        sub_queries: list[SubQueryItem] = await self._decomposer.decompose(
            question_clean
        )
        logger.info(
            "SubQuery decomposition yielded %d sub-queries for trace_id=%s",
            len(sub_queries),
            trace_id,
        )

        # ------------------------------------------------------------------
        # Step 2: Parallel Hybrid Retrieval per sub-query (k=4 raw candidates)
        # ------------------------------------------------------------------
        retriever = await get_user_retriever_async(user_id, k=4, plain_mode=False)

        async def _fetch_candidates(sq: SubQueryItem) -> tuple[SubQueryItem, list[Document]]:
            try:
                raw_res = await retrieve_hybrid_documents(
                    sq.query,
                    retriever,
                    enable_hyde=False,
                    enable_multi_query=False,
                )
                docs = list(raw_res.documents)
                if authorized_doc_ids:
                    auth_set = set(authorized_doc_ids)
                    docs = [d for d in docs if str(get_document_id(d.metadata) or "") in auth_set]
                return sq, docs[:4]
            except Exception as ret_err:
                logger.warning(
                    "Retrieval failed for sub-query '%s': %s", sq.query, ret_err
                )
                return sq, []

        retrieval_tasks = [_fetch_candidates(sq) for sq in sub_queries]
        retrieval_results = await asyncio.gather(*retrieval_tasks)

        # ------------------------------------------------------------------
        # Step 3: Sequential Reranking (Top-2 per sub-query branch)
        # ------------------------------------------------------------------
        reranker_instance = (
            self._reranker or DocumentReranker.get_instance()
        )
        selected_candidates: list[tuple[Document, float, SubQueryItem]] = []
        branch_diagnostics: list[dict[str, Any]] = []

        for sq, branch_docs in retrieval_results:
            if not branch_docs:
                branch_diagnostics.append(
                    {
                        "subquery_id": sq.id,
                        "query": sq.query,
                        "focus": sq.focus,
                        "raw_candidate_count": 0,
                        "reranked_count": 0,
                        "scores": [],
                    }
                )
                continue

            try:
                # Sequential call to prevent GPU VRAM concurrency spike
                scored_docs = reranker_instance.rerank(
                    sq.query, branch_docs, top_k=2
                )
            except Exception as rerank_err:
                logger.warning(
                    "Sequential rerank fallback for '%s': %s", sq.query, rerank_err
                )
                scored_docs = [(doc, 0.5) for doc in branch_docs[:2]]

            scores = []
            for doc, score in scored_docs:
                selected_candidates.append((doc, score, sq))
                scores.append(round(float(score), 4))

            branch_diagnostics.append(
                {
                    "subquery_id": sq.id,
                    "query": sq.query,
                    "focus": sq.focus,
                    "raw_candidate_count": len(branch_docs),
                    "reranked_count": len(scored_docs),
                    "scores": scores,
                }
            )

        # ------------------------------------------------------------------
        # Step 4: Evidence Deduplication & Context Formatting
        # ------------------------------------------------------------------
        unique_docs_map: dict[str, Document] = {}
        rendered_blocks: list[str] = []
        doc_ids_seen: set[str] = set()

        for doc, score, sq in selected_candidates:
            h = _doc_hash(doc)
            doc_id = str(get_document_id(doc.metadata) or "")
            if doc_id:
                doc_ids_seen.add(doc_id)

            if h in unique_docs_map:
                continue

            unique_docs_map[h] = doc
            ref_idx = len(unique_docs_map)
            doc_title = _extract_doc_title(doc)
            page = doc.metadata.get("page", "N/A") if doc.metadata else "N/A"

            block = (
                f"### 檢索來源證據 [Ref {ref_idx}]\n"
                f"- 文檔名稱 (Document Name): {doc_title}\n"
                f"- 文檔 ID (Doc ID): {doc_id}\n"
                f"- 頁碼 (Page): {page}\n"
                f"- 關聯子查詢焦點 (Target Focus): {sq.focus}\n"
                f"- 相關度評分 (Rerank Score): {score:.4f}\n"
                f"- 內容 (Content):\n"
                f'"""\n{doc.page_content.strip()}\n"""'
            )
            rendered_blocks.append(block)

        context_text = (
            "\n\n".join(rendered_blocks)
            if rendered_blocks
            else "（知識庫中未檢索到相關文檔片段）"
        )

        subqueries_overview = "\n".join(
            f"{sq.id}. [{sq.target_entity}] {sq.focus} (查詢: {sq.query})"
            for sq in sub_queries
        )

        # ------------------------------------------------------------------
        # Step 5: Final Grounded Synthesis (LLM Generation)
        # ------------------------------------------------------------------
        prompt_reg = get_agentic_v10_prompt_registry()
        synth_sys = prompt_reg.get("grounded_synthesis_system").template
        synth_user = format_agentic_v10_prompt(
            "grounded_synthesis_user",
            question=question_clean,
            subqueries_overview=subqueries_overview,
            context_text=context_text,
        )

        messages = [
            {"role": "system", "content": synth_sys},
            {"role": "user", "content": synth_user},
        ]

        synth_llm = get_llm(purpose="synthesizer")
        usage_dict: dict[str, Any] = {}
        try:
            if hasattr(synth_llm, "ainvoke"):
                synth_resp = await synth_llm.ainvoke(messages)
            else:
                synth_resp = synth_llm.invoke(messages)

            raw_content = getattr(synth_resp, "content", synth_resp)
            if isinstance(raw_content, str):
                answer_text = raw_content.strip()
            elif isinstance(raw_content, list):
                answer_text = "".join(
                    part.get("text", str(part)) if isinstance(part, dict) else str(part)
                    for part in raw_content
                ).strip()
            else:
                answer_text = str(raw_content).strip()

            # Capture token usage if available from LangChain metadata
            usage_meta = getattr(synth_resp, "usage_metadata", None)
            if isinstance(usage_meta, dict):
                usage_dict = usage_meta
            else:
                resp_meta = getattr(synth_resp, "response_metadata", {})
                if isinstance(resp_meta, dict) and "token_usage" in resp_meta:
                    usage_dict = resp_meta["token_usage"]

        except Exception as synth_err:
            logger.error("Final synthesis LLM invocation failed: %s", synth_err)
            answer_text = (
                "生成回答時發生錯誤，但檢索已完成。請參考上述檢索來源文檔。"
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        final_docs = list(unique_docs_map.values())
        final_doc_ids = list(doc_ids_seen)

        # ------------------------------------------------------------------
        # Step 6: Construct Structured Observability Trace & RAGResult
        # ------------------------------------------------------------------
        agent_trace = {
            "trace_id": trace_id,
            "mode": "agentic",
            "agentic_execution_version": "v10",
            "execution_profile": AGENTIC_V10_EXECUTION_PROFILE,
            "context_policy_version": AGENTIC_V10_CONTEXT_POLICY_VERSION,
            "response_status": "complete" if final_docs else "qualified_partial",
            "latency_ms": duration_ms,
            "agentic_v10": {
                "sub_queries": [sq.model_dump() for sq in sub_queries],
                "branch_diagnostics": branch_diagnostics,
                "total_unique_evidence_count": len(final_docs),
                "referenced_doc_ids": final_doc_ids,
            },
        }

        logger.info(
            "Agentic v10 execution finished in %.2f ms, %d unique docs cited",
            duration_ms,
            len(final_docs),
        )

        return RAGResult(
            answer=answer_text,
            source_doc_ids=final_doc_ids,
            documents=final_docs,
            usage=usage_dict,
            agent_trace=agent_trace,
        )
