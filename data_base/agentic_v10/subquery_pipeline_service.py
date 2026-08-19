"""Evaluation-only Agentic RAG v10: fail-soft sub-query retrieval and synthesis."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

from core.prompt_loader import (
    format_agentic_v10_prompt,
    get_agentic_v10_prompt_registry,
)
from core.providers import get_llm
from data_base.agentic_v10.subquery_decomposer import (
    SubQueryDecomposer,
    SubQueryDecompositionTrace,
    SubQueryItem,
)
from data_base.document_metadata import get_document_id
from data_base.rag_pipeline_schemas import RAGResult
from data_base.rag_retrieval import retrieve_hybrid_documents
from data_base.reranker import DocumentReranker
from data_base.vector_store_manager import (
    get_user_retriever_async,
    load_user_vector_documents_async,
)

logger = logging.getLogger(__name__)

AGENTIC_V10_EXECUTION_PROFILE = "agentic_eval_v10_top1_raw_same_document_neighbors"
AGENTIC_V10_CONTEXT_POLICY_VERSION = "v10_raw_top1_same_document_neighbors"
AGENTIC_V10_TRACE_SCHEMA_VERSION = "4"


def _document_title(document: Document) -> str:
    metadata = document.metadata or {}
    for key in ("title", "filename", "file_name", "source_name", "source", "doc_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.replace("\\", "/").split("/")[-1]
    return f"Doc_{str(get_document_id(metadata) or '')[:8]}"


def _document_key(document: Document) -> str:
    doc_id = str(get_document_id(document.metadata) or "")
    return hashlib.sha256(f"{doc_id}:{document.page_content.strip()}".encode("utf-8")).hexdigest()


def _trace_document(document: Document, score: float | None = None) -> dict[str, Any]:
    metadata = document.metadata if isinstance(document.metadata, dict) else {}
    row: dict[str, Any] = {
        "doc_id": str(get_document_id(metadata) or ""),
        "title": _document_title(document),
        "page": metadata.get("page"),
        "metadata": dict(metadata),
        "content": document.page_content,
    }
    if score is not None:
        row["rerank_score"] = round(float(score), 6)
    return row


def _normalize_usage(response: Any) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None)
    if not isinstance(metadata, dict):
        response_metadata = getattr(response, "response_metadata", {})
        metadata = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}
    if not isinstance(metadata, dict):
        return {}

    def value(*names: str) -> int:
        for name in names:
            raw = metadata.get(name)
            if isinstance(raw, (int, float)) and raw >= 0:
                return int(raw)
        return 0

    prompt = value("input_tokens", "prompt_tokens")
    completion = value("output_tokens", "completion_tokens")
    total = value("total_tokens") or prompt + completion
    usage = {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}
    reasoning = value("reasoning_tokens")
    if reasoning:
        usage["reasoning_tokens"] = reasoning
    return usage


class AgenticV10PipelineService:
    """Run the v10 retrieval policy without v9 admission or sufficiency gates."""

    def __init__(self, decomposer: SubQueryDecomposer | None = None, reranker: DocumentReranker | None = None) -> None:
        self._decomposer = decomposer or SubQueryDecomposer()
        self._reranker = reranker

    async def execute(
        self,
        *,
        question: str,
        user_id: str,
        authorized_doc_ids: list[str] | None = None,
        setup_snapshot: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> RAGResult:
        start = time.perf_counter()
        trace_id = trace_id or str(uuid4())
        question = question.strip()
        decomposition = await self._decompose(question)
        retriever: Any | None = None
        retriever_error: str | None = None
        try:
            retriever = await get_user_retriever_async(user_id, k=4, plain_mode=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("v10 retriever initialization failed: %s", exc)
            retriever_error = type(exc).__name__
        allowed = set(authorized_doc_ids or [])

        async def retrieve_branch(item: SubQueryItem) -> tuple[SubQueryItem, list[Document], str | None]:
            if retriever is None:
                return item, [], retriever_error or "RetrieverUnavailable"
            try:
                response = await retrieve_hybrid_documents(item.query, retriever, enable_hyde=False, enable_multi_query=False)
                documents = list(response.documents)
                if allowed:
                    documents = [doc for doc in documents if str(get_document_id(doc.metadata) or "") in allowed]
                return item, documents[:4], None
            except Exception as exc:  # noqa: BLE001
                logger.warning("v10 retrieval failed for %s: %s", item.id, exc)
                return item, [], type(exc).__name__

        retrieved = await asyncio.gather(*(retrieve_branch(item) for item in decomposition.sub_queries))
        reranker = self._reranker or DocumentReranker.get_instance()
        selected: list[tuple[Document, float, SubQueryItem]] = []
        branches: list[dict[str, Any]] = []
        for item, candidates, retrieval_error in retrieved:
            rerank_error: str | None = None
            rerank_started = time.perf_counter()
            try:
                ranked = (
                    reranker.rerank_with_scores(item.query, candidates, top_k=1)
                    if candidates
                    else []
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("v10 rerank failed for %s: %s", item.id, exc)
                ranked = [(document, 0.5) for document in candidates[:1]]
                rerank_error = type(exc).__name__
            selected.extend((document, float(score), item) for document, score in ranked)
            branches.append({
                "subquery_id": item.id,
                "query": item.query,
                "focus": item.focus,
                "target_entity": item.target_entity,
                "raw_candidates": [_trace_document(document) for document in candidates],
                "reranked_candidates": [_trace_document(document, score) for document, score in ranked],
                "retrieval_error": retrieval_error,
                "rerank_error": rerank_error,
                "rerank_latency_ms": (time.perf_counter() - rerank_started) * 1000,
            })

        unique: dict[str, tuple[Document, float, SubQueryItem]] = {}
        for document, score, item in selected:
            unique.setdefault(_document_key(document), (document, score, item))
        top1_records = list(unique.values())
        neighbors, neighbor_lookup_error = await self._same_document_neighbors(
            user_id=user_id,
            selected_documents=[document for document, _, _ in top1_records],
            authorized_doc_ids=allowed,
        )
        context_records: list[tuple[Document, float | None, str]] = [
            (document, score, "reranked_top1")
            for document, score, _ in top1_records
        ]
        known_context_keys = {_document_key(document) for document, _, _ in context_records}
        for document, neighbor_of in neighbors:
            key = _document_key(document)
            if key in known_context_keys:
                continue
            known_context_keys.add(key)
            context_records.append((document, None, "same_document_neighbor"))
        final_documents = [document for document, _, _ in context_records]
        source_doc_ids = list(
            dict.fromkeys(
                str(get_document_id(document.metadata) or "")
                for document in final_documents
                if get_document_id(document.metadata)
            )
        )
        reference_by_document_key = {
            _document_key(document): index
            for index, (document, _, _) in enumerate(context_records, start=1)
        }
        neighbor_parent_keys: dict[str, list[str]] = {}
        for document, parent_key in neighbors:
            neighbor_parent_keys.setdefault(_document_key(document), []).append(parent_key)
        source_document_mapping = [
            {
                "reference_id": f"[Ref {index}]",
                "context_origin": origin,
                "document": _trace_document(document, score),
                "selected_by_subquery_ids": [
                    item.id
                    for candidate, _, item in selected
                    if _document_key(candidate) == _document_key(document)
                ],
                "neighbor_of_reference_ids": [
                    f"[Ref {reference_by_document_key[parent_key]}]"
                    for parent_key in neighbor_parent_keys.get(_document_key(document), [])
                    if parent_key in reference_by_document_key
                ],
            }
            for index, (document, score, origin) in enumerate(context_records, start=1)
        ]
        for branch in branches:
            matching = next(
                (
                    document
                    for document, _, item in selected
                    if item.id == branch["subquery_id"]
                ),
                None,
            )
            branch["selected_reference_id"] = (
                f"[Ref {reference_by_document_key[_document_key(matching)]}]"
                if matching is not None
                else None
            )
            branch["same_document_neighbors"] = [
                _trace_document(document)
                for document, neighbor_of in neighbors
                if neighbor_of == _document_key(matching)
            ] if matching is not None else []

        context_blocks = [
            self._context_block(
                index=index,
                document=document,
                score=score,
                origin=origin,
            )
            for index, (document, score, origin) in enumerate(context_records, start=1)
        ]
        context_text = "\n\n".join(context_blocks)
        overview = "\n".join(
            f"{item.id}. [{item.target_entity}] {item.focus} (查詢: {item.query})"
            for item in decomposition.sub_queries
        )
        synthesis_messages = self._synthesis_messages(question, overview, context_text)
        synthesis_usage: dict[str, int] = {}
        synthesis_error: str | None = None
        if final_documents:
            answer, synthesis_usage, synthesis_error = await self._synthesize(synthesis_messages)
            response_status = "qualified_partial" if synthesis_error else "complete"
        else:
            answer = "目前知識庫沒有可用的相關證據，因此無法根據文件回答此問題。請補充或上傳相關文獻後再試。"
            response_status = "qualified_partial"
        duration_ms = (time.perf_counter() - start) * 1000
        trace = {
            "trace_id": trace_id,
            "mode": "agentic",
            "agentic_execution_version": "v10",
            "execution_profile": AGENTIC_V10_EXECUTION_PROFILE,
            "context_policy_version": AGENTIC_V10_CONTEXT_POLICY_VERSION,
            "response_status": response_status,
            "latency_ms": duration_ms,
            "agentic_v10": {
                "schema_version": AGENTIC_V10_TRACE_SCHEMA_VERSION,
                "response_status": response_status,
                "setup_snapshot": dict(setup_snapshot or {}),
                "decomposition": decomposition.model_dump(mode="json"),
                "branches": branches,
                "deduplicated_evidence": [
                    {
                        "reference_id": f"[Ref {index}]",
                        "context_origin": origin,
                        **_trace_document(document, score),
                    }
                    for index, (document, score, origin) in enumerate(context_records, start=1)
                ],
                "source_document_mapping": source_document_mapping,
                "context_pack": {
                    "strategy": "top1_raw_same_document_neighbors",
                    "rendered_context": context_text,
                    "top1_evidence_count": len(top1_records),
                    "neighbor_evidence_count": len(context_records) - len(top1_records),
                    "neighbor_lookup_error": neighbor_lookup_error,
                    "source_doc_ids": source_doc_ids,
                },
                "synthesis": {
                    "prompt_messages": synthesis_messages,
                    "token_usage": synthesis_usage,
                    "failure_diagnostic": synthesis_error,
                },
            },
        }
        return RAGResult(
            answer=answer,
            source_doc_ids=source_doc_ids,
            documents=final_documents,
            usage=synthesis_usage,
            agent_trace=trace,
        )

    async def _decompose(self, question: str) -> SubQueryDecompositionTrace:
        if isinstance(self._decomposer, SubQueryDecomposer):
            return await self._decomposer.decompose_with_trace(question)
        items = await self._decomposer.decompose(question)
        return SubQueryDecompositionTrace(sub_queries=items)

    @staticmethod
    def _chunk_position(document: Document) -> tuple[int, int] | None:
        metadata = document.metadata or {}
        page = metadata.get("page_number", metadata.get("page"))
        chunk_index = metadata.get("chunk_index_in_page")
        if isinstance(page, bool) or isinstance(chunk_index, bool):
            return None
        try:
            return int(page), int(chunk_index)
        except (TypeError, ValueError):
            return None

    @classmethod
    async def _same_document_neighbors(
        cls,
        *,
        user_id: str,
        selected_documents: list[Document],
        authorized_doc_ids: set[str],
    ) -> tuple[list[tuple[Document, str]], str | None]:
        if not selected_documents:
            return [], None
        try:
            all_documents = await load_user_vector_documents_async(user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("v10 neighbor lookup failed: %s", exc)
            return [], type(exc).__name__

        selected_doc_ids = {
            str(get_document_id(document.metadata) or "")
            for document in selected_documents
        }
        grouped: dict[str, list[Document]] = {}
        for document in all_documents:
            document_id = str(get_document_id(document.metadata) or "")
            if document_id not in selected_doc_ids:
                continue
            if authorized_doc_ids and document_id not in authorized_doc_ids:
                continue
            if cls._chunk_position(document) is None:
                continue
            grouped.setdefault(document_id, []).append(document)
        for documents in grouped.values():
            documents.sort(key=lambda document: (cls._chunk_position(document), _document_key(document)))

        neighbors: list[tuple[Document, str]] = []
        for selected in selected_documents:
            document_id = str(get_document_id(selected.metadata) or "")
            siblings = grouped.get(document_id, [])
            selected_key = _document_key(selected)
            try:
                selected_index = next(
                    index
                    for index, candidate in enumerate(siblings)
                    if _document_key(candidate) == selected_key
                )
            except StopIteration:
                continue
            for neighbor_index in (selected_index - 1, selected_index + 1):
                if 0 <= neighbor_index < len(siblings):
                    neighbors.append((siblings[neighbor_index], selected_key))
        return neighbors, None

    @staticmethod
    def _context_block(
        *,
        index: int,
        document: Document,
        score: float | None,
        origin: str,
    ) -> str:
        metadata = document.metadata or {}
        score_line = f"- Rerank Score: {score:.4f}\n" if score is not None else ""
        return (
            f"### 檢索來源證據 [Ref {index}]\n- 文檔名稱: {_document_title(document)}\n"
            f"- 文檔 ID: {get_document_id(metadata) or ''}\n"
            f"- 頁碼: {metadata.get('page_number', metadata.get('page', 'N/A'))}\n"
            f"- Context role: {origin}\n{score_line}"
            f"- 內容:\n\"\"\"\n{document.page_content.strip()}\n\"\"\""
        )

    @staticmethod
    def _synthesis_messages(
        question: str, overview: str, context_text: str
    ) -> list[dict[str, str]]:
        registry = get_agentic_v10_prompt_registry()
        return [
            {"role": "system", "content": registry.get("grounded_synthesis_system").template},
            {
                "role": "user",
                "content": format_agentic_v10_prompt(
                    "grounded_synthesis_user",
                    question=question,
                    subqueries_overview=overview,
                    context_text=context_text,
                ),
            },
        ]

    @staticmethod
    async def _synthesize(
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, int], str | None]:
        try:
            llm = get_llm(purpose="synthesizer")
            response = await (llm.ainvoke(messages) if hasattr(llm, "ainvoke") else llm.invoke(messages))
            content = getattr(response, "content", response)
            if isinstance(content, list):
                text = "".join(part.get("text", str(part)) if isinstance(part, dict) else str(part) for part in content)
            else:
                text = str(content)
            return text.strip(), _normalize_usage(response), None
        except Exception as exc:  # noqa: BLE001
            logger.error("v10 synthesis failed: %s", exc)
            return (
                "生成回答時發生錯誤，但檢索已完成；請參考已保存的檢索來源。",
                {},
                type(exc).__name__,
            )
