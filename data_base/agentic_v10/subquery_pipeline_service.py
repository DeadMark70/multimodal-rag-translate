"""Evaluation-only Agentic RAG v10: fail-soft sub-query retrieval and synthesis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from typing import Any, Literal
from uuid import uuid4

from langchain_core.documents import Document
from pydantic import BaseModel, Field

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
from data_base.vector_store_manager import get_user_retriever_async

logger = logging.getLogger(__name__)

AGENTIC_V10_EXECUTION_PROFILE = "agentic_v10_top1_map_reduce_evidence_cards"
AGENTIC_V10_CONTEXT_POLICY_VERSION = "v10_map_reduce_evidence_cards"
AGENTIC_V10_TRACE_SCHEMA_VERSION = "2"


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


def _merge_usage(*usage_rows: dict[str, int]) -> dict[str, int]:
    """Aggregate normalized per-call usage without assuming provider-specific keys."""
    merged: dict[str, int] = {}
    for row in usage_rows:
        for key, value in row.items():
            if isinstance(value, int) and value >= 0:
                merged[key] = merged.get(key, 0) + value
    return merged


def _failure_diagnostic(exc: Exception) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _reference_number(value: str) -> int | None:
    match = re.fullmatch(r"\s*\[?\s*ref\s*(\d+)\s*\]?\s*", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


class EvidenceFinding(BaseModel):
    statement: str
    reference_ids: list[str] = Field(min_length=1)


class SubqueryEvidenceCard(BaseModel):
    status: Literal["summarized", "no_evidence", "raw_fallback"]
    subquery_id: str
    target_entity: str
    focus: str
    supported_findings: list[EvidenceFinding] = Field(default_factory=list)
    missing_or_unsupported: list[str] = Field(default_factory=list)


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
                ranked = [(document, 0.5) for document in candidates[:2]]
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

        selected_by_subquery = {
            item.id: (document, score, item)
            for document, score, item in selected
        }
        unique: dict[str, tuple[Document, float, SubQueryItem]] = {}
        for document, score, item in selected:
            unique.setdefault(_document_key(document), (document, score, item))
        final_documents = [value[0] for value in unique.values()]
        source_doc_ids = list(
            dict.fromkeys(
                str(get_document_id(document.metadata) or "")
                for document in final_documents
                if get_document_id(document.metadata)
            )
        )
        reference_by_document_key = {
            _document_key(document): index
            for index, (document, _, _) in enumerate(unique.values(), start=1)
        }
        source_document_mapping = [
            {
                "reference_id": f"[Ref {index}]",
                "document": _trace_document(document, score),
                "selected_by_subquery_ids": [
                    item.id
                    for candidate, _, item in selected
                    if _document_key(candidate) == _document_key(document)
                ],
            }
            for index, (document, score, _) in enumerate(unique.values(), start=1)
        ]
        for branch in branches:
            selection = selected_by_subquery.get(branch["subquery_id"])
            branch["selected_reference_id"] = (
                f"[Ref {reference_by_document_key[_document_key(selection[0])]}]"
                if selection is not None
                else None
            )

        map_llm: Any | None = None
        map_llm_error: str | None = None
        if selected_by_subquery:
            try:
                base_llm = get_llm(purpose="summary")
                map_llm = base_llm.with_structured_output(
                    SubqueryEvidenceCard,
                    method="json_schema",
                    include_raw=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("v10 map-stage initialization failed: %s", exc)
                map_llm_error = _failure_diagnostic(exc)

        mapped = await asyncio.gather(
            *(
                self._map_evidence_card(
                    item=item,
                    selection=selected_by_subquery.get(item.id),
                    reference_by_document_key=reference_by_document_key,
                    map_llm=map_llm,
                    initialization_error=map_llm_error,
                )
                for item in decomposition.sub_queries
            )
        )
        evidence_cards: list[dict[str, Any]] = []
        map_usage: dict[str, int] = {}
        for branch, (card, map_trace) in zip(branches, mapped, strict=True):
            branch["map"] = map_trace
            evidence_cards.append(card)
            map_usage = _merge_usage(map_usage, map_trace["token_usage"])

        overview = "\n".join(f"{item.id}. [{item.target_entity}] {item.focus} (查詢: {item.query})" for item in decomposition.sub_queries)
        evidence_cards_text = json.dumps(evidence_cards, ensure_ascii=False, indent=2)
        synthesis_messages = self._synthesis_messages(question, overview, evidence_cards_text)
        synthesis_usage: dict[str, int] = {}
        synthesis_error: str | None = None
        if final_documents:
            answer, synthesis_usage, synthesis_error = await self._synthesize(synthesis_messages)
            response_status = "qualified_partial" if synthesis_error else "complete"
        else:
            answer = "目前知識庫沒有可用的相關證據，因此無法根據文件回答此問題。請補充或上傳相關文獻後再試。"
            response_status = "qualified_partial"
        usage = _merge_usage(map_usage, synthesis_usage)
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
                        **_trace_document(document, score),
                    }
                    for index, (document, score, _) in enumerate(unique.values(), start=1)
                ],
                "source_document_mapping": source_document_mapping,
                "map_stage": {
                    "structured_output_method": "json_schema",
                    "token_usage": map_usage,
                    "card_count": len(evidence_cards),
                    "fallback_count": sum(card["status"] == "raw_fallback" for card in evidence_cards),
                    "failure_count": sum(
                        bool(branch["map"].get("failure_diagnostic"))
                        for branch in branches
                    ),
                },
                "context_pack": {
                    "evidence_cards": evidence_cards,
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
            usage=usage,
            agent_trace=trace,
        )

    async def _decompose(self, question: str) -> SubQueryDecompositionTrace:
        if isinstance(self._decomposer, SubQueryDecomposer):
            return await self._decomposer.decompose_with_trace(question)
        items = await self._decomposer.decompose(question)
        return SubQueryDecompositionTrace(sub_queries=items)

    @staticmethod
    def _context_block(index: int, document: Document, score: float, item: SubQueryItem) -> str:
        metadata = document.metadata or {}
        return (
            f"### 檢索來源證據 [Ref {index}]\n- 文檔名稱: {_document_title(document)}\n"
            f"- 文檔 ID: {get_document_id(metadata) or ''}\n- 頁碼: {metadata.get('page', 'N/A')}\n"
            f"- 子查詢焦點: {item.focus}\n- Rerank Score: {score:.4f}\n- 內容:\n\"\"\"\n{document.page_content.strip()}\n\"\"\""
        )

    @staticmethod
    def _no_evidence_card(item: SubQueryItem) -> dict[str, Any]:
        return SubqueryEvidenceCard(
            status="no_evidence",
            subquery_id=item.id,
            target_entity=item.target_entity,
            focus=item.focus,
            missing_or_unsupported=["此子問題沒有 rerank 後的可用來源證據。"],
        ).model_dump(mode="json")

    @classmethod
    async def _map_evidence_card(
        cls,
        *,
        item: SubQueryItem,
        selection: tuple[Document, float, SubQueryItem] | None,
        reference_by_document_key: dict[str, int],
        map_llm: Any | None,
        initialization_error: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if selection is None:
            return cls._no_evidence_card(item), {
                "status": "no_evidence",
                "prompt_messages": [],
                "token_usage": {},
                "latency_ms": 0.0,
                "failure_diagnostic": None,
            }

        document, score, _ = selection
        reference_number = reference_by_document_key[_document_key(document)]
        reference_id = f"[Ref {reference_number}]"
        evidence_block = cls._context_block(reference_number, document, score, item)
        messages = cls._evidence_card_messages(item, evidence_block)
        started = time.perf_counter()
        if map_llm is None:
            return cls._raw_fallback_card(item, reference_id, evidence_block), {
                "status": "raw_fallback",
                "prompt_messages": messages,
                "token_usage": {},
                "latency_ms": (time.perf_counter() - started) * 1000,
                "failure_diagnostic": initialization_error or "StructuredOutputUnavailable",
            }

        call_usage: dict[str, int] = {}
        try:
            response = await (
                map_llm.ainvoke(messages)
                if hasattr(map_llm, "ainvoke")
                else map_llm.invoke(messages)
            )
            parsed, raw_response, parsing_error = cls._structured_output_parts(response)
            call_usage = _normalize_usage(raw_response)
            if parsing_error:
                raise ValueError(f"StructuredOutputInvalid: {type(parsing_error).__name__}")
            card = SubqueryEvidenceCard.model_validate(parsed)
            if card.status == "raw_fallback":
                raise ValueError("StructuredOutputInvalid: raw_fallback is pipeline-owned")
            canonical_findings: list[EvidenceFinding] = []
            for finding in card.supported_findings:
                reference_numbers = {
                    _reference_number(value) for value in finding.reference_ids
                }
                if reference_numbers != {reference_number}:
                    raise ValueError("StructuredOutputInvalid: reference_ids do not match source")
                canonical_findings.append(
                    finding.model_copy(update={"reference_ids": [reference_id]})
                )
            card = card.model_copy(
                update={
                    "subquery_id": item.id,
                    "target_entity": item.target_entity,
                    "focus": item.focus,
                    "supported_findings": canonical_findings,
                }
            )
            return card.model_dump(mode="json"), {
                "status": card.status,
                "prompt_messages": messages,
                "token_usage": call_usage,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "failure_diagnostic": None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("v10 map-stage failed for %s: %s", item.id, exc)
            return cls._raw_fallback_card(item, reference_id, evidence_block), {
                "status": "raw_fallback",
                "prompt_messages": messages,
                "token_usage": call_usage,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "failure_diagnostic": _failure_diagnostic(exc),
            }

    @staticmethod
    def _structured_output_parts(response: Any) -> tuple[Any, Any, Any | None]:
        if isinstance(response, dict) and {"parsed", "raw"}.issubset(response):
            return response.get("parsed"), response.get("raw"), response.get("parsing_error")
        return response, response, None

    @staticmethod
    def _raw_fallback_card(
        item: SubQueryItem, reference_id: str, evidence_block: str
    ) -> dict[str, Any]:
        card = SubqueryEvidenceCard(
            status="raw_fallback",
            subquery_id=item.id,
            target_entity=item.target_entity,
            focus=item.focus,
            missing_or_unsupported=[
                "Evidence card structured output failed; use the marked raw source only."
            ],
        ).model_dump(mode="json")
        card["reference_ids"] = [reference_id]
        card["raw_evidence_block"] = evidence_block
        return card

    @staticmethod
    def _evidence_card_messages(
        item: SubQueryItem, evidence_block: str
    ) -> list[dict[str, str]]:
        registry = get_agentic_v10_prompt_registry()
        return [
            {"role": "system", "content": registry.get("subquery_evidence_card_system").template},
            {
                "role": "user",
                "content": format_agentic_v10_prompt(
                    "subquery_evidence_card_user",
                    subquery_id=item.id,
                    target_entity=item.target_entity,
                    focus=item.focus,
                    query=item.query,
                    evidence_block=evidence_block,
                ),
            },
        ]

    @staticmethod
    def _synthesis_messages(
        question: str, overview: str, evidence_cards: str
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
                    evidence_cards=evidence_cards,
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
