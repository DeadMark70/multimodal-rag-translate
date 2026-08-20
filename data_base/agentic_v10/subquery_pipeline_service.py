"""Evaluation-only Agentic RAG v10 with one fail-soft evidence drill-down."""

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
from data_base.vector_store_manager import (
    get_user_retriever_async,
    load_user_vector_documents_async,
)

logger = logging.getLogger(__name__)

AGENTIC_V10_EXECUTION_PROFILE = (
    "agentic_eval_v10_top1_neighbors_conditional_drilldown"
)
AGENTIC_V10_CONTEXT_POLICY_VERSION = (
    "v10_top1_neighbors_conditional_drilldown_ledger_matrix"
)
AGENTIC_V10_TRACE_SCHEMA_VERSION = "5"

_AUDIT_REFERENCE_ID_PATTERN = re.compile(
    r"^\s*(?:\[Ref\s+([1-9]\d*)\]|Ref\s+([1-9]\d*))\s*$"
)


class CoverageRequirement(BaseModel):
    """One explicit answer obligation identified from the user's question."""

    id: str
    entity: str
    criterion: str


class EntityCriterionMatrixCell(BaseModel):
    """Coverage state for one explicit question requirement."""

    requirement_id: str
    coverage: Literal["supported", "partial", "missing"]
    reference_ids: list[str] = Field(default_factory=list)


class ExtractiveEvidenceLedgerEntry(BaseModel):
    """A source reference mapped to the requirements it supports."""

    reference_id: str
    requirement_ids: list[str] = Field(default_factory=list)


class PriorityGap(BaseModel):
    """The single highest-priority missing fact to retrieve once."""

    requirement_id: str
    missing_information: str
    retrieval_query: str


class CoverageAuditResponse(BaseModel):
    """Native JSON Schema response used to route the bounded v10 workflow."""

    needs_drill_down: Literal[0, 1]
    answer: str | None = None
    requirements: list[CoverageRequirement] = Field(default_factory=list)
    entity_criterion_matrix: list[EntityCriterionMatrixCell] = Field(
        default_factory=list
    )
    extractive_evidence_ledger: list[ExtractiveEvidenceLedgerEntry] = Field(
        default_factory=list
    )
    priority_gap: PriorityGap | None = None


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


def _merge_usage(*usages: dict[str, int]) -> dict[str, int]:
    """Sum independently captured provider usage without inventing fields."""
    merged: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                merged[key] = merged.get(key, 0) + max(value, 0)
    return merged


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip()


ContextEntry = tuple[int, Document, float | None, str]


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

        initial_context_entries: list[ContextEntry] = [
            (index, document, score, origin)
            for index, (document, score, origin) in enumerate(
                context_records, start=1
            )
        ]
        context_text = self._render_context_entries(initial_context_entries)
        overview = "\n".join(
            f"{item.id}. [{item.target_entity}] {item.focus} (查詢: {item.query})"
            for item in decomposition.sub_queries
        )
        audit_messages = self._coverage_audit_messages(
            question=question,
            overview=overview,
            context_text=context_text,
        )
        audit, audit_usage, audit_error, audit_latency_ms = (
            await self._coverage_audit(audit_messages)
            if initial_context_entries
            else (None, {}, "no_initial_evidence", 0.0)
        )
        reference_validation = (
            self._validate_coverage_audit(
                audit=audit,
                valid_reference_ids={
                    f"[Ref {reference_id}]"
                    for reference_id, _, _, _ in initial_context_entries
                },
            )
            if audit is not None
            else {
                "validated": False,
                "failure_reason": audit_error or "audit_unavailable",
                "valid_reference_ids": [
                    f"[Ref {reference_id}]"
                    for reference_id, _, _, _ in initial_context_entries
                ],
                "ledger_reference_ids": [],
            }
        )
        final_context_entries = list(initial_context_entries)
        final_context_text = context_text
        drill_down: dict[str, Any] | None = None
        route = "no_initial_evidence"
        synthesis_messages: list[dict[str, str]] = []
        synthesis_usage: dict[str, int] = {}
        synthesis_error: str | None = None
        if not initial_context_entries:
            answer = "目前知識庫沒有可用的相關證據，因此無法根據文件回答此問題。請補充或上傳相關文獻後再試。"
            response_status = "qualified_partial"
        elif not reference_validation["validated"]:
            route = "audit_fallback_raw_synthesis"
            synthesis_messages = self._synthesis_messages(question, overview, context_text)
            answer, synthesis_usage, synthesis_error = await self._synthesize(
                synthesis_messages
            )
            response_status = "qualified_partial" if synthesis_error else "complete"
        elif audit.needs_drill_down == 0:
            route = "audit_answer"
            answer = str(audit.answer).strip()
            synthesis_messages = audit_messages
            synthesis_usage = audit_usage
            response_status = "complete"
        else:
            route = "conditional_drill_down"
            ledger_reference_ids = set(reference_validation["ledger_reference_ids"])
            final_context_entries = [
                entry
                for entry in initial_context_entries
                if f"[Ref {entry[0]}]" in ledger_reference_ids
            ]
            drill_down, drill_entries = await self._run_drill_down(
                user_id=user_id,
                retriever=retriever,
                retriever_error=retriever_error,
                allowed_doc_ids=allowed,
                reranker=reranker,
                priority_gap=audit.priority_gap,
                initial_entries=initial_context_entries,
            )
            final_context_entries.extend(drill_entries)
            final_context_text = self._render_context_entries(final_context_entries)
            if final_context_entries:
                synthesis_messages = self._drilldown_synthesis_messages(
                    question=question,
                    matrix=audit.entity_criterion_matrix,
                    ledger=audit.extractive_evidence_ledger,
                    context_text=final_context_text,
                )
                answer, synthesis_usage, synthesis_error = await self._synthesize(
                    synthesis_messages
                )
            else:
                answer = "目前沒有可用的來源證據可補足此問題的關鍵缺口，因此無法根據文件完成回答。"
                synthesis_error = "NoFinalEvidence"
            response_status = (
                "complete"
                if drill_down["new_evidence_count"] and not synthesis_error
                else "qualified_partial"
            )
        final_documents = [document for _, document, _, _ in final_context_entries]
        source_doc_ids = list(
            dict.fromkeys(
                str(get_document_id(document.metadata) or "")
                for document in final_documents
                if get_document_id(document.metadata)
            )
        )
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
                    "strategy": "top1_raw_same_document_neighbors_conditional_drilldown",
                    "rendered_context": context_text,
                    "final_rendered_context": final_context_text,
                    "top1_evidence_count": len(top1_records),
                    "neighbor_evidence_count": len(context_records) - len(top1_records),
                    "final_evidence_count": len(final_context_entries),
                    "neighbor_lookup_error": neighbor_lookup_error,
                    "source_doc_ids": source_doc_ids,
                },
                "coverage_audit": {
                    "route": route,
                    "prompt_messages": audit_messages,
                    "structured_payload": (
                        audit.model_dump(mode="json") if audit is not None else None
                    ),
                    "reference_validation": reference_validation,
                    "token_usage": audit_usage,
                    "latency_ms": audit_latency_ms,
                    "failure_diagnostic": audit_error,
                    "unresolved_requirement_count": (
                        sum(
                            cell.coverage != "supported"
                            for cell in audit.entity_criterion_matrix
                        )
                        if audit is not None
                        else None
                    ),
                },
                "drill_down": drill_down,
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
            usage=(
                _merge_usage(audit_usage, synthesis_usage)
                if route == "conditional_drill_down"
                else synthesis_usage
            ),
            agent_trace=trace,
        )

    async def _decompose(self, question: str) -> SubQueryDecompositionTrace:
        if isinstance(self._decomposer, SubQueryDecomposer):
            return await self._decomposer.decompose_with_trace(question)
        items = await self._decomposer.decompose(question)
        return SubQueryDecompositionTrace(sub_queries=items)

    async def _run_drill_down(
        self,
        *,
        user_id: str,
        retriever: Any | None,
        retriever_error: str | None,
        allowed_doc_ids: set[str],
        reranker: DocumentReranker,
        priority_gap: PriorityGap | None,
        initial_entries: list[ContextEntry],
    ) -> tuple[dict[str, Any], list[ContextEntry]]:
        """Retrieve exactly one additional Top-1 branch for a validated gap."""
        started = time.perf_counter()
        query = priority_gap.retrieval_query if priority_gap is not None else ""
        candidates: list[Document] = []
        retrieval_error: str | None = None
        if retriever is None:
            retrieval_error = retriever_error or "RetrieverUnavailable"
        else:
            try:
                response = await retrieve_hybrid_documents(
                    query,
                    retriever,
                    enable_hyde=False,
                    enable_multi_query=False,
                )
                candidates = list(response.documents)
                if allowed_doc_ids:
                    candidates = [
                        document
                        for document in candidates
                        if str(get_document_id(document.metadata) or "")
                        in allowed_doc_ids
                    ]
                candidates = candidates[:4]
            except Exception as exc:  # noqa: BLE001
                logger.warning("v10 drill-down retrieval failed: %s", exc)
                retrieval_error = type(exc).__name__
        rerank_error: str | None = None
        try:
            ranked = (
                reranker.rerank_with_scores(query, candidates, top_k=1)
                if candidates
                else []
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("v10 drill-down rerank failed: %s", exc)
            ranked = [(document, 0.5) for document in candidates[:1]]
            rerank_error = type(exc).__name__
        selected_documents = [document for document, _ in ranked]
        neighbors, neighbor_lookup_error = await self._same_document_neighbors(
            user_id=user_id,
            selected_documents=selected_documents,
            authorized_doc_ids=allowed_doc_ids,
        )
        known_keys = {
            _document_key(document) for _, document, _, _ in initial_entries
        }
        entries: list[ContextEntry] = []
        next_reference_id = len(initial_entries) + 1
        for document, score in ranked:
            key = _document_key(document)
            if key not in known_keys:
                known_keys.add(key)
                entries.append(
                    (next_reference_id, document, float(score), "drilldown_top1")
                )
                next_reference_id += 1
        for document, _ in neighbors:
            key = _document_key(document)
            if key not in known_keys:
                known_keys.add(key)
                entries.append(
                    (
                        next_reference_id,
                        document,
                        None,
                        "drilldown_same_document_neighbor",
                    )
                )
                next_reference_id += 1
        return {
            "attempted": True,
            "priority_gap": (
                priority_gap.model_dump(mode="json") if priority_gap else None
            ),
            "query": query,
            "raw_candidates": [_trace_document(document) for document in candidates],
            "reranked_candidates": [
                _trace_document(document, score) for document, score in ranked
            ],
            "retrieval_error": retrieval_error,
            "rerank_error": rerank_error,
            "neighbor_lookup_error": neighbor_lookup_error,
            "new_evidence_count": len(entries),
            "source_document_mapping": [
                {
                    "reference_id": f"[Ref {reference_id}]",
                    "context_origin": origin,
                    "document": _trace_document(document, score),
                }
                for reference_id, document, score, origin in entries
            ],
            "latency_ms": (time.perf_counter() - started) * 1000,
        }, entries

    @staticmethod
    def _validate_coverage_audit(
        *,
        audit: CoverageAuditResponse,
        valid_reference_ids: set[str],
    ) -> dict[str, Any]:
        normalized_references = (
            AgenticV10PipelineService._normalize_audit_reference_ids(audit)
        )
        requirement_ids = [requirement.id for requirement in audit.requirements]
        requirement_set = set(requirement_ids)
        failure_reason: str | None = None
        if not requirement_ids or len(requirement_ids) != len(requirement_set):
            failure_reason = "invalid_requirements"
        elif not audit.entity_criterion_matrix:
            failure_reason = "missing_matrix"
        elif (
            len(audit.entity_criterion_matrix) != len(requirement_set)
            or {
                cell.requirement_id for cell in audit.entity_criterion_matrix
            }
            != requirement_set
        ):
            failure_reason = "incomplete_matrix"
        elif any(
            cell.requirement_id not in requirement_set
            or any(reference_id not in valid_reference_ids for reference_id in cell.reference_ids)
            or (
                cell.coverage in {"supported", "partial"}
                and not cell.reference_ids
            )
            for cell in audit.entity_criterion_matrix
        ):
            failure_reason = "invalid_matrix_reference"
        elif any(
            entry.reference_id not in valid_reference_ids
            or not entry.requirement_ids
            or any(
                requirement_id not in requirement_set
                for requirement_id in entry.requirement_ids
            )
            for entry in audit.extractive_evidence_ledger
        ):
            failure_reason = "invalid_ledger_reference"
        else:
            ledger_pairs = {
                (entry.reference_id, requirement_id)
                for entry in audit.extractive_evidence_ledger
                for requirement_id in entry.requirement_ids
            }
            if any(
                (reference_id, cell.requirement_id) not in ledger_pairs
                for cell in audit.entity_criterion_matrix
                for reference_id in cell.reference_ids
            ):
                failure_reason = "matrix_ledger_mismatch"
            elif audit.needs_drill_down == 0 and (
                not audit.answer
                or any(
                    cell.coverage != "supported"
                    for cell in audit.entity_criterion_matrix
                )
            ):
                failure_reason = "invalid_answer_route"
            elif audit.needs_drill_down == 1 and (
                audit.answer is not None
                or audit.priority_gap is None
                or audit.priority_gap.requirement_id not in requirement_set
                or not audit.priority_gap.retrieval_query.strip()
                or not any(
                    cell.requirement_id == audit.priority_gap.requirement_id
                    and cell.coverage != "supported"
                    for cell in audit.entity_criterion_matrix
                )
            ):
                failure_reason = "invalid_drill_down_route"
        return {
            "validated": failure_reason is None,
            "failure_reason": failure_reason,
            "valid_reference_ids": sorted(valid_reference_ids),
            "normalized_references": normalized_references,
            "ledger_reference_ids": list(
                dict.fromkeys(
                    entry.reference_id for entry in audit.extractive_evidence_ledger
                )
            ),
        }

    @staticmethod
    def _normalize_audit_reference_ids(
        audit: CoverageAuditResponse,
    ) -> list[dict[str, str]]:
        """Canonicalize only the two Gemini reference forms we explicitly allow."""

        normalized: list[dict[str, str]] = []

        def normalize(value: str, location: str) -> str:
            match = _AUDIT_REFERENCE_ID_PATTERN.fullmatch(value)
            if match is None:
                return value
            canonical = f"[Ref {match.group(1) or match.group(2)}]"
            if canonical != value:
                normalized.append(
                    {"location": location, "original": value, "canonical": canonical}
                )
            return canonical

        for cell_index, cell in enumerate(audit.entity_criterion_matrix):
            cell.reference_ids = [
                normalize(reference_id, f"matrix[{cell_index}].reference_ids[{reference_index}]")
                for reference_index, reference_id in enumerate(cell.reference_ids)
            ]
        for entry_index, entry in enumerate(audit.extractive_evidence_ledger):
            entry.reference_id = normalize(
                entry.reference_id,
                f"ledger[{entry_index}].reference_id",
            )
        return normalized

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

    @classmethod
    def _render_context_entries(cls, entries: list[ContextEntry]) -> str:
        return "\n\n".join(
            cls._context_block(
                index=reference_id,
                document=document,
                score=score,
                origin=origin,
            )
            for reference_id, document, score, origin in entries
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
    def _coverage_audit_messages(
        *, question: str, overview: str, context_text: str
    ) -> list[dict[str, str]]:
        registry = get_agentic_v10_prompt_registry()
        return [
            {
                "role": "system",
                "content": registry.get("coverage_audit_system").template,
            },
            {
                "role": "user",
                "content": format_agentic_v10_prompt(
                    "coverage_audit_user",
                    question=question,
                    subqueries_overview=overview,
                    context_text=context_text,
                ),
            },
        ]

    @staticmethod
    def _drilldown_synthesis_messages(
        *,
        question: str,
        matrix: list[EntityCriterionMatrixCell],
        ledger: list[ExtractiveEvidenceLedgerEntry],
        context_text: str,
    ) -> list[dict[str, str]]:
        registry = get_agentic_v10_prompt_registry()
        return [
            {
                "role": "system",
                "content": registry.get("drilldown_synthesis_system").template,
            },
            {
                "role": "user",
                "content": format_agentic_v10_prompt(
                    "drilldown_synthesis_user",
                    question=question,
                    matrix_text=json.dumps(
                        [cell.model_dump(mode="json") for cell in matrix],
                        ensure_ascii=False,
                    ),
                    ledger_text=json.dumps(
                        [entry.model_dump(mode="json") for entry in ledger],
                        ensure_ascii=False,
                    ),
                    context_text=context_text,
                ),
            },
        ]

    @staticmethod
    async def _coverage_audit(
        messages: list[dict[str, str]],
    ) -> tuple[CoverageAuditResponse | None, dict[str, int], str | None, float]:
        started = time.perf_counter()
        try:
            llm = get_llm(purpose="synthesizer")
            if not hasattr(llm, "with_structured_output"):
                raise TypeError("StructuredOutputUnavailable")
            structured_llm = llm.with_structured_output(
                CoverageAuditResponse,
                method="json_schema",
                include_raw=True,
            )
            response = await (
                structured_llm.ainvoke(messages)
                if hasattr(structured_llm, "ainvoke")
                else structured_llm.invoke(messages)
            )
            parsed = response.get("parsed") if isinstance(response, dict) else response
            raw = response.get("raw") if isinstance(response, dict) else response
            if parsed is None:
                raise ValueError("StructuredOutputInvalid")
            audit = (
                parsed
                if isinstance(parsed, CoverageAuditResponse)
                else CoverageAuditResponse.model_validate(parsed)
            )
            return (
                audit,
                _normalize_usage(raw),
                None,
                (time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("v10 coverage audit failed; using raw synthesis: %s", exc)
            return (
                None,
                {},
                type(exc).__name__,
                (time.perf_counter() - started) * 1000,
            )

    @staticmethod
    async def _synthesize(
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, int], str | None]:
        try:
            llm = get_llm(purpose="synthesizer")
            response = await (llm.ainvoke(messages) if hasattr(llm, "ainvoke") else llm.invoke(messages))
            return _response_text(response), _normalize_usage(response), None
        except Exception as exc:  # noqa: BLE001
            logger.error("v10 synthesis failed: %s", exc)
            return (
                "生成回答時發生錯誤，但檢索已完成；請參考已保存的檢索來源。",
                {},
                type(exc).__name__,
            )
