"""GraphRAG policy, context, evidence, and observability runtime."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple, Union
from uuid import uuid4

from langchain_core.documents import Document

from data_base.agentic_v9.schemas import LlmInvoker
from data_base.document_metadata import get_document_id
from evaluation.schemas import EvaluationGraphEvent, EvaluationGraphEvidenceItem
from graph_rag.anchor_resolver import (
    ChunkAnchorResolver,
    ChunkLookup,
    VectorStoreChunkLookup,
)
from graph_rag.feature_flags import get_graph_feature_flags
from graph_rag.generic_mode import (
    GenericGraphRouter,
    GraphEvidence,
    GraphQueryHints,
    GraphRouteDecision,
    estimate_token_count,
    merge_graph_evidence,
    merge_graph_evidence_bundle,
)
from graph_rag.schemas import GraphEvidenceBundle, is_graph_evidence_item_eligible
from graph_rag.store import GraphStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, Optional[Dict[str, Any]]], Awaitable[None]]

# GraphRAG-related keywords for auto mode detection
_GRAPH_KEYWORDS = [
    "關係",
    "連結",
    "趨勢",
    "比較",
    "對比",
    "這些論文",
    "這幾篇",
    "跨文件",
    "綜合",
    "relationship",
    "connection",
    "trend",
    "compare",
    "across",
    "these papers",
    "multi-document",
]

DEFAULT_GRAPH_LOCAL_HOPS = 2
DEFAULT_GRAPH_LOCAL_MAX_NODES = 20

class GraphContextDetails(NamedTuple):
    """Graph retrieval execution metadata used for observability writes."""

    route_decision: GraphRouteDecision
    matched_entity_ids: List[str]
    community_ids: List[int]
    candidate_evidence_count: int
    graph_latency_ms: int
    graph_index_version: Optional[int] = None


@dataclass(frozen=True, slots=True)
class GraphNeedDecision:
    """Classify whether GraphRAG may contribute to this answer path."""

    use_graph: bool
    role: Literal["skip", "locator", "planning"]
    locator_only: bool
    final_graph_context_allowed: bool
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class GraphExecutionStrategy:
    """Choose the only graph path allowed to affect answer context."""

    strategy: Literal["skip", "source_expand", "raw_legacy"]
    gate_decision: Optional[GraphNeedDecision]
    reason: str


@dataclass(frozen=True, slots=True)
class GraphEvidenceLifecycle:
    """Track graph evidence item IDs through the answer-context lifecycle."""

    candidate_item_ids: List[str]
    resolved_item_ids: List[str]
    scope_approved_item_ids: List[str]
    scored_item_ids: List[str]
    packed_item_ids: List[str]
    used_as_locator: bool = False
    graph_to_chunk_attempted: bool = False

    def to_router_reason(self) -> str:
        return "; ".join(
            (
                f"candidate_ids={','.join(self.candidate_item_ids)}",
                f"resolved_ids={','.join(self.resolved_item_ids)}",
                f"scope_approved_ids={','.join(self.scope_approved_item_ids)}",
                f"scored_ids={','.join(self.scored_item_ids)}",
                f"packed_ids={','.join(self.packed_item_ids)}",
            )
        )


def _classify_graph_need(
    question: str,
    manual_override: bool = False,
    asset_registry_available: bool = False,
) -> GraphNeedDecision:
    """Return deterministic graph-use semantics without reading process-global state."""
    normalized = question.lower()
    if manual_override:
        return GraphNeedDecision(True, "locator", True, False, 1.0, "manual override")

    exact_markers = (
        "table",
        "figure",
        "formula",
        "flops",
        "params",
        "exact",
        "numeric",
        "number",
        "數值",
        "公式",
        "表格",
    )
    planning_markers = (
        "evolution",
        "roadmap",
        "research plan",
        "technical evolution",
        "技術演進",
        "規劃",
    )
    graph_markers = (
        "compare",
        "relationship",
        "relation",
        "claim",
        "scope",
        "contradict",
        "across papers",
        "跨文獻",
        "關係",
        "主張",
        "範圍",
    )

    if any(marker in normalized for marker in exact_markers):
        if asset_registry_available:
            return GraphNeedDecision(
                True,
                "locator",
                True,
                False,
                0.55,
                "exact extraction; graph may locate table/formula but cannot answer directly",
            )
        return GraphNeedDecision(
            False,
            "skip",
            False,
            False,
            0.2,
            "exact extraction without usable graph asset locator",
        )
    if any(marker in normalized for marker in planning_markers):
        return GraphNeedDecision(
            True,
            "planning",
            True,
            False,
            0.65,
            "technical evolution requires graph planning only",
        )
    if any(marker in normalized for marker in graph_markers):
        return GraphNeedDecision(
            True,
            "locator",
            False,
            True,
            0.8,
            "relationship or claim-scope query",
        )
    return GraphNeedDecision(
        False, "skip", False, False, 0.3, "no graph-specific intent"
    )


def _graph_execution_strategy(
    *,
    question: str,
    flags: Any,
    graph_evidence_mode: str,
    manual_override: bool,
    asset_registry_available: bool,
    oracle_graph_decision: bool | None = None,
) -> GraphExecutionStrategy:
    """Prevent structured graph modes from ever falling back to raw prompts."""
    if graph_evidence_mode == "planning_only":
        return GraphExecutionStrategy("skip", None, "planning_only")
    if oracle_graph_decision is False:
        return GraphExecutionStrategy("skip", None, "oracle_router_skip")
    if oracle_graph_decision is True:
        manual_override = True

    structured_mode_requested = (
        flags.graph_auto_gate_enabled
        or flags.graph_evidence_locator_enabled
        or flags.graph_to_chunk_enabled
        or graph_evidence_mode in {"locator_only", "locator_to_chunk", "router_auto"}
    )
    gate_decision: Optional[GraphNeedDecision] = None

    if flags.graph_auto_gate_enabled:
        gate_decision = _classify_graph_need(
            question,
            manual_override=manual_override,
            asset_registry_available=asset_registry_available,
        )
        if gate_decision.role == "planning":
            return GraphExecutionStrategy("skip", gate_decision, "planning_only")
        if not gate_decision.use_graph:
            return GraphExecutionStrategy("skip", gate_decision, "auto_gate_skip")
        if flags.graph_to_chunk_enabled:
            return GraphExecutionStrategy(
                "source_expand", gate_decision, "auto_source_expand"
            )
        return GraphExecutionStrategy(
            "skip", gate_decision, "auto_requires_source_expand"
        )

    if flags.graph_to_chunk_enabled:
        return GraphExecutionStrategy("source_expand", None, "source_expand_enabled")
    if structured_mode_requested:
        return GraphExecutionStrategy("skip", None, "locator_requires_source_expand")
    if flags.graph_raw_current_enabled and graph_evidence_mode == "raw_current":
        return GraphExecutionStrategy("raw_legacy", None, "explicit_legacy_raw")
    return GraphExecutionStrategy("skip", None, "raw_legacy_not_explicit")

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


async def _resolve_graph_route_decision(
    question: str,
    search_mode: str,
    status: Any,
    graph_execution_hints: Optional[Dict[str, Any]],
    llm_invoker: LlmInvoker | None = None,
) -> Tuple[GraphRouteDecision, bool, bool]:
    """Resolve the common route used by raw and structured graph retrieval."""
    effective_mode = "generic" if search_mode == "auto" else search_mode
    hints = _filter_graph_query_hints(graph_execution_hints)
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
        decision = await GenericGraphRouter(llm_invoker=llm_invoker).route(
            question,
            has_communities=has_communities,
            hints=hints,
        )
    return decision, has_hierarchy, has_communities


async def _get_graph_context_legacy_raw(
    question: str,
    user_id: str,
    search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
    return_evidence: bool = False,
    return_details: bool = False,
) -> Union[
    str,
    Tuple[str, List[GraphEvidence]],
    Tuple[str, List[GraphEvidence], Optional[GraphContextDetails]],
]:
    """
    Get raw context from knowledge graph for legacy compatibility.

    Args:
        question: User's question.
        user_id: User's ID.
        search_mode: Search mode (`generic` recommended; `auto/local/global/hybrid` are legacy compatibility values).
        graph_execution_hints: Optional internal routing hints for generic mode.

    Returns:
        Graph context string or `(context, evidence_units)` when `return_evidence=True`.
    """
    started_at = time.perf_counter()
    try:
        from graph_rag.store import GraphStore
        from graph_rag.global_search import global_search_evidence
        from graph_rag.local_search import local_search_evidence

        store = GraphStore(user_id)

        # Check if graph exists
        status = store.get_status()
        if not status.has_graph or status.node_count == 0:
            logger.debug(f"No graph data for user {user_id}")
            if return_evidence:
                if return_details:
                    return "", [], None
                return "", []
            return ""

        if status.needs_optimization:
            logger.info(
                "Graph metadata for user %s is stale; skipping automatic chat-path optimization and waiting for explicit maintenance",
                user_id,
            )

        decision, has_hierarchy, has_communities = await _resolve_graph_route_decision(
            question,
            search_mode,
            status,
            graph_execution_hints,
        )

        local_evidence = []
        global_evidence = []
        node_ids: List[str] = []
        community_ids: List[int] = []

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
                level=1
                if (has_hierarchy and decision.query_kind == "summary")
                else None,
            )
            if decision.query_kind == "summary" and has_hierarchy and community_ids:
                selected_leaf_ids = []
                for community_id in community_ids:
                    parent = next(
                        (
                            community
                            for community in store.get_communities(level=1)
                            if community.id == community_id
                        ),
                        None,
                    )
                    if not parent or not parent.child_ids:
                        continue
                    selected_leaf_ids.extend(parent.child_ids[:2])
                if selected_leaf_ids:
                    leaf_communities = [
                        community
                        for community in store.get_communities(level=0)
                        if community.id in selected_leaf_ids
                    ]
                    for leaf in leaf_communities:
                        text = (
                            f"{leaf.title or f'社群 {leaf.id}'}: {leaf.summary or ''}"
                        )
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
                logger.debug(
                    "Generic global search used %s communities", len(community_ids)
                )

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
        details = GraphContextDetails(
            route_decision=decision,
            matched_entity_ids=list(node_ids),
            community_ids=list(community_ids),
            candidate_evidence_count=len(local_evidence) + len(global_evidence),
            graph_latency_ms=max(int((time.perf_counter() - started_at) * 1000), 0),
            graph_index_version=getattr(status, "index_version", None),
        )
        if return_evidence:
            if return_details:
                return merged_context, list(merged_units), details
            return merged_context, list(merged_units)
        return merged_context

    except Exception as e:
        logger.warning(f"Graph context retrieval failed: {e}")
        if return_evidence:
            if return_details:
                return "", [], None
            return "", []
        return ""


async def _get_graph_evidence_bundle(
    question: str,
    user_id: str,
    search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
    chunk_lookup: Optional[ChunkLookup] = None,
    llm_invoker: LlmInvoker | None = None,
) -> GraphEvidenceBundle:
    """Build a structured graph bundle without calling the compatibility wrapper."""
    try:
        from graph_rag.store import GraphStore
        from graph_rag.global_search import global_search_hints
        from graph_rag.local_search import local_search_evidence_items

        store = GraphStore(user_id)
        status = store.get_status()
        if not status.has_graph or status.node_count == 0:
            return GraphEvidenceBundle(query=question, route="none")

        decision, has_hierarchy, has_communities = await _resolve_graph_route_decision(
            question,
            search_mode,
            status,
            graph_execution_hints,
            llm_invoker=llm_invoker,
        )
        evidence_items = []
        hints = []
        if decision.path in ("local-first", "blended"):
            evidence_items, _ = await local_search_evidence_items(
                store,
                question,
                user_id=user_id,
                anchor_resolver=ChunkAnchorResolver(
                    chunk_lookup or VectorStoreChunkLookup()
                ),
                hops=decision.hops,
                max_nodes=decision.max_nodes,
            )

        if decision.path in ("global-first", "blended") and has_communities:
            _, hints, _ = await global_search_hints(
                store,
                question,
                max_communities=decision.max_communities,
                level=1
                if (has_hierarchy and decision.query_kind == "summary")
                else None,
                generate_answers=False,
            )
        return merge_graph_evidence_bundle(
            hints=hints,
            evidence_items=evidence_items,
            token_budget=decision.token_budget,
            query=question,
            route=decision.path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Graph evidence bundle retrieval failed: %s", exc)
        return GraphEvidenceBundle(query=question, route="none")


async def get_graph_evidence_bundle(
    *,
    question: str,
    user_id: str,
    search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
    chunk_lookup: Optional[ChunkLookup] = None,
    llm_invoker: LlmInvoker | None = None,
) -> GraphEvidenceBundle:
    """Public adapter for callers that need the normal evidence-locator bundle."""
    return await _get_graph_evidence_bundle(
        question=question,
        user_id=user_id,
        search_mode=search_mode,
        graph_execution_hints=graph_execution_hints,
        chunk_lookup=chunk_lookup,
        llm_invoker=llm_invoker,
    )


def _render_graph_bundle_for_legacy_prompt(bundle: GraphEvidenceBundle) -> str:
    """Render only source-resolved bundle evidence for legacy prompt consumers."""
    rendered_items = [
        item.evidence_quote
        for item in bundle.final_context_items
        if is_graph_evidence_item_eligible(item) and item.evidence_quote
    ]
    if not rendered_items:
        return ""
    return "=== Graph Evidence ===\n" + "\n".join(rendered_items)


async def _get_graph_context(
    question: str,
    user_id: str,
    search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
    return_evidence: bool = False,
    return_details: bool = False,
) -> Union[
    str,
    Tuple[str, List[GraphEvidence]],
    Tuple[str, List[GraphEvidence], Optional[GraphContextDetails]],
]:
    """Return legacy graph context contracts through the configured graph path."""
    flags = get_graph_feature_flags(_graph_feature_flag_config(graph_execution_hints))
    if not flags.graph_evidence_locator_enabled:
        return await _get_graph_context_legacy_raw(
            question=question,
            user_id=user_id,
            search_mode=search_mode,
            graph_execution_hints=graph_execution_hints,
            return_evidence=return_evidence,
            return_details=return_details,
        )

    bundle = await _get_graph_evidence_bundle(
        question=question,
        user_id=user_id,
        search_mode=search_mode,
        graph_execution_hints=graph_execution_hints,
    )
    context = _render_graph_bundle_for_legacy_prompt(bundle)
    if return_evidence:
        if return_details:
            return context, [], None
        return context, []
    return context


def _to_graph_evidence_documents(evidence_units: List[GraphEvidence]) -> List[Document]:
    """Convert graph evidence units into evaluation-visible documents."""
    evidence_documents: List[Document] = []
    for evidence in evidence_units:
        text = str(getattr(evidence, "text", "") or "")
        if not text:
            continue
        evidence_documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "graph_evidence",
                    "evidence_type": str(getattr(evidence, "evidence_type", "unknown")),
                    "evidence_id": str(getattr(evidence, "evidence_id", "")),
                    "score": getattr(evidence, "score", None),
                    "graph_metadata": dict(getattr(evidence, "metadata", {}) or {}),
                },
            )
        )
    return evidence_documents


def _summarize_graph_evidence_for_log(
    evidence_units: List[GraphEvidence],
) -> dict[str, object]:
    return {
        "node_count": sum(
            1 for item in evidence_units if item.evidence_type == "local_node"
        ),
        "edge_count": sum(
            1 for item in evidence_units if item.evidence_type == "local_edge"
        ),
        "community_count": sum(
            1
            for item in evidence_units
            if item.evidence_type in {"community_summary", "community_answer"}
        ),
        "graph_context_tokens": sum(item.token_estimate for item in evidence_units),
    }


def _normalize_evaluation_metadata(
    mode_hints: Optional[Dict[str, Any]],
    graph_execution_hints: Optional[Dict[str, Any]],
) -> dict[str, Any]:
    for source in (mode_hints, graph_execution_hints):
        if not isinstance(source, dict):
            continue
        payload = source.get("evaluation_metadata")
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _filter_graph_query_hints(
    graph_execution_hints: Optional[Dict[str, Any]],
) -> GraphQueryHints:
    if not isinstance(graph_execution_hints, dict):
        return GraphQueryHints()
    allowed_keys = GraphQueryHints.__dataclass_fields__.keys()
    filtered = {
        key: value
        for key, value in graph_execution_hints.items()
        if key in allowed_keys
    }
    return GraphQueryHints(**filtered)


def _graph_feature_flag_snapshot(
    mode_hints: Optional[Dict[str, Any]],
    graph_execution_hints: Optional[Dict[str, Any]],
    evaluation_metadata: dict[str, Any],
) -> dict[str, bool]:
    source: dict[str, object] = {}
    for candidate in (mode_hints, graph_execution_hints, evaluation_metadata):
        if not isinstance(candidate, dict):
            continue
        flags = candidate.get("graph_feature_flags")
        if isinstance(flags, dict):
            source.update(flags)
    return get_graph_feature_flags(source).to_snapshot()


def _graph_feature_flag_config(
    graph_execution_hints: Optional[Dict[str, Any]],
) -> Dict[str, object]:
    """Read graph flags from the established nested and direct hint forms."""
    if not isinstance(graph_execution_hints, dict):
        return {}
    nested_flags = graph_execution_hints.get("graph_feature_flags")
    config = dict(nested_flags) if isinstance(nested_flags, dict) else {}
    config.update(
        {
            key: value
            for key, value in graph_execution_hints.items()
            if key.startswith("graph_")
        }
    )
    return config


def _hint_enabled(source: Optional[Dict[str, Any]], key: str) -> bool:
    """Read an explicit boolean execution hint without consulting global state."""
    if not isinstance(source, dict):
        return False
    value = source.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _graph_gate_inputs(
    graph_execution_hints: Optional[Dict[str, Any]],
    mode_hints: Optional[Dict[str, Any]],
    graph_flags: Any,
    *,
    asset_probe_result: bool = False,
) -> tuple[bool, bool]:
    """Resolve manual override plus a registry-derived asset availability result."""
    sources = (graph_execution_hints, mode_hints)
    manual_override = any(
        _hint_enabled(source, "graph_manual_override")
        or _hint_enabled(source, "manual_graph_override")
        for source in sources
    )
    asset_registry_available = (
        bool(graph_flags.graph_asset_graph_enabled) and asset_probe_result
    )
    return manual_override, asset_registry_available


def _oracle_graph_decision(
    graph_execution_hints: Optional[Dict[str, Any]],
    mode_hints: Optional[Dict[str, Any]],
) -> bool | None:
    """Read an explicit per-question oracle label used only by evaluation modes."""
    for source in (graph_execution_hints, mode_hints):
        if not isinstance(source, dict):
            continue
        value = source.get("graph_oracle_decision")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"use", "true", "1"}:
                return True
            if normalized in {"skip", "false", "0"}:
                return False
    return None


_CLAIM_SCOPE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "the",
    "to",
    "use",
    "what",
    "which",
    "with",
}


def _claim_scope_approves_chunk(question: str, chunk: Any) -> bool:
    """Require graph-located evidence to share claim-bearing terms with the query."""
    query_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9_]+", question)
        if len(term) > 2 and term.lower() not in _CLAIM_SCOPE_STOPWORDS
    }
    if not query_terms:
        return True
    item = chunk.evidence_item
    evidence_text = " ".join(
        filter(
            None,
            (
                item.evidence_quote,
                item.summary,
                item.relation_type,
                chunk.document.page_content,
            ),
        )
    ).lower()
    return any(term in evidence_text for term in query_terms)


def _required_modalities_for_question(question: str) -> list[str]:
    """Infer lightweight modality requirements for the graph score bonus."""
    lowered_question = question.lower()
    modalities: list[str] = []
    if any(token in lowered_question for token in ("table", "表格", "表")):
        modalities.append("table")
    if any(token in lowered_question for token in ("figure", "image", "圖", "圖片")):
        modalities.append("image")
    return modalities


def _request_scoped_graph_asset_probe(
    *,
    user_id: str,
    question: str,
    documents: List[Document],
    requested_doc_ids: Optional[List[str]],
) -> bool:
    """Resolve asset availability from actual registry entries in this request's scope."""
    doc_scope = set(requested_doc_ids or [])
    if not doc_scope:
        doc_scope = {
            doc_id
            for document in documents
            if (doc_id := get_document_id(document.metadata))
        }
    requested_types: set[str] = set()
    lowered_question = question.lower()
    if any(token in lowered_question for token in ("table", "表格", "表")):
        requested_types.add("table")
    if any(token in lowered_question for token in ("figure", "image", "圖", "圖片")):
        requested_types.add("figure")
    if any(token in lowered_question for token in ("formula", "equation", "公式")):
        requested_types.add("formula")
    return GraphStore(user_id).has_usable_asset_links(doc_scope, requested_types)


def _graph_evidence_mode(
    mode_hints: Optional[Dict[str, Any]],
    graph_execution_hints: Optional[Dict[str, Any]],
    evaluation_metadata: dict[str, Any],
) -> str:
    for candidate in (evaluation_metadata, mode_hints, graph_execution_hints):
        if not isinstance(candidate, dict):
            continue
        value = candidate.get("graph_evidence_mode")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "raw_current"


def _graph_provenance_status(
    *,
    source_doc_ids: List[str],
    source_chunk_ids: List[str],
    pages: List[int],
    asset_ids: List[str],
) -> str:
    if source_doc_ids and source_chunk_ids:
        return "full"
    if source_doc_ids or source_chunk_ids or pages or asset_ids:
        return "partial"
    return "missing"


def _graph_evidence_item_id(graph_event_id: str, evidence_id: str) -> str:
    return f"{graph_event_id}:{evidence_id}"


def _build_graph_evidence_items(
    *,
    graph_event_id: str,
    evidence_units: List[GraphEvidence],
    graph_evidence_mode: str,
    created_at: datetime,
    lifecycle: Optional[GraphEvidenceLifecycle] = None,
) -> List[EvaluationGraphEvidenceItem]:
    items: List[EvaluationGraphEvidenceItem] = []
    used_as_locator = lifecycle.used_as_locator if lifecycle is not None else False
    packed_item_ids = (
        set(lifecycle.packed_item_ids)
        if lifecycle is not None
        else {unit.evidence_id for unit in evidence_units}
    )
    for unit in evidence_units:
        metadata = dict(unit.metadata or {})
        node_ids: List[str] = []
        edge_ids: List[str] = []
        relation_path: List[str] = []
        source_doc_ids = [str(item) for item in metadata.get("doc_ids", []) if item]
        source_chunk_ids = [str(item) for item in metadata.get("chunk_ids", []) if item]
        pages = [
            int(item) for item in metadata.get("pages", []) if isinstance(item, int)
        ]
        asset_ids = [str(item) for item in metadata.get("asset_ids", []) if item]

        if unit.evidence_type == "local_node":
            node_id = metadata.get("node_id")
            if node_id:
                node_ids.append(str(node_id))
        elif unit.evidence_type == "local_edge":
            source_id = metadata.get("source_id")
            target_id = metadata.get("target_id")
            if source_id:
                node_ids.append(str(source_id))
            if target_id:
                node_ids.append(str(target_id))
            edge_ids.append(unit.evidence_id)
            if source_id and target_id:
                relation_path = [str(source_id), str(target_id)]

        items.append(
            EvaluationGraphEvidenceItem(
                graph_evidence_item_id=_graph_evidence_item_id(
                    graph_event_id, unit.evidence_id
                ),
                graph_event_id=graph_event_id,
                node_ids=node_ids,
                edge_ids=edge_ids,
                relation_path=relation_path,
                source_doc_ids=source_doc_ids,
                source_chunk_ids=source_chunk_ids,
                pages=pages,
                asset_ids=asset_ids,
                confidence=max(0.0, min(float(unit.score), 1.0)),
                provenance_status=_graph_provenance_status(
                    source_doc_ids=source_doc_ids,
                    source_chunk_ids=source_chunk_ids,
                    pages=pages,
                    asset_ids=asset_ids,
                ),
                used_as_locator=used_as_locator,
                packed_in_context=unit.evidence_id in packed_item_ids,
                used_in_answer=False,
                supported_claim_ids=[],
                created_at=created_at,
            )
        )
    return items


def _graph_evidence_units_from_bundle(
    bundle: GraphEvidenceBundle,
    *,
    items: Optional[List[Any]] = None,
) -> List[GraphEvidence]:
    """Adapt source-backed final bundle items for the established recorder."""
    units: List[GraphEvidence] = []
    source_items = items if items is not None else bundle.final_context_items
    for item in source_items:
        evidence_type = "local_edge" if item.edge_ids else "local_node"
        units.append(
            GraphEvidence(
                evidence_id=item.item_id,
                evidence_type=evidence_type,
                text=item.evidence_quote or item.summary,
                score=item.confidence,
                token_estimate=estimate_token_count(
                    item.evidence_quote or item.summary
                ),
                metadata={
                    "doc_ids": list(item.source_doc_ids),
                    "chunk_ids": list(item.source_chunk_ids),
                    "asset_ids": list(item.asset_ids),
                    "pages": list(item.pages),
                    "source_id": item.node_ids[0] if item.node_ids else None,
                    "target_id": item.node_ids[1] if len(item.node_ids) > 1 else None,
                },
            )
        )
    return units


def _graph_context_details_for_bundle(
    bundle: GraphEvidenceBundle,
    graph_need_decision: Optional[GraphNeedDecision],
    lifecycle: GraphEvidenceLifecycle,
    graph_latency_ms: int,
) -> GraphContextDetails:
    """Expose structured bundle counts through the existing graph-event contract."""
    route = (
        bundle.route
        if bundle.route in {"local-first", "global-first", "blended"}
        else "local-first"
    )
    candidate_count = len(lifecycle.candidate_item_ids)
    reason_parts = []
    if graph_need_decision is not None:
        reason_parts.append(f"gate={graph_need_decision.reason}")
    if not lifecycle.packed_item_ids:
        reason_parts.append("fallback=no_packed_graph_chunks")
    reason_parts.extend(("strategy=source_expand", lifecycle.to_router_reason()))
    return GraphContextDetails(
        route_decision=GraphRouteDecision(
            query_kind="relation",
            path=route,
            router_reason="; ".join(reason_parts),
        ),
        matched_entity_ids=[],
        community_ids=[],
        candidate_evidence_count=candidate_count,
        graph_latency_ms=graph_latency_ms,
    )


def _graph_fallback_context_details(
    *,
    reason: str,
    graph_latency_ms: int,
    lifecycle: GraphEvidenceLifecycle,
) -> GraphContextDetails:
    return GraphContextDetails(
        route_decision=GraphRouteDecision(
            query_kind="relation",
            path="skip",
            router_reason=(
                f"strategy=source_expand; fallback={reason}; "
                f"{lifecycle.to_router_reason()}"
            ),
        ),
        matched_entity_ids=[],
        community_ids=[],
        candidate_evidence_count=len(lifecycle.candidate_item_ids),
        graph_latency_ms=graph_latency_ms,
    )


async def _record_graph_observability(
    *,
    question: str,
    graph_search_mode: str,
    graph_execution_hints: Optional[Dict[str, Any]],
    mode_hints: Optional[Dict[str, Any]],
    graph_context_details: Optional[GraphContextDetails],
    graph_evidence_units: List[GraphEvidence],
    lifecycle: Optional[GraphEvidenceLifecycle] = None,
) -> None:
    summary = _summarize_graph_evidence_for_log(graph_evidence_units)
    evaluation_metadata = _normalize_evaluation_metadata(
        mode_hints, graph_execution_hints
    )
    if not evaluation_metadata:
        logger.debug(
            "Graph retrieval observability skipped because evaluation metadata was absent: %s",
            summary,
        )
        return

    run_id = str(evaluation_metadata.get("run_id") or "").strip()
    campaign_id = str(evaluation_metadata.get("campaign_id") or "").strip()
    if not run_id or not campaign_id:
        logger.debug(
            "Graph retrieval observability skipped because run_id/campaign_id were missing: %s",
            {
                **summary,
                "run_id_present": bool(run_id),
                "campaign_id_present": bool(campaign_id),
            },
        )
        return

    if graph_context_details is None:
        logger.debug(
            "Graph retrieval observability skipped because graph context details were unavailable: %s",
            summary,
        )
        return

    created_at = datetime.now(timezone.utc)
    feature_flags = _graph_feature_flag_snapshot(
        mode_hints,
        graph_execution_hints,
        evaluation_metadata,
    )
    evidence_mode = _graph_evidence_mode(
        mode_hints,
        graph_execution_hints,
        evaluation_metadata,
    )
    event_id = str(uuid4())
    evidence_items = _build_graph_evidence_items(
        graph_event_id=event_id,
        evidence_units=graph_evidence_units,
        graph_evidence_mode=evidence_mode,
        created_at=created_at,
        lifecycle=lifecycle,
    )
    graph_to_chunk_attempted = (
        lifecycle.graph_to_chunk_attempted if lifecycle is not None else False
    )
    candidate_item_ids = (
        set(lifecycle.candidate_item_ids)
        if lifecycle is not None
        else {unit.evidence_id for unit in graph_evidence_units}
    )
    resolved_item_ids = (
        set(lifecycle.resolved_item_ids)
        if lifecycle is not None
        else {
            unit.evidence_id
            for unit in graph_evidence_units
            if unit.metadata.get("chunk_ids")
        }
    )
    resolved_candidate_item_ids = candidate_item_ids.intersection(resolved_item_ids)
    candidate_count = len(candidate_item_ids)
    graph_snapshot_version = evaluation_metadata.get("graph_snapshot_version")
    if (
        not graph_snapshot_version
        and graph_context_details.graph_index_version is not None
    ):
        graph_snapshot_version = f"index-v{graph_context_details.graph_index_version}"
    graph_schema_version = evaluation_metadata.get("graph_schema_version")
    if not graph_schema_version and feature_flags.get("graph_schema_v1_enabled"):
        graph_schema_version = "graph-schema-v1"

    event = EvaluationGraphEvent(
        graph_event_id=event_id,
        run_id=run_id,
        campaign_id=campaign_id,
        span_id=str(evaluation_metadata.get("span_id") or "") or None,
        graph_query=question,
        graph_search_mode=graph_search_mode,
        graph_evidence_mode=evidence_mode,
        graph_route=graph_context_details.route_decision.path,
        router_reason=graph_context_details.route_decision.router_reason,
        graph_feature_flags=feature_flags,
        graph_snapshot_version=str(graph_snapshot_version)
        if graph_snapshot_version
        else None,
        graph_schema_version=str(graph_schema_version)
        if graph_schema_version
        else None,
        graph_extraction_prompt_version=(
            str(evaluation_metadata.get("graph_extraction_prompt_version"))
            if evaluation_metadata.get("graph_extraction_prompt_version")
            else None
        ),
        matched_entity_ids=list(graph_context_details.matched_entity_ids),
        community_ids=list(graph_context_details.community_ids),
        node_count=int(summary["node_count"]),
        edge_count=int(summary["edge_count"]),
        path_count=int(summary["edge_count"]),
        graph_latency_ms=graph_context_details.graph_latency_ms,
        graph_context_tokens=int(summary["graph_context_tokens"]),
        graph_to_chunk_success_rate=(
            (len(resolved_candidate_item_ids) / candidate_count)
            if graph_to_chunk_attempted and candidate_count
            else None
        ),
        graph_noise_ratio=(
            (candidate_count - len(resolved_candidate_item_ids)) / candidate_count
            if candidate_count
            else None
        ),
        created_at=created_at,
    )

    try:
        from evaluation.observability_storage import EvaluationObservabilityRepository

        repository = EvaluationObservabilityRepository()
        await repository.record_graph_event(event)
        await repository.record_graph_evidence_items(evidence_items)
    except Exception:
        logger.warning("Failed to persist graph retrieval observability", exc_info=True)
