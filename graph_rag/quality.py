"""Deterministic static and runtime GraphRAG quality calculations."""

from __future__ import annotations

from collections.abc import Iterable

from evaluation.observability_storage import (
    EvaluationGraphEventRepository,
    EvaluationGraphEvidenceItemRepository,
)
from graph_rag.schemas import (
    GraphQualityIssue,
    GraphQualityResponse,
    GraphRuntimeQualityResponse,
)
from graph_rag.store import GraphStore


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def compute_graph_quality(store: GraphStore) -> GraphQualityResponse:
    """Summarize graph-store quality without inferring runtime retrieval behavior."""
    nodes = store.get_all_nodes()
    edges = store.get_all_edges()
    node_count = len(nodes)
    edge_count = len(edges)
    edge_ids = [store.edge_id(edge.source_id, edge.target_id, edge.relation) for edge in edges]
    provenance_count = sum(
        store.get_edge_provenance_status(edge_id) == "full" for edge_id in edge_ids
    )
    orphan_count = sum(store.graph.degree(node.id) == 0 for node in nodes)
    method_nodes = [node for node in nodes if node.entity_type.value == "method"]
    canonical_methods = {
        entity.canonical_id
        for entity in store.canonical_entities.values()
        if entity.entity_type.value == "method"
    }
    duplicate_methods = (
        max(0, len(method_nodes) - len(canonical_methods))
        if canonical_methods
        else 0
    )
    generic_relations = sum(edge.relation in {"uses", "related"} for edge in edges)
    issues: list[GraphQualityIssue] = []
    if edge_count and provenance_count < edge_count:
        issues.append(GraphQualityIssue(code="missing_edge_provenance", severity="warning", message="Some graph edges lack full provenance.", recommended_action="Rebuild affected documents with schema-first extraction."))
    if orphan_count:
        issues.append(GraphQualityIssue(code="orphan_nodes", severity="info", message="The graph contains nodes without edges.", recommended_action="Review raw candidates or rebuild the source document."))
    claim_scope_missing_count = sum(
        entity.entity_type.value == "claim" and not entity.identity_key
        for entity in store.canonical_entities.values()
    )
    if generic_relations / edge_count > 0.25 if edge_count else False:
        issues.append(GraphQualityIssue(code="generic_relations", severity="warning", message="Generic graph relations reduce evidence precision.", recommended_action="Review raw candidates and keep only schema-v1 relation types."))
    if duplicate_methods:
        issues.append(GraphQualityIssue(code="duplicate_methods", severity="warning", message="Method nodes appear to duplicate canonical identities.", recommended_action="Review alias mappings before merging methods."))
    if claim_scope_missing_count:
        issues.append(GraphQualityIssue(code="claims_missing_scope", severity="critical", message="Claim nodes are missing an explicit claim identity scope.", recommended_action="Re-extract affected claims with type, scope, and source document."))
    score_penalty = {"info": 5, "warning": 10, "critical": 30}
    score = max(0, 100 - sum(score_penalty[issue.severity] for issue in issues))
    return GraphQualityResponse(
        score=score,
        num_nodes=node_count,
        num_edges=edge_count,
        edge_with_provenance_ratio=(provenance_count / edge_count if edge_count else 1.0),
        generic_relation_ratio=(generic_relations / edge_count if edge_count else 0.0),
        duplicate_method_node_ratio=(duplicate_methods / len(method_nodes) if method_nodes else 0.0),
        orphan_node_ratio=(orphan_count / node_count if node_count else 0.0),
        claim_scope_missing_count=claim_scope_missing_count,
        issues=issues,
    )


def compute_graph_runtime_quality(
    *,
    campaign_id: str | None,
    community_summary_used_as_evidence_count: int = 0,
    unsupported_graph_claim_rate: float | None = None,
    graph_context_noise_ratio: float | None = None,
    unresolved_anchor_count: int = 0,
    graph_to_chunk_success_rate: float | None = None,
) -> GraphRuntimeQualityResponse:
    """Build runtime violations from evaluation-derived values, never GraphStore guesses."""
    issues: list[GraphQualityIssue] = []
    if community_summary_used_as_evidence_count:
        issues.append(GraphQualityIssue(code="community_summary_used_as_evidence", severity="critical", message="Community summaries were recorded as final evidence.", recommended_action="Inspect graph evidence eligibility and context packing."))
    if unresolved_anchor_count:
        issues.append(GraphQualityIssue(code="unresolved_anchors", severity="warning", message="Some graph anchors could not resolve to source evidence.", recommended_action="Rebuild stale graph provenance or inspect source chunks."))
    return GraphRuntimeQualityResponse(campaign_id=campaign_id, community_summary_used_as_evidence_count=community_summary_used_as_evidence_count, unsupported_graph_claim_rate=unsupported_graph_claim_rate, graph_context_noise_ratio=graph_context_noise_ratio, unresolved_anchor_count=unresolved_anchor_count, graph_to_chunk_success_rate=graph_to_chunk_success_rate, issues=issues)


async def compute_campaign_runtime_quality(
    campaign_id: str,
    *,
    event_repository: EvaluationGraphEventRepository | None = None,
    evidence_repository: EvaluationGraphEvidenceItemRepository | None = None,
) -> GraphRuntimeQualityResponse:
    """Aggregate runtime quality only from persisted evaluation observability rows."""
    events_by_run = await (event_repository or EvaluationGraphEventRepository()).list_graph_events_for_campaign(campaign_id)
    evidence_by_run = await (evidence_repository or EvaluationGraphEvidenceItemRepository()).list_graph_evidence_items_for_campaign(campaign_id)
    events = [event for rows in events_by_run.values() for event in rows]
    evidence_items = [item for rows in evidence_by_run.values() for item in rows]
    final_items = [item for item in evidence_items if item.used_in_answer]
    unsupported_count = sum(item.provenance_status != "full" for item in final_items)
    unresolved_anchor_count = sum(
        item.provenance_status != "full" for item in evidence_items
    )
    community_summary_used_as_evidence_count = sum(
        not item.source_chunk_ids and not item.asset_ids for item in final_items
    )
    return compute_graph_runtime_quality(
        campaign_id=campaign_id,
        community_summary_used_as_evidence_count=community_summary_used_as_evidence_count,
        unsupported_graph_claim_rate=(
            unsupported_count / len(final_items) if final_items else None
        ),
        graph_context_noise_ratio=_mean(event.graph_noise_ratio for event in events),
        unresolved_anchor_count=unresolved_anchor_count,
        graph_to_chunk_success_rate=_mean(
            event.graph_to_chunk_success_rate for event in events
        ),
    )
