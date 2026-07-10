"""Pure response builder for GraphRAG query diagnostics."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from graph_rag.schemas import GraphDebugSearchResponse, GraphEvidenceBundle, is_graph_evidence_item_eligible
from graph_rag.store import GraphStore


def build_debug_search_response(*, query: str, bundle: GraphEvidenceBundle, entity_links: list[dict[str, object]]) -> GraphDebugSearchResponse:
    """Expose hints and evidence while retaining the final-context eligibility boundary."""
    return GraphDebugSearchResponse(
        query=query,
        route=bundle.route,
        entity_links=entity_links,
        hints=list(bundle.hints),
        evidence_items=list(bundle.evidence_items),
        final_context_items=[item for item in bundle.final_context_items if is_graph_evidence_item_eligible(item)],
    )


BundleLoader = Callable[[str, str, str], Awaitable[GraphEvidenceBundle]]


async def run_debug_search(
    *,
    user_id: str,
    query: str,
    search_mode: str,
    bundle_loader: BundleLoader | None = None,
) -> GraphDebugSearchResponse:
    """Run the normal evidence-locator path and expose its safe diagnostic output."""
    if bundle_loader is None:
        from data_base.RAG_QA_service import get_graph_evidence_bundle

        async def bundle_loader(
            question: str, owner_id: str, mode: str
        ) -> GraphEvidenceBundle:
            return await get_graph_evidence_bundle(
                question=question,
                user_id=owner_id,
                search_mode=mode,
            )

    store = GraphStore(user_id)
    entity_links: list[dict[str, object]] = []
    for node_id in store.find_canonical_nodes_in_text(query):
        node = store.get_node(node_id)
        canonical = store.canonical_entities.get(node_id)
        if node is None:
            continue
        entity_links.append(
            {
                "node_id": node_id,
                "label": node.label,
                "entity_type": node.entity_type.value,
                "aliases": list(canonical.aliases) if canonical else [],
            }
        )
    bundle = await bundle_loader(query, user_id, search_mode)
    return build_debug_search_response(
        query=query,
        bundle=bundle,
        entity_links=entity_links,
    )
