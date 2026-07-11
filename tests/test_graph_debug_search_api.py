from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from graph_rag.debug import build_debug_search_response
from graph_rag.schemas import GraphDebugSearchResponse, GraphEvidenceBundle, GraphQualityResponse
from main import app


def test_debug_response_explains_final_context_eligibility() -> None:
    response = build_debug_search_response(
        query="Weak-Mamba first claim scope",
        bundle=GraphEvidenceBundle(
            query="Weak-Mamba first claim scope",
            route="local-first",
        ),
        entity_links=[],
    )

    assert response.query == "Weak-Mamba first claim scope"
    assert response.route == "local-first"
    assert response.final_context_items == []


def test_quality_runtime_and_debug_routes_are_user_scoped_services() -> None:
    app.dependency_overrides[get_current_user_id] = lambda: "test-user-graph"
    static_quality = GraphQualityResponse(
        score=100,
        num_nodes=0,
        num_edges=0,
        edge_with_provenance_ratio=1.0,
        generic_relation_ratio=0.0,
        duplicate_method_node_ratio=0.0,
        orphan_node_ratio=0.0,
    )
    debug_response = GraphDebugSearchResponse(
        query="Weak-Mamba",
        route="none",
    )
    campaign_repository = MagicMock()
    campaign_repository.get = AsyncMock()
    try:
        with (
            patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
            patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
            patch(
                "graph_rag.router.CampaignRepository",
                return_value=campaign_repository,
            ),
            patch("graph_rag.router.compute_graph_quality", return_value=static_quality) as quality,
            patch(
                "graph_rag.router.compute_campaign_runtime_quality",
                new=AsyncMock(return_value={"campaign_id": "campaign-1", "issues": []}),
            ) as runtime_quality,
            patch(
                "graph_rag.router.run_debug_search",
                new=AsyncMock(return_value=debug_response),
            ) as debug_search,
            TestClient(app) as client,
        ):
            quality_response = client.get("/graph/quality")
            runtime_response = client.get("/graph/runtime-quality?campaign_id=campaign-1")
            debug_response_http = client.post(
                "/graph/debug/search",
                json={"query": "Weak-Mamba", "search_mode": "generic"},
            )
    finally:
        app.dependency_overrides = {}

    assert quality_response.status_code == 200
    assert runtime_response.status_code == 200
    assert debug_response_http.status_code == 200
    quality.assert_called_once()
    campaign_repository.get.assert_awaited_once_with(
        user_id="test-user-graph",
        campaign_id="campaign-1",
    )
    runtime_quality.assert_awaited_once_with("campaign-1")
    debug_search.assert_awaited_once_with(
        user_id="test-user-graph",
        query="Weak-Mamba",
        search_mode="generic",
    )
