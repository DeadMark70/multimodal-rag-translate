from __future__ import annotations

import pytest

from graph_rag.quality import compute_graph_quality, compute_graph_runtime_quality
from graph_rag.schemas import EntityType, GraphNode
from graph_rag.store import GraphStore


def test_static_quality_reports_empty_graph_metrics(tmp_path) -> None:
    store = GraphStore("user-1", storage_dir=tmp_path)

    report = compute_graph_quality(store)

    assert report.num_nodes == 0
    assert report.num_edges == 0
    assert report.score == 100


def test_static_quality_does_not_treat_legacy_methods_as_duplicates(tmp_path) -> None:
    store = GraphStore("user-1", storage_dir=tmp_path)
    store.add_node(
        GraphNode(
            id="method-1",
            label="Legacy Method",
            entity_type=EntityType.METHOD,
            doc_ids=["doc-1"],
        )
    )

    report = compute_graph_quality(store)

    assert report.duplicate_method_node_ratio == 0.0
    assert not any(issue.code == "duplicate_methods" for issue in report.issues)


def test_runtime_quality_flags_community_summary_as_final_evidence() -> None:
    report = compute_graph_runtime_quality(
        campaign_id="campaign-1",
        community_summary_used_as_evidence_count=1,
    )

    assert any(
        issue.code == "community_summary_used_as_evidence" for issue in report.issues
    )


class _EventRepository:
    async def list_graph_events_for_campaign(self, campaign_id: str):  # noqa: ANN201
        assert campaign_id == "campaign-1"
        return {
            "run-1": [
                type(
                    "Event",
                    (),
                    {
                        "graph_to_chunk_success_rate": 0.5,
                        "graph_noise_ratio": 0.25,
                    },
                )()
            ]
        }


class _EvidenceRepository:
    async def list_graph_evidence_items_for_campaign(self, campaign_id: str):  # noqa: ANN201
        assert campaign_id == "campaign-1"
        return {
            "run-1": [
                type(
                    "Evidence",
                    (),
                    {
                        "provenance_status": "missing",
                        "used_in_answer": True,
                        "source_chunk_ids": [],
                        "asset_ids": [],
                    },
                )()
            ]
        }


@pytest.mark.asyncio
async def test_campaign_runtime_quality_aggregates_observability_rows() -> None:
    from graph_rag.quality import compute_campaign_runtime_quality

    report = await compute_campaign_runtime_quality(
        campaign_id="campaign-1",
        event_repository=_EventRepository(),
        evidence_repository=_EvidenceRepository(),
    )

    assert report.graph_to_chunk_success_rate == 0.5
    assert report.graph_context_noise_ratio == 0.25
    assert report.unresolved_anchor_count == 1
    assert report.community_summary_used_as_evidence_count == 1
