import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from data_base.RAG_QA_service import _build_graph_evidence_items, _filter_graph_query_hints
from evaluation import db as evaluation_db
from evaluation.observability_storage import (
    EvaluationGraphEventRepository,
    EvaluationGraphEvidenceItemRepository,
)
from evaluation.schemas import EvaluationGraphEvent, EvaluationGraphEvidenceItem
from graph_rag.generic_mode import GraphEvidence


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _seed_campaign(campaign_id: str) -> None:
    now = _now().isoformat()
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            INSERT INTO campaigns (
                id, user_id, name, status, phase, config_json, completed_units, total_units,
                evaluation_completed_units, evaluation_total_units, current_question_id,
                current_mode, error_message, cancel_requested, created_at, started_at,
                completed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, NULL, NULL, NULL, 0, ?, NULL, NULL, ?)
            """,
            (
                campaign_id,
                "user-a",
                "Graph event repository test",
                "pending",
                "execution",
                "{}",
                now,
                now,
            ),
        )
        await connection.commit()


def test_graph_event_schema_records_route_and_success_rates() -> None:
    event = EvaluationGraphEvent(
        graph_event_id="ge-1",
        run_id="run-1",
        campaign_id="camp-1",
        span_id="span-1",
        graph_query="compare MedSAM and SAM-Med3D",
        graph_search_mode="generic",
        graph_evidence_mode="locator_to_chunk",
        graph_route="blended",
        router_reason="relation query with communities",
        graph_feature_flags={"graph_to_chunk_enabled": True},
        graph_snapshot_version="v003",
        graph_schema_version="graph-schema-v1",
        graph_extraction_prompt_version="graph-extract-v2",
        matched_entity_ids=["method:medsam"],
        community_ids=[3],
        node_count=2,
        edge_count=1,
        path_count=1,
        graph_latency_ms=42,
        graph_context_tokens=120,
        graph_to_chunk_success_rate=1.0,
        graph_noise_ratio=0.0,
    )

    assert event.graph_route == "blended"
    assert event.graph_evidence_mode == "locator_to_chunk"
    assert event.graph_feature_flags["graph_to_chunk_enabled"] is True
    assert event.graph_to_chunk_success_rate == 1.0


def test_graph_evidence_item_schema_tracks_context_lifecycle() -> None:
    item = EvaluationGraphEvidenceItem(
        graph_evidence_item_id="gei-1",
        graph_event_id="ge-1",
        node_ids=["method:medsam"],
        edge_ids=["edge-1"],
        relation_path=["method:medsam", "paper_proposes_method", "paper:medsam"],
        source_doc_ids=["doc-1"],
        source_chunk_ids=["chunk-1"],
        pages=[4],
        asset_ids=[],
        confidence=0.88,
        provenance_status="full",
        used_as_locator=True,
        packed_in_context=True,
        used_in_answer=False,
        supported_claim_ids=[],
    )

    assert item.provenance_status == "full"
    assert item.packed_in_context is True


@pytest.mark.asyncio
async def test_graph_observability_repositories_round_trip_graph_rows(
    monkeypatch,
) -> None:
    db_path = Path(os.environ["TEMP"]) / f"graph-events-{uuid4().hex}.sqlite3"
    try:
        monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", db_path)
        await _seed_campaign("campaign-graph")

        event_repository = EvaluationGraphEventRepository()
        item_repository = EvaluationGraphEvidenceItemRepository()
        created_at = _now()

        event = EvaluationGraphEvent(
            graph_event_id="ge-1",
            run_id="run-1",
            campaign_id="campaign-graph",
            span_id="span-1",
            graph_query="compare MedSAM and SAM-Med3D",
            graph_search_mode="generic",
            graph_evidence_mode="raw_current",
            graph_route="blended",
            router_reason="relation query with communities",
            graph_feature_flags={"graph_raw_current_enabled": True},
            graph_snapshot_version="index-v2",
            graph_schema_version="graph-schema-v1",
            graph_extraction_prompt_version="graph-extract-v2",
            matched_entity_ids=["node_method_medsam"],
            community_ids=[3],
            node_count=2,
            edge_count=1,
            path_count=1,
            graph_latency_ms=42,
            graph_context_tokens=120,
            graph_to_chunk_success_rate=None,
            graph_noise_ratio=0.25,
            created_at=created_at,
        )
        item = EvaluationGraphEvidenceItem(
            graph_evidence_item_id="gei-1",
            graph_event_id="ge-1",
            node_ids=["node_method_medsam"],
            edge_ids=["edge-1"],
            relation_path=["node_method_medsam", "compares_to", "node_method_sammed3d"],
            source_doc_ids=["doc-1"],
            source_chunk_ids=[],
            pages=[4],
            asset_ids=[],
            confidence=0.88,
            provenance_status="partial",
            used_as_locator=False,
            packed_in_context=True,
            used_in_answer=False,
            supported_claim_ids=[],
            created_at=created_at,
        )

        await event_repository.record_graph_event(event)
        await item_repository.record_graph_evidence_items([item])

        run_events = await event_repository.list_graph_events_for_run("run-1")
        campaign_events = await event_repository.list_graph_events_for_campaign("campaign-graph")
        run_items = await item_repository.list_graph_evidence_items_for_run("run-1")
        campaign_items = await item_repository.list_graph_evidence_items_for_campaign("campaign-graph")

        assert run_events[0].graph_feature_flags["graph_raw_current_enabled"] is True
        assert run_events[0].matched_entity_ids == ["node_method_medsam"]
        assert set(campaign_events) == {"run-1"}
        assert campaign_events["run-1"][0].graph_event_id == "ge-1"
        assert run_items[0].pages == [4]
        assert run_items[0].provenance_status == "partial"
        assert set(campaign_items) == {"run-1"}
        assert campaign_items["run-1"][0].graph_evidence_item_id == "gei-1"
    finally:
        db_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_graph_evidence_items_keep_history_across_events_with_shared_source_ids(
    monkeypatch,
) -> None:
    db_path = Path(os.environ["TEMP"]) / f"graph-evidence-history-{uuid4().hex}.sqlite3"
    try:
        monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", db_path)
        await _seed_campaign("campaign-graph-history")

        event_repository = EvaluationGraphEventRepository()
        item_repository = EvaluationGraphEvidenceItemRepository()
        created_at = _now()

        first_event = EvaluationGraphEvent(
            graph_event_id="ge-1",
            run_id="run-1",
            campaign_id="campaign-graph-history",
            graph_query="first graph query",
            graph_search_mode="generic",
            graph_evidence_mode="raw_current",
            graph_route="local-first",
            created_at=created_at,
        )
        second_event = EvaluationGraphEvent(
            graph_event_id="ge-2",
            run_id="run-1",
            campaign_id="campaign-graph-history",
            graph_query="second graph query",
            graph_search_mode="generic",
            graph_evidence_mode="raw_current",
            graph_route="local-first",
            created_at=created_at,
        )
        shared_evidence = GraphEvidence(
            evidence_id="shared-edge-1",
            evidence_type="local_edge",
            text="Shared relation evidence",
            score=0.91,
            token_estimate=6,
            metadata={"source_id": "node-a", "target_id": "node-b"},
        )

        first_items = _build_graph_evidence_items(
            graph_event_id=first_event.graph_event_id,
            evidence_units=[shared_evidence],
            graph_evidence_mode="raw_current",
            created_at=created_at,
        )
        second_items = _build_graph_evidence_items(
            graph_event_id=second_event.graph_event_id,
            evidence_units=[shared_evidence],
            graph_evidence_mode="raw_current",
            created_at=created_at,
        )

        await event_repository.record_graph_event(first_event)
        await event_repository.record_graph_event(second_event)
        await item_repository.record_graph_evidence_items(first_items)
        await item_repository.record_graph_evidence_items(second_items)

        run_items = await item_repository.list_graph_evidence_items_for_run("run-1")
        items_by_event = {
            item.graph_event_id: item.graph_evidence_item_id for item in run_items
        }

        assert len(run_items) == 2
        assert items_by_event == {
            "ge-1": "ge-1:shared-edge-1",
            "ge-2": "ge-2:shared-edge-1",
        }
    finally:
        db_path.unlink(missing_ok=True)


def test_filter_graph_query_hints_ignores_observability_metadata() -> None:
    hints = _filter_graph_query_hints(
        {
            "stage_hint": "verification",
            "prefer_local": True,
            "evaluation_metadata": {"run_id": "run-1"},
            "graph_feature_flags": {"graph_raw_current_enabled": True},
            "unexpected": "ignored",
        }
    )

    assert hints.stage_hint == "verification"
    assert hints.prefer_local is True
    assert hints.task_type_hint is None
    assert hints.prefer_global is False
