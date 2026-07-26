"""Idempotent bounded visual-manifest backfill."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from core import uploads
from graph_rag.store import GraphStore
from scripts.backfill_visual_asset_manifest import backfill_visual_asset_manifest


def test_backfill_is_idempotent_and_manifest_contains_no_base64(
    tmp_path,
    monkeypatch,
) -> None:
    upload_root = tmp_path / "uploads"
    document_dir = upload_root / "user-1" / "doc-1"
    document_dir.mkdir(parents=True)
    Image.new("RGB", (9, 6), "white").save(document_dir / "page.png")
    monkeypatch.setattr(uploads, "BASE_UPLOAD_FOLDER", str(upload_root))
    store = GraphStore("user-1", storage_dir=tmp_path / "graph")

    first = backfill_visual_asset_manifest(
        user_id="user-1",
        doc_id="doc-1",
        document_dir=document_dir,
        store=store,
        max_assets=10,
    )
    second = backfill_visual_asset_manifest(
        user_id="user-1",
        doc_id="doc-1",
        document_dir=document_dir,
        store=store,
        max_assets=10,
    )

    assert first.status == "completed"
    assert first.added_count == 1
    assert second.status == "completed"
    assert second.added_count == 0
    assert second.existing_count == 1
    assert len(store.get_asset_links_for_doc("doc-1")) == 1
    serialized = (tmp_path / "graph" / "graph.asset_links.json").read_text("utf-8")
    assert "base64" not in serialized
    assert "data:image" not in serialized


def test_backfill_returns_explicit_unavailable_result_and_obeys_bound(
    tmp_path,
    monkeypatch,
) -> None:
    upload_root = tmp_path / "uploads"
    document_dir = upload_root / "user-1" / "doc-1"
    document_dir.mkdir(parents=True)
    monkeypatch.setattr(uploads, "BASE_UPLOAD_FOLDER", str(upload_root))
    store = GraphStore("user-1", storage_dir=tmp_path / "graph")

    unavailable = backfill_visual_asset_manifest(
        user_id="user-1",
        doc_id="doc-1",
        document_dir=document_dir,
        store=store,
        max_assets=1,
    )
    Image.new("RGB", (4, 4), "white").save(document_dir / "a.png")
    Image.new("RGB", (4, 4), "black").save(document_dir / "b.png")
    bounded = backfill_visual_asset_manifest(
        user_id="user-1",
        doc_id="doc-1",
        document_dir=document_dir,
        store=store,
        max_assets=1,
    )

    assert unavailable.status == "visual_assets_unavailable"
    assert unavailable.scanned_count == 0
    assert bounded.status == "completed"
    assert bounded.scanned_count == 1
    assert bounded.added_count == 1
    payload = json.loads(
        (tmp_path / "graph" / "graph.asset_links.json").read_text("utf-8")
    )
    assert len(payload["assets"]) == 1


def test_backfill_rejects_file_symlink_escape_before_reading(
    tmp_path,
    monkeypatch,
) -> None:
    upload_root = tmp_path / "uploads"
    document_dir = upload_root / "user-1" / "doc-1"
    document_dir.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (5, 5), "red").save(outside)
    link = document_dir / "page.png"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    monkeypatch.setattr(uploads, "BASE_UPLOAD_FOLDER", str(upload_root))
    store = GraphStore("user-1", storage_dir=tmp_path / "graph")

    result = backfill_visual_asset_manifest(
        user_id="user-1",
        doc_id="doc-1",
        document_dir=document_dir,
        store=store,
        max_assets=10,
    )

    assert result.status == "visual_assets_unavailable"
    assert result.added_count == 0
    assert store.get_asset_links_for_doc("doc-1") == []
