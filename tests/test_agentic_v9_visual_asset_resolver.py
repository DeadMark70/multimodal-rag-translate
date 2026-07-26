"""Authorized manifest-to-asset resolution for Agentic v9."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

from core import uploads
from data_base.agentic_v9.asset_locator import AssetLocator
from data_base.agentic_v9.schemas import (
    RequiredSlot,
    ResolvedSourceScope,
    RetrievalTask,
)
from data_base.agentic_v9.visual_asset_resolver import VisualAssetResolver
from graph_rag.schemas import GraphAssetLink


def _task(
    *,
    authorized_doc_ids: list[str] | None = None,
    locator_hints: list[str] | None = None,
) -> RetrievalTask:
    return RetrievalTask(
        task_id="Q1:round-1:source-group-1",
        round_id="round-1",
        query_id="Q1",
        query="Read Figure 1.",
        target_slot_ids=["S1"],
        source_scope=ResolvedSourceScope(
            authorized_doc_ids=authorized_doc_ids or ["doc-1"]
        ),
        locator_hints=locator_hints or ["Figure 1"],
        visual_required=True,
    )


def _link(
    *,
    asset_id: str = "asset-1",
    doc_id: str = "doc-1",
    reference: str = "user-1/doc-1/page.png",
    caption: str = "Figure 1",
    width: int = 12,
    height: int = 8,
) -> GraphAssetLink:
    return GraphAssetLink(
        asset_id=asset_id,
        doc_id=doc_id,
        page=2,
        asset_type="figure",
        caption=caption,
        storage_reference=reference,
        width=width,
        height=height,
        source_chunk_id=f"graph:asset:{asset_id}",
        asset_parse_status="parsed",
    )


def _write_png(upload_root: Path, relative: str, size: tuple[int, int] = (12, 8)) -> bytes:
    path = upload_root / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path.read_bytes()


def test_authorized_manifest_locator_loads_into_asset_locator(
    tmp_path,
    monkeypatch,
) -> None:
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(uploads, "BASE_UPLOAD_FOLDER", str(upload_root))
    payload = _write_png(upload_root, "user-1/doc-1/page.png")

    resolution = VisualAssetResolver().resolve(
        user_id="user-1",
        task=_task(),
        links=[_link()],
        slot_ids_by_asset={"asset-1": ["S1"]},
    )
    located = AssetLocator().locate(task=_task(), assets=resolution.assets)

    assert base64.b64decode(resolution.assets[0].page_image_base64) == payload
    assert [asset.asset_id for asset in located.located_assets] == ["asset-1"]
    assert resolution.diagnostics.model_dump() == {
        "manifest_count": 1,
        "authorized_count": 1,
        "locator_match_count": 1,
        "loaded_count": 1,
        "selected_count": 1,
        "dropped_count": 0,
        "evidence_packet_count": 0,
        "covered_slot_count": 1,
        "terminal_reason": None,
    }


@pytest.mark.parametrize(
    ("links", "task", "max_bytes", "expected_reason"),
    [
        ([], _task(), 1_000_000, "asset_manifest_empty"),
        ([_link(doc_id="doc-2")], _task(), 1_000_000, "source_not_authorized"),
        (
            [_link(caption="Figure 9")],
            _task(locator_hints=["Figure 1"]),
            1_000_000,
            "locator_not_matched",
        ),
        ([_link(reference="user-1/doc-1/missing.png")], _task(), 1_000_000, "asset_load_failed"),
        ([_link()], _task(), 4, "asset_exceeds_cap"),
    ],
)
def test_resolver_reports_each_terminal_diagnostic(
    tmp_path,
    monkeypatch,
    links,
    task,
    max_bytes,
    expected_reason,
) -> None:
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(uploads, "BASE_UPLOAD_FOLDER", str(upload_root))
    _write_png(upload_root, "user-1/doc-1/page.png")

    result = VisualAssetResolver(max_encoded_bytes=max_bytes).resolve(
        user_id="user-1",
        task=task,
        links=links,
    )

    assert result.assets == ()
    assert result.diagnostics.terminal_reason == expected_reason


@pytest.mark.parametrize(
    "reference",
    [
        "../doc-2/secret.png",
        "user-1/doc-2/secret.png",
        "other-user/doc-1/secret.png",
        "C:/outside/secret.png",
    ],
)
def test_resolver_rejects_traversal_and_cross_document_references(
    tmp_path,
    monkeypatch,
    reference: str,
) -> None:
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(uploads, "BASE_UPLOAD_FOLDER", str(upload_root))
    _write_png(upload_root, "user-1/doc-2/secret.png")

    result = VisualAssetResolver().resolve(
        user_id="user-1",
        task=_task(),
        links=[_link(reference=reference)],
    )

    assert result.assets == ()
    assert result.diagnostics.terminal_reason == "asset_load_failed"


def test_resolve_task_uses_bounded_manifest_lookup(
    tmp_path,
    monkeypatch,
) -> None:
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(uploads, "BASE_UPLOAD_FOLDER", str(upload_root))
    _write_png(upload_root, "user-1/doc-1/page.png")

    class Store:
        def lookup_asset_links(self, **kwargs):
            assert kwargs["authorized_doc_ids"] == {"doc-1"}
            assert kwargs["figure_id"] == "Figure 1"
            assert kwargs["limit"] == 3
            return [_link()]

    result = VisualAssetResolver(store_factory=lambda _user_id: Store()).resolve_task(
        user_id="user-1",
        task=_task(),
        slots=[
            RequiredSlot(
                slot_id="S1",
                description="Figure 1 result",
                locator_hints=["Figure 1"],
                authorized_source_doc_ids=["doc-1"],
                visual_policy="required",
            )
        ],
    )

    assert [asset.asset_id for asset in result.assets] == ["asset-1"]
    assert result.assets[0].slot_ids == ["S1"]
