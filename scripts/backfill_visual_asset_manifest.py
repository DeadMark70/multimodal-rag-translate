"""Bounded, idempotent GraphAssetLink backfill for existing upload images."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from core import uploads
from data_base.agentic_v9.schemas import VisualAssetBackfillResult
from graph_rag.schemas import GraphAssetLink
from graph_rag.store import GraphStore

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def backfill_visual_asset_manifest(
    *,
    user_id: str,
    doc_id: str,
    document_dir: str | Path,
    store: GraphStore | None = None,
    max_assets: int = 100,
) -> VisualAssetBackfillResult:
    """Add at most ``max_assets`` deterministic image rows and never duplicate."""
    if not 1 <= max_assets <= 1_000:
        raise ValueError("max_assets must be between 1 and 1000")
    document_path = Path(document_dir).resolve()
    upload_root = Path(uploads.ensure_upload_root()).resolve()
    expected_document_path = (upload_root / user_id / doc_id).resolve()
    if document_path != expected_document_path:
        raise ValueError("document directory is outside the authorized upload scope")
    image_paths = list(_bounded_image_paths(document_path, limit=max_assets))
    if not image_paths:
        return VisualAssetBackfillResult(status="visual_assets_unavailable")

    graph_store = store or GraphStore(user_id)
    existing_links = graph_store.get_asset_links_for_doc(doc_id)
    existing_references = {
        link.storage_reference for link in existing_links if link.storage_reference
    }
    added_count = 0
    existing_count = 0
    for image_path in image_paths:
        reference = image_path.relative_to(upload_root).as_posix()
        if reference in existing_references:
            existing_count += 1
            continue
        try:
            resolved_image_path = uploads.resolve_upload_storage_reference(
                user_id=user_id,
                doc_id=doc_id,
                storage_reference=reference,
            )
            payload = resolved_image_path.read_bytes()
            with Image.open(resolved_image_path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError, ValueError):
            continue
        asset_id = (
            "asset:backfill:"
            + hashlib.sha256(f"{doc_id}|{reference}".encode("utf-8")).hexdigest()[:20]
        )
        graph_store.record_asset_link(
            GraphAssetLink(
                asset_id=asset_id,
                doc_id=doc_id,
                asset_type="figure",
                storage_reference=reference,
                sha256=hashlib.sha256(payload).hexdigest(),
                width=width,
                height=height,
                asset_parse_status="not_attempted",
            )
        )
        existing_references.add(reference)
        added_count += 1
    if added_count:
        graph_store.save_sidecars()
    if added_count == 0 and existing_count == 0:
        return VisualAssetBackfillResult(
            status="visual_assets_unavailable",
            scanned_count=len(image_paths),
        )
    return VisualAssetBackfillResult(
        status="completed",
        scanned_count=len(image_paths),
        added_count=added_count,
        existing_count=existing_count,
    )


def _bounded_image_paths(document_dir: Path, *, limit: int) -> Iterator[Path]:
    selected = 0
    for root, directories, files in os.walk(document_dir):
        directories.sort()
        for filename in sorted(files):
            path = Path(root) / filename
            if path.suffix.casefold() not in _IMAGE_SUFFIXES:
                continue
            yield path
            selected += 1
            if selected == limit:
                return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--document-dir", required=True)
    parser.add_argument("--max-assets", type=int, default=100)
    args = parser.parse_args()
    result = backfill_visual_asset_manifest(
        user_id=args.user_id,
        doc_id=args.doc_id,
        document_dir=args.document_dir,
        max_assets=args.max_assets,
    )
    sys.stdout.write(result.model_dump_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
