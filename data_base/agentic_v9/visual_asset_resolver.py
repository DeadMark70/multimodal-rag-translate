"""Resolve authorized GraphAssetLink rows into bounded visual candidates."""

from __future__ import annotations

import base64
import hashlib
import io
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from core.uploads import resolve_upload_storage_reference
from data_base.agentic_v9.asset_locator import VisualAssetCandidate
from data_base.agentic_v9.schemas import (
    EvidenceSource,
    RequiredSlot,
    RetrievalTask,
)
from graph_rag.schemas import GraphAssetLink
from graph_rag.store import GraphStore

StoreFactory = Callable[[str], GraphStore]


class VisualAssetResolutionDiagnostics(BaseModel):
    """Counts and one terminal reason for deterministic asset resolution."""

    model_config = ConfigDict(extra="forbid")

    manifest_count: int | None = Field(default=None, ge=0)
    authorized_count: int | None = Field(default=None, ge=0)
    locator_match_count: int | None = Field(default=None, ge=0)
    loaded_count: int | None = Field(default=None, ge=0)
    selected_count: int | None = Field(default=None, ge=0)
    dropped_count: int | None = Field(default=None, ge=0)
    evidence_packet_count: int | None = Field(default=0, ge=0)
    covered_slot_count: int | None = Field(default=0, ge=0)
    terminal_reason: str | None = None


class VisualAssetResolution(BaseModel):
    """Loaded candidates plus complete deterministic diagnostics."""

    model_config = ConfigDict(extra="forbid")

    assets: tuple[VisualAssetCandidate, ...] = ()
    diagnostics: VisualAssetResolutionDiagnostics


class VisualAssetResolver:
    """Authorize, locate, cap, and load only selected manifest assets."""

    def __init__(
        self,
        *,
        store_factory: StoreFactory = GraphStore,
        max_assets_per_run: int = 3,
        max_encoded_bytes: int = 1_000_000,
        max_image_width: int = 2_048,
        max_image_height: int = 2_048,
    ) -> None:
        if not 1 <= max_assets_per_run <= 3:
            raise ValueError("max_assets_per_run must be between 1 and 3")
        if min(max_encoded_bytes, max_image_width, max_image_height) < 1:
            raise ValueError("visual asset caps must be positive")
        self._store_factory = store_factory
        self._max_assets_per_run = max_assets_per_run
        self._max_encoded_bytes = max_encoded_bytes
        self._max_image_width = max_image_width
        self._max_image_height = max_image_height

    def resolve_task(
        self,
        *,
        user_id: str,
        task: RetrievalTask,
        slots: Sequence[RequiredSlot],
    ) -> VisualAssetResolution:
        """Read a bounded manifest, then preserve authorization and locator stages."""
        store = self._store_factory(user_id)
        links = store.get_asset_links(limit=100)
        slot_ids_by_asset: dict[str, list[str]] = {}
        for slot in slots:
            authorized = set(
                slot.authorized_source_doc_ids or task.source_scope.authorized_doc_ids
            )
            locators = slot.locator_hints or task.locator_hints
            for link in links:
                if link.doc_id not in authorized:
                    continue
                if _matches_locators(locators, link):
                    slot_ids_by_asset.setdefault(link.asset_id, []).append(slot.slot_id)
        return self.resolve(
            user_id=user_id,
            task=task,
            links=links,
            slot_ids_by_asset=slot_ids_by_asset,
        )

    def resolve(
        self,
        *,
        user_id: str,
        task: RetrievalTask,
        links: Iterable[GraphAssetLink],
        slot_ids_by_asset: Mapping[str, list[str]] | None = None,
    ) -> VisualAssetResolution:
        """Resolve supplied manifest rows without loading rejected assets."""
        manifest = sorted(links, key=lambda link: link.asset_id)
        authorized_ids = set(task.source_scope.authorized_doc_ids)
        authorized = [link for link in manifest if link.doc_id in authorized_ids]
        matched = [
            link
            for link in authorized
            if (
                link.asset_id in slot_ids_by_asset
                if slot_ids_by_asset is not None
                else _matches_task_locator(task, link)
            )
        ]
        selected_links = matched[: self._max_assets_per_run]
        assets: list[VisualAssetCandidate] = []
        load_failures = 0
        cap_failures = 0
        for link in selected_links:
            result = self._load_candidate(
                user_id=user_id,
                task=task,
                link=link,
                slot_ids=(
                    (slot_ids_by_asset or {}).get(link.asset_id) or task.target_slot_ids
                ),
            )
            if isinstance(result, VisualAssetCandidate):
                assets.append(result)
            elif result == "asset_exceeds_cap":
                cap_failures += 1
            else:
                load_failures += 1
        terminal_reason = _terminal_reason(
            manifest_count=len(manifest),
            authorized_count=len(authorized),
            locator_match_count=len(matched),
            loaded_count=len(assets),
            load_failures=load_failures,
            cap_failures=cap_failures,
        )
        covered_slots = {slot_id for asset in assets for slot_id in asset.slot_ids}
        return VisualAssetResolution(
            assets=tuple(assets),
            diagnostics=VisualAssetResolutionDiagnostics(
                manifest_count=len(manifest),
                authorized_count=len(authorized),
                locator_match_count=len(matched),
                loaded_count=len(assets),
                selected_count=len(selected_links),
                dropped_count=len(manifest) - len(assets),
                covered_slot_count=len(covered_slots),
                terminal_reason=terminal_reason,
            ),
        )

    def _load_candidate(
        self,
        *,
        user_id: str,
        task: RetrievalTask,
        link: GraphAssetLink,
        slot_ids: list[str],
    ) -> VisualAssetCandidate | str:
        if not link.storage_reference or link.page is None:
            return "asset_load_failed"
        try:
            path = resolve_upload_storage_reference(
                user_id=user_id,
                doc_id=link.doc_id,
                storage_reference=link.storage_reference,
            )
            size = path.stat().st_size
        except (OSError, ValueError):
            return "asset_load_failed"
        if size > self._max_encoded_bytes:
            return "asset_exceeds_cap"
        try:
            payload = path.read_bytes()
            width, height = _image_dimensions(path, payload, link)
        except (OSError, ValueError, UnidentifiedImageError):
            return "asset_load_failed"
        if link.sha256 and hashlib.sha256(payload).hexdigest() != link.sha256:
            return "asset_load_failed"
        if width > self._max_image_width or height > self._max_image_height:
            return "asset_exceeds_cap"
        return VisualAssetCandidate(
            asset_id=link.asset_id,
            source=EvidenceSource(
                doc_id=link.doc_id,
                chunk_id=link.source_chunk_id,
                asset_id=link.asset_id,
            ),
            pdf_page_index=link.page,
            slot_ids=list(dict.fromkeys(slot_ids)),
            figure_id=link.caption if link.asset_type == "figure" else None,
            table_id=link.caption if link.asset_type == "table" else None,
            formula_id=link.formula_id if link.asset_type == "formula" else None,
            bbox=tuple(link.bbox) if link.bbox and len(link.bbox) == 4 else None,
            page_image_base64=base64.b64encode(payload).decode("ascii"),
            page_encoded_bytes=len(payload),
            page_width=width,
            page_height=height,
        )


def _image_dimensions(
    path: Path,
    payload: bytes,
    link: GraphAssetLink,
) -> tuple[int, int]:
    if link.width is not None and link.height is not None:
        return link.width, link.height
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
    if width < 1 or height < 1:
        raise ValueError(f"invalid image dimensions for {path.name}")
    return width, height


def _lookup_arguments(locator: str) -> dict[str, object] | None:
    normalized = _normalized(locator)
    page_match = re.fullmatch(r"(?:pdf)?page(\d+)", normalized)
    if page_match:
        return {"page": int(page_match.group(1))}
    if normalized.startswith(("figure", "fig")):
        return {"figure_id": locator}
    if normalized.startswith("table"):
        return {"table_id": locator}
    if normalized.startswith(("formula", "equation")):
        return {"formula_id": locator}
    return None


def _matches_task_locator(task: RetrievalTask, link: GraphAssetLink) -> bool:
    return _matches_locators(task.locator_hints, link)


def _matches_locators(locators: Sequence[str], link: GraphAssetLink) -> bool:
    if not locators:
        return True
    for locator in locators:
        lookup = _lookup_arguments(locator)
        if lookup is None:
            continue
        if "page" in lookup and link.page == lookup["page"]:
            return True
        if "figure_id" in lookup and link.asset_type == "figure":
            return _identifier_matches(link.caption, locator)
        if "table_id" in lookup and link.asset_type == "table":
            return _identifier_matches(link.caption, locator)
        if "formula_id" in lookup and link.asset_type == "formula":
            return _identifier_matches(link.formula_id, locator)
    return False


def _identifier_matches(value: str | None, requested: str) -> bool:
    return bool(value and _normalized(value) == _normalized(requested))


def _normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _terminal_reason(
    *,
    manifest_count: int,
    authorized_count: int,
    locator_match_count: int,
    loaded_count: int,
    load_failures: int,
    cap_failures: int,
) -> str | None:
    if loaded_count:
        return None
    if manifest_count == 0:
        return "asset_manifest_empty"
    if authorized_count == 0:
        return "source_not_authorized"
    if locator_match_count == 0:
        return "locator_not_matched"
    if cap_failures and not load_failures:
        return "asset_exceeds_cap"
    return "asset_load_failed"


__all__ = [
    "VisualAssetResolution",
    "VisualAssetResolutionDiagnostics",
    "VisualAssetResolver",
]
