"""Bounded source-asset selection for Agentic v9 visual evidence work."""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from data_base.agentic_v9.schemas import EvidenceSource, RetrievalTask, SourceLocator


class VisualAssetCandidate(BaseModel):
    """One retrieved visual asset with a page-level, source-bound locator."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    source: EvidenceSource
    pdf_page_index: int | None = Field(default=None, ge=0)
    slot_ids: list[str] = Field(min_length=1)
    figure_id: str | None = None
    table_id: str | None = None
    formula_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    page_image_base64: str = Field(min_length=1)
    page_encoded_bytes: int = Field(ge=0)
    page_width: int = Field(ge=1)
    page_height: int = Field(ge=1)
    crop_image_base64: str | None = None
    crop_encoded_bytes: int | None = Field(default=None, ge=0)
    crop_width: int | None = Field(default=None, ge=1)
    crop_height: int | None = Field(default=None, ge=1)


class LocatedVisualAsset(BaseModel):
    """A cap-compliant page or crop that may be shown to the visual model."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    source: EvidenceSource
    slot_ids: tuple[str, ...]
    pdf_page_index: int
    locator: SourceLocator
    image_base64: str
    encoded_bytes: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    crop_used: bool


class DroppedVisualAsset(BaseModel):
    """Persistable reason why an asset was not sent to a visual model."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    reason: str


class AssetLocationResult(BaseModel):
    """Typed asset selection result, including every drop decision."""

    model_config = ConfigDict(extra="forbid")

    located_assets: tuple[LocatedVisualAsset, ...] = ()
    dropped_assets: tuple[DroppedVisualAsset, ...] = ()


class AssetLocator:
    """Locate and bound visual assets before an Agentic v9 model invocation."""

    def __init__(
        self,
        *,
        max_assets_per_run: int = 3,
        max_encoded_bytes: int = 1_000_000,
        max_image_width: int = 2_048,
        max_image_height: int = 2_048,
    ) -> None:
        if max_assets_per_run < 1 or max_assets_per_run > 3:
            raise ValueError("max_assets_per_run must be between 1 and 3")
        if min(max_encoded_bytes, max_image_width, max_image_height) < 1:
            raise ValueError("visual asset caps must be positive")
        self._max_assets_per_run = max_assets_per_run
        self._max_encoded_bytes = max_encoded_bytes
        self._max_image_width = max_image_width
        self._max_image_height = max_image_height

    def locate(
        self,
        *,
        task: RetrievalTask,
        assets: Iterable[VisualAssetCandidate],
        slot_priorities: Mapping[str, int] | None = None,
    ) -> AssetLocationResult:
        """Return only authorized, located, cap-compliant assets for ``task``.

        A candidate represents exactly one PDF page.  This prevents a visual
        invocation from silently scanning additional pages of the same asset.
        """
        authorized_doc_ids = set(task.source_scope.authorized_doc_ids)
        priorities = slot_priorities or {}
        eligible: list[tuple[int, int, LocatedVisualAsset]] = []
        dropped: list[DroppedVisualAsset] = []
        for sequence, asset in enumerate(assets):
            rejection = self._rejection_reason(
                task=task, asset=asset, authorized_doc_ids=authorized_doc_ids
            )
            if rejection is not None:
                dropped.append(DroppedVisualAsset(asset_id=asset.asset_id, reason=rejection))
                continue
            located = self._located_asset(asset)
            if located is None:
                dropped.append(
                    DroppedVisualAsset(
                        asset_id=asset.asset_id, reason="encoded_image_invalid"
                    )
                )
                continue
            if located.encoded_bytes > self._max_encoded_bytes:
                dropped.append(
                    DroppedVisualAsset(
                        asset_id=asset.asset_id, reason="encoded_bytes_exceed_cap"
                    )
                )
                continue
            if (
                located.width > self._max_image_width
                or located.height > self._max_image_height
            ):
                dropped.append(
                    DroppedVisualAsset(
                        asset_id=asset.asset_id, reason="image_dimensions_exceed_cap"
                    )
                )
                continue
            priority = self._slot_priority(task, asset.slot_ids, priorities)
            eligible.append((priority, sequence, located))

        eligible.sort(key=lambda item: (-item[0], item[1], item[2].asset_id))
        selected = eligible[: self._max_assets_per_run]
        dropped.extend(
            DroppedVisualAsset(asset_id=item.asset_id, reason="asset_limit_priority")
            for _, _, item in eligible[self._max_assets_per_run :]
        )
        return AssetLocationResult(
            located_assets=tuple(item for _, _, item in selected),
            dropped_assets=tuple(dropped),
        )

    def _rejection_reason(
        self,
        *,
        task: RetrievalTask,
        asset: VisualAssetCandidate,
        authorized_doc_ids: set[str],
    ) -> str | None:
        if asset.source.doc_id not in authorized_doc_ids:
            return "source_not_authorized"
        if asset.pdf_page_index is None:
            return "page_not_located"
        if not set(asset.slot_ids).intersection(task.target_slot_ids):
            return "target_slot_not_requested"
        if not self._matches_locator_hint(task, asset):
            return "asset_locator_not_matched"
        return None

    @staticmethod
    def _matches_locator_hint(task: RetrievalTask, asset: VisualAssetCandidate) -> bool:
        hints = {hint.strip().lower() for hint in task.locator_hints}
        if not hints:
            return any((asset.figure_id, asset.table_id, asset.formula_id, asset.bbox))
        return (
            ("figure" in hints and bool(asset.figure_id))
            or ("table" in hints and bool(asset.table_id))
            or ("formula" in hints and bool(asset.formula_id))
            or ("bbox" in hints and asset.bbox is not None)
        )

    def _located_asset(self, asset: VisualAssetCandidate) -> LocatedVisualAsset | None:
        use_crop = asset.bbox is not None and asset.crop_image_base64 is not None
        image = asset.crop_image_base64 if use_crop else asset.page_image_base64
        width = asset.crop_width if use_crop else asset.page_width
        height = asset.crop_height if use_crop else asset.page_height
        if image is None or width is None or height is None or asset.pdf_page_index is None:
            return None
        encoded_bytes = _encoded_bytes(image)
        if encoded_bytes is None:
            return None
        return LocatedVisualAsset(
            asset_id=asset.asset_id,
            source=asset.source.model_copy(update={"asset_id": asset.asset_id}),
            slot_ids=tuple(asset.slot_ids),
            pdf_page_index=asset.pdf_page_index,
            locator=SourceLocator(
                pdf_page_index=asset.pdf_page_index,
                figure_id=asset.figure_id,
                table_id=asset.table_id,
                section=asset.formula_id,
                bbox=asset.bbox,
            ),
            image_base64=image,
            encoded_bytes=encoded_bytes,
            width=width,
            height=height,
            crop_used=use_crop,
        )

    @staticmethod
    def _slot_priority(
        task: RetrievalTask, slot_ids: list[str], priorities: Mapping[str, int]
    ) -> int:
        default_priorities = {
            slot_id: len(task.target_slot_ids) - index
            for index, slot_id in enumerate(task.target_slot_ids)
        }
        return max(
            priorities.get(slot_id, default_priorities.get(slot_id, 0))
            for slot_id in slot_ids
            if slot_id in task.target_slot_ids
        )


def _encoded_bytes(value: str) -> int | None:
    try:
        return len(base64.b64decode(value, validate=True))
    except (ValueError, TypeError):
        return None


__all__ = [
    "AssetLocationResult",
    "AssetLocator",
    "DroppedVisualAsset",
    "LocatedVisualAsset",
    "VisualAssetCandidate",
]
