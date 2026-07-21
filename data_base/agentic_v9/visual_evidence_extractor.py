"""Evidence-only visual extraction through the budgeted Agentic v9 boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from data_base.agentic_v9.asset_locator import (
    AssetLocator,
    DroppedVisualAsset,
    LocatedVisualAsset,
    VisualAssetCandidate,
)
from data_base.agentic_v9.schemas import EvidencePacket, LlmInvoker, RetrievalTask


class VisualExtractionPolicy(BaseModel):
    """Runtime-independent visual limits; Setup limits remain invoker-owned."""

    model_config = ConfigDict(extra="forbid")

    max_assets_per_run: int = 3
    max_encoded_bytes: int = 1_000_000
    max_image_width: int = 2_048
    max_image_height: int = 2_048
    max_concurrency: int = 1
    timeout_s: float = 12.0


class VisualEvidenceExtractionResult(BaseModel):
    """Packets and persisted asset-level decisions from one visual stage."""

    model_config = ConfigDict(extra="forbid")

    packets: tuple[EvidencePacket, ...] = ()
    located_assets: tuple[LocatedVisualAsset, ...] = ()
    dropped_assets: tuple[DroppedVisualAsset, ...] = ()


class VisualEvidenceExtractor:
    """Call only a budgeted v9 visual phase and accept only EvidencePackets."""

    def __init__(
        self,
        invoker: LlmInvoker,
        *,
        policy: VisualExtractionPolicy | None = None,
        locator: AssetLocator | None = None,
    ) -> None:
        self._invoker = invoker
        self._policy = policy or VisualExtractionPolicy()
        self._validate_policy(self._policy)
        self._locator = locator or AssetLocator(
            max_assets_per_run=self._policy.max_assets_per_run,
            max_encoded_bytes=self._policy.max_encoded_bytes,
            max_image_width=self._policy.max_image_width,
            max_image_height=self._policy.max_image_height,
        )
        self._semaphore = asyncio.Semaphore(self._policy.max_concurrency)

    async def extract(
        self,
        *,
        task: RetrievalTask,
        assets: Iterable[VisualAssetCandidate],
        question_fragment: str,
        slot_priorities: Mapping[str, int] | None = None,
        cancellation_token: Any = None,
    ) -> VisualEvidenceExtractionResult:
        """Locate assets first, then return only source-bound evidence packets."""
        if not task.visual_required:
            return VisualEvidenceExtractionResult(
                dropped_assets=tuple(
                    DroppedVisualAsset(
                        asset_id=asset.asset_id, reason="visual_not_required"
                    )
                    for asset in assets
                )
            )
        located = self._locator.locate(
            task=task, assets=assets, slot_priorities=slot_priorities
        )
        dropped = list(located.dropped_assets)
        packets: list[EvidencePacket] = []
        for asset in located.located_assets:
            if _is_cancelled(cancellation_token):
                dropped.append(DroppedVisualAsset(asset_id=asset.asset_id, reason="cancelled"))
                continue
            packet, drop = await self._extract_asset(
                task=task,
                asset=asset,
                question_fragment=question_fragment,
                cancellation_token=cancellation_token,
            )
            if packet is not None:
                packets.append(packet)
            if drop is not None:
                dropped.append(drop)
        return VisualEvidenceExtractionResult(
            packets=tuple(packets),
            located_assets=located.located_assets,
            dropped_assets=tuple(dropped),
        )

    async def _extract_asset(
        self,
        *,
        task: RetrievalTask,
        asset: LocatedVisualAsset,
        question_fragment: str,
        cancellation_token: Any,
    ) -> tuple[EvidencePacket | None, DroppedVisualAsset | None]:
        if _is_cancelled(cancellation_token):
            return None, DroppedVisualAsset(asset_id=asset.asset_id, reason="cancelled")
        try:
            async with self._semaphore:
                if _is_cancelled(cancellation_token):
                    return None, DroppedVisualAsset(
                        asset_id=asset.asset_id, reason="cancelled"
                    )
                response = await asyncio.wait_for(
                    self._invoker.invoke(
                        phase="visual_extract",
                        purpose="visual_evidence_extraction",
                        messages=_messages(task, asset, question_fragment),
                    ),
                    timeout=self._policy.timeout_s,
                )
        except asyncio.TimeoutError:
            return None, DroppedVisualAsset(asset_id=asset.asset_id, reason="visual_timeout")
        except asyncio.CancelledError:
            raise
        except Exception:
            return None, DroppedVisualAsset(
                asset_id=asset.asset_id, reason="visual_invocation_failed"
            )

        packet = _parse_packet(response)
        if packet is None or not _is_bound_to_task(packet, task, asset):
            return None, DroppedVisualAsset(
                asset_id=asset.asset_id, reason="invalid_evidence_packet"
            )
        return packet, None

    @staticmethod
    def _validate_policy(policy: VisualExtractionPolicy) -> None:
        if not 1 <= policy.max_assets_per_run <= 3:
            raise ValueError("max_assets_per_run must be between 1 and 3")
        if min(
            policy.max_encoded_bytes,
            policy.max_image_width,
            policy.max_image_height,
            policy.max_concurrency,
        ) < 1 or policy.timeout_s <= 0:
            raise ValueError("visual extraction limits must be positive")


def _messages(
    task: RetrievalTask, asset: LocatedVisualAsset, question_fragment: str
) -> list[dict[str, Any]]:
    fragment = question_fragment.strip() or task.query
    binding = {
        "task_id": task.task_id,
        "round_id": task.round_id,
        "query_id": task.query_id,
        "target_slot_ids": list(asset.slot_ids),
        "question_fragment": fragment,
        "asset_id": asset.asset_id,
        "source": asset.source.model_dump(mode="json"),
        "locator": asset.locator.model_dump(mode="json"),
    }
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one EvidencePacket JSON object and no prose, answer, "
                "markdown, or fields outside that schema. Report only directly visible "
                "source-bound evidence; unknown values must not be invented."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(binding, ensure_ascii=False)},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + asset.image_base64
                    },
                },
            ],
        },
    ]


def _parse_packet(response: Any) -> EvidencePacket | None:
    content = getattr(response, "content", response)
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return None
    if not isinstance(content, dict):
        return None
    if not set(content).issubset(EvidencePacket.model_fields):
        return None
    try:
        return EvidencePacket.model_validate(content)
    except ValidationError:
        return None


def _is_bound_to_task(
    packet: EvidencePacket, task: RetrievalTask, asset: LocatedVisualAsset
) -> bool:
    return (
        packet.task_id == task.task_id
        and packet.round_id == task.round_id
        and packet.query_id == task.query_id
        and bool(set(packet.slot_ids).intersection(asset.slot_ids))
        and set(packet.slot_ids).issubset(set(task.target_slot_ids))
        and _same_source_provenance(packet.source, asset.source)
        and packet.locator.pdf_page_index == asset.pdf_page_index
        and packet.locator.figure_id == asset.locator.figure_id
        and packet.locator.table_id == asset.locator.table_id
        and packet.locator.bbox == asset.locator.bbox
    )


def _same_source_provenance(packet_source: Any, asset_source: Any) -> bool:
    """Compare model-supplied provenance to the selected source identity.

    A visual model may receive a span hash after identifying its visible span,
    but it must not rewrite any supplied source-identity field.
    """
    return packet_source.model_dump(exclude={"source_span_hash"}) == asset_source.model_dump(
        exclude={"source_span_hash"}
    )


def _is_cancelled(cancellation_token: Any) -> bool:
    if cancellation_token is None:
        return False
    for name in ("is_set", "is_cancelled", "cancelled"):
        value = getattr(cancellation_token, name, None)
        if callable(value) and value():
            return True
    return False


__all__ = [
    "VisualEvidenceExtractionResult",
    "VisualEvidenceExtractor",
    "VisualExtractionPolicy",
]
