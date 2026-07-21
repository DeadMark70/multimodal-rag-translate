"""Contracts for bounded, evidence-only visual extraction in Agentic v9."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest

from data_base.agentic_v9.asset_locator import AssetLocator, VisualAssetCandidate
from data_base.agentic_v9.schemas import (
    EvidenceScope,
    EvidenceSource,
    RetrievalTask,
    ResolvedSourceScope,
    SourceLocator,
)
from data_base.agentic_v9.visual_evidence_extractor import (
    VisualEvidenceExtractor,
    VisualExtractionPolicy,
)


def _encoded(payload: bytes = b"visual-evidence") -> str:
    return base64.b64encode(payload).decode("ascii")


def _task(*, visual_required: bool = True) -> RetrievalTask:
    return RetrievalTask(
        task_id="Q15:round-1:source-group-1",
        round_id="round-1",
        query_id="Q15",
        query="What is Figure 1(b)'s mIoU?",
        target_slot_ids=["primary", "secondary"],
        source_scope=ResolvedSourceScope(authorized_doc_ids=["paper-a"]),
        locator_hints=["figure", "table", "formula"],
        visual_required=visual_required,
    )


def _asset(
    asset_id: str,
    *,
    slot_ids: list[str] | None = None,
    page: int | None = 2,
    bbox: tuple[float, float, float, float] | None = (1, 2, 30, 40),
    encoded_image: str | None = None,
    crop_image: str | None = None,
    width: int = 320,
    height: int = 240,
    doc_id: str = "paper-a",
    figure_id: str | None = "Figure 1(b)",
    table_id: str | None = None,
    formula_id: str | None = None,
) -> VisualAssetCandidate:
    return VisualAssetCandidate(
        asset_id=asset_id,
        source=EvidenceSource(doc_id=doc_id, asset_id=asset_id),
        pdf_page_index=page,
        slot_ids=slot_ids or ["primary"],
        figure_id=figure_id,
        table_id=table_id,
        formula_id=formula_id,
        bbox=bbox,
        page_image_base64=encoded_image or _encoded(),
        page_encoded_bytes=len(base64.b64decode(encoded_image or _encoded())),
        page_width=width,
        page_height=height,
        crop_image_base64=crop_image,
        crop_encoded_bytes=(len(base64.b64decode(crop_image)) if crop_image else None),
        crop_width=100 if crop_image else None,
        crop_height=80 if crop_image else None,
    )


class RecordingInvoker:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def invoke(
        self, *, phase: str, purpose: str, messages: list[dict[str, Any]]
    ) -> Any:
        self.calls.append(
            {"phase": phase, "purpose": purpose, "messages": messages}
        )
        return self.response


def _packet_json(asset: VisualAssetCandidate) -> str:
    return json.dumps(
        {
            "schema_version": "1",
            "evidence_id": "visual-packet-1",
            "task_id": "Q15:round-1:source-group-1",
            "round_id": "round-1",
            "query_id": "Q15",
            "slot_ids": ["primary"],
            "statement": "Figure 1(b) reports an mIoU of 0.82.",
            "support_type": "direct",
            "source": asset.source.model_dump(),
            "scope": EvidenceScope().model_dump(),
            "locator": SourceLocator(
                pdf_page_index=asset.pdf_page_index,
                figure_id=asset.figure_id,
                bbox=asset.bbox,
            ).model_dump(),
        }
    )


def test_locator_prefers_bbox_crop_after_locating_an_authorized_page_and_figure() -> None:
    task = _task()
    crop = _encoded(b"crop")
    asset = _asset("figure-1", crop_image=crop)

    result = AssetLocator().locate(task=task, assets=[asset])

    assert len(result.located_assets) == 1
    located = result.located_assets[0]
    assert located.asset_id == "figure-1"
    assert located.pdf_page_index == 2
    assert located.crop_used is True
    assert located.image_base64 == crop
    assert located.locator.figure_id == "Figure 1(b)"
    assert result.dropped_assets == ()


def test_locator_drops_unlocated_unauthorized_and_overflow_assets_by_slot_priority() -> None:
    task = _task()
    assets = [
        _asset("low", slot_ids=["secondary"]),
        _asset("high", slot_ids=["primary"]),
        _asset("third", slot_ids=["primary"]),
        _asset("fourth", slot_ids=["primary"]),
        _asset("no-page", page=None),
        _asset("other-doc", doc_id="paper-b"),
    ]

    result = AssetLocator(max_assets_per_run=3).locate(
        task=task,
        assets=assets,
        slot_priorities={"primary": 10, "secondary": 1},
    )

    assert [item.asset_id for item in result.located_assets] == ["high", "third", "fourth"]
    assert {(item.asset_id, item.reason) for item in result.dropped_assets} == {
        ("low", "asset_limit_priority"),
        ("no-page", "page_not_located"),
        ("other-doc", "source_not_authorized"),
    }


def test_locator_enforces_encoded_byte_and_dimension_caps_before_invocation() -> None:
    task = _task()
    oversized = _asset("oversized", encoded_image=_encoded(b"0123456789"))
    too_wide = _asset("too-wide", width=1000, encoded_image=_encoded(b"tiny"))

    result = AssetLocator(
        max_encoded_bytes=8,
        max_image_width=500,
        max_image_height=500,
    ).locate(task=task, assets=[oversized, too_wide])

    assert result.located_assets == ()
    assert {(item.asset_id, item.reason) for item in result.dropped_assets} == {
        ("oversized", "encoded_bytes_exceed_cap"),
        ("too-wide", "image_dimensions_exceed_cap"),
    }


@pytest.mark.asyncio
async def test_extractor_invokes_only_v9_visual_phase_with_locator_bound_packet_json() -> None:
    task = _task()
    asset = _asset("figure-1", crop_image=_encoded(b"crop"))
    invoker = RecordingInvoker(_packet_json(asset))

    result = await VisualEvidenceExtractor(invoker).extract(
        task=task,
        assets=[asset],
        question_fragment="Figure 1(b) mIoU",
    )

    assert [packet.evidence_id for packet in result.packets] == ["visual-packet-1"]
    assert result.dropped_assets == ()
    assert len(invoker.calls) == 1
    call = invoker.calls[0]
    assert call["phase"] == "visual_extract"
    assert call["purpose"] == "visual_evidence_extraction"
    prompt = call["messages"]
    assert prompt[0]["content"].startswith("Return exactly one EvidencePacket JSON object")
    assert prompt[1]["content"][0]["text"].find("Figure 1(b) mIoU") >= 0
    assert "data:image/png;base64," in prompt[1]["content"][1]["image_url"]["url"]
    assert "answer" not in result.model_dump()


@pytest.mark.asyncio
async def test_extractor_rejects_visual_output_that_contains_an_answer_field() -> None:
    task = _task()
    asset = _asset("figure-1")
    response = json.loads(_packet_json(asset))
    response["answer"] = "The answer is 0.82."

    result = await VisualEvidenceExtractor(RecordingInvoker(response)).extract(
        task=task,
        assets=[asset],
        question_fragment="mIoU",
    )

    assert result.packets == ()
    assert [(item.asset_id, item.reason) for item in result.dropped_assets] == [
        ("figure-1", "invalid_evidence_packet")
    ]


@pytest.mark.asyncio
async def test_extractor_persists_timeout_and_cancellation_without_legacy_visual_synthesis() -> None:
    task = _task()
    asset = _asset("figure-1")

    class SlowInvoker(RecordingInvoker):
        async def invoke(self, **kwargs: Any) -> Any:
            await asyncio.sleep(0.05)
            return await super().invoke(**kwargs)

    timeout_result = await VisualEvidenceExtractor(
        SlowInvoker(_packet_json(asset)),
        policy=VisualExtractionPolicy(timeout_s=0.001),
    ).extract(task=task, assets=[asset], question_fragment="mIoU")
    assert timeout_result.packets == ()
    assert [(item.asset_id, item.reason) for item in timeout_result.dropped_assets] == [
        ("figure-1", "visual_timeout")
    ]

    cancelled = asyncio.Event()
    cancelled.set()
    invoker = RecordingInvoker(_packet_json(asset))
    cancelled_result = await VisualEvidenceExtractor(invoker).extract(
        task=task,
        assets=[asset],
        question_fragment="mIoU",
        cancellation_token=cancelled,
    )
    assert cancelled_result.packets == ()
    assert [(item.asset_id, item.reason) for item in cancelled_result.dropped_assets] == [
        ("figure-1", "cancelled")
    ]
    assert invoker.calls == []
