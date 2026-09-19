"""Tests for Datalab provider injection paths."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.providers import RealDatalabProvider
from multimodal_rag.structure_analyzer import StructureAnalyzer
from pdfserviceMD.PDF_OCR_services import _ocr_with_datalab_api


class StubDatalabProvider:
    """In-memory Datalab provider used to avoid external API calls."""

    def __init__(self) -> None:
        self.layout_called_with: str | None = None
        self.ocr_called_with: str | None = None

    def is_configured(self) -> bool:
        return True

    async def request_ocr_markdown(self, pdf_path: str) -> dict[str, Any]:
        self.ocr_called_with = pdf_path
        return {
            "markdown": "stub markdown content",
            "page_count": 1,
            "images": {},
            "status": "complete",
        }

    def request_layout_analysis(self, pdf_path: str) -> dict[str, Any]:
        self.layout_called_with = pdf_path
        return {"pages": [{"page_number": 1, "markdown": "stub page"}]}


def test_structure_analyzer_uses_injected_provider() -> None:
    """StructureAnalyzer should call injected provider, not global HTTP client."""
    provider = StubDatalabProvider()
    analyzer = StructureAnalyzer(datalab_provider=provider)
    result = analyzer._call_datalab_layout_api("dummy.pdf")
    assert result["pages"][0]["page_number"] == 1
    assert provider.layout_called_with == "dummy.pdf"


def test_ocr_service_uses_injected_provider() -> None:
    """OCR flow should work with injected provider and no real Datalab call."""
    provider = StubDatalabProvider()
    root = Path(__file__).resolve().parent
    fake_pdf = root / f"demo-{uuid.uuid4().hex}.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 stub")

    try:
        markdown = _ocr_with_datalab_api(str(fake_pdf), datalab_provider=provider)
        assert "[[PAGE_1]]" in markdown
        assert "stub markdown content" in markdown
        assert provider.ocr_called_with == str(fake_pdf)
    finally:
        fake_pdf.unlink(missing_ok=True)


@pytest.mark.parametrize("queued", [False, True])
def test_api_ocr_submission_and_polling_without_local_marker(monkeypatch, tmp_path, queued):
    from pdfserviceMD import PDF_OCR_services as ocr

    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF-1.4 test fixture")
    calls = []
    complete = {
        "status": "complete", "page_count": 2, "images": {},
        "markdown": "{0}" + "-" * 48 + "\nFirst page\n{1}" + "-" * 48 + "\nSecond page",
    }

    def handle(request):
        calls.append(request)
        assert request.headers["X-Api-Key"] == "unit-test"
        if request.method == "POST":
            assert b"%PDF-1.4 test fixture" in request.content
            assert b"balanced" in request.content
            if queued:
                return httpx.Response(200, json={
                    "request_check_url": "https://ocr.example.test/check/1",
                })
        return httpx.Response(200, json=complete)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        "core.providers.httpx.AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    monkeypatch.setattr("core.providers.asyncio.sleep", AsyncMock())
    monkeypatch.setenv("DATALAB_API_KEY", "unit-test")
    monkeypatch.setenv("DATALAB_API_URL", "https://ocr.example.test/marker")
    monkeypatch.setattr(ocr, "USE_LOCAL_MARKER", False)

    def reject_local(*args, **kwargs):
        pytest.fail("API OCR must not load local Marker")

    monkeypatch.setattr(ocr, "_ocr_with_local_marker", reject_local)
    markdown = ocr.ocr_service_sync(str(pdf), datalab_provider=RealDatalabProvider())
    assert "[[PAGE_1]]" in markdown and "[[PAGE_2]]" in markdown
    assert "First page" in markdown and "Second page" in markdown
    assert [call.method for call in calls] == (["POST", "GET"] if queued else ["POST"])
