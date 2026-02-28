"""Tests for Datalab provider injection paths."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

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
