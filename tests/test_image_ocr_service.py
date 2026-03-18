"""Regression tests for image OCR device policy and side-effect safety."""

# Standard library
import importlib
import os
from unittest.mock import MagicMock

# Third-party
import image_service.ocr_service as ocr_service


def _reload_ocr_service():
    """Reload OCR service module so env-driven defaults are refreshed per test."""
    module = importlib.reload(ocr_service)
    module._ocr_engine = None
    return module


def test_import_does_not_mutate_cuda_visible_devices(monkeypatch):
    """Importing OCR service should not rewrite global CUDA visibility env vars."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    module = _reload_ocr_service()

    assert os.getenv("CUDA_VISIBLE_DEVICES") == "0"
    module._ocr_engine = None


def test_get_ocr_engine_respects_cpu_policy(monkeypatch):
    """IMAGE_OCR_DEVICE=cpu should place OCR engine on CPU explicitly."""
    monkeypatch.setenv("IMAGE_OCR_DEVICE", "cpu")
    module = _reload_ocr_service()

    mock_engine = MagicMock()
    mock_engine.to.return_value = mock_engine
    mock_predictor = MagicMock(return_value=mock_engine)
    monkeypatch.setattr(module, "_load_ocr_predictor_factory", lambda: mock_predictor)
    monkeypatch.setattr(module, "_probe_cuda_state", lambda: (True, 1, None))

    engine = module._get_ocr_engine()

    assert engine is mock_engine
    mock_engine.to.assert_called_once_with("cpu")
    module._ocr_engine = None


def test_auto_policy_uses_cuda_when_available(monkeypatch):
    """IMAGE_OCR_DEVICE=auto should select CUDA when a usable device exists."""
    monkeypatch.setenv("IMAGE_OCR_DEVICE", "auto")
    module = _reload_ocr_service()
    monkeypatch.setattr(module, "_probe_cuda_state", lambda: (True, 1, None))

    assert module._resolve_ocr_device() == ("cuda", None)
    module._ocr_engine = None


def test_cuda_policy_falls_back_when_unavailable(monkeypatch):
    """IMAGE_OCR_DEVICE=cuda should degrade to CPU with explicit reason when unavailable."""
    monkeypatch.setenv("IMAGE_OCR_DEVICE", "cuda")
    module = _reload_ocr_service()
    monkeypatch.setattr(module, "_probe_cuda_state", lambda: (False, 0, "probe_failed"))

    assert module._resolve_ocr_device() == ("cpu", "cuda_requested_unavailable")
    module._ocr_engine = None
