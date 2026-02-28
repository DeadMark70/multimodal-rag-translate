"""
Provider registry and interfaces for external dependencies.

This module centralizes LLM and Datalab provider selection so tests can run
without touching real external APIs.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import httpx

from core.llm_factory import LLMPurpose, get_llm as get_real_llm

logger = logging.getLogger(__name__)


def _is_true(name: str, default: str = "false") -> bool:
    """Parse boolean-like env vars."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class ProviderError(RuntimeError):
    """Raised when a provider operation fails."""


@runtime_checkable
class LLMProvider(Protocol):
    """LLM provider interface."""

    def get_llm(self, purpose: LLMPurpose, model_name: Optional[str] = None) -> Any:
        """Return an LLM client for the requested purpose."""


@runtime_checkable
class DatalabProvider(Protocol):
    """Datalab provider interface for OCR and layout analysis."""

    def is_configured(self) -> bool:
        """Whether this provider has enough config to call real API."""

    async def request_ocr_markdown(self, pdf_path: str) -> dict[str, Any]:
        """Run OCR flow and return Datalab JSON result."""

    def request_layout_analysis(self, pdf_path: str) -> dict[str, Any]:
        """Run layout analysis and return Datalab JSON result."""


class RealLLMProvider:
    """Production LLM provider backed by core.llm_factory."""

    def get_llm(self, purpose: LLMPurpose, model_name: Optional[str] = None) -> Any:
        return get_real_llm(purpose, model_name=model_name)


class _FakeLLMResponse:
    """Simple response object compatible with call sites expecting `.content`."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {"total_tokens": 0}


class _FakeLLM:
    """Minimal async LLM used in tests/fake mode."""

    def __init__(self, purpose: LLMPurpose) -> None:
        self.model = f"fake-{purpose}"
        self._purpose = purpose

    async def ainvoke(self, _messages: Any) -> _FakeLLMResponse:
        return _FakeLLMResponse(
            f"[TEST_MODE] Fake provider response for purpose={self._purpose}"
        )


class FakeLLMProvider:
    """LLM provider that never calls external APIs."""

    def __init__(self) -> None:
        self._instances: dict[str, _FakeLLM] = {}

    def get_llm(self, purpose: LLMPurpose, model_name: Optional[str] = None) -> Any:
        cache_key = f"{purpose}:{model_name or ''}"
        if cache_key not in self._instances:
            self._instances[cache_key] = _FakeLLM(purpose)
        return self._instances[cache_key]


class RealDatalabProvider:
    """Production Datalab provider backed by HTTP API calls."""

    def __init__(self) -> None:
        self._api_url = os.getenv("DATALAB_API_URL", "https://www.datalab.to/api/v1/marker")
        self._api_key = os.getenv("DATALAB_API_KEY", "")
        self._timeout = 300.0

    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def request_ocr_markdown(self, pdf_path: str) -> dict[str, Any]:
        if not self.is_configured():
            raise ProviderError("DATALAB_API_KEY not configured")

        poll_interval = 2.0
        max_poll_attempts = 150

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}
        headers = {"X-Api-Key": self._api_key}
        data = {
            "output_format": "markdown",
            "mode": "balanced",
            "paginate": True,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                submit_response = await client.post(
                    self._api_url,
                    files=files,
                    headers=headers,
                    data=data,
                )
                submit_response.raise_for_status()
                submit_data = submit_response.json()

                if submit_data.get("status") == "complete" or submit_data.get("markdown"):
                    return submit_data

                request_check_url = submit_data.get("request_check_url")
                if not request_check_url:
                    raise ProviderError(
                        f"API did not return request_check_url: {submit_data}"
                    )

                for _ in range(max_poll_attempts):
                    await asyncio.sleep(poll_interval)
                    poll_response = await client.get(request_check_url, headers=headers)
                    poll_response.raise_for_status()
                    result = poll_response.json()
                    status = result.get("status", "unknown")

                    if status == "complete":
                        return result
                    if status == "failed":
                        raise ProviderError(
                            f"Processing failed: {result.get('error', 'Unknown error')}"
                        )

                raise ProviderError(
                    f"Timeout: Processing did not complete after {max_poll_attempts * poll_interval}s"
                )
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"Datalab API returned {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                raise ProviderError(f"Datalab API request failed: {exc}") from exc

    def request_layout_analysis(self, pdf_path: str) -> dict[str, Any]:
        if not self.is_configured():
            raise ProviderError("DATALAB_API_KEY not configured")

        with open(pdf_path, "rb") as f:
            files = {"file": ("document.pdf", f, "application/pdf")}
            headers = {"X-Api-Key": self._api_key}
            data = {
                "output_format": "json",
                "extract_images": True,
            }

            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        self._api_url,
                        files=files,
                        headers=headers,
                        data=data,
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"Datalab layout API returned {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                raise ProviderError(f"Datalab layout API request failed: {exc}") from exc


class FakeDatalabProvider:
    """Datalab provider that blocks real API calls."""

    def is_configured(self) -> bool:
        return True

    async def request_ocr_markdown(self, pdf_path: str) -> dict[str, Any]:
        raise ProviderError(
            f"External Datalab OCR is disabled in test/fake mode (pdf_path={pdf_path})"
        )

    def request_layout_analysis(self, pdf_path: str) -> dict[str, Any]:
        raise ProviderError(
            f"External Datalab layout analysis is disabled in test/fake mode (pdf_path={pdf_path})"
        )


@dataclass
class ProviderRegistry:
    """Container for active providers."""

    llm_provider: LLMProvider
    datalab_provider: DatalabProvider


_registry: Optional[ProviderRegistry] = None


def configure_providers(use_fake: Optional[bool] = None) -> ProviderRegistry:
    """
    Configure global provider registry.

    Args:
        use_fake: Force fake/real mode. If omitted, infer from env.
    """
    global _registry

    if use_fake is None:
        use_fake = _is_true("TEST_MODE") or _is_true("USE_FAKE_PROVIDERS")

    if use_fake:
        _registry = ProviderRegistry(
            llm_provider=FakeLLMProvider(),
            datalab_provider=FakeDatalabProvider(),
        )
    else:
        _registry = ProviderRegistry(
            llm_provider=RealLLMProvider(),
            datalab_provider=RealDatalabProvider(),
        )

    logger.info("Provider registry configured (fake=%s)", use_fake)
    return _registry


def _get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = configure_providers()
    return _registry


def get_llm(purpose: LLMPurpose, model_name: Optional[str] = None) -> Any:
    """Return LLM client from active provider registry."""
    return _get_registry().llm_provider.get_llm(purpose, model_name=model_name)


def get_datalab_provider() -> DatalabProvider:
    """Return Datalab provider from active registry."""
    return _get_registry().datalab_provider


def using_fake_providers() -> bool:
    """Return whether fake providers are currently active."""
    return isinstance(_get_registry().llm_provider, FakeLLMProvider)
