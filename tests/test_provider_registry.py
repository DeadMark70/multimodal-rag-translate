"""Tests for provider registry behavior in test mode."""

import asyncio

import pytest

from core.providers import (
    ProviderError,
    configure_providers,
    get_datalab_provider,
    get_llm,
    get_llm_provider_name,
    using_fake_providers,
)


def test_configure_fake_providers_returns_fake_registry() -> None:
    """Registry should switch to fake providers when forced."""
    configure_providers(use_fake=True)
    assert using_fake_providers() is True

    llm = get_llm("rag_qa")
    response = asyncio.run(llm.ainvoke([]))
    assert "[TEST_MODE] Fake provider response" in response.content


def test_fake_datalab_provider_blocks_external_calls() -> None:
    """Fake Datalab provider should never execute real network calls."""
    configure_providers(use_fake=True)
    provider = get_datalab_provider()

    with pytest.raises(ProviderError, match="disabled in test/fake mode"):
        asyncio.run(provider.request_ocr_markdown("fake.pdf"))


def test_real_registry_reports_production_provider_identity() -> None:
    """Production mode should retain its independent provider identity."""
    try:
        configure_providers(use_fake=False)
        assert get_llm_provider_name() == "google"
    finally:
        configure_providers(use_fake=True)
