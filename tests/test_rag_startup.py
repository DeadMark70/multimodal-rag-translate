"""Startup tests for RAG component warmup."""

# Standard library
from unittest.mock import AsyncMock, patch

# Third-party
import pytest
from fastapi import FastAPI


@pytest.mark.asyncio
async def test_startup_initializes_reranker() -> None:
    """Startup should warm the reranker after embeddings and LLM."""
    from data_base import router as db_router

    with patch.object(db_router, "initialize_embeddings", new=AsyncMock()) as mock_embeddings:
        with patch.object(db_router, "initialize_llm_service", new=AsyncMock()) as mock_llm:
            with patch.object(db_router, "initialize_reranker", new=AsyncMock()) as mock_reranker:
                await db_router.on_startup_rag_init()

    mock_embeddings.assert_awaited_once()
    mock_llm.assert_awaited_once()
    mock_reranker.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_degrades_when_reranker_warmup_fails(caplog: pytest.LogCaptureFixture) -> None:
    """Reranker warmup failure should not stop overall RAG startup."""
    from data_base import router as db_router

    caplog.set_level("WARNING")

    with patch.object(db_router, "initialize_embeddings", new=AsyncMock()) as mock_embeddings:
        with patch.object(db_router, "initialize_llm_service", new=AsyncMock()) as mock_llm:
            with patch.object(db_router, "initialize_reranker", new=AsyncMock(side_effect=RuntimeError("hf download failed"))):
                await db_router.on_startup_rag_init()

    mock_embeddings.assert_awaited_once()
    mock_llm.assert_awaited_once()
    assert "Reranker warmup failed; continuing without reranking" in caplog.text
    assert "reranker_active" in caplog.text


@pytest.mark.asyncio
async def test_app_lifespan_recovers_inflight_campaigns_after_init_db() -> None:
    from core import app_factory

    recover_mock = AsyncMock()
    fake_engine = type("FakeCampaignEngine", (), {"recover_inflight_campaigns": recover_mock})()

    with (
        patch("evaluation.db.init_db", new=AsyncMock()) as mock_init_db,
        patch("evaluation.router.get_campaign_engine", return_value=fake_engine),
        patch.object(app_factory, "_ensure_base_directories"),
        patch.object(app_factory, "_initialize_external_clients"),
        patch.object(app_factory, "_initialize_rag_components", new=AsyncMock()),
        patch.object(app_factory, "_warm_up_pdf_ocr", new=AsyncMock()),
    ):
        async with app_factory.app_lifespan(FastAPI()):
            pass

    mock_init_db.assert_awaited_once()
    recover_mock.assert_awaited_once()
