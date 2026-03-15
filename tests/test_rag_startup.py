"""Startup tests for RAG component warmup."""

# Standard library
from unittest.mock import AsyncMock, patch

# Third-party
import pytest


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
