from unittest.mock import AsyncMock, patch

import pytest

import data_base.router as db_router
from experiments.evaluation_pipeline import EvaluationPipeline


@pytest.mark.asyncio
async def test_single_run():
    with patch.object(db_router, "on_startup_rag_init", new=AsyncMock()) as mock_init:
        await db_router.on_startup_rag_init()

    pipeline = EvaluationPipeline()
    assert pipeline.models == ["gemini-2.0-flash-lite"]

    mocked_result = {
        "answer": "mocked answer",
        "usage": {"total_tokens": 12},
        "retrieved_contexts": [{"text": "ctx", "metadata": {"source": "mock"}}],
    }
    pipeline.run_tier = AsyncMock(return_value=mocked_result)

    result = await pipeline.run_tier(
        "Naive RAG",
        "What is the primary contribution of the SwinUNETR paper?",
        "gemini-2.0-flash-lite",
    )

    pipeline.run_tier.assert_awaited_once_with(
        "Naive RAG",
        "What is the primary contribution of the SwinUNETR paper?",
        "gemini-2.0-flash-lite",
    )
    mock_init.assert_awaited_once()
    assert "total_tokens" in result["usage"]
    assert "retrieved_contexts" in result
