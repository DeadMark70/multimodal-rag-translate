"""
Unit tests for the Jina-based reranker module.
"""

# Standard library
import logging
import os
from unittest.mock import MagicMock, patch

# Third-party
import pytest
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def reset_reranker_singleton():
    """Reset singleton state between tests."""
    from data_base.reranker import DocumentReranker

    DocumentReranker._instance = None
    DocumentReranker._model = None
    DocumentReranker._model_name = None
    DocumentReranker._device = None
    DocumentReranker._init_error = None
    DocumentReranker._device_reason = None
    DocumentReranker._device_policy = "auto"
    yield
    DocumentReranker._instance = None
    DocumentReranker._model = None
    DocumentReranker._model_name = None
    DocumentReranker._device = None
    DocumentReranker._init_error = None
    DocumentReranker._device_reason = None
    DocumentReranker._device_policy = "auto"


class TestDocumentRerankerUnit:
    """Unit tests for convenience helpers and singleton state."""

    def test_rerank_empty_list(self):
        """Empty document lists should remain empty."""
        from data_base.reranker import rerank_documents

        result = rerank_documents("query", [], top_k=6, enabled=False)
        assert result == []

    def test_rerank_disabled(self):
        """Disabled reranking should preserve original order."""
        from data_base.reranker import rerank_documents

        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
            Document(page_content="Doc 3", metadata={}),
        ]

        result = rerank_documents("query", docs, top_k=2, enabled=False)

        assert len(result) == 2
        assert result[0].page_content == "Doc 1"
        assert result[1].page_content == "Doc 2"

    def test_rerank_top_k_truncation(self):
        """Disabled reranking should still respect top_k."""
        from data_base.reranker import rerank_documents

        docs = [Document(page_content=f"Doc {i}", metadata={}) for i in range(10)]

        result = rerank_documents("query", docs, top_k=3, enabled=False)

        assert len(result) == 3

    def test_runtime_metadata_before_initialization(self):
        """Runtime metadata should expose inactive reranker state."""
        from data_base.reranker import DocumentReranker

        assert not DocumentReranker.is_initialized()
        assert DocumentReranker.runtime_metadata(reason="test") == {
            "reranker_active": False,
            "reranker_model": "jinaai/jina-reranker-v3",
            "reranker_device": None,
            "reranker_reason": "test",
        }


class TestDocumentRerankerWithMock:
    """Tests with a mocked Jina reranker model."""

    def test_rerank_with_mock_model(self):
        """Rerank should sort by descending relevance score."""
        from data_base.reranker import DocumentReranker

        mock_model = MagicMock()
        mock_model.rerank.return_value = [
            {"index": 0, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.7},
            {"index": 1, "relevance_score": 0.5},
        ]

        docs = [
            Document(page_content="High relevance", metadata={"id": 1}),
            Document(page_content="Low relevance", metadata={"id": 2}),
            Document(page_content="Medium relevance", metadata={"id": 3}),
        ]

        reranker = object.__new__(DocumentReranker)
        DocumentReranker._instance = reranker
        DocumentReranker._model = mock_model
        DocumentReranker._model_name = "jinaai/jina-reranker-v3"
        DocumentReranker._device = "cpu"
        DocumentReranker._device_policy = "cpu"

        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0].metadata["id"] == 1
        assert result[1].metadata["id"] == 3

    def test_rerank_with_scores_mock(self):
        """rerank_with_scores should keep scores paired with documents."""
        from data_base.reranker import DocumentReranker

        mock_model = MagicMock()
        mock_model.rerank.return_value = [
            {"index": 1, "relevance_score": 0.8},
            {"index": 0, "relevance_score": 0.3},
        ]

        docs = [
            Document(page_content="First", metadata={}),
            Document(page_content="Second", metadata={}),
        ]

        reranker = object.__new__(DocumentReranker)
        DocumentReranker._instance = reranker
        DocumentReranker._model = mock_model
        DocumentReranker._model_name = "jinaai/jina-reranker-v3"
        DocumentReranker._device = "cpu"
        DocumentReranker._device_policy = "cpu"

        result = reranker.rerank_with_scores("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0][0].page_content == "Second"
        assert result[0][1] == 0.8
        assert result[1][0].page_content == "First"
        assert result[1][1] == 0.3

    def test_rerank_error_handling(self):
        """Model errors should degrade to original order."""
        from data_base.reranker import DocumentReranker

        mock_model = MagicMock()
        mock_model.rerank = MagicMock(side_effect=RuntimeError("Model error"))

        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
        ]

        reranker = object.__new__(DocumentReranker)
        DocumentReranker._instance = reranker
        DocumentReranker._model = mock_model
        DocumentReranker._model_name = "jinaai/jina-reranker-v3"
        DocumentReranker._device = "cpu"
        DocumentReranker._device_policy = "cpu"

        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0].page_content == "Doc 1"

    def test_rerank_logs_runtime_device(self, caplog):
        """Rerank logs should expose the active runtime device and candidate count."""
        from data_base.reranker import DocumentReranker

        mock_model = MagicMock()
        mock_model.rerank.return_value = [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.4},
        ]

        docs = [
            Document(page_content="First", metadata={}),
            Document(page_content="Second", metadata={}),
        ]

        reranker = object.__new__(DocumentReranker)
        DocumentReranker._instance = reranker
        DocumentReranker._model = mock_model
        DocumentReranker._model_name = "jinaai/jina-reranker-v3"
        DocumentReranker._device = "cpu"
        DocumentReranker._device_reason = "cuda_unavailable"
        DocumentReranker._device_policy = "cpu"

        with caplog.at_level(logging.INFO, logger="data_base.reranker"):
            reranker.rerank("query", docs, top_k=2)

        assert "Running rerank (reranker_device=cpu, reranker_reason=cuda_unavailable, candidate_count=2, top_k=2)" in caplog.text


class TestRerankerSingleton:
    """Tests for initialization and device selection."""

    def test_cuda_device_count_retries_after_explicit_init(self):
        """CUDA device counting should retry once after explicit init."""
        from data_base.reranker import _cuda_device_count

        with patch("data_base.reranker.torch.cuda.is_available", return_value=True), patch(
            "data_base.reranker.torch.cuda.device_count",
            side_effect=[0, 1],
        ) as mock_device_count, patch(
            "data_base.reranker.torch.cuda.init",
        ) as mock_cuda_init:
            assert _cuda_device_count() == 1

        assert mock_device_count.call_count == 2
        mock_cuda_init.assert_called_once()

    def test_cuda_device_count_can_recover_after_false_is_available(self):
        """CUDA probing should not trust an early false is_available() result."""
        from data_base.reranker import _cuda_device_count

        with patch("data_base.reranker.torch.cuda.is_available", return_value=False), patch(
            "data_base.reranker.torch.cuda.device_count",
            side_effect=[0, 1],
        ) as mock_device_count, patch(
            "data_base.reranker.torch.cuda.init",
        ) as mock_cuda_init:
            assert _cuda_device_count() == 1

        assert mock_device_count.call_count == 2
        mock_cuda_init.assert_called_once()

    def test_is_initialized_before_creation(self):
        """is_initialized returns False before model load."""
        from data_base.reranker import DocumentReranker

        assert not DocumentReranker.is_initialized()

    def test_singleton_uses_cpu_fallback(self):
        """Model initialization should fall back to CPU when CUDA is unavailable."""
        from data_base.reranker import DocumentReranker

        mock_model = MagicMock()
        with patch("data_base.reranker.torch.cuda.is_available", return_value=False), patch(
            "data_base.reranker.torch.cuda.device_count",
            return_value=0,
        ), patch(
            "data_base.reranker.torch.cuda.init",
            side_effect=RuntimeError("cuda unavailable"),
        ):
            with patch("data_base.reranker.AutoModel.from_pretrained", return_value=mock_model) as mock_from_pretrained:
                r1 = DocumentReranker()
                r2 = DocumentReranker()

        assert r1 is r2
        assert mock_from_pretrained.call_count == 1
        assert DocumentReranker.is_initialized()
        assert DocumentReranker.runtime_metadata()["reranker_device"] == "cpu"
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()

    def test_singleton_uses_cpu_when_gpu_memory_is_too_small(self):
        """Auto device selection should keep low-VRAM GPUs off the reranker path."""
        from data_base.reranker import DocumentReranker

        mock_model = MagicMock()
        with patch("data_base.reranker._DEFAULT_MIN_GPU_MEMORY_GB", 8.0), patch(
            "data_base.reranker.torch.cuda.is_available",
            return_value=True,
        ), patch(
            "data_base.reranker._gpu_total_memory_gb",
            return_value=7.0,
        ), patch(
            "data_base.reranker.AutoModel.from_pretrained",
            return_value=mock_model,
        ):
            DocumentReranker()

        assert DocumentReranker.runtime_metadata() == {
            "reranker_active": True,
            "reranker_model": "jinaai/jina-reranker-v3",
            "reranker_device": "cpu",
            "reranker_reason": "low_vram_7.0gb",
        }
        mock_model.to.assert_called_once_with("cpu")

    def test_singleton_uses_cpu_when_cuda_reports_no_valid_device(self):
        """Device selection should degrade to CPU when CUDA is flagged but no device is usable."""
        from data_base.reranker import DocumentReranker

        mock_model = MagicMock()
        with patch(
            "data_base.reranker.torch.cuda.is_available",
            return_value=True,
        ), patch(
            "data_base.reranker.torch.cuda.device_count",
            return_value=0,
        ), patch(
            "data_base.reranker.AutoModel.from_pretrained",
            return_value=mock_model,
        ):
            DocumentReranker()

        assert DocumentReranker.runtime_metadata() == {
            "reranker_active": True,
            "reranker_model": "jinaai/jina-reranker-v3",
            "reranker_device": "cpu",
            "reranker_reason": "cuda_unavailable",
        }
        mock_model.to.assert_called_once_with("cpu")

    def test_singleton_marks_masked_cuda_env(self):
        """Masked CUDA visibility should produce an explicit diagnostic reason."""
        from data_base.reranker import DocumentReranker

        mock_model = MagicMock()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False), patch(
            "data_base.reranker.torch.cuda.is_available",
            return_value=True,
        ), patch(
            "data_base.reranker.torch.cuda.device_count",
            return_value=0,
        ), patch(
            "data_base.reranker.torch.cuda.init",
            side_effect=RuntimeError("masked"),
        ), patch(
            "data_base.reranker.AutoModel.from_pretrained",
            return_value=mock_model,
        ):
            DocumentReranker()

        assert DocumentReranker.runtime_metadata() == {
            "reranker_active": True,
            "reranker_model": "jinaai/jina-reranker-v3",
            "reranker_device": "cpu",
            "reranker_reason": "cuda_masked_by_env",
        }
        mock_model.to.assert_called_once_with("cpu")

    def test_singleton_retries_on_cpu_after_cuda_oom(self):
        """Warmup should retry on CPU after a CUDA OOM during model load."""
        from data_base.reranker import DocumentReranker

        gpu_model = MagicMock()
        gpu_model.to.side_effect = RuntimeError("CUDA out of memory")
        cpu_model = MagicMock()
        with patch("data_base.reranker._select_runtime_device", return_value=("cuda", None)), patch(
            "data_base.reranker.AutoModel.from_pretrained",
            side_effect=[gpu_model, cpu_model],
        ):
            DocumentReranker()

        assert DocumentReranker.runtime_metadata() == {
            "reranker_active": True,
            "reranker_model": "jinaai/jina-reranker-v3",
            "reranker_device": "cpu",
            "reranker_reason": "cuda_oom_fallback",
        }
        cpu_model.to.assert_called_once_with("cpu")
        cpu_model.eval.assert_called_once()
