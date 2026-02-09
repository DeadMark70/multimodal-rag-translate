"""
Unit Tests for Semantic Chunker

Tests the SemanticTextChunker class and related functions.
"""

# Standard library
import pytest

# Third-party

# Local application
from data_base.semantic_chunker import (
    SemanticTextChunker,
    _split_into_sentences,
    _calculate_cosine_distances,
    _get_breakpoint_threshold,
    create_semantic_chunker,
)


class TestSplitIntoSentences:
    """Tests for the _split_into_sentences function."""

    def test_split_chinese_sentences(self):
        """Tests splitting Chinese text with Chinese punctuation."""
        text = "這是第一句。這是第二句！這是第三句？"
        result = _split_into_sentences(text)
        
        # Note: The regex keeps punctuation with each sentence
        assert len(result) == 3
        assert "這是第一句" in result[0]
        assert "這是第二句" in result[1]
        assert "這是第三句" in result[2]

    def test_split_english_sentences(self):
        """Tests splitting English text."""
        text = "This is the first sentence. This is the second! And the third?"
        result = _split_into_sentences(text)
        
        assert len(result) == 3
        assert "first sentence" in result[0]
        assert "second" in result[1]
        assert "third" in result[2]

    def test_split_mixed_sentences(self):
        """Tests splitting mixed Chinese and English text."""
        text = "這是中文。This is English. 再來一句中文。"
        result = _split_into_sentences(text)
        
        assert len(result) == 3

    def test_empty_text(self):
        """Tests empty text input."""
        result = _split_into_sentences("")
        assert result == []

    def test_single_sentence(self):
        """Tests text with single sentence."""
        text = "只有一句話"
        result = _split_into_sentences(text)
        
        assert len(result) == 1
        assert result[0] == "只有一句話"


class TestCalculateCosineDistances:
    """Tests for the _calculate_cosine_distances function."""

    def test_calculate_distances(self):
        """Tests calculating distances between embeddings."""
        import numpy as np
        
        # Create test embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Same as first - distance should be 0
            [0.0, 1.0, 0.0],  # Different - distance should be 1
        ])
        
        distances = _calculate_cosine_distances(embeddings)
        
        assert len(distances) == 2
        assert distances[0] < 0.01  # Nearly 0 for identical vectors
        assert distances[1] > 0.99  # Nearly 1 for orthogonal vectors

    def test_single_embedding(self):
        """Tests with single embedding (no distances)."""
        import numpy as np
        
        embeddings = np.array([[1.0, 0.0, 0.0]])
        distances = _calculate_cosine_distances(embeddings)
        
        assert len(distances) == 0


class TestGetBreakpointThreshold:
    """Tests for the _get_breakpoint_threshold function."""

    def test_percentile_threshold(self):
        """Tests percentile-based threshold calculation."""
        import numpy as np
        
        distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        threshold = _get_breakpoint_threshold(distances, "percentile", 90.0)
        
        assert 0.85 <= threshold <= 0.95

    def test_standard_deviation_threshold(self):
        """Tests standard deviation-based threshold."""
        import numpy as np
        
        distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        threshold = _get_breakpoint_threshold(distances, "standard_deviation", 1.0)
        
        # mean = 0.3, std ≈ 0.158, so threshold ≈ 0.458
        assert 0.4 <= threshold <= 0.5

    def test_interquartile_threshold(self):
        """Tests interquartile-based threshold."""
        import numpy as np
        
        distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        threshold = _get_breakpoint_threshold(distances, "interquartile", 1.5)
        
        assert threshold > 0.5

    def test_invalid_threshold_type(self):
        """Tests error on invalid threshold type."""
        import numpy as np
        
        distances = np.array([0.1, 0.2, 0.3])
        
        with pytest.raises(ValueError):
            _get_breakpoint_threshold(distances, "invalid_type", 90.0)


class TestSemanticTextChunker:
    """Tests for the SemanticTextChunker class."""

    def test_init(self, mock_embeddings):
        """Tests chunker initialization."""
        chunker = SemanticTextChunker(
            embeddings=mock_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90.0,
            min_chunk_size=100,
            max_chunk_size=2000,
        )
        
        assert chunker._embeddings == mock_embeddings
        assert chunker._threshold_type == "percentile"
        assert chunker._threshold_amount == 90.0

    def test_split_text_empty(self, mock_embeddings):
        """Tests splitting empty text."""
        chunker = SemanticTextChunker(mock_embeddings)
        
        result = chunker.split_text("", {})
        assert result == []

    def test_split_text_short(self, mock_embeddings):
        """Tests splitting text shorter than min_chunk_size."""
        chunker = SemanticTextChunker(mock_embeddings, min_chunk_size=100)
        
        result = chunker.split_text("短文", {"test": True})
        
        assert len(result) == 1
        assert result[0].page_content == "短文"
        assert result[0].metadata["test"]

    def test_split_text_single_sentence(self, mock_embeddings):
        """Tests splitting text with single sentence."""
        chunker = SemanticTextChunker(mock_embeddings, min_chunk_size=10)
        
        result = chunker.split_text(
            "這是一個測試句子但沒有句號所以只有一句",
            {"page": 1}
        )
        
        assert len(result) == 1

    def test_split_text_preserves_metadata(self, mock_embeddings):
        """Tests that metadata is preserved in chunks."""
        chunker = SemanticTextChunker(mock_embeddings, min_chunk_size=10)
        
        metadata = {"page": 1, "doc_id": "test-123"}
        result = chunker.split_text(
            "第一句話。第二句話。第三句話。",
            metadata
        )
        
        for doc in result:
            assert doc.metadata["page"] == 1
            assert doc.metadata["doc_id"] == "test-123"
            assert doc.metadata["chunking_method"] == "semantic"


class TestCreateSemanticChunker:
    """Tests for the create_semantic_chunker factory function."""

    def test_create_chunker(self, mock_embeddings):
        """Tests factory function creates chunker correctly."""
        chunker = create_semantic_chunker(
            embeddings=mock_embeddings,
            threshold_type="percentile",
            threshold_amount=85.0,
            min_size=50,
            max_size=1500,
        )
        
        assert isinstance(chunker, SemanticTextChunker)
        assert chunker._threshold_amount == 85.0
        assert chunker._min_chunk_size == 50
        assert chunker._max_chunk_size == 1500
