"""
Semantic Text Chunker

Provides semantic-aware text chunking using embedding similarity breakpoints.
Unlike mechanical chunking with fixed character counts, this module splits
text at natural semantic boundaries.
"""

# Standard library
import logging
import re
from typing import List, Optional, Literal

# Third-party
import numpy as np
from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)

# Type definitions
BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]


def _split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using regex patterns.
    
    Handles Chinese and English punctuation.
    
    Args:
        text: Input text to split.
        
    Returns:
        List of sentence strings.
    """
    # Pattern for sentence-ending punctuation (Chinese and English)
    sentence_endings = r'(?<=[。！？.!?])\s*'
    
    # Split by sentence endings
    sentences = re.split(sentence_endings, text)
    
    # Filter empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def _calculate_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculates cosine distances between consecutive embeddings.
    
    Args:
        embeddings: Array of shape (n_sentences, embedding_dim).
        
    Returns:
        Array of shape (n_sentences - 1,) with distances.
    """
    distances = []
    for i in range(len(embeddings) - 1):
        # Cosine similarity
        similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
        )
        # Convert to distance
        distance = 1 - similarity
        distances.append(distance)
    
    return np.array(distances)


def _get_breakpoint_threshold(
    distances: np.ndarray,
    threshold_type: BreakpointThresholdType,
    threshold_amount: float,
) -> float:
    """
    Calculates the breakpoint threshold based on distance distribution.
    
    Args:
        distances: Array of cosine distances.
        threshold_type: Method for calculating threshold.
        threshold_amount: Parameter for the threshold method.
        
    Returns:
        Threshold value for determining breakpoints.
    """
    if threshold_type == "percentile":
        return float(np.percentile(distances, threshold_amount))
    elif threshold_type == "standard_deviation":
        return float(np.mean(distances) + threshold_amount * np.std(distances))
    elif threshold_type == "interquartile":
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        return float(q3 + threshold_amount * iqr)
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")


class SemanticTextChunker:
    """
    Semantic-aware text chunker using embedding similarity breakpoints.
    
    Instead of fixed character counts, splits at semantic boundaries
    detected via embedding similarity changes between sentences.
    
    Attributes:
        _embeddings: HuggingFace embedding model instance.
        _threshold_type: Method for calculating breakpoints.
        _threshold_amount: Parameter for the threshold method.
        _min_chunk_size: Minimum characters per chunk.
        _max_chunk_size: Maximum characters per chunk.
    """
    
    def __init__(
        self,
        embeddings,  # Any embedding model with embed_documents method
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: float = 90.0,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ) -> None:
        """
        Initializes the semantic chunker.
        
        Args:
            embeddings: HuggingFace embedding model (injected to avoid circular imports).
            breakpoint_threshold_type: Method for breakpoint detection.
                - "percentile": Use percentile of distance distribution.
                - "standard_deviation": Mean + N*std.
                - "interquartile": Q3 + N*IQR.
            breakpoint_threshold_amount: Parameter for threshold calculation.
                - For percentile: 90 means 90th percentile.
                - For std: 1.0 means mean + 1*std.
                - For IQR: 1.5 is common (like outlier detection).
            min_chunk_size: Minimum characters per chunk.
            max_chunk_size: Maximum characters per chunk.
        """
        self._embeddings = embeddings
        self._threshold_type = breakpoint_threshold_type
        self._threshold_amount = breakpoint_threshold_amount
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        
        logger.info(
            f"SemanticTextChunker initialized: "
            f"threshold_type={breakpoint_threshold_type}, "
            f"threshold_amount={breakpoint_threshold_amount}"
        )
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merges chunks that are smaller than min_chunk_size.
        
        Args:
            chunks: List of chunk strings.
            
        Returns:
            List with small chunks merged into neighbors.
        """
        if not chunks:
            return chunks
        
        merged = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            if len(current) < self._min_chunk_size:
                # Merge with next chunk
                current = current + " " + chunk
            else:
                merged.append(current)
                current = chunk
        
        # Add the last chunk
        merged.append(current)
        
        return merged
    
    def _split_large_chunks(self, chunks: List[str]) -> List[str]:
        """
        Splits chunks that exceed max_chunk_size.
        
        Uses paragraph and sentence boundaries when possible.
        
        Args:
            chunks: List of chunk strings.
            
        Returns:
            List with large chunks split.
        """
        result = []
        
        for chunk in chunks:
            if len(chunk) <= self._max_chunk_size:
                result.append(chunk)
            else:
                # Try splitting by paragraphs first
                paragraphs = chunk.split("\n\n")
                current = ""
                
                for para in paragraphs:
                    if len(current) + len(para) <= self._max_chunk_size:
                        current = current + "\n\n" + para if current else para
                    else:
                        if current:
                            result.append(current.strip())
                        
                        # If paragraph itself is too large, split by sentences
                        if len(para) > self._max_chunk_size:
                            sentences = _split_into_sentences(para)
                            sent_current = ""
                            
                            for sent in sentences:
                                if len(sent_current) + len(sent) <= self._max_chunk_size:
                                    sent_current = sent_current + " " + sent if sent_current else sent
                                else:
                                    if sent_current:
                                        result.append(sent_current.strip())
                                    sent_current = sent
                            
                            if sent_current:
                                current = sent_current.strip()
                            else:
                                current = ""
                        else:
                            current = para
                
                if current:
                    result.append(current.strip())
        
        return result
    
    def split_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> List[Document]:
        """
        Splits text into semantically coherent chunks.
        
        Args:
            text: Full text to split.
            metadata: Base metadata to attach to all chunks.
            
        Returns:
            List of LangChain Documents with semantic chunks.
        """
        if metadata is None:
            metadata = {}
        
        # Handle empty or very short text
        if not text or len(text.strip()) < self._min_chunk_size:
            if text.strip():
                return [Document(page_content=text.strip(), metadata=metadata.copy())]
            return []
        
        # Step 1: Split into sentences
        sentences = _split_into_sentences(text)
        
        if len(sentences) <= 1:
            # Only one sentence, return as single chunk
            return [Document(page_content=text.strip(), metadata=metadata.copy())]
        
        logger.debug(f"Split text into {len(sentences)} sentences")
        
        # Step 2: Compute embeddings for all sentences
        try:
            sentence_embeddings = self._embeddings.embed_documents(sentences)
            sentence_embeddings = np.array(sentence_embeddings)
        except RuntimeError as e:
            logger.error(f"Embedding computation failed: {e}")
            # Fallback: return as single chunk
            return [Document(page_content=text.strip(), metadata=metadata.copy())]
        
        # Step 3: Calculate cosine distances between consecutive sentences
        distances = _calculate_cosine_distances(sentence_embeddings)
        
        if len(distances) == 0:
            return [Document(page_content=text.strip(), metadata=metadata.copy())]
        
        # Step 4: Determine breakpoint threshold
        threshold = _get_breakpoint_threshold(
            distances, self._threshold_type, self._threshold_amount
        )
        
        logger.debug(f"Breakpoint threshold: {threshold:.4f}")
        
        # Step 5: Find breakpoints (where distance exceeds threshold)
        breakpoint_indices = [i for i, d in enumerate(distances) if d > threshold]
        
        logger.debug(f"Found {len(breakpoint_indices)} breakpoints")
        
        # Step 6: Create chunks based on breakpoints
        chunks = []
        start_idx = 0
        
        for bp_idx in breakpoint_indices:
            # Chunk includes sentences from start_idx to bp_idx (inclusive)
            chunk_sentences = sentences[start_idx:bp_idx + 1]
            chunk_text = " ".join(chunk_sentences)
            chunks.append(chunk_text)
            start_idx = bp_idx + 1
        
        # Add remaining sentences as final chunk
        if start_idx < len(sentences):
            chunk_text = " ".join(sentences[start_idx:])
            chunks.append(chunk_text)
        
        # Step 7: Post-process chunks
        chunks = self._merge_small_chunks(chunks)
        chunks = self._split_large_chunks(chunks)
        
        logger.debug(f"Final chunk count: {len(chunks)}")
        
        # Step 8: Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunking_method"] = "semantic"
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        return documents


def create_semantic_chunker(
    embeddings,  # Any embedding model with embed_documents method
    threshold_type: BreakpointThresholdType = "percentile",
    threshold_amount: float = 90.0,
    min_size: int = 100,
    max_size: int = 2000,
) -> SemanticTextChunker:
    """
    Factory function to create a SemanticTextChunker.
    
    Args:
        embeddings: HuggingFace embedding model.
        threshold_type: Breakpoint detection method.
        threshold_amount: Threshold parameter.
        min_size: Minimum chunk size.
        max_size: Maximum chunk size.
        
    Returns:
        Configured SemanticTextChunker instance.
    """
    return SemanticTextChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=threshold_amount,
        min_chunk_size=min_size,
        max_chunk_size=max_size,
    )
