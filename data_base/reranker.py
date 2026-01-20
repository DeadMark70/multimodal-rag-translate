"""
Document Reranker Module (Local - ms-marco)

Provides Cross-Encoder based document reranking for improved retrieval precision.
Uses ms-marco-MiniLM-L-12-v2 for compliant relevance scoring.

Microsoft/SBERT (USA/Germany) - Apache 2.0 License
"""

# Standard library
import logging
from typing import List, Optional, Tuple

# Third-party
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# Configure logging
logger = logging.getLogger(__name__)

# Default reranker model (Microsoft - compliant)
_DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class DocumentReranker:
    """
    Cross-encoder based document reranker.
    
    Unlike bi-encoders that encode query and document separately,
    cross-encoders jointly encode (query, document) pairs for
    more accurate relevance scoring at the cost of speed.
    
    Implements singleton pattern since the model is large (~1GB) and
    expensive to load.
    
    Attributes:
        _model: The CrossEncoder model instance.
    """
    
    _instance: Optional["DocumentReranker"] = None
    _model: Optional[CrossEncoder] = None
    
    def __new__(cls, model_name: str = _DEFAULT_RERANKER_MODEL) -> "DocumentReranker":
        """
        Creates or returns singleton instance.
        
        Args:
            model_name: HuggingFace model name for reranking.
            
        Returns:
            The singleton DocumentReranker instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model(model_name)
        return cls._instance
    
    def _init_model(self, model_name: str) -> None:
        """
        Initializes the CrossEncoder model.
        
        Args:
            model_name: HuggingFace model name.
        """
        logger.info(f"Loading reranker model: {model_name}...")
        try:
            self._model = CrossEncoder(model_name)
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise RuntimeError(f"Reranker model initialization failed: {e}")
    
    @classmethod
    def get_instance(cls) -> "DocumentReranker":
        """
        Returns the singleton reranker instance.
        
        Creates the instance if it doesn't exist.
        
        Returns:
            The singleton DocumentReranker instance.
        """
        if cls._instance is None:
            cls._instance = DocumentReranker()
        return cls._instance
    
    @classmethod
    def is_initialized(cls) -> bool:
        """
        Checks if the reranker is initialized.
        
        Returns:
            True if initialized, False otherwise.
        """
        return cls._instance is not None and cls._model is not None
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 6,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """
        Reranks documents by relevance to query.
        
        Args:
            query: User question.
            documents: Candidate documents from initial retrieval.
            top_k: Number of documents to return.
            score_threshold: Minimum score to keep document (default None).
            
        Returns:
            Top-k documents sorted by relevance score (descending).
        """
        if not documents:
            return []
        
        if self._model is None:
            logger.warning("Reranker model not initialized, returning original order")
            return documents[:top_k]
        
        # Create (query, doc_content) pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Score all pairs
        try:
            scores = self._model.predict(pairs)
        except RuntimeError as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
        
        # Pair documents with scores and sort
        scored_docs: List[Tuple[Document, float]] = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold if provided
        filtered_docs = scored_docs
        if score_threshold is not None:
            filtered_docs = [
                (doc, score) for doc, score in scored_docs 
                if score >= score_threshold
            ]
            if len(filtered_docs) < len(scored_docs):
                logger.debug(f"Threshold {score_threshold} filtered out {len(scored_docs) - len(filtered_docs)} docs")
        
        logger.debug(f"Reranked {len(documents)} docs, returning top {top_k}")
        
        return [doc for doc, _ in filtered_docs[:top_k]]
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 6,
    ) -> List[Tuple[Document, float]]:
        """
        Reranks documents and returns with scores.
        
        Args:
            query: User question.
            documents: Candidate documents from initial retrieval.
            top_k: Number of documents to return.
            
        Returns:
            List of (Document, score) tuples sorted by relevance.
        """
        if not documents:
            return []
        
        if self._model is None:
            logger.warning("Reranker model not initialized, returning original order")
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        pairs = [(query, doc.page_content) for doc in documents]
        
        try:
            scores = self._model.predict(pairs)
        except RuntimeError as e:
            logger.error(f"Reranking failed: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        scored_docs: List[Tuple[Document, float]] = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]


async def initialize_reranker(model_name: str = _DEFAULT_RERANKER_MODEL) -> None:
    """
    Initializes the reranker model.
    
    Should be called during application startup.
    
    Args:
        model_name: HuggingFace model name.
    """
    from fastapi.concurrency import run_in_threadpool
    
    logger.info("Initializing reranker...")
    await run_in_threadpool(DocumentReranker, model_name)
    logger.info("Reranker initialized")


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 6,
    enabled: bool = True,
) -> List[Document]:
    """
    Convenience function to rerank documents.
    
    Args:
        query: User question.
        documents: Candidate documents.
        top_k: Number to return.
        enabled: If False, returns documents unchanged.
        
    Returns:
        Reranked documents (or original if disabled).
    """
    if not enabled:
        return documents[:top_k]
    
    if not DocumentReranker.is_initialized():
        logger.warning("Reranker not initialized, skipping reranking")
        return documents[:top_k]
    
    reranker = DocumentReranker.get_instance()
    return reranker.rerank(query, documents, top_k)
