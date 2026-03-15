"""
Document reranker module backed by Jina Reranker v3.

Provides local document reranking for retrieval precision improvements while
keeping a stable wrapper API for the rest of the application.
"""

# Standard library
import logging
from typing import Any, List, Optional, Tuple

# Third-party
import torch
from langchain_core.documents import Document
from transformers import AutoModel

# Configure logging
logger = logging.getLogger(__name__)

# Default reranker model
_DEFAULT_RERANKER_MODEL = "jinaai/jina-reranker-v3"


def _select_device() -> str:
    """Choose the best available runtime device for reranking."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class DocumentReranker:
    """
    Jina-based document reranker.

    The wrapper keeps the previous singleton-style API so the rest of the
    backend can switch model providers without changing call sites.
    """

    _instance: Optional["DocumentReranker"] = None
    _model: Optional[Any] = None
    _model_name: Optional[str] = None
    _device: Optional[str] = None
    _init_error: Optional[str] = None

    def __new__(cls, model_name: str = _DEFAULT_RERANKER_MODEL) -> "DocumentReranker":
        """Create or return the singleton reranker instance."""
        if cls._instance is None:
            instance = super().__new__(cls)
            try:
                instance._init_model(model_name)
            except Exception as exc:
                cls._instance = None
                cls._model = None
                cls._model_name = model_name
                cls._device = None
                cls._init_error = str(exc)
                raise
            cls._instance = instance
        return cls._instance

    def _init_model(self, model_name: str) -> None:
        """Initialize the Jina reranker model."""
        device = _select_device()
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info(
            "Loading reranker model: %s (device=%s)",
            model_name,
            device,
        )

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model.to(device)
        model.eval()

        type(self)._model = model
        type(self)._model_name = model_name
        type(self)._device = device
        type(self)._init_error = None
        logger.info(
            "Reranker model loaded successfully (reranker_active=%s, reranker_model=%s, reranker_device=%s)",
            True,
            model_name,
            device,
        )

    @classmethod
    def get_instance(cls) -> "DocumentReranker":
        """Return the singleton reranker instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = DocumentReranker()
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Check whether the reranker is ready for inference."""
        return cls._instance is not None and cls._model is not None

    @classmethod
    def runtime_metadata(cls, reason: Optional[str] = None) -> dict[str, Any]:
        """Return structured runtime metadata for observability."""
        return {
            "reranker_active": cls.is_initialized(),
            "reranker_model": cls._model_name or _DEFAULT_RERANKER_MODEL,
            "reranker_device": cls._device,
            "reranker_reason": reason or cls._init_error,
        }

    def _run_rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """Run the underlying model and normalize its results."""
        if not documents:
            return []

        model = type(self)._model
        if model is None:
            logger.warning(
                "Reranker model not initialized: %s",
                type(self).runtime_metadata(reason="not_initialized"),
            )
            return [(doc, 0.0) for doc in documents[:top_k]]

        doc_texts = [doc.page_content for doc in documents]

        try:
            with torch.inference_mode():
                results = model.rerank(
                    query=query,
                    documents=doc_texts,
                    top_n=min(top_k, len(documents)),
                )
        except RuntimeError as exc:
            logger.error("Reranking failed: %s", exc)
            return [(doc, 0.0) for doc in documents[:top_k]]

        normalized: List[Tuple[Document, float]] = []
        for item in results:
            index = int(item["index"])
            score = float(item["relevance_score"])
            normalized.append((documents[index], score))

        normalized.sort(key=lambda entry: entry[1], reverse=True)
        return normalized[:top_k]

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 6,
    ) -> List[Document]:
        """Rerank documents by relevance to the query."""
        return [doc for doc, _ in self._run_rerank(query, documents, top_k)]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 6,
    ) -> List[Tuple[Document, float]]:
        """Rerank documents and return paired relevance scores."""
        return self._run_rerank(query, documents, top_k)


async def initialize_reranker(model_name: str = _DEFAULT_RERANKER_MODEL) -> None:
    """Initialize the reranker model in a threadpool during startup."""
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
    """Convenience function to rerank documents or return the original slice."""
    if not enabled:
        return documents[:top_k]

    if not DocumentReranker.is_initialized():
        logger.warning(
            "Reranker not initialized, skipping reranking: %s",
            DocumentReranker.runtime_metadata(reason="not_initialized"),
        )
        return documents[:top_k]

    reranker = DocumentReranker.get_instance()
    return reranker.rerank(query, documents, top_k)
