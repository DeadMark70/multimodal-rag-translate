"""
Document reranker module backed by Jina Reranker v3.

Provides local document reranking for retrieval precision improvements while
keeping a stable wrapper API for the rest of the application.
"""

# Standard library
import gc
import logging
import os
from typing import Any, List, Optional, Tuple

# Third-party
import torch
from langchain_core.documents import Document
from transformers import AutoModel

# Configure logging
logger = logging.getLogger(__name__)

# Default reranker model
_DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v3")
_DEFAULT_RERANKER_DEVICE_POLICY = os.getenv("RERANKER_DEVICE", "auto").strip().lower()
_DEFAULT_MIN_GPU_MEMORY_GB = float(os.getenv("RERANKER_MIN_GPU_GB", "7.5"))


def _is_cuda_oom_error(exc: RuntimeError) -> bool:
    """Return True when a RuntimeError represents CUDA OOM."""
    message = str(exc).lower()
    return "out of memory" in message or "cuda" in message and "memory" in message


def _clear_cuda_memory() -> None:
    """Best-effort release of cached CUDA memory."""
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass

    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def _gpu_total_memory_gb() -> Optional[float]:
    """Return total memory of the first CUDA device in GiB when available."""
    if not torch.cuda.is_available():
        return None

    try:
        if torch.cuda.device_count() < 1:
            return None
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except (AssertionError, RuntimeError):
        return None


def _normalize_device_policy(policy: Optional[str]) -> str:
    """Normalize reranker device policy to a supported value."""
    normalized = (policy or _DEFAULT_RERANKER_DEVICE_POLICY).strip().lower()
    return normalized if normalized in {"auto", "cpu", "cuda"} else "auto"


def _select_runtime_device(policy: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Choose a safe reranker device and expose the reason when CPU is selected."""
    normalized_policy = _normalize_device_policy(policy)

    if normalized_policy == "cpu":
        return "cpu", "manual_cpu_override"

    if normalized_policy == "cuda":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda", None
        return "cpu", "cuda_requested_unavailable"

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return "cpu", "cuda_unavailable"

    total_memory_gb = _gpu_total_memory_gb()
    if total_memory_gb is not None and total_memory_gb < _DEFAULT_MIN_GPU_MEMORY_GB:
        return "cpu", f"low_vram_{total_memory_gb:.1f}gb"

    return "cuda", None


def _select_device() -> str:
    """Choose the best available runtime device for reranking."""
    device, _ = _select_runtime_device()
    return device


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
    _device_reason: Optional[str] = None

    def __new__(
        cls,
        model_name: str = _DEFAULT_RERANKER_MODEL,
        device_policy: Optional[str] = None,
    ) -> "DocumentReranker":
        """Create or return the singleton reranker instance."""
        if cls._instance is None:
            instance = super().__new__(cls)
            try:
                instance._init_model(model_name, device_policy)
            except Exception as exc:
                cls._instance = None
                cls._model = None
                cls._model_name = model_name
                cls._device = None
                cls._init_error = str(exc)
                cls._device_reason = None
                raise
            cls._instance = instance
        return cls._instance

    @staticmethod
    def _load_model(model_name: str, device: str) -> Any:
        """Load and place the reranker model on the requested device."""
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model.to(device)
        model.eval()
        return model

    @classmethod
    def _swap_to_cpu_model(
        cls,
        *,
        model_name: str,
        reason: str,
    ) -> Any:
        """Release any GPU model state and reload the reranker on CPU."""
        current_model = cls._model
        cls._model = None

        if current_model is not None:
            del current_model

        gc.collect()
        _clear_cuda_memory()

        cpu_model = cls._load_model(model_name, "cpu")
        cls._model = cpu_model
        cls._device = "cpu"
        cls._device_reason = reason
        cls._init_error = None
        logger.warning(
            "Reranker switched to CPU fallback (reason=%s, reranker_model=%s)",
            reason,
            model_name,
        )
        return cpu_model

    def _init_model(self, model_name: str, device_policy: Optional[str] = None) -> None:
        """Initialize the Jina reranker model."""
        device, device_reason = _select_runtime_device(device_policy)
        logger.info(
            "Loading reranker model: %s (device=%s, reason=%s)",
            model_name,
            device,
            device_reason or "default",
        )

        try:
            model = self._load_model(model_name, device)
        except RuntimeError as exc:
            if device == "cuda" and _is_cuda_oom_error(exc):
                logger.warning(
                    "Reranker warmup hit CUDA OOM; retrying on CPU: %s",
                    exc,
                )
                model = type(self)._swap_to_cpu_model(
                    model_name=model_name,
                    reason="cuda_oom_fallback",
                )
                device = "cpu"
                device_reason = "cuda_oom_fallback"
            else:
                raise

        type(self)._model = model
        type(self)._model_name = model_name
        type(self)._device = device
        type(self)._init_error = None
        type(self)._device_reason = device_reason
        logger.info(
            "Reranker model loaded successfully (reranker_active=%s, reranker_model=%s, reranker_device=%s, reranker_reason=%s)",
            True,
            model_name,
            device,
            device_reason,
        )

    @classmethod
    def get_instance(cls, device_policy: Optional[str] = None) -> "DocumentReranker":
        """Return the singleton reranker instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = DocumentReranker(device_policy=device_policy)
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
            "reranker_reason": reason or cls._device_reason or cls._init_error,
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
            if type(self)._device == "cuda" and _is_cuda_oom_error(exc):
                logger.warning("CUDA OOM during reranking; retrying on CPU: %s", exc)
                model = type(self)._swap_to_cpu_model(
                    model_name=type(self)._model_name or _DEFAULT_RERANKER_MODEL,
                    reason="cuda_oom_fallback",
                )
                with torch.inference_mode():
                    results = model.rerank(
                        query=query,
                        documents=doc_texts,
                        top_n=min(top_k, len(documents)),
                    )
            else:
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


async def initialize_reranker(
    model_name: str = _DEFAULT_RERANKER_MODEL,
    device_policy: Optional[str] = None,
) -> None:
    """Initialize the reranker model in a threadpool during startup."""
    from fastapi.concurrency import run_in_threadpool

    logger.info("Initializing reranker...")
    await run_in_threadpool(DocumentReranker, model_name, device_policy)
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
