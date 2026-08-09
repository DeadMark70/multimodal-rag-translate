"""Shared value objects for the legacy RAG retrieval and generation boundary.

These schemas deliberately keep retrieval artifacts separate from generated answer
artifacts.  ``RAGResult`` remains the public legacy projection until the wrapper
is extracted in a later refactor step.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional

from langchain_core.documents import Document


class RAGResult(NamedTuple):
    """Result from RAG question answering with optional documents."""

    answer: str
    source_doc_ids: List[str]
    documents: List[Document]
    usage: Dict[str, int] = {}
    thought_process: Optional[str] = None
    tool_calls: List[dict] = []
    agent_trace: Optional[dict] = None
    visual_verification_meta: Optional[Dict[str, Any]] = None


ProgressCallback = Callable[[str, Optional[Dict[str, Any]]], Awaitable[None]]


@dataclass(slots=True)
class RagRetrievalResult:
    """Evidence and observability data produced before answer generation.

    ``metadata`` holds retrieval-specific observability such as query origins,
    candidate ranks, filtering outcomes, and graph-location details.  It must
    not contain a generated answer.
    """

    documents: list[Document] = field(default_factory=list)
    source_doc_ids: list[str] = field(default_factory=list)
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class GeneratedRagAnswer:
    """Answer-generation output prior to projection into legacy ``RAGResult``."""

    answer: str
    usage: dict[str, int] = field(default_factory=dict)
    thought_process: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: dict[str, Any] | None = None
    visual_verification_meta: dict[str, Any] | None = None


__all__ = [
    "GeneratedRagAnswer",
    "ProgressCallback",
    "RAGResult",
    "RagRetrievalResult",
]
