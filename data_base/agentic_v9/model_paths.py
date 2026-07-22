"""Injected v9 adapters for every query-time model phase."""

from __future__ import annotations

from typing import Any

from core.prompt_loader import format_rag_pipeline_prompt
from data_base.agentic_v9.schemas import LlmInvoker


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    return content.strip() if isinstance(content, str) else str(content).strip()


class V9QueryRewriter:
    """Budgeted HyDE and multi-query rewrites for corrective retrieval."""

    def __init__(self, invoker: LlmInvoker) -> None:
        self._invoker = invoker

    async def hyde(self, question: str) -> str:
        response = await self._invoker.invoke(
            phase="query_rewrite",
            purpose="query_rewrite",
            messages=[
                {
                    "role": "user",
                    "content": format_rag_pipeline_prompt("hyde", question=question),
                }
            ],
        )
        return _response_text(response) or question

    async def multi_query(self, question: str, *, max_queries: int = 4) -> list[str]:
        response = await self._invoker.invoke(
            phase="query_rewrite",
            purpose="query_rewrite",
            messages=[
                {
                    "role": "user",
                    "content": format_rag_pipeline_prompt(
                        "multi_query", question=question
                    ),
                }
            ],
        )
        queries = [question]
        for line in _response_text(response).splitlines():
            candidate = line.strip().lstrip("0123456789.)").strip()
            if line[:1].isdigit() and len(candidate) > 5:
                queries.append(candidate)
        return queries[: max_queries + 1]


class V9CragJudge:
    """Budgeted relevance judgment for v9 corrective retrieval."""

    def __init__(self, invoker: LlmInvoker) -> None:
        self._invoker = invoker

    async def judge(self, messages: list[dict[str, Any]]) -> Any:
        return await self._invoker.invoke(
            phase="retrieval_judge", purpose="retrieval_judge", messages=messages
        )


class V9VisualHelper:
    """Budgeted visual extraction helper for v9-only visual work."""

    def __init__(self, invoker: LlmInvoker) -> None:
        self._invoker = invoker

    async def extract(self, messages: list[dict[str, Any]]) -> Any:
        return await self._invoker.invoke(
            phase="visual_extract", purpose="visual_analysis", messages=messages
        )


class V9EvidenceExtractor:
    """Budgeted evidence extraction helper for v9 retrieval results."""

    def __init__(self, invoker: LlmInvoker) -> None:
        self._invoker = invoker

    async def extract(self, messages: list[dict[str, Any]]) -> Any:
        return await self._invoker.invoke(
            phase="evidence_extract", purpose="evidence_extraction", messages=messages
        )


class V9ConflictArbiter:
    """Budgeted conflict arbitration helper for scope-aware evidence."""

    def __init__(self, invoker: LlmInvoker) -> None:
        self._invoker = invoker

    async def arbitrate(self, messages: list[dict[str, Any]]) -> Any:
        return await self._invoker.invoke(
            phase="conflict_arbitration",
            purpose="conflict_arbitration",
            messages=messages,
        )


class V9ClaimVerifier:
    """Budgeted high-risk claim verification helper."""

    def __init__(self, invoker: LlmInvoker) -> None:
        self._invoker = invoker

    async def verify(self, messages: list[dict[str, Any]]) -> Any:
        return await self._invoker.invoke(
            phase="claim_verifier", purpose="claim_verifier", messages=messages
        )


class V9FinalAnswerRenderer:
    """Budgeted final-answer rendering helper."""

    def __init__(self, invoker: LlmInvoker) -> None:
        self._invoker = invoker

    async def render(self, messages: list[dict[str, Any]]) -> Any:
        return await self._invoker.invoke(
            phase="final_answer", purpose="final_answer", messages=messages
        )
