"""Fail-soft sub-query decomposition for the evaluation-only v10 runtime."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from core.prompt_loader import (
    format_agentic_v10_prompt,
    get_agentic_v10_prompt_registry,
)
from core.providers import get_llm

logger = logging.getLogger(__name__)


class SubQueryItem(BaseModel):
    """One focused retrieval query generated for a research question."""

    id: str = Field(description="Stable sub-query identifier")
    query: str = Field(description="Dense academic English retrieval query")
    focus: str = Field(description="Traditional Chinese focus description")
    target_entity: str = Field(default="", description="Primary target entity")


class SubQueryDecompositionResponse(BaseModel):
    """Structured LLM response containing two to five retrieval queries."""

    sub_queries: list[SubQueryItem] = Field(min_length=2, max_length=5)


class SubQueryDecompositionTrace(BaseModel):
    """Durable decomposition diagnostics kept inside the flexible v10 trace."""

    sub_queries: list[SubQueryItem]
    used_fallback: bool = False
    fallback_reason: str | None = None
    prompt_messages: list[dict[str, str]] = Field(default_factory=list)


def _clean_english_keywords(text: str) -> str:
    cleaned = re.sub(r"[^\w\s\-\.\,\(\)\/]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) > 3 else text.strip()


def _fallback_subqueries(question: str) -> list[SubQueryItem]:
    """Create deterministic queries when the decomposition provider is unavailable."""
    entities = re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+|\+\+|\d+D|\d+)?\b", question)
    ignored = {"and", "the", "for", "with", "table", "figure", "compare"}
    unique = list(dict.fromkeys(item for item in entities if len(item) >= 3 and item.lower() not in ignored))
    if len(unique) >= 2:
        items = [
            SubQueryItem(
                id=f"SQ{index}",
                query=f"{entity} architecture mechanism benchmark performance",
                focus=f"檢索 {entity} 的架構機制與性能數據",
                target_entity=entity,
            )
            for index, entity in enumerate(unique[:4], start=1)
        ]
        items.append(
            SubQueryItem(
                id=f"SQ{len(items) + 1}",
                query=f"{' '.join(unique[:3])} comparison evaluation table",
                focus="檢索各實體之對比評估與表格數據",
                target_entity="Comparison",
            )
        )
        return items[:5]
    keywords = _clean_english_keywords(question)
    return [
        SubQueryItem(
            id="SQ1",
            query=f"{keywords} core method architecture mechanism",
            focus="核心架構與方法機制檢索",
            target_entity="Core Method",
        ),
        SubQueryItem(
            id="SQ2",
            query=f"{keywords} experiment results table benchmark ablation",
            focus="實驗結果與消融指標檢索",
            target_entity="Evaluation",
        ),
    ]


class SubQueryDecomposer:
    """Turn a complex research question into two to five retrieval branches."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm = llm_client

    def _get_llm(self) -> Any:
        return self._llm if self._llm is not None else get_llm(purpose="planner")

    async def decompose(self, question: str) -> list[SubQueryItem]:
        """Return sub-queries while preserving the compact pre-v10 call contract."""
        return (await self.decompose_with_trace(question)).sub_queries

    async def decompose_with_trace(self, question: str) -> SubQueryDecompositionTrace:
        """Return sub-queries plus prompts and fallback diagnostics for export."""
        question = question.strip()
        if not question:
            return SubQueryDecompositionTrace(sub_queries=[])
        registry = get_agentic_v10_prompt_registry()
        messages = [
            {"role": "system", "content": registry.get("subquery_decomposition_system").template},
            {"role": "user", "content": format_agentic_v10_prompt("subquery_decomposition_user", question=question)},
        ]
        try:
            llm = self._get_llm()
            if hasattr(llm, "with_structured_output"):
                try:
                    response = await llm.with_structured_output(SubQueryDecompositionResponse).ainvoke(messages)
                    if response and response.sub_queries:
                        return SubQueryDecompositionTrace(sub_queries=response.sub_queries[:5], prompt_messages=messages)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("v10 structured decomposition failed: %s", exc)
            response = await (llm.ainvoke(messages) if hasattr(llm, "ainvoke") else llm.invoke(messages))
            parsed = self._extract_json(str(getattr(response, "content", response)))
            if parsed and "sub_queries" in parsed:
                validated = SubQueryDecompositionResponse.model_validate(parsed)
                return SubQueryDecompositionTrace(sub_queries=validated.sub_queries[:5], prompt_messages=messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("v10 decomposition failed; using fallback: %s", exc)
            return SubQueryDecompositionTrace(sub_queries=_fallback_subqueries(question), used_fallback=True, fallback_reason=type(exc).__name__, prompt_messages=messages)
        return SubQueryDecompositionTrace(sub_queries=_fallback_subqueries(question), used_fallback=True, fallback_reason="invalid_response", prompt_messages=messages)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        block = re.search(r"```(?:json)?\s*(.*?)\s*```", text.strip(), re.DOTALL)
        candidates = [block.group(1)] if block else []
        raw = re.search(r"(\{.*\})", text, re.DOTALL)
        if raw:
            candidates.append(raw.group(1))
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                return {"sub_queries": data}
            if isinstance(data, dict):
                return data
        return None
