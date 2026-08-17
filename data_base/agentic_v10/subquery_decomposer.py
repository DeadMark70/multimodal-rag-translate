"""Sub-query decomposer for Agentic RAG v10 using Structured Output."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.prompt_loader import format_agentic_v10_prompt, get_agentic_v10_prompt_registry
from core.providers import get_llm

logger = logging.getLogger(__name__)


class SubQueryItem(BaseModel):
    """A single focused search sub-query."""

    id: str = Field(description="Sub-query ID (e.g. SQ1, SQ2)")
    query: str = Field(
        description="Dense academic English search keywords for retrieval"
    )
    focus: str = Field(
        description="Traditional Chinese summary of what this sub-query aims to resolve"
    )
    target_entity: str = Field(
        default="",
        description="Target model name, method, metric, or module",
    )


class SubQueryDecompositionResponse(BaseModel):
    """Structured response containing 2 to 5 sub-queries."""

    sub_queries: list[SubQueryItem] = Field(
        min_length=2,
        max_length=5,
        description="List of 2 to 5 academic English search sub-queries",
    )


def _clean_english_keywords(text: str) -> str:
    """Extract and normalize ASCII/English keywords from a string."""
    cleaned = re.sub(r"[^\w\s\-\.\,\(\)\/]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) > 3 else text.strip()


def _fallback_subqueries(question: str) -> list[SubQueryItem]:
    """Deterministic fallback sub-queries if LLM decomposition fails."""
    # Look for common medical AI entity patterns (e.g., SAMed, MedSAM, SwinUNETR, nnFormer, etc.)
    entities = re.findall(
        r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+|\+\+|\d+D|\d+)?\b", question
    )
    unique_entities = list(
        dict.fromkeys(
            e for e in entities if len(e) >= 3 and e.lower() not in {"and", "the", "for", "with", "table", "figure"}
        )
    )

    if len(unique_entities) >= 2:
        items = []
        for idx, ent in enumerate(unique_entities[:4], start=1):
            items.append(
                SubQueryItem(
                    id=f"SQ{idx}",
                    query=f"{ent} architecture mechanism benchmark performance",
                    focus=f"檢索 {ent} 的架構機制與性能數據",
                    target_entity=ent,
                )
            )
        items.append(
            SubQueryItem(
                id=f"SQ{len(items)+1}",
                query=f"{' '.join(unique_entities[:3])} comparison evaluation table",
                focus="檢索各實體之對比評估與表格數據",
                target_entity="Comparison",
            )
        )
        return items[:5]

    english_part = _clean_english_keywords(question)
    return [
        SubQueryItem(
            id="SQ1",
            query=f"{english_part} core method architecture mechanism",
            focus="核心架構與方法機制檢索",
            target_entity="Core Method",
        ),
        SubQueryItem(
            id="SQ2",
            query=f"{english_part} experiment results table benchmark ablation",
            focus="實驗結果與消融指標檢索",
            target_entity="Evaluation",
        ),
    ]


class SubQueryDecomposer:
    """Decomposes complex user queries into 2~5 focused search sub-queries."""

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        self._llm = llm_client

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        return get_llm(purpose="planner")

    async def decompose(self, question: str) -> list[SubQueryItem]:
        """Decompose question into 2~5 English sub-queries with fail-soft fallback."""
        if not question or not question.strip():
            return []

        llm = self._get_llm()

        try:
            prompt_reg = get_agentic_v10_prompt_registry()
            sys_def = prompt_reg.get("subquery_decomposition_system")
            user_msg = format_agentic_v10_prompt(
                "subquery_decomposition_user", question=question.strip()
            )

            messages = [
                {"role": "system", "content": sys_def.template},
                {"role": "user", "content": user_msg},
            ]

            # Try structured output first if supported by model client
            if hasattr(llm, "with_structured_output"):
                try:
                    structured_llm = llm.with_structured_output(
                        SubQueryDecompositionResponse
                    )
                    res: SubQueryDecompositionResponse = await structured_llm.ainvoke(
                        messages
                    )
                    if res and res.sub_queries:
                        logger.info(
                            "SubQueryDecomposer succeeded via structured output: %d sub-queries",
                            len(res.sub_queries),
                        )
                        return res.sub_queries[:5]
                except Exception as structured_err:
                    logger.warning(
                        "with_structured_output failed; falling back to direct JSON invoke: %s",
                        structured_err,
                    )

            # Fallback to direct invocation
            if hasattr(llm, "ainvoke"):
                resp = await llm.ainvoke(messages)
            else:
                resp = llm.invoke(messages)

            resp_text = getattr(resp, "content", str(resp))
            parsed = self._extract_json(resp_text)
            if parsed and "sub_queries" in parsed:
                validated = SubQueryDecompositionResponse.model_validate(parsed)
                if validated.sub_queries:
                    return validated.sub_queries[:5]

        except Exception as exc:
            logger.warning(
                "SubQueryDecomposer error (%s: %s); applying deterministic fallback.",
                type(exc).__name__,
                exc,
            )

        return _fallback_subqueries(question)

    def _extract_json(self, text: str) -> Optional[dict[str, Any]]:
        """Extract JSON object from string."""
        text = text.strip()
        # Look for code block ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Look for raw { ... }
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None
