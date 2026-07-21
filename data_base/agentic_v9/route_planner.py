"""Deterministic-first planning of bounded, evidence-only retrieval contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from data_base.agentic_v9.schemas import (
    AgenticV9Route,
    LlmInvoker,
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
)


_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "agentic_v9_route_planner.json"
_ROUTE_VALUES: set[str] = {
    "single_lookup",
    "bounded_compare",
    "exact_structured",
    "multi_document_exact",
    "multi_hop",
    "graph_relational",
}
_ENTITY_PATTERN = re.compile(r"(?<![\w-])[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*(?![\w-])")
_LOCATOR_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("figure", ("figure", "fig.", "圖")),
    ("table", ("table", "表")),
    ("appendix", ("appendix", "附錄")),
    ("formula", ("formula", "equation", "theorem", "公式", "定理")),
    ("page", ("page", "頁")),
)


@dataclass(frozen=True, slots=True)
class _RouteBudget:
    max_retrieval_rounds: int
    max_repair_rounds: int
    max_llm_calls: int
    runtime_token_budget: int


_ROUTE_BUDGETS: dict[AgenticV9Route, _RouteBudget] = {
    "single_lookup": _RouteBudget(1, 0, 2, 30_000),
    "bounded_compare": _RouteBudget(2, 1, 2, 40_000),
    "exact_structured": _RouteBudget(1, 1, 2, 40_000),
    "multi_document_exact": _RouteBudget(2, 1, 2, 50_000),
    "multi_hop": _RouteBudget(2, 1, 2, 50_000),
    "graph_relational": _RouteBudget(1, 1, 3, 50_000),
}


class _PlannerDecision(BaseModel):
    """The deliberately tiny schema accepted from the ambiguity-only model."""

    model_config = ConfigDict(extra="forbid")

    route: Literal[
        "single_lookup",
        "bounded_compare",
        "exact_structured",
        "multi_document_exact",
        "multi_hop",
        "graph_relational",
    ]


class RoutePlanner:
    """Create a retrieval-only ``QueryContract`` without generating an answer.

    Stable route shapes are classified locally.  The optional injected invoker
    is used once, through the established budgeted v9 boundary, only when the
    deterministic rules cannot classify the question.
    """

    def __init__(self, *, llm_invoker: LlmInvoker | None = None) -> None:
        self._llm_invoker = llm_invoker

    async def plan(
        self,
        *,
        question: str,
        resolved_source_scope: ResolvedSourceScope,
    ) -> QueryContract:
        """Return the bounded retrieval contract for one authorized question."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        route = _deterministic_route(normalized_question)
        planner_call_used = route is None
        if route is None:
            route = await self._resolve_ambiguous_route(
                question=normalized_question,
                resolved_source_scope=resolved_source_scope,
            )
        return _build_contract(
            question=normalized_question,
            route=route,
            resolved_source_scope=resolved_source_scope,
            planner_call_used=planner_call_used,
        )

    async def _resolve_ambiguous_route(
        self,
        *,
        question: str,
        resolved_source_scope: ResolvedSourceScope,
    ) -> AgenticV9Route:
        if self._llm_invoker is None:
            # No model boundary means ambiguity must still remain bounded and
            # retrieval-safe; a single lookup is the least expansive fallback.
            return "single_lookup"

        prompt = _load_prompt()
        messages = [
            {"role": "system", "content": prompt["system"]},
            {
                "role": "user",
                "content": prompt["user_template"].format(
                    question=question,
                    authorized_doc_ids=", ".join(
                        resolved_source_scope.authorized_doc_ids
                    )
                    or "none",
                ),
            },
        ]
        response = await self._llm_invoker.invoke(
            phase="route_plan",
            purpose="resolve_ambiguous_query_contract",
            messages=messages,
        )
        return _parse_route(response)


def _deterministic_route(question: str) -> AgenticV9Route | None:
    normalized = question.casefold()
    entities = _extract_entities(question)
    locator_hints = _locator_hints(normalized)

    if _contains_any(normalized, ("lineage path", "graph path", "relationship path", "技術脈絡", "關係路徑")):
        return "graph_relational"
    if len(locator_hints) >= 2 and len(entities) >= 2:
        return "multi_document_exact"
    if _contains_any(normalized, ("from ", " 到 ", "從 ", "演進", "取捨")) or (
        len(entities) >= 3
        and _contains_any(normalized, ("compare", "which", "哪個", "如何"))
    ):
        return "multi_hop"
    if locator_hints or _contains_any(normalized, ("calculate", "how many", "多少", "計算", "檢索")):
        return "exact_structured"
    if _contains_any(normalized, ("compare", "versus", " vs ", "which performs", "哪個", "是否互斥")):
        return "bounded_compare"
    if _contains_any(normalized, ("what is", "what are", "find ", "是什麼", "什麼是")):
        return "single_lookup"
    return None


def _build_contract(
    *,
    question: str,
    route: AgenticV9Route,
    resolved_source_scope: ResolvedSourceScope,
    planner_call_used: bool,
) -> QueryContract:
    entities = _extract_entities(question)
    locators = _locator_hints(question.casefold()) or ["source passage for each target slot"]
    budget = _ROUTE_BUDGETS[route]
    visual_required = any(hint in {"figure", "table"} for hint in locators)
    required_slots = [
        RequiredSlot(
            slot_id="slot-1",
            description=_intent_for_route(route, question),
            entity_ids=entities,
            locator_hints=locators,
        )
    ]
    if route in {"bounded_compare", "multi_document_exact", "multi_hop", "graph_relational"}:
        required_slots.append(
            RequiredSlot(
                slot_id="slot-2",
                description="Resolve source-bound scope, relationship, and qualification constraints.",
                entity_ids=entities,
                locator_hints=locators,
            )
        )

    return QueryContract(
        route=route,
        intent=_intent_for_route(route, question),
        required_slots=required_slots,
        entities=entities,
        locator_hints=locators,
        visual_required=visual_required,
        evidence_extraction_required=True,
        max_retrieval_rounds=budget.max_retrieval_rounds,
        max_repair_rounds=budget.max_repair_rounds,
        max_llm_calls=budget.max_llm_calls + int(planner_call_used),
        runtime_token_budget=budget.runtime_token_budget,
        resolved_source_scope=resolved_source_scope,
        strategy_tier="deterministic" if not planner_call_used else "budgeted_ambiguity",
    )


def _intent_for_route(route: AgenticV9Route, question: str) -> str:
    labels = {
        "single_lookup": "Locate one source-bound fact",
        "bounded_compare": "Compare a bounded set of source-bound claims",
        "exact_structured": "Extract exact structured values and locators",
        "multi_document_exact": "Extract exact values across named source groups",
        "multi_hop": "Resolve a source-bound multi-document relationship",
        "graph_relational": "Locate a graph relationship before source retrieval",
    }
    return f"{labels[route]}: {question}"


def _extract_entities(question: str) -> list[str]:
    """Keep only stable technical identifiers; ordinary prose is not an entity."""
    ignored = {"according", "and", "appendix", "compare", "figure", "find", "from", "please", "table", "the", "to", "what", "which"}
    return list(
        dict.fromkeys(
            value
            for value in _ENTITY_PATTERN.findall(question)
            if value.casefold() not in ignored
            and (any(character.isupper() for character in value) or "-" in value or any(character.isdigit() for character in value))
        )
    )


def _locator_hints(normalized_question: str) -> list[str]:
    return [
        label
        for label, terms in _LOCATOR_TERMS
        if _contains_any(normalized_question, terms)
    ]


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.casefold() in text for term in terms)


def _load_prompt() -> dict[str, str]:
    payload = json.loads(_PROMPT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload.get("system"), str) or not isinstance(payload.get("user_template"), str):
        raise ValueError("route planner prompt is incomplete")
    return payload


def _parse_route(response: Any) -> AgenticV9Route:
    content = response
    if isinstance(response, dict) and "content" in response:
        content = response["content"]
    elif hasattr(response, "content"):
        content = response.content
    if not isinstance(content, str):
        raise ValueError("route planner response must contain JSON text")
    try:
        decision = _PlannerDecision.model_validate_json(content)
    except ValueError as error:
        raise ValueError("route planner response is not a valid route decision") from error
    if decision.route not in _ROUTE_VALUES:  # Defensive guard if route literals evolve.
        raise ValueError("route planner selected an unsupported route")
    return decision.route


async def plan_query_contract(
    *,
    question: str,
    resolved_source_scope: ResolvedSourceScope,
    llm_invoker: LlmInvoker | None = None,
) -> QueryContract:
    """Convenience boundary for callers that do not need a retained planner."""
    return await RoutePlanner(llm_invoker=llm_invoker).plan(
        question=question,
        resolved_source_scope=resolved_source_scope,
    )


__all__ = ["RoutePlanner", "plan_query_contract"]
