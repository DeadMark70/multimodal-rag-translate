"""Question-only planning of answer-free atomic Agentic v9 contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from core.prompt_loader import PromptRegistry
from data_base.agentic_v9.schemas import (
    AgenticV9Route,
    BudgetExceededError,
    LlmInvoker,
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
    RouteDecision,
)
from data_base.agentic_v9.slot_constraints import canonical_structured_locator

_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "agentic_v9_contract_planner.json"
)
_PROMPT_KEY = "atomic_contract_planning"
_ENTITY_PATTERN = re.compile(
    r"(?<![\w-])[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*(?![\w-])"
)
_EXACT_LOCATOR_PATTERN = re.compile(
    r"\b(Figure|Fig\.|Table|Appendix|Formula|Equation|Theorem|Page|Section)"
    r"\s*([A-Za-z0-9]+(?:\([a-z]\))?)",
    re.IGNORECASE,
)
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
    "single_lookup": _RouteBudget(1, 0, 3, 30_000),
    "bounded_compare": _RouteBudget(2, 1, 3, 40_000),
    "exact_structured": _RouteBudget(1, 1, 3, 40_000),
    "multi_document_exact": _RouteBudget(2, 1, 3, 50_000),
    "multi_hop": _RouteBudget(2, 1, 3, 50_000),
    "graph_relational": _RouteBudget(1, 1, 4, 50_000),
}


class _PlannerSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    source_name_hints: list[str]
    authorized_source_doc_ids: list[str]
    locator_hints: list[str]
    expected_answer_type: Literal[
        "number",
        "equation",
        "definition",
        "comparison",
        "explanation",
        "text",
    ]
    depends_on_slot_ids: list[str]
    visual_policy: Literal["never", "preferred", "required"]


class _PlannerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_route: Literal[
        "single_lookup",
        "bounded_compare",
        "exact_structured",
        "multi_document_exact",
        "multi_hop",
        "graph_relational",
    ]
    slots: list[_PlannerSlot]
    route_reason: str
    confidence: float


@dataclass(frozen=True, slots=True)
class _AmbiguityResult:
    route: AgenticV9Route
    slots: list[RequiredSlot] | None
    route_reason: str
    confidence: float
    decision_source: Literal["llm_planner", "safe_fallback"]
    fallback_reason: str | None


class QuestionContractPlanner:
    """Build v2 contracts from the question, authorized sources, and setup only."""

    def __init__(self, *, llm_invoker: LlmInvoker | None = None) -> None:
        self._llm_invoker = llm_invoker

    async def plan(
        self,
        *,
        question: str,
        authorized_source_names: list[str],
        authorized_source_doc_ids: list[str],
        setup_policy: dict[str, Any],
        authorized_source_name_to_doc_ids: dict[str, list[str]] | None = None,
    ) -> QueryContract:
        del setup_policy  # Task 6 applies these authoritative provider limits.
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        source_mapping = _source_mapping(
            source_names=authorized_source_names,
            source_doc_ids=authorized_source_doc_ids,
            authoritative=authorized_source_name_to_doc_ids,
        )
        ordered_source_doc_ids = [
            source_mapping[name][0]
            for name in authorized_source_names
            if source_mapping.get(name)
        ]
        if not ordered_source_doc_ids:
            ordered_source_doc_ids = list(authorized_source_doc_ids)
        mapping_missing = bool(authorized_source_names) and (
            set(source_mapping) != set(authorized_source_names)
        )

        route = _deterministic_route(normalized_question)
        ambiguity: _AmbiguityResult | None = None
        planner_call_requested = False
        if mapping_missing:
            ambiguity = _safe_ambiguity_result("authoritative_source_mapping_missing")
            if route is None:
                route = ambiguity.route
        elif route is None:
            planner_call_requested = True
            ambiguity = await self._resolve_ambiguous_contract(
                question=normalized_question,
                authorized_source_names=authorized_source_names,
                authorized_source_doc_ids=authorized_source_doc_ids,
                authorized_source_name_to_doc_ids=source_mapping,
            )
            route = ambiguity.route
        if mapping_missing:
            slots = [
                _slot(
                    description=(
                        "Resolve the bounded requirement without assuming "
                        "source-name-to-document-ID pairing."
                    ),
                    answer_type="text",
                    source_names=authorized_source_names,
                    source_doc_ids=authorized_source_doc_ids,
                )
            ]
            matched_rules = ["authoritative_source_mapping_missing"]
        elif ambiguity and ambiguity.slots:
            slots = ambiguity.slots
            matched_rules = ["llm_atomic_decomposition"]
        else:
            slots, matched_rules = _decompose(
                question=normalized_question,
                source_names=authorized_source_names,
                source_doc_ids=ordered_source_doc_ids,
            )
        if not slots:
            slots = [
                _slot(
                    description="Resolve the source-bound requirement in the question.",
                    answer_type="text",
                    source_names=authorized_source_names,
                    source_doc_ids=ordered_source_doc_ids,
                )
            ]
        if len(slots) == 1 and route in {
            "bounded_compare",
            "multi_document_exact",
            "multi_hop",
            "graph_relational",
        }:
            slots.append(
                _slot(
                    description="Resolve the independent source-bound comparison or qualification.",
                    answer_type="comparison",
                    source_names=authorized_source_names,
                    source_doc_ids=ordered_source_doc_ids,
                )
            )
        slots = [
            slot.model_copy(update={"slot_id": f"S{index}"})
            for index, slot in enumerate(slots[:8], 1)
        ]
        locators = list(
            dict.fromkeys(locator for slot in slots for locator in slot.locator_hints)
        ) or ["source passage for each target slot"]
        entities = _extract_entities(normalized_question)
        budget = _ROUTE_BUDGETS[route]
        decision_source = ambiguity.decision_source if ambiguity else "deterministic"
        decision = RouteDecision(
            selected_route=route,
            decision_source=decision_source,
            matched_rules=matched_rules,
            candidate_routes=_candidate_routes(route, matched_rules),
            route_reason=(
                ambiguity.route_reason
                if ambiguity
                else _route_reason(route, matched_rules)
            ),
            planner_call_used=(
                planner_call_requested and self._llm_invoker is not None
            ),
            fallback_reason=ambiguity.fallback_reason if ambiguity else None,
            confidence=ambiguity.confidence if ambiguity else 1.0,
        )
        visual_requested = any(
            slot.visual_policy in {"preferred", "required"} for slot in slots
        )
        visual_required = any(slot.visual_policy == "required" for slot in slots)
        return QueryContract(
            contract_version="2",
            route=route,
            intent=_intent_for_route(route, normalized_question),
            required_slots=slots,
            entities=entities,
            locator_hints=locators,
            visual_requested=visual_requested,
            visual_required=visual_required,
            evidence_extraction_required=True,
            max_retrieval_rounds=budget.max_retrieval_rounds,
            max_repair_rounds=budget.max_repair_rounds,
            max_llm_calls=budget.max_llm_calls + int(planner_call_requested),
            runtime_token_budget=budget.runtime_token_budget,
            resolved_source_scope=ResolvedSourceScope(
                requested_source_names=authorized_source_names,
                resolved_doc_ids=authorized_source_doc_ids,
                authorized_doc_ids=authorized_source_doc_ids,
                source_name_to_doc_ids=source_mapping,
            ),
            strategy_tier=(
                "budgeted_ambiguity"
                if planner_call_requested
                else "safe_fallback"
                if mapping_missing
                else "deterministic"
            ),
            route_decision=decision,
            slot_plan_status=(
                "degraded" if decision_source == "safe_fallback" else "complete"
            ),
        )

    async def _resolve_ambiguous_contract(
        self,
        *,
        question: str,
        authorized_source_names: list[str],
        authorized_source_doc_ids: list[str],
        authorized_source_name_to_doc_ids: dict[str, list[str]],
    ) -> _AmbiguityResult:
        if self._llm_invoker is None:
            return _safe_ambiguity_result("planner_unavailable")
        prompt = PromptRegistry(_PROMPT_PATH).format(
            _PROMPT_KEY,
            question=question,
            authorized_source_names=json.dumps(
                authorized_source_names, ensure_ascii=False
            ),
            authorized_doc_ids=json.dumps(authorized_source_doc_ids),
            authorized_source_mapping=json.dumps(
                authorized_source_name_to_doc_ids,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        try:
            response = await self._llm_invoker.invoke(
                phase="contract_planning",
                purpose="atomic_contract_planning",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            decision = _parse_decision(response)
            _validate_planner_scope(
                decision,
                authorized_source_names=authorized_source_names,
                authorized_source_doc_ids=authorized_source_doc_ids,
                authorized_source_name_to_doc_ids=(authorized_source_name_to_doc_ids),
            )
            _validate_answer_free(decision, question=question)
        except TimeoutError:
            return _safe_ambiguity_result("planner_timeout")
        except BudgetExceededError:
            return _safe_ambiguity_result("planner_budget_rejected")
        except _UnauthorizedSourceExpansion:
            return _safe_ambiguity_result("unauthorized_source_expansion")
        except (TypeError, ValueError):
            return _safe_ambiguity_result("invalid_planner_output")
        return _AmbiguityResult(
            route=decision.selected_route,
            slots=[
                RequiredSlot(
                    slot_id=f"S{index}",
                    description=slot.description,
                    source_name_hints=slot.source_name_hints,
                    authorized_source_doc_ids=slot.authorized_source_doc_ids,
                    locator_hints=slot.locator_hints,
                    expected_answer_type=slot.expected_answer_type,
                    depends_on_slot_ids=slot.depends_on_slot_ids,
                    visual_policy=slot.visual_policy,
                )
                for index, slot in enumerate(decision.slots[:8], 1)
            ],
            route_reason=decision.route_reason,
            confidence=max(0.0, min(decision.confidence, 1.0)),
            decision_source="llm_planner",
            fallback_reason=None,
        )


def _decompose(
    *, question: str, source_names: list[str], source_doc_ids: list[str]
) -> tuple[list[RequiredSlot], list[str]]:
    matched_rules: list[str] = []
    if len(source_names) > 1:
        matched_rules.append("multiple_named_sources")

    clauses = _split_clauses(question)
    if len(clauses) > 1:
        matched_rules.append("numbered_subquestions")
    slots: list[RequiredSlot] = []
    for clause in clauses:
        parallel = _parallel_value_descriptions(clause)
        if parallel:
            matched_rules.append("parallel_values")
            descriptions = parallel
        else:
            clause_locators = _exact_locators(clause)
            descriptions = (
                [
                    f"Resolve the requirement at {locator}."
                    for locator in clause_locators
                ]
                if len(clause_locators) > 1
                else [_answer_free_description(clause)]
            )
        for description in descriptions:
            slots.append(
                _slot(
                    description=description,
                    answer_type=_answer_type(description),
                    source_names=_source_hints(description, source_names),
                    source_doc_ids=_source_ids_for_hints(
                        _source_hints(description, source_names),
                        source_names,
                        source_doc_ids,
                    ),
                    locators=_exact_locators(clause),
                )
            )
    return slots, list(dict.fromkeys(matched_rules))


def _slot(
    *,
    description: str,
    answer_type: str,
    source_names: list[str],
    source_doc_ids: list[str],
    locators: list[str] | None = None,
) -> RequiredSlot:
    return RequiredSlot(
        slot_id="pending",
        description=description,
        source_name_hints=source_names,
        authorized_source_doc_ids=source_doc_ids,
        locator_hints=locators or [],
        expected_answer_type=answer_type,
        visual_policy=(
            "preferred"
            if any(
                locator.casefold().startswith(("figure", "table"))
                for locator in locators or []
            )
            else "never"
        ),
    )


def _split_clauses(question: str) -> list[str]:
    numbered = re.split(
        r"(?:^|\s)(?:\d+[.)]|[-*•])\s*|;\s*|(?=\band explain Equation\b)",
        question,
        flags=re.IGNORECASE,
    )
    clauses = [value.strip(" ;") for value in numbered if value.strip(" ;")]
    if len(clauses) > 1 and clauses[0].casefold().startswith(("using ", "from ")):
        clauses = clauses[1:]
    return clauses or [question]


def _parallel_value_descriptions(clause: str) -> list[str]:
    match = re.search(
        r"\b([A-Za-z][A-Za-z0-9 .+-]*)\s+(?:and|,)\s+"
        r"([A-Za-z][A-Za-z0-9 .+-]*)\s+(Dice|value|score)s?\b",
        clause,
        re.IGNORECASE,
    )
    if not match:
        return []
    metric = match.group(3)
    return [
        f"Retrieve the {match.group(1).strip()} {metric}.",
        f"Retrieve the {match.group(2).strip()} {metric}.",
    ]


def _answer_free_description(clause: str) -> str:
    cleaned = clause.strip().rstrip("?.")
    return f"Resolve: {cleaned}."


def _answer_type(text: str) -> str:
    normalized = text.casefold()
    if any(term in normalized for term in ("equation", "formula")):
        return "equation"
    if any(term in normalized for term in ("meaning", "define", "what is |")):
        return "definition"
    if any(term in normalized for term in ("compare", "which")):
        return "comparison"
    if any(term in normalized for term in ("why", "reason", "explain")):
        return "explanation"
    if any(term in normalized for term in ("dice", "score", "value", "how many")):
        return "number"
    return "text"


def _exact_locators(text: str) -> list[str]:
    locators: list[str] = []
    for match in _EXACT_LOCATOR_PATTERN.finditer(text):
        locator = (
            f"{match.group(1).title().replace('Fig.', 'Figure')} {match.group(2)}"
        )
        if match.group(1).casefold() == "section":
            if canonical_structured_locator(locator) is None:
                continue
            if re.match(r"\s+[A-Za-z]", text[match.end() :]):
                continue
        locators.append(locator)
    return list(dict.fromkeys(locators))


def _source_hints(text: str, source_names: list[str]) -> list[str]:
    matched = [
        name
        for name in source_names
        if Path(name).stem.casefold() in text.casefold()
        or text.casefold() in Path(name).stem.casefold()
    ]
    return matched or source_names


def _source_ids_for_hints(
    hints: list[str], source_names: list[str], source_doc_ids: list[str]
) -> list[str]:
    mapping = dict(zip(source_names, source_doc_ids, strict=False))
    return [mapping[name] for name in hints if name in mapping] or source_doc_ids


def _deterministic_route(question: str) -> AgenticV9Route | None:
    normalized = question.casefold()
    entities = _extract_entities(question)
    locator_hints = _locator_hints(normalized)
    if _contains_any(
        normalized,
        ("lineage path", "graph path", "relationship path", "關係路徑", "譜系路徑"),
    ):
        return "graph_relational"
    if len(locator_hints) >= 2 and len(entities) >= 2:
        return "multi_document_exact"
    if _contains_any(normalized, ("from ", "至 ", "從", "源自", "追溯")) or (
        len(entities) >= 3
        and _contains_any(normalized, ("compare", "which", "比較", "判斷"))
    ):
        return "multi_hop"
    if locator_hints or _contains_any(
        normalized, ("calculate", "how many", "計算", "列出", "擷取")
    ):
        return "exact_structured"
    if _contains_any(
        normalized,
        ("compare", "versus", " vs ", "which performs", "比較", "是否優於"),
    ):
        return "bounded_compare"
    if _contains_any(normalized, ("what is", "what are", "find ", "什麼是", "何謂")):
        return "single_lookup"
    return None


def _extract_entities(question: str) -> list[str]:
    ignored = {
        "according",
        "and",
        "appendix",
        "compare",
        "figure",
        "find",
        "from",
        "please",
        "table",
        "the",
        "to",
        "what",
        "which",
    }
    return list(
        dict.fromkeys(
            value
            for value in _ENTITY_PATTERN.findall(question)
            if value.casefold() not in ignored
            and (
                any(character.isupper() for character in value)
                or "-" in value
                or any(character.isdigit() for character in value)
            )
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


def _candidate_routes(
    route: AgenticV9Route, matched_rules: list[str]
) -> list[AgenticV9Route]:
    candidates: list[AgenticV9Route] = [route]
    if "multiple_named_sources" in matched_rules and route != "exact_structured":
        candidates.append("exact_structured")
    return candidates


def _route_reason(route: AgenticV9Route, matched_rules: list[str]) -> str:
    suffix = f" Matched: {', '.join(matched_rules)}." if matched_rules else ""
    return f"Selected {route} from question-only deterministic analysis.{suffix}"


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


class _UnauthorizedSourceExpansion(ValueError):
    """Planner attempted to add a source outside the authorized intersection."""


def _parse_decision(response: Any) -> _PlannerDecision:
    content = response
    if isinstance(response, dict) and "content" in response:
        content = response["content"]
    elif hasattr(response, "content"):
        content = response.content
    if not isinstance(content, str):
        raise ValueError("contract planner response must contain JSON text")
    try:
        decision = _PlannerDecision.model_validate_json(content)
    except ValueError as error:
        raise ValueError("invalid contract planner response") from error
    if not 1 <= len(decision.slots) <= 8:
        raise ValueError("contract planner must return one to eight slots")
    return decision


def _validate_planner_scope(
    decision: _PlannerDecision,
    *,
    authorized_source_names: list[str],
    authorized_source_doc_ids: list[str],
    authorized_source_name_to_doc_ids: dict[str, list[str]],
) -> None:
    allowed_names = set(authorized_source_names)
    allowed_ids = set(authorized_source_doc_ids)
    for slot in decision.slots:
        if not set(slot.source_name_hints) <= allowed_names:
            raise _UnauthorizedSourceExpansion
        if not set(slot.authorized_source_doc_ids) <= allowed_ids:
            raise _UnauthorizedSourceExpansion
        paired_ids = {
            doc_id
            for name in slot.source_name_hints
            for doc_id in authorized_source_name_to_doc_ids.get(name, ())
        }
        if not set(slot.authorized_source_doc_ids) <= paired_ids:
            raise _UnauthorizedSourceExpansion


def _validate_answer_free(decision: _PlannerDecision, *, question: str) -> None:
    question_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", question))
    planner_text = [decision.route_reason]
    for index, slot in enumerate(decision.slots, 1):
        planner_text.extend([slot.description, *slot.locator_hints])
        valid_dependencies = {f"S{prior}" for prior in range(1, index)}
        if not set(slot.depends_on_slot_ids) <= valid_dependencies:
            raise ValueError("slot dependency must reference an earlier slot")
        for locator in slot.locator_hints:
            if not _valid_locator_hint(locator):
                raise ValueError("invalid planner locator hint")
    for text in planner_text:
        authored_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        if not authored_numbers <= question_numbers:
            raise ValueError("planner text contains an answer-like value")


def _valid_locator_hint(value: str) -> bool:
    """Admit only bounded source-location descriptions, never free-form values."""
    normalized = value.strip().casefold()
    if not normalized or len(normalized) > 120:
        return False
    if normalized.startswith("section"):
        return canonical_structured_locator(value) is not None
    return bool(
        re.match(
            r"^(figure|fig\.?|table|appendix|formula|equation|theorem|page|section)\b",
            normalized,
        )
        or any(
            term in normalized
            for term in ("source passage", "regional impurity equation", "matrix")
        )
    )


def _source_mapping(
    *,
    source_names: list[str],
    source_doc_ids: list[str],
    authoritative: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    if authoritative is not None:
        allowed_ids = set(source_doc_ids)
        allowed_names = set(source_names)
        return {
            name: list(dict.fromkeys(doc_ids))
            for name, doc_ids in authoritative.items()
            if name in allowed_names and doc_ids and set(doc_ids) <= allowed_ids
        }
    if len(source_names) == 1 and len(source_doc_ids) == 1:
        return {source_names[0]: [source_doc_ids[0]]}
    return {}


def _safe_ambiguity_result(reason: str) -> _AmbiguityResult:
    return _AmbiguityResult(
        route="single_lookup",
        slots=None,
        route_reason="Ambiguity planning failed; use bounded safe retrieval.",
        confidence=0.0,
        decision_source="safe_fallback",
        fallback_reason=reason,
    )


__all__ = ["QuestionContractPlanner"]
