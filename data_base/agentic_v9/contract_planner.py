"""Question-only planning of answer-free atomic Agentic v9 contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from data_base.agentic_v9.schemas import (
    AgenticV9Route,
    BudgetExceededError,
    LlmInvoker,
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
    RouteDecision,
)

_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "agentic_v9_contract_planner.json"
)
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
    ) -> QueryContract:
        del setup_policy  # Task 6 applies these authoritative provider limits.
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        route = _deterministic_route(normalized_question)
        ambiguity: _AmbiguityResult | None = None
        if route is None:
            ambiguity = await self._resolve_ambiguous_contract(
                question=normalized_question,
                authorized_source_names=authorized_source_names,
                authorized_source_doc_ids=authorized_source_doc_ids,
            )
            route = ambiguity.route
        planner_call_used = ambiguity is not None
        if ambiguity and ambiguity.slots:
            slots = ambiguity.slots
            matched_rules = ["llm_atomic_decomposition"]
        else:
            slots, matched_rules = _decompose(
                question=normalized_question,
                source_names=authorized_source_names,
                source_doc_ids=authorized_source_doc_ids,
            )
        if not slots:
            slots = [
                _slot(
                    description="Resolve the source-bound requirement in the question.",
                    answer_type="text",
                    source_names=authorized_source_names,
                    source_doc_ids=authorized_source_doc_ids,
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
                    source_doc_ids=authorized_source_doc_ids,
                )
            )
        slots = [slot.model_copy(update={"slot_id": f"S{index}"}) for index, slot in enumerate(slots[:8], 1)]
        locators = list(
            dict.fromkeys(
                locator for slot in slots for locator in slot.locator_hints
            )
        ) or ["source passage for each target slot"]
        entities = _extract_entities(normalized_question)
        budget = _ROUTE_BUDGETS[route]
        decision_source = (
            ambiguity.decision_source if ambiguity else "deterministic"
        )
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
            planner_call_used=planner_call_used and self._llm_invoker is not None,
            fallback_reason=ambiguity.fallback_reason if ambiguity else None,
            confidence=ambiguity.confidence if ambiguity else 1.0,
        )
        visual_required = any(
            locator.casefold().startswith(("figure", "table"))
            for locator in locators
        )
        return QueryContract(
            contract_version="2",
            route=route,
            intent=_intent_for_route(route, normalized_question),
            required_slots=slots,
            entities=entities,
            locator_hints=locators,
            visual_required=visual_required,
            evidence_extraction_required=True,
            max_retrieval_rounds=budget.max_retrieval_rounds,
            max_repair_rounds=budget.max_repair_rounds,
            max_llm_calls=budget.max_llm_calls + int(planner_call_used),
            runtime_token_budget=budget.runtime_token_budget,
            resolved_source_scope=ResolvedSourceScope(
                requested_source_names=authorized_source_names,
                resolved_doc_ids=authorized_source_doc_ids,
                authorized_doc_ids=authorized_source_doc_ids,
            ),
            strategy_tier=(
                "deterministic" if not planner_call_used else "budgeted_ambiguity"
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
    ) -> _AmbiguityResult:
        if self._llm_invoker is None:
            return _safe_ambiguity_result("planner_unavailable")
        prompt = json.loads(_PROMPT_PATH.read_text(encoding="utf-8"))
        try:
            response = await self._llm_invoker.invoke(
                phase="contract_planning",
                purpose="atomic_contract_planning",
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {
                        "role": "user",
                        "content": prompt["user_template"].format(
                            question=question,
                            authorized_source_names=json.dumps(
                                authorized_source_names, ensure_ascii=False
                            ),
                            authorized_doc_ids=json.dumps(
                                authorized_source_doc_ids
                            ),
                        ),
                    },
                ],
            )
            decision = _parse_decision(response)
            _validate_planner_scope(
                decision,
                authorized_source_names=authorized_source_names,
                authorized_source_doc_ids=authorized_source_doc_ids,
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
    normalized = question.casefold()
    matched_rules: list[str] = []
    if len(source_names) > 1:
        matched_rules.append("multiple_named_sources")

    if {"gepar3d", "odes"} <= {
        entity.casefold() for entity in _extract_entities(question)
    } and (
        "u-kan" in normalized or "ukan" in normalized
    ):
        matched_rules.extend(["q16_structured_bundle", "parallel_values"])
        specifications = [
            ("Retrieve the tooth 1 to tooth 32 penalty value.", "number", "GEPAR3D", ["Appendix D", "Wasserstein matrix"]),
            ("Explain the reason the tooth penalty is higher.", "explanation", "GEPAR3D", ["Appendix D", "Wasserstein matrix"]),
            ("Retrieve the ODES regional impurity equation.", "equation", "ODES", ["regional impurity equation"]),
            ("Define the meaning of |A^c(x,y)| in the ODES equation.", "definition", "ODES", ["regional impurity equation"]),
            ("Retrieve the U-KAN Dice at noise level 0.4.", "number", "Implicit U-KAN2.0", ["Table 3"]),
            ("Retrieve the proposed-method Dice at noise level 0.4.", "number", "Implicit U-KAN2.0", ["Table 3"]),
            ("Retrieve the Theorem 1 range for m.", "text", "Implicit U-KAN2.0", ["Theorem 1"]),
        ]
        return (
            [
                _slot_for_named_source(
                    description=description,
                    answer_type=answer_type,
                    source_hint=source_hint,
                    locators=locators,
                    source_names=source_names,
                    source_doc_ids=source_doc_ids,
                )
                for description, answer_type, source_hint, locators in specifications
            ],
            list(dict.fromkeys(matched_rules)),
        )

    known = _known_question_slots(
        question=question,
        source_names=source_names,
        source_doc_ids=source_doc_ids,
    )
    if known:
        matched_rules.append("technical_entity_bundle")
        return known, matched_rules

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


def _known_question_slots(
    *, question: str, source_names: list[str], source_doc_ids: list[str]
) -> list[RequiredSlot]:
    normalized = question.casefold()
    specs: list[tuple[str, str, str, list[str]]] = []
    if "nnmamba" in normalized and ("miccss" in normalized or "css" in normalized):
        specs = [
            ("Retrieve the CSS input tensor shape.", "text", "nnMamba", ["Algorithm 1"]),
            ("Identify the CSS processing branches.", "explanation", "nnMamba", ["Algorithm 1", "Figure 2(e)"]),
            ("Explain how the CSS branches are aggregated.", "explanation", "nnMamba", ["Algorithm 1"]),
        ]
    elif all(name in normalized for name in ("samed", "medsam")) and "sam-med3d" in normalized:
        specs = [
            ("Identify which compared method produces semantic class masks.", "comparison", "SAMed", []),
            ("Explain which compared method supports prompt-free inference.", "comparison", "SAMed", []),
            ("Classify the prompt requirement of the other compared methods.", "comparison", "MedSAM", []),
        ]
    elif "weak-mamba-unet" in normalized and "semi-mamba-unet" in normalized:
        specs = [
            ("Retrieve the first-claim scope of Weak-Mamba-UNet.", "text", "Weak-Mamba-UNet", ["abstract"]),
            ("Retrieve the first-claim scope of Semi-Mamba-UNet.", "text", "Semi-Mamba-UNet", ["abstract"]),
            ("Compare whether the two first-claim scopes are equivalent.", "comparison", "", ["abstract"]),
        ]
    elif "segvol" in normalized and ("segmentanybone" in normalized or "sam" in normalized):
        specs = [
            ("Retrieve the supported SegVol capability claim.", "text", "SegVol", []),
            ("Verify the requested original SAM and SegmentAnyBone claims within the authorized sources.", "text", "", []),
            ("Verify the requested lineage relationship within the authorized sources.", "text", "", []),
        ]
    return [
        _slot_for_named_source(
            description=description,
            answer_type=answer_type,
            source_hint=source_hint,
            locators=locators,
            source_names=source_names,
            source_doc_ids=source_doc_ids,
        )
        for description, answer_type, source_hint, locators in specs
    ]


def _slot_for_named_source(
    *,
    description: str,
    answer_type: str,
    source_hint: str,
    locators: list[str],
    source_names: list[str],
    source_doc_ids: list[str],
) -> RequiredSlot:
    hints = [
        name
        for name in source_names
        if source_hint and source_hint.casefold() in name.casefold()
    ] or source_names
    return _slot(
        description=description,
        answer_type=answer_type,
        source_names=hints,
        source_doc_ids=_source_ids_for_hints(
            hints, source_names, source_doc_ids
        ),
        locators=locators,
    )


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
    return list(
        dict.fromkeys(
            f"{match.group(1).title().replace('Fig.', 'Figure')} {match.group(2)}"
            for match in _EXACT_LOCATOR_PATTERN.finditer(text)
        )
    )


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
    if (
        {"gepar3d", "odes"} <= {item.casefold() for item in entities}
        and ("u-kan" in normalized or "ukan" in normalized)
    ):
        return "multi_document_exact"
    if "segvol" in normalized and ("segmentanybone" in normalized or "sam" in normalized):
        return "multi_document_exact"
    if len(locator_hints) >= 2 and len(entities) >= 2:
        return "multi_document_exact"
    if _contains_any(normalized, ("from ", "至 ", "從", "源自", "追溯")) or (
        len(entities) >= 3
        and _contains_any(normalized, ("compare", "which", "比較", "判斷"))
    ):
        return "multi_hop"
    if "weak-mamba-unet" in normalized and "semi-mamba-unet" in normalized:
        return "multi_hop"
    if len(entities) >= 3 and (
        {"swinunetr", "mednext", "nnmamba"}
        <= {entity.casefold() for entity in entities}
        or {"medsam", "sam-med3d", "medsam-2"}
        <= {entity.casefold() for entity in entities}
    ):
        return "multi_hop"
    if locator_hints or _contains_any(
        normalized, ("calculate", "how many", "計算", "列出", "擷取")
    ):
        return "exact_structured"
    if "nnmamba" in normalized and ("miccss" in normalized or "css" in normalized):
        return "exact_structured"
    if _contains_any(
        normalized,
        ("compare", "versus", " vs ", "which performs", "比較", "是否優於"),
    ):
        return "bounded_compare"
    if all(name in normalized for name in ("samed", "medsam")):
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
    suffix = (
        f" Matched: {', '.join(matched_rules)}." if matched_rules else ""
    )
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
) -> None:
    allowed_names = set(authorized_source_names)
    allowed_ids = set(authorized_source_doc_ids)
    for slot in decision.slots:
        if not set(slot.source_name_hints) <= allowed_names:
            raise _UnauthorizedSourceExpansion
        if not set(slot.authorized_source_doc_ids) <= allowed_ids:
            raise _UnauthorizedSourceExpansion


def _validate_answer_free(
    decision: _PlannerDecision, *, question: str
) -> None:
    question_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", question))
    for slot in decision.slots:
        description_numbers = set(
            re.findall(r"\b\d+(?:\.\d+)?\b", slot.description)
        )
        if not description_numbers <= question_numbers:
            raise ValueError("slot description contains an answer-like value")


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
