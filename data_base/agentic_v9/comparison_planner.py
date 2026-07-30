"""Fail-soft semantic planning for explicit Agentic v9 comparisons."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from data_base.agentic_v9.schemas import (
    ComparisonPlan,
    ComparisonPlannerFallbackReason,
    ComparisonPlannerOutcome,
    ComparisonSubject,
    LlmInvoker,
    QueryContract,
    RequiredSlot,
)


_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "agentic_v9_comparison_planner.json"
)
_FENCED_JSON = re.compile(
    r"^\s*```(?:json)?\s*(?P<payload>\{.*\})\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_RELATIVE_MARKERS = (
    " compare ",
    " compared ",
    " versus ",
    " vs ",
    "which is",
    "which performs",
    "better",
    "worse",
    "higher",
    "lower",
    "highest",
    "lowest",
    "most efficient",
    "比較",
    "相比",
    "相較",
    "哪個",
    "哪一個",
    "誰有",
    "誰是",
    "更好",
    "更高",
    "更低",
    "最高",
    "最低",
    "是否成立",
    "能否支持",
    "應選",
    "互斥",
)


class _PlannerPayload(BaseModel):
    """Provider JSON before it is promoted into a comparison plan."""

    model_config = ConfigDict(extra="forbid")

    is_comparison: bool
    subjects: list[ComparisonSubject] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list, max_length=12)
    qualification: str | None = Field(default=None, max_length=512)


def is_suspected_comparison(question: str) -> bool:
    """Return whether a bounded semantic comparison check is warranted."""
    normalized = f" {question.strip().casefold()} "
    return any(marker in normalized for marker in _RELATIVE_MARKERS)


class ComparisonPlanner:
    """Invoke one answer-free comparison planner and classify safe fallback."""

    def __init__(self, *, llm_invoker: LlmInvoker) -> None:
        self._llm_invoker = llm_invoker

    async def plan(
        self,
        *,
        question: str,
        authorized_source_names: Sequence[str],
        timeout_seconds: float,
    ) -> ComparisonPlannerOutcome:
        """Return a validated plan or a non-throwing fallback result."""
        started_at = perf_counter()
        if timeout_seconds <= 0:
            return _fallback("timeout", started_at)
        messages = _planner_messages(
            question=question,
            authorized_source_names=authorized_source_names,
        )
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await self._llm_invoker.invoke(
                    phase="comparison_plan",
                    purpose="agentic_v9_comparison_plan",
                    messages=messages,
                )
        except TimeoutError:
            return _fallback("timeout", started_at)
        except Exception:
            return _fallback("provider_error", started_at)

        try:
            content = _response_text(response)
            decoded = json.loads(_json_text(content))
        except (json.JSONDecodeError, TypeError):
            return _fallback("invalid_response", started_at)
        try:
            payload = _PlannerPayload.model_validate(decoded)
        except ValidationError:
            return _fallback("schema_violation", started_at)

        if not payload.is_comparison:
            return _fallback("not_comparison", started_at)
        try:
            plan = ComparisonPlan(
                subjects=payload.subjects,
                dimensions=payload.dimensions,
                qualification=payload.qualification,
            )
            _reject_invented_numbers(question, plan)
        except (ValidationError, ValueError):
            return _fallback("schema_violation", started_at)
        return ComparisonPlannerOutcome(
            status="planned",
            plan=plan,
            latency_ms=_elapsed_ms(started_at),
        )


def apply_comparison_overlay(
    contract: QueryContract,
    plan: ComparisonPlan,
) -> QueryContract:
    """Add subject slots without changing route or source authority."""
    slots = [
        RequiredSlot(
            slot_id=f"comparison-subject:{subject.subject_id}",
            description=_subject_slot_description(subject, plan),
            entity_ids=[subject.display_name, *subject.aliases],
            expected_answer_type="comparison",
        )
        for subject in plan.subjects
    ]
    return contract.model_copy(
        update={
            "required_slots": slots,
            "comparison_plan": plan,
            "max_repair_rounds": max(contract.max_repair_rounds, 1),
        }
    )


def _subject_slot_description(
    subject: ComparisonSubject,
    plan: ComparisonPlan,
) -> str:
    dimensions = ", ".join(plan.dimensions) or "requested comparison dimensions"
    return f"Find evidence for {subject.display_name} about {dimensions}."


def _planner_messages(
    *,
    question: str,
    authorized_source_names: Sequence[str],
) -> list[dict[str, Any]]:
    prompt = json.loads(_PROMPT_PATH.read_text(encoding="utf-8"))
    safe_scope = ", ".join(
        name.strip() for name in authorized_source_names if name.strip()
    )
    return [
        {"role": "system", "content": prompt["system"]},
        {
            "role": "user",
            "content": prompt["user_template"].format(
                question=question,
                authorized_source_names=safe_scope or "not provided",
            ),
        },
    ]


def _response_text(response: Any) -> str:
    content = response
    if isinstance(response, dict) and "content" in response:
        content = response["content"]
    elif hasattr(response, "content"):
        content = response.content
    if not isinstance(content, str):
        raise TypeError("comparison planner response must contain text")
    return content


def _json_text(content: str) -> str:
    stripped = content.strip()
    fenced = _FENCED_JSON.fullmatch(stripped)
    return fenced.group("payload") if fenced else stripped


def _reject_invented_numbers(question: str, plan: ComparisonPlan) -> None:
    allowed = set(_NUMBER.findall(question))
    supplied = {
        number
        for subject in plan.subjects
        for number in _NUMBER.findall(subject.retrieval_query)
    }
    supplied.update(
        number for dimension in plan.dimensions for number in _NUMBER.findall(dimension)
    )
    if supplied.difference(allowed):
        raise ValueError("comparison planner introduced numeric result values")


def _fallback(
    reason: ComparisonPlannerFallbackReason,
    started_at: float,
) -> ComparisonPlannerOutcome:
    return ComparisonPlannerOutcome(
        status="fallback",
        fallback_reason=reason,
        latency_ms=_elapsed_ms(started_at),
    )


def _elapsed_ms(started_at: float) -> float:
    return max((perf_counter() - started_at) * 1000, 0)


__all__ = [
    "ComparisonPlanner",
    "apply_comparison_overlay",
    "is_suspected_comparison",
]
