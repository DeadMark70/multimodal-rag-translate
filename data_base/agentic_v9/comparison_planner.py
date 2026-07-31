"""Fail-soft semantic planning for explicit Agentic v9 comparisons."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Sequence
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from data_base.agentic_v9.schemas import (
    ComparisonPlan,
    ComparisonPlannerDiagnosticStage,
    ComparisonPlannerFallbackReason,
    ComparisonPlannerOutcome,
    ComparisonPlannerValidationIssue,
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
_NUMBER = re.compile(
    r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])"
)
_VS_MARKER = re.compile(
    r"(?<!\w)vs\.?(?=$|[\s,;:!?。！？、])",
    re.IGNORECASE,
)
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


class _PlannerSubjectPayload(BaseModel):
    """Untrusted provider transport before deterministic subject promotion."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        min_length=1,
        max_length=160,
        description="Exact independent entity name copied from the question.",
    )
    query: str = Field(
        min_length=1,
        max_length=512,
        description="Retrieval query that explicitly names this entity.",
    )


class _PlannerPayload(BaseModel):
    """Provider JSON before it is promoted into a comparison plan."""

    model_config = ConfigDict(extra="ignore")

    is_comparison: bool = Field(
        description="Whether the question compares two or more independent entities."
    )
    subjects: list[_PlannerSubjectPayload] = Field(default_factory=list, max_length=4)
    dimensions: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Comparison dimensions stated by the question, as short strings.",
    )
    qualification: str | None = Field(
        default=None,
        max_length=512,
        description="Optional scope or qualification stated by the question.",
    )


def comparison_planner_response_schema() -> dict[str, Any]:
    """Return the compact provider transport schema for native JSON binding."""
    return _PlannerPayload.model_json_schema()


def is_suspected_comparison(question: str) -> bool:
    """Return whether a bounded semantic comparison check is warranted."""
    normalized = f" {question.strip().casefold()} "
    return bool(_VS_MARKER.search(question)) or any(
        marker in normalized for marker in _RELATIVE_MARKERS
    )


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
        try:
            messages = _planner_messages(
                question=question,
                authorized_source_names=authorized_source_names,
            )
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
            return _fallback(
                "invalid_response",
                started_at,
                stage="response_decode",
            )
        try:
            payload = _PlannerPayload.model_validate(decoded)
        except ValidationError as error:
            return _fallback(
                "schema_violation",
                started_at,
                stage="transport_schema",
                validation_issues=_validation_issues(error),
            )

        if not payload.is_comparison:
            return _fallback("not_comparison", started_at)
        subjects = _validated_subjects(question, payload.subjects)
        if len(subjects) < 2 or len(subjects) != len(payload.subjects):
            return _fallback(
                "invalid_subjects",
                started_at,
                stage="subject_validation",
            )
        try:
            plan = ComparisonPlan(
                subjects=subjects,
                dimensions=payload.dimensions,
                qualification=payload.qualification,
            )
        except ValidationError as error:
            return _fallback(
                "schema_violation",
                started_at,
                stage="trusted_plan_validation",
                validation_issues=_validation_issues(error),
            )
        try:
            _reject_invented_numbers(question, plan)
        except ValueError:
            return _fallback(
                "schema_violation",
                started_at,
                stage="numeric_guard",
            )
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


def _normalized_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _stable_subject_id(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).strip().casefold()
    ascii_slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    if ascii_slug:
        return ascii_slug[:80]
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"subject-{digest}"


def _contains_explicit_span(question: str, span: str) -> bool:
    normalized_question = unicodedata.normalize("NFKC", question).casefold()
    normalized_span = unicodedata.normalize("NFKC", span).strip().casefold()
    if not normalized_span:
        return False
    for match in re.finditer(re.escape(normalized_span), normalized_question):
        before = normalized_question[match.start() - 1] if match.start() else ""
        after = (
            normalized_question[match.end()]
            if match.end() < len(normalized_question)
            else ""
        )
        left_boundary_required = (
            normalized_span[0].isascii() and normalized_span[0].isalnum()
        )
        right_boundary_required = (
            normalized_span[-1].isascii() and normalized_span[-1].isalnum()
        )
        left_ok = not left_boundary_required or not (
            before.isascii() and before.isalnum()
        )
        right_ok = not right_boundary_required or not (
            after.isascii() and after.isalnum()
        )
        if left_ok and right_ok:
            return True
    return False


def _validated_subjects(
    question: str,
    candidates: Sequence[_PlannerSubjectPayload],
) -> list[ComparisonSubject]:
    accepted: list[ComparisonSubject] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not _contains_explicit_span(question, candidate.name):
            continue
        identity = _normalized_identity(candidate.name)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        try:
            accepted.append(
                ComparisonSubject(
                    subject_id=_stable_subject_id(candidate.name),
                    display_name=candidate.name,
                    aliases=[],
                    retrieval_query=candidate.query,
                )
            )
        except (ValidationError, ValueError):
            continue
    return accepted


def _fallback(
    reason: ComparisonPlannerFallbackReason,
    started_at: float,
    *,
    stage: ComparisonPlannerDiagnosticStage | None = None,
    validation_issues: Sequence[ComparisonPlannerValidationIssue] = (),
) -> ComparisonPlannerOutcome:
    return ComparisonPlannerOutcome(
        status="fallback",
        fallback_reason=reason,
        fallback_stage=stage,
        validation_issues=list(validation_issues),
        latency_ms=_elapsed_ms(started_at),
    )


def _validation_issues(
    error: ValidationError,
) -> list[ComparisonPlannerValidationIssue]:
    issues: set[tuple[str, str]] = set()
    for row in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        path = ".".join(_safe_diagnostic_segment(part) for part in row["loc"])
        issue_type = _safe_diagnostic_segment(row.get("type") or "unknown")
        issues.add((path[:160] or "root", issue_type[:80] or "unknown"))
    return [
        ComparisonPlannerValidationIssue(path=path, type=issue_type)
        for path, issue_type in sorted(issues)[:8]
    ]


def _safe_diagnostic_segment(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_") or "unknown"


def _elapsed_ms(started_at: float) -> float:
    return max((perf_counter() - started_at) * 1000, 0)


__all__ = [
    "ComparisonPlanner",
    "apply_comparison_overlay",
    "comparison_planner_response_schema",
    "is_suspected_comparison",
]
