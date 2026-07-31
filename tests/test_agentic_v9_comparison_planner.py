"""Comparison-specialization contracts for Agentic v9."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from core.prompt_loader import PromptConfigError
import data_base.agentic_v9.comparison_planner as comparison_planner_module
from data_base.agentic_v9.comparison_planner import (
    ComparisonPlanner,
    apply_comparison_overlay,
    is_suspected_comparison,
)
from data_base.agentic_v9.schemas import (
    ComparisonPlan,
    ComparisonPlannerOutcome,
    ComparisonPlannerValidationIssue,
    ComparisonSubject,
    QueryContract,
    ResolvedSourceScope,
)


Q4 = (
    "結合 Params 與 FLOPs，比較 nnMamba 與 EfficientMedNeXt-L "
    "在 3D 醫療分割的計算效率。"
)


def test_comparison_planner_response_schema_requires_string_dimensions() -> None:
    assert hasattr(comparison_planner_module, "comparison_planner_response_schema")
    schema = comparison_planner_module.comparison_planner_response_schema()

    assert schema["properties"]["dimensions"]["items"]["type"] == "string"
    subject_ref = schema["properties"]["subjects"]["items"]["$ref"]
    subject_key = subject_ref.rsplit("/", 1)[-1]
    subject_schema = schema["$defs"][subject_key]
    assert set(subject_schema["properties"]) == {"name", "query"}


class _Invoker:
    def __init__(
        self,
        response: object = None,
        *,
        error: Exception | None = None,
        delay_s: float = 0,
    ) -> None:
        self.response = response
        self.error = error
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []

    async def invoke(
        self, *, phase: str, purpose: str, messages: list[dict[str, Any]]
    ) -> object:
        self.calls.append(
            {"phase": phase, "purpose": purpose, "messages": messages}
        )
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        return self.response


def _subject(
    subject_id: str,
    display_name: str,
    *,
    aliases: list[str] | None = None,
    query: str | None = None,
) -> ComparisonSubject:
    return ComparisonSubject(
        subject_id=subject_id,
        display_name=display_name,
        aliases=aliases or [],
        retrieval_query=query or f"{display_name} parameters FLOPs",
    )


def _planner_subject(
    name: str,
    *,
    retrieval_query: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "query": retrieval_query or name,
    }


def _payload(
    *,
    subjects: list[dict[str, object]] | None = None,
    dimensions: list[object] | None = None,
) -> str:
    return json.dumps(
        {
            "is_comparison": True,
            "subjects": subjects
            or [
                _planner_subject(
                    "nnMamba",
                    retrieval_query="nnMamba parameters FLOPs",
                ),
                _planner_subject(
                    "EfficientMedNeXt-L",
                    retrieval_query="EfficientMedNeXt-L parameters FLOPs",
                ),
            ],
            "dimensions": dimensions
            or ["parameters", "FLOPs", "computational efficiency"],
            "qualification": "cross-paper relative comparison",
        }
    )


@pytest.mark.asyncio
async def test_planner_rejects_dimension_objects_at_transport_boundary() -> None:
    response = _payload(
        dimensions=[
            {"name": "parameters"},
            {"dimension": "FLOPs", "description": "reported compute"},
            {"label": "computational efficiency"},
        ]
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "schema_violation"
    assert outcome.fallback_stage == "transport_schema"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dimension",
    [
        {"name": "parameters", "label": "FLOPs"},
        {"description": "parameters"},
        123,
        ["parameters"],
    ],
)
async def test_planner_rejects_ambiguous_or_unsupported_dimension_objects(
    dimension: object,
) -> None:
    outcome = await ComparisonPlanner(
        llm_invoker=_Invoker(_payload(dimensions=[dimension]))
    ).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "schema_violation"
    assert outcome.fallback_stage == "transport_schema"


@pytest.mark.parametrize(
    "question",
    [
        Q4,
        "SwinUNETR 和 MedNeXt 哪個表現更好？",
        "Model A、Model B、Model C 的 latency 應如何比較？",
        "nnMamba versus EfficientMedNeXt-L: which is more efficient?",
        "Model A vs. Model B",
        "Model A vs。Model B",
        "這項『最高效率』主張是否成立？",
    ],
)
def test_suspected_comparison_markers_cover_judgment_questions(
    question: str,
) -> None:
    assert is_suspected_comparison(question)


@pytest.mark.parametrize(
    "question",
    [
        "請找出 nnMamba 的 Params 與 FLOPs。",
        "Summarize SwinUNETR architecture.",
        "列出 Dice、latency 與 memory 指標。",
    ],
)
def test_metrics_without_relative_judgment_do_not_trigger_planner(
    question: str,
) -> None:
    assert not is_suspected_comparison(question)


def test_comparison_subject_normalizes_aliases_and_requires_subject_query() -> None:
    subject = _subject(
        " NN-MAMBA ",
        "nnMamba",
        aliases=["nnMamba", " nnMamba ", "Mamba model"],
    )

    assert subject.subject_id == "nn-mamba"
    assert subject.aliases == ["Mamba model"]

    with pytest.raises(ValidationError):
        _subject("nnmamba", "nnMamba", query="parameters FLOPs")


def test_comparison_plan_requires_two_to_four_unique_subjects() -> None:
    with pytest.raises(ValidationError):
        ComparisonPlan(subjects=[_subject("a", "Model A")])

    with pytest.raises(ValidationError):
        ComparisonPlan(
            subjects=[
                _subject("same", "Model A"),
                _subject("same", "Model B"),
            ]
        )

    with pytest.raises(ValidationError):
        ComparisonPlan(
            subjects=[
                _subject("a", "Model A"),
                _subject("b", "Model B"),
                _subject("c", "Model C"),
                _subject("d", "Model D"),
                _subject("e", "Model E"),
            ]
        )


def test_planned_outcome_rejects_fallback_diagnostics() -> None:
    with pytest.raises(ValidationError):
        ComparisonPlannerOutcome(
            status="planned",
            plan=ComparisonPlan(
                subjects=[_subject("a", "Model A"), _subject("b", "Model B")]
            ),
            fallback_stage="transport_schema",
            validation_issues=[
                ComparisonPlannerValidationIssue(
                    path="subjects.0.subject_role",
                    type="missing",
                )
            ],
            latency_ms=1,
        )


def test_comparison_models_forbid_unknown_and_source_fields() -> None:
    with pytest.raises(ValidationError):
        ComparisonSubject.model_validate(
            {
                "subject_id": "nnmamba",
                "display_name": "nnMamba",
                "retrieval_query": "nnMamba efficiency",
                "winner": True,
            }
        )

    with pytest.raises(ValidationError):
        _subject(
            "nnmamba",
            "nnMamba",
            aliases=["2402.03526v2nnMamba.pdf"],
        )


@pytest.mark.parametrize(
    "subject",
    [
        {
            "subject_id": "blank-name",
            "display_name": "   ",
            "aliases": [],
            "retrieval_query": "blank-name evidence",
        },
        {
            "subject_id": "unsafe/id:part",
            "display_name": "Unsafe",
            "aliases": [],
            "retrieval_query": "Unsafe evidence",
        },
        {
            "subject_id": "oversized-alias",
            "display_name": "Model",
            "aliases": ["x" * 161],
            "retrieval_query": "Model evidence",
        },
        {
            "subject_id": "embedded-file",
            "display_name": "Model",
            "aliases": [],
            "retrieval_query": "Model evidence from secret-paper.pdf section",
        },
    ],
)
def test_comparison_subject_rejects_adversarial_identity_text(
    subject: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ComparisonSubject.model_validate(subject)


def test_legacy_serialization_omits_absent_comparison_plan() -> None:
    contract = QueryContract(route="single_lookup", intent="legacy")

    assert "comparison_plan" not in contract.model_dump(mode="json")


@pytest.mark.asyncio
async def test_valid_planner_response_identifies_subjects_not_dimensions() -> None:
    invoker = _Invoker(_payload())
    outcome = await ComparisonPlanner(llm_invoker=invoker).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "planned"
    assert outcome.fallback_reason is None
    assert outcome.plan is not None
    assert [item.display_name for item in outcome.plan.subjects] == [
        "nnMamba",
        "EfficientMedNeXt-L",
    ]
    assert "Params" not in [item.display_name for item in outcome.plan.subjects]
    assert invoker.calls == [
        {
            "phase": "comparison_plan",
            "purpose": "agentic_v9_comparison_plan",
            "messages": invoker.calls[0]["messages"],
        }
    ]


@pytest.mark.asyncio
async def test_transport_variation_promotes_question_anchored_model_subjects() -> None:
    response = json.dumps(
        {
            "is_comparison": True,
            "subjects": [
                {
                    "name": "nnMamba",
                    "query": "nnMamba Params FLOPs",
                    "rationale": "named model in the question",
                },
                {
                    "name": "EfficientMedNeXt-L",
                    "query": "EfficientMedNeXt-L Params FLOPs",
                    "rationale": "named architecture in the question",
                },
            ],
            "dimensions": ["Params", "FLOPs"],
            "qualification": "cross-paper relative comparison",
            "explanation": "retrieval planning only",
        }
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "planned"
    assert outcome.fallback_reason is None
    assert outcome.plan is not None
    assert [subject.display_name for subject in outcome.plan.subjects] == [
        "nnMamba",
        "EfficientMedNeXt-L",
    ]
    assert [subject.subject_id for subject in outcome.plan.subjects] == [
        "nnmamba",
        "efficientmednext-l",
    ]


@pytest.mark.asyncio
async def test_transport_schema_failure_exposes_only_safe_validation_issues() -> None:
    secret = "secret-paper.pdf"
    response = json.dumps(
        {
            "is_comparison": True,
            "subjects": [
                {
                    "name": secret,
                },
                _planner_subject("EfficientMedNeXt-L"),
            ],
            "dimensions": ["Params", "FLOPs"],
        }
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "schema_violation"
    assert outcome.fallback_stage == "transport_schema"
    assert [issue.model_dump(mode="json") for issue in outcome.validation_issues] == [
        {"path": "subjects.0.query", "type": "missing"}
    ]
    serialized = outcome.model_dump_json()
    assert secret not in serialized
    assert "Field required" not in serialized
    assert "errors.pydantic.dev" not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected_stage"),
    [
        ("not-json", "response_decode"),
        (_payload(subjects=[_planner_subject("nnMamba")]), "subject_validation"),
        (_payload(dimensions=["x" * 161]), "trusted_plan_validation"),
        (
            _payload(
                subjects=[
                    _planner_subject(
                        "nnMamba",
                        retrieval_query="nnMamba Params 999",
                    ),
                    _planner_subject(
                        "EfficientMedNeXt-L",
                    ),
                ]
            ),
            "numeric_guard",
        ),
    ],
)
async def test_planner_fallback_identifies_the_failing_boundary(
    response: str,
    expected_stage: str,
) -> None:
    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_stage == expected_stage


@pytest.mark.asyncio
async def test_provider_subject_id_cannot_control_trusted_subject_id() -> None:
    response = _payload(
        subjects=[
            {"name": "nnMamba", "query": "nnMamba", "subject_id": "secret-paper.pdf"},
            _planner_subject("EfficientMedNeXt-L"),
        ]
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "planned"
    assert outcome.plan is not None
    assert [subject.subject_id for subject in outcome.plan.subjects] == [
        "nnmamba",
        "efficientmednext-l",
    ]


@pytest.mark.asyncio
async def test_one_entity_claim_arbitration_is_not_subject_comparison() -> None:
    question = (
        "Does MedSAM-2 support single-prompt segmentation, and how does "
        "initial bounding box prompt quality affect it?"
    )
    response = json.dumps(
        {
            "is_comparison": False,
            "subjects": [],
            "dimensions": [],
        }
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question=question,
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "not_comparison"
    assert outcome.plan is None


@pytest.mark.asyncio
async def test_unanchored_entity_subject_is_rejected() -> None:
    response = _payload(
        subjects=[
            _planner_subject("nnMamba"),
            _planner_subject("InventedModel"),
        ]
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question="Compare nnMamba and EfficientMedNeXt-L.",
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "invalid_subjects"


@pytest.mark.asyncio
async def test_any_invalid_candidate_rejects_the_entire_subject_plan() -> None:
    question = (
        "Compare nnMamba and EfficientMedNeXt-L for computational efficiency "
        "under a low-memory condition."
    )
    response = _payload(
        subjects=[
            _planner_subject("nnMamba"),
            _planner_subject("EfficientMedNeXt-L"),
            _planner_subject("InventedCondition"),
        ]
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question=question,
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "invalid_subjects"
    assert outcome.plan is None


@pytest.mark.asyncio
async def test_valid_three_entity_lineage_comparison_remains_planned() -> None:
    question = "Compare the technical lineage of SAM, SegmentAnyBone, and SegVol."
    response = _payload(
        subjects=[
            _planner_subject("SAM"),
            _planner_subject("SegmentAnyBone"),
            _planner_subject("SegVol"),
        ],
        dimensions=["technical lineage"],
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question=question,
        timeout_seconds=1,
    )

    assert outcome.status == "planned"
    assert outcome.plan is not None
    assert [subject.subject_id for subject in outcome.plan.subjects] == [
        "sam",
        "segmentanybone",
        "segvol",
    ]


@pytest.mark.asyncio
async def test_planner_accepts_one_fenced_json_object() -> None:
    outcome = await ComparisonPlanner(
        llm_invoker=_Invoker(f"```json\n{_payload()}\n```")
    ).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "planned"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("not json", "invalid_response"),
        (
            json.dumps(
                {
                    "is_comparison": True,
                    "subjects": [
                        _planner_subject("Only"),
                    ],
                    "dimensions": [],
                }
            ),
            "invalid_subjects",
        ),
        (json.dumps({"is_comparison": False}), "not_comparison"),
    ],
)
async def test_planner_returns_safe_parse_fallback(
    response: str, reason: str
) -> None:
    invoker = _Invoker(response)
    outcome = await ComparisonPlanner(llm_invoker=invoker).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == reason
    assert outcome.plan is None
    assert len(invoker.calls) == 1


@pytest.mark.asyncio
async def test_planner_rejects_numeric_values_not_present_in_question() -> None:
    payload = _payload(
        subjects=[
            _planner_subject(
                "nnMamba",
                retrieval_query="nnMamba parameters 999",
            ),
            _planner_subject(
                "EfficientMedNeXt-L",
                retrieval_query="EfficientMedNeXt-L FLOPs",
            ),
        ]
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(payload)).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "schema_violation"


@pytest.mark.asyncio
@pytest.mark.parametrize("invented_number", ["1e9", "-0.4", ".75", "+12.0"])
async def test_planner_rejects_invented_numeric_formats(
    invented_number: str,
) -> None:
    payload = _payload(
        subjects=[
            _planner_subject(
                "nnMamba",
                retrieval_query=f"nnMamba parameters {invented_number}",
            ),
            _planner_subject(
                "EfficientMedNeXt-L",
                retrieval_query="EfficientMedNeXt-L FLOPs",
            ),
        ]
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(payload)).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "schema_violation"


@pytest.mark.asyncio
async def test_planner_preserves_numeric_tokens_copied_from_question() -> None:
    question = "在 noise 0.4 時比較 Model A 與 Model B"
    payload = _payload(
        subjects=[
            _planner_subject(
                "Model A",
                retrieval_query="Model A noise 0.4",
            ),
            _planner_subject(
                "Model B",
                retrieval_query="Model B noise 0.4",
            ),
        ],
        dimensions=["noise 0.4"],
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(payload)).plan(
        question=question,
        timeout_seconds=1,
    )

    assert outcome.status == "planned"


@pytest.mark.asyncio
async def test_planner_timeout_and_provider_error_are_fail_soft() -> None:
    timed_out = await ComparisonPlanner(
        llm_invoker=_Invoker(_payload(), delay_s=0.05)
    ).plan(
        question=Q4,
        timeout_seconds=0.001,
    )
    failed = await ComparisonPlanner(
        llm_invoker=_Invoker(error=RuntimeError("provider exploded"))
    ).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert timed_out.fallback_reason == "timeout"
    assert failed.fallback_reason == "provider_error"


@pytest.mark.asyncio
async def test_missing_prompt_is_fail_soft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoker = _Invoker(_payload())

    def _missing_prompt(*_: object, **__: object) -> str:
        raise PromptConfigError("missing comparison prompt")

    monkeypatch.setattr(
        comparison_planner_module,
        "format_agentic_rag_prompt",
        _missing_prompt,
    )

    outcome = await ComparisonPlanner(llm_invoker=invoker).plan(
        question=Q4,
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "provider_error"
    assert invoker.calls == []


def test_comparison_overlay_preserves_authority_and_builds_subject_slots() -> None:
    scope = ResolvedSourceScope(
        authorized_doc_ids=["doc-a", "doc-b"],
        resolved_doc_ids=["doc-a", "doc-b"],
    )
    contract = QueryContract(
        route="exact_structured",
        intent="base intent",
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=3,
        runtime_token_budget=40_000,
        resolved_source_scope=scope,
    )
    plan = ComparisonPlan(
        subjects=[
            _subject("nnmamba", "nnMamba"),
            _subject("efficientmednext_l", "EfficientMedNeXt-L"),
        ],
        dimensions=["parameters", "FLOPs"],
    )

    overlaid = apply_comparison_overlay(contract, plan)

    assert overlaid.route == "exact_structured"
    assert overlaid.resolved_source_scope == scope
    assert overlaid.max_repair_rounds == 1
    assert overlaid.comparison_plan == plan
    assert [slot.slot_id for slot in overlaid.required_slots] == [
        "comparison-subject:nnmamba",
        "comparison-subject:efficientmednext_l",
    ]
    assert overlaid.required_slots[0].entity_ids == ["nnMamba"]
