"""Bounded, behavior-neutral requirement diagnostics for Agentic v9.

This module deliberately does not participate in routing, retrieval,
sufficiency, capability execution, context packing, or generation.  It only
projects what the question appears to require and which representations were
present in the already-retrieved documents.
"""

from __future__ import annotations

import re
import hashlib
from collections.abc import Sequence
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from data_base.agentic_v9.requirement_decomposition import (
    ConstraintKind,
    DecomposedRequirement,
    DecompositionConfidence,
    DecompositionMethod,
    decompose_question,
)
from data_base.document_metadata import get_document_id


AnswerKind = Literal[
    "number",
    "equation",
    "definition",
    "comparison",
    "explanation",
    "text",
]
InformationNeed = Literal[
    "plain_text",
    "text_structured",
    "markdown_table",
    "visual_pattern",
]
VisualPrecision = Literal["none", "qualitative", "exact"]
VisualDecision = Literal["not_requested", "optional", "required"]
CoverageStatus = Literal["missing", "candidate", "supported"]
RepresentationKind = Literal[
    "plain_text",
    "markdown_table",
    "image_summary",
    "visual_asset",
]


class ShadowRequirement(BaseModel):
    """One question-derived requirement with candidate-only diagnostics."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(pattern=r"^R[1-8]$")
    text: str = Field(min_length=1, max_length=512)
    answer_kind: AnswerKind
    information_need: InformationNeed
    information_needs: list[InformationNeed] = Field(
        min_length=1,
        max_length=4,
    )
    decomposition_method: DecompositionMethod
    decomposition_confidence: DecompositionConfidence
    visual_precision: VisualPrecision = "none"
    visual_decision: VisualDecision = "not_requested"
    visual_reason: str = Field(min_length=1, max_length=96)
    importance: Literal["core"] = "core"
    coverage_status: CoverageStatus
    available_representations: list[RepresentationKind] = Field(
        default_factory=list,
        max_length=4,
    )
    candidate_evidence_refs: list[str] = Field(default_factory=list, max_length=8)


class RequirementShadowSummary(BaseModel):
    """Bounded aggregate that never promotes candidate evidence to support."""

    model_config = ConfigDict(extra="forbid")

    requirement_count: int = Field(ge=0, le=8)
    candidate_count: int = Field(ge=0, le=8)
    missing_count: int = Field(ge=0, le=8)
    supported_count: int = Field(default=0, ge=0, le=8)
    visual_required_count: int = Field(ge=0, le=8)
    constraint_count: int = Field(ge=0, le=8)
    low_confidence_count: int = Field(ge=0, le=8)
    truncated_requirement_count: int = Field(ge=0)
    truncated_constraint_count: int = Field(ge=0)


class ShadowResponseConstraint(BaseModel):
    """A question-derived synthesis constraint, separate from obligations."""

    model_config = ConfigDict(extra="forbid")

    constraint_id: str = Field(pattern=r"^C[1-8]$")
    kind: ConstraintKind
    text: str = Field(min_length=1, max_length=512)


class RequirementShadowAnalysis(BaseModel):
    """Versioned shadow output safe to persist in an Agentic v9 trace."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["shadow_requirements_v2"] = "shadow_requirements_v2"
    behavior_influence: Literal[False] = False
    support_assessment: Literal["candidate_only"] = "candidate_only"
    requirements: list[ShadowRequirement] = Field(default_factory=list, max_length=8)
    response_constraints: list[ShadowResponseConstraint] = Field(
        default_factory=list,
        max_length=8,
    )
    truncated: bool = False
    summary: RequirementShadowSummary


_NUMBERED_REQUIREMENT = re.compile(
    r"(?:^|[\s：:；;])(?:[（(]?)(\d{1,2})(?:[.)、）])\s*"
)
_MARKDOWN_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_TABLE_TERMS = re.compile(r"\btable\b|表格|資料表", re.IGNORECASE)
_FIGURE_TERMS = re.compile(
    r"\bfig(?:ure)?\.?\s*\d*|熱力圖|折線圖|曲線圖|散點圖|影像圖|圖像|圖中",
    re.IGNORECASE,
)
_FORMULA_TERMS = re.compile(
    r"\b(?:formula|equation|theorem)\b|公式|方程|定理",
    re.IGNORECASE,
)
_EXACT_VISUAL_TERMS = re.compile(
    r"精確|準確|確切|數值|數字|座標|像素|色階值|\bepoch\s*\d+",
    re.IGNORECASE,
)
_DEFINITION_TERMS = re.compile(r"定義|何謂|\bdefine[ds]?\b", re.IGNORECASE)
_NUMBER_TERMS = re.compile(
    r"多少|幾(?:個|項|種|何)?|數值|分數|比例|範圍|邊界|\b(?:dice|miou|score|value)\b",
    re.IGNORECASE,
)
_COMPARISON_TERMS = re.compile(
    r"比較|差別|差異|取捨|何者|哪個|\b(?:compare|versus|vs\.?|which)\b",
    re.IGNORECASE,
)
_EXPLANATION_TERMS = re.compile(
    r"為何|為什麼|原因|機制|如何運作|怎麼運作|\b(?:why|explain|mechanism|how)\b",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-.^][A-Za-z0-9]+)*|\d+(?:\.\d+)?")
_REPRESENTATION_ORDER: tuple[RepresentationKind, ...] = (
    "plain_text",
    "markdown_table",
    "image_summary",
    "visual_asset",
)


def build_requirement_shadow(
    *, question: str, documents: Sequence[Document]
) -> RequirementShadowAnalysis:
    """Build observational requirements without changing execution behavior."""
    decomposition = decompose_question(question)
    document_projections_by_ref: dict[str, _DocumentProjection] = {}
    for index, document in enumerate(documents, start=1):
        projection = _DocumentProjection.from_document(document, index)
        document_projections_by_ref.setdefault(projection.evidence_ref, projection)
    document_projections = list(document_projections_by_ref.values())
    requirements = [
        _build_requirement(index, requirement, document_projections)
        for index, requirement in enumerate(decomposition.requirements, start=1)
    ]
    constraints = [
        ShadowResponseConstraint(
            constraint_id=f"C{index}", kind=constraint.kind, text=constraint.text
        )
        for index, constraint in enumerate(decomposition.response_constraints, start=1)
    ]
    return RequirementShadowAnalysis(
        requirements=requirements,
        response_constraints=constraints,
        truncated=bool(
            decomposition.truncated_requirement_count
            or decomposition.truncated_constraint_count
        ),
        summary=RequirementShadowSummary(
            requirement_count=len(requirements),
            candidate_count=sum(
                item.coverage_status == "candidate" for item in requirements
            ),
            missing_count=sum(
                item.coverage_status == "missing" for item in requirements
            ),
            supported_count=0,
            visual_required_count=sum(
                item.visual_decision == "required" for item in requirements
            ),
            constraint_count=len(constraints),
            low_confidence_count=sum(
                item.decomposition_confidence == "low" for item in requirements
            ),
            truncated_requirement_count=decomposition.truncated_requirement_count,
            truncated_constraint_count=decomposition.truncated_constraint_count,
        ),
    )


class _DocumentProjection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    evidence_ref: str
    text: str
    representations: list[RepresentationKind]

    @classmethod
    def from_document(cls, document: Document, index: int) -> _DocumentProjection:
        metadata = document.metadata or {}
        content = document.page_content or ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        doc_id = get_document_id(metadata)
        chunk_id = str(
            metadata.get("chunk_id") or metadata.get("unique_chunk_id") or ""
        ).strip()
        if doc_id and chunk_id:
            evidence_ref = f"{doc_id}:{chunk_id}"
        elif doc_id:
            evidence_ref = f"{doc_id}:content-{content_hash[:16]}"
        else:
            evidence_ref = f"content:{content_hash}"
        return cls(
            evidence_ref=evidence_ref,
            text=content,
            representations=_document_representations(document),
        )


def _build_requirement(
    index: int,
    requirement: DecomposedRequirement,
    documents: Sequence[_DocumentProjection],
) -> ShadowRequirement:
    text = requirement.text
    information_needs = _information_needs(text)
    information_need = information_needs[0]
    visual_precision = _visual_precision(text, information_needs)
    candidates = [
        document
        for document in documents
        if _is_candidate(text, information_needs, document)
    ][:8]
    representations = [
        representation
        for representation in _REPRESENTATION_ORDER
        if any(representation in document.representations for document in candidates)
    ]
    visual_decision, visual_reason = _visual_decision(
        information_needs=information_needs,
        visual_precision=visual_precision,
        representations=representations,
    )
    return ShadowRequirement(
        requirement_id=f"R{index}",
        text=text,
        answer_kind=_answer_kind(text),
        information_need=information_need,
        information_needs=information_needs,
        decomposition_method=requirement.method,
        decomposition_confidence=requirement.confidence,
        visual_precision=visual_precision,
        visual_decision=visual_decision,
        visual_reason=visual_reason,
        coverage_status="candidate" if candidates else "missing",
        available_representations=representations,
        candidate_evidence_refs=[item.evidence_ref for item in candidates],
    )


def _decompose_explicit_requirements(question: str) -> list[str]:
    normalized = " ".join(question.strip().split())
    if not normalized:
        return []
    matches = list(_NUMBERED_REQUIREMENT.finditer(normalized))
    if len(matches) >= 2:
        requirements = []
        for position, match in enumerate(matches):
            end = (
                matches[position + 1].start()
                if position + 1 < len(matches)
                else len(normalized)
            )
            value = normalized[match.end() : end].strip(" ：:；;。")
            if value:
                requirements.append(value)
        if requirements:
            return requirements
    semicolon_parts = [
        value.strip(" ：:；;。")
        for value in re.split(r"[；;]\s*", normalized)
        if value.strip(" ：:；;。")
    ]
    return semicolon_parts if len(semicolon_parts) > 1 else [normalized]


def _answer_kind(text: str) -> AnswerKind:
    if _DEFINITION_TERMS.search(text):
        return "definition"
    if _NUMBER_TERMS.search(text):
        return "number"
    if _FORMULA_TERMS.search(text):
        return "equation"
    if _COMPARISON_TERMS.search(text):
        return "comparison"
    if _EXPLANATION_TERMS.search(text):
        return "explanation"
    return "text"


def _information_needs(text: str) -> list[InformationNeed]:
    matches: list[tuple[int, InformationNeed]] = []
    for information_need, pattern in (
        ("markdown_table", _TABLE_TERMS),
        ("visual_pattern", _FIGURE_TERMS),
        ("text_structured", _FORMULA_TERMS),
    ):
        match = pattern.search(text)
        if match is not None:
            matches.append((match.start(), information_need))
    if not matches:
        return ["plain_text"]
    matches.sort(key=lambda item: item[0])
    return [information_need for _, information_need in matches]


def _visual_precision(
    text: str, information_needs: Sequence[InformationNeed]
) -> VisualPrecision:
    if "visual_pattern" not in information_needs:
        return "none"
    return "exact" if _EXACT_VISUAL_TERMS.search(text) else "qualitative"


def _visual_decision(
    *,
    information_needs: Sequence[InformationNeed],
    visual_precision: VisualPrecision,
    representations: Sequence[RepresentationKind],
) -> tuple[VisualDecision, str]:
    if "visual_pattern" not in information_needs:
        return "not_requested", "text_representation_expected"
    if visual_precision == "exact":
        return "required", "exact_visual_information_requested"
    if "image_summary" in representations:
        return "optional", "qualitative_image_summary_available"
    return "required", "visual_pattern_without_summary"


def _document_representations(document: Document) -> list[RepresentationKind]:
    metadata = document.metadata or {}
    text = document.page_content or ""
    values: set[RepresentationKind] = set()
    if _MARKDOWN_TABLE_LINE.search(text):
        values.add("markdown_table")
    source = str(metadata.get("source") or "").casefold()
    media_type = str(
        metadata.get("type") or metadata.get("asset_type") or ""
    ).casefold()
    if source == "image" or media_type in {"figure", "image", "chart", "plot"}:
        values.add("image_summary")
        if metadata.get("image_path") or metadata.get("asset_id"):
            values.add("visual_asset")
    if not values:
        values.add("plain_text")
    return [item for item in _REPRESENTATION_ORDER if item in values]


def _is_candidate(
    requirement_text: str,
    information_needs: Sequence[InformationNeed],
    document: _DocumentProjection,
) -> bool:
    if (
        "markdown_table" in information_needs
        and "markdown_table" in document.representations
    ):
        return True
    if "visual_pattern" in information_needs and any(
        item in document.representations for item in ("image_summary", "visual_asset")
    ):
        return True
    anchors = {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(requirement_text)
        if len(token) > 1
    }
    if not anchors:
        return bool(document.text.strip())
    normalized_document = document.text.casefold()
    return any(anchor in normalized_document for anchor in anchors)


__all__ = [
    "RequirementShadowAnalysis",
    "RequirementShadowSummary",
    "ShadowRequirement",
    "build_requirement_shadow",
]
