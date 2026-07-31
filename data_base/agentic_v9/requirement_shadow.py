"""Bounded, behavior-neutral requirement diagnostics for Agentic v9.

This module deliberately does not participate in routing, retrieval,
sufficiency, capability execution, context packing, or generation.  It only
projects what the question appears to require and which representations were
present in the already-retrieved documents.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


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


class RequirementShadowAnalysis(BaseModel):
    """Versioned shadow output safe to persist in an Agentic v9 trace."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["shadow_requirements_v1"] = "shadow_requirements_v1"
    behavior_influence: Literal[False] = False
    support_assessment: Literal["candidate_only"] = "candidate_only"
    requirements: list[ShadowRequirement] = Field(default_factory=list, max_length=8)
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
    requirement_texts = _decompose_explicit_requirements(question)[:8]
    document_projections_by_ref: dict[str, _DocumentProjection] = {}
    for index, document in enumerate(documents, start=1):
        projection = _DocumentProjection.from_document(document, index)
        document_projections_by_ref.setdefault(projection.evidence_ref, projection)
    document_projections = list(document_projections_by_ref.values())
    requirements = [
        _build_requirement(index, text, document_projections)
        for index, text in enumerate(requirement_texts, start=1)
    ]
    return RequirementShadowAnalysis(
        requirements=requirements,
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
        doc_id = str(metadata.get("doc_id") or metadata.get("source_id") or "unknown")
        chunk_id = str(metadata.get("chunk_id") or f"chunk-{index}")
        return cls(
            evidence_ref=f"{doc_id}:{chunk_id}",
            text=document.page_content or "",
            representations=_document_representations(document),
        )


def _build_requirement(
    index: int,
    text: str,
    documents: Sequence[_DocumentProjection],
) -> ShadowRequirement:
    information_need = _information_need(text)
    visual_precision = _visual_precision(text, information_need)
    candidates = [
        document
        for document in documents
        if _is_candidate(text, information_need, document)
    ][:8]
    representations = [
        representation
        for representation in _REPRESENTATION_ORDER
        if any(representation in document.representations for document in candidates)
    ]
    visual_decision, visual_reason = _visual_decision(
        information_need=information_need,
        visual_precision=visual_precision,
        representations=representations,
    )
    return ShadowRequirement(
        requirement_id=f"R{index}",
        text=text,
        answer_kind=_answer_kind(text),
        information_need=information_need,
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


def _information_need(text: str) -> InformationNeed:
    if _TABLE_TERMS.search(text):
        return "markdown_table"
    if _FIGURE_TERMS.search(text):
        return "visual_pattern"
    if _FORMULA_TERMS.search(text):
        return "text_structured"
    return "plain_text"


def _visual_precision(text: str, information_need: InformationNeed) -> VisualPrecision:
    if information_need != "visual_pattern":
        return "none"
    return "exact" if _EXACT_VISUAL_TERMS.search(text) else "qualitative"


def _visual_decision(
    *,
    information_need: InformationNeed,
    visual_precision: VisualPrecision,
    representations: Sequence[RepresentationKind],
) -> tuple[VisualDecision, str]:
    if information_need != "visual_pattern":
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
    information_need: InformationNeed,
    document: _DocumentProjection,
) -> bool:
    if (
        information_need == "markdown_table"
        and "markdown_table" in document.representations
    ):
        return True
    if information_need == "visual_pattern" and any(
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
