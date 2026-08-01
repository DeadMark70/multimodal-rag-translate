"""Deterministic, behavior-neutral question decomposition primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


DecompositionMethod = Literal[
    "numbered", "coordinated", "entity_distributive", "fallback"
]
DecompositionConfidence = Literal["high", "medium", "low"]
ConstraintKind = Literal[
    "conditional_scope",
    "output_format",
    "prohibition",
    "allowed_labels",
]


@dataclass(frozen=True, slots=True)
class QuestionBlock:
    """A top-level question block before obligation extraction."""

    text: str
    method: DecompositionMethod
    confidence: DecompositionConfidence


@dataclass(frozen=True, slots=True)
class DecomposedRequirement:
    """One answer obligation produced by the shadow classifier."""

    text: str
    method: DecompositionMethod
    confidence: DecompositionConfidence


@dataclass(frozen=True, slots=True)
class ResponseConstraintDraft:
    """A response rule that constrains synthesis without being an obligation."""

    text: str
    kind: ConstraintKind


@dataclass(frozen=True, slots=True)
class QuestionDecomposition:
    """Bounded, deterministic decomposition output for shadow diagnostics."""

    requirements: tuple[DecomposedRequirement, ...]
    response_constraints: tuple[ResponseConstraintDraft, ...]
    truncated_requirement_count: int = 0
    truncated_constraint_count: int = 0


_NUMBERED_MARKER = re.compile(
    r"(?P<arabic>\d{1,2})[.、]|(?P<chinese>[一二三四五六七八九十]+)、|"
    r"（(?P<chinese_parenthetical>[一二三四五六七八九十]+)）"
)
_BOUNDARY_CHARS = frozenset("：:；;。！？!?\n")
_CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def split_top_level_blocks(question: str) -> tuple[QuestionBlock, ...]:
    """Split only a validated top-level numbered sequence.

    Numeric markers inside identifiers, decimals, and ordinary parenthetical
    text are intentionally not accepted as boundaries. Ambiguous input is
    returned as one low-confidence fallback block.
    """

    normalized = " ".join(question.strip().split())
    if not normalized:
        return ()

    candidates = [
        match
        for match in _NUMBERED_MARKER.finditer(normalized)
        if _is_top_level_marker(normalized, match.start())
    ]
    values = [_marker_value(match) for match in candidates]
    expected = list(range(1, len(values) + 1))
    if len(candidates) < 2 or values != expected:
        return (
            QuestionBlock(
                text=normalized,
                method="fallback",
                confidence="low",
            ),
        )

    blocks: list[QuestionBlock] = []
    for index, marker in enumerate(candidates):
        end = (
            candidates[index + 1].start()
            if index + 1 < len(candidates)
            else len(normalized)
        )
        text = normalized[marker.end() : end].strip()
        if text:
            blocks.append(
                QuestionBlock(
                    text=text,
                    method="numbered",
                    confidence="high",
                )
            )
    return tuple(blocks) or (
        QuestionBlock(text=normalized, method="fallback", confidence="low"),
    )


_CONSTRAINT_PATTERNS: tuple[tuple[ConstraintKind, re.Pattern[str]], ...] = (
    (
        "conditional_scope",
        re.compile(
            r"(?:若|如果)[^。！？!?；;]*(?:必須|必须|需要|則|则)[^。！？!?；;]*"
        ),
    ),
    (
        "output_format",
        re.compile(r"(?:請|请)(?:以|按)[^。！？!?；;]*?(?:格式|方式)[^。！？!?；;]*"),
    ),
    (
        "prohibition",
        re.compile(r"(?:不要|不得|不可|不應|不应)[^。！？!?；;]*"),
    ),
    (
        "allowed_labels",
        re.compile(
            r"(?:^|[：:；;])\s*[A-ZＡ-Ｚ]\s*[.．、:：]"
            r"[^。！？!?；;]*[；;]\s*[A-ZＡ-Ｚ]\s*[.．、:：]"
            r"[^。！？!?；;]*"
        ),
    ),
)
_QUESTION_BREAK = re.compile(r"(?<=[？?])\s*")
_CONTINUATION = re.compile(r"\s*(?:此外|並且|并且|另外|同時|同时)\s*，?\s*")
_COORDINATION = re.compile(
    r"\s+(?:與|和|及|以及|并|and)\s+|\s*(?:與|和|及|以及|并)\s*|\s*/\s*"
)
_ENTITY_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_ENTITY_STOPWORDS = {
    "a",
    "b",
    "and",
    "based",
    "compare",
    "only",
    "the",
    "with",
}


def decompose_question(
    question: str,
    *,
    max_requirements: int = 8,
    max_constraints: int = 8,
) -> QuestionDecomposition:
    """Classify answer obligations and response constraints conservatively.

    This is deliberately deterministic and template-light.  It is intended for
    shadow diagnostics and planning hints, not as a retrieval or sufficiency
    gate.  Ambiguous clauses remain a single low-confidence obligation.
    """

    normalized = " ".join(question.strip().split())
    if not normalized:
        return QuestionDecomposition((), ())

    masked, constraints = _extract_constraints(normalized)
    blocks = split_top_level_blocks(masked)
    requirement_drafts: list[DecomposedRequirement] = []
    for block in blocks:
        requirement_drafts.extend(_classify_block(block))

    deduped = _dedupe_requirements(requirement_drafts)
    bounded_requirements = deduped[: max(0, max_requirements)]
    bounded_constraints = constraints[: max(0, max_constraints)]
    return QuestionDecomposition(
        requirements=tuple(bounded_requirements),
        response_constraints=tuple(bounded_constraints),
        truncated_requirement_count=max(0, len(deduped) - len(bounded_requirements)),
        truncated_constraint_count=max(0, len(constraints) - len(bounded_constraints)),
    )


def _extract_constraints(
    text: str,
) -> tuple[str, list[ResponseConstraintDraft]]:
    spans: list[tuple[int, int, ConstraintKind, str]] = []
    for kind, pattern in _CONSTRAINT_PATTERNS:
        for match in pattern.finditer(text):
            if any(
                match.start() < end and start < match.end() for start, end, *_ in spans
            ):
                continue
            spans.append((match.start(), match.end(), kind, match.group(0).strip()))

    spans.sort(key=lambda item: item[0])
    masked_parts: list[str] = []
    cursor = 0
    constraints: list[ResponseConstraintDraft] = []
    for start, end, kind, matched in spans:
        masked_parts.append(text[cursor:start])
        masked_parts.append(" ")
        cursor = end
        constraints.append(ResponseConstraintDraft(text=matched, kind=kind))
    masked_parts.append(text[cursor:])
    return " ".join("".join(masked_parts).split()), _dedupe_constraints(constraints)


def _classify_block(block: QuestionBlock) -> list[DecomposedRequirement]:
    text = block.text.strip(" ：:，,；;。！？?!")
    if not text:
        return []

    entities = _extract_entities(text)
    distributive = bool(
        re.search(r"每個|每一|各自|分別|另外|唯一符合|哪一個|respectively", text, re.I)
    )
    if distributive and 2 <= len(entities) <= 6:
        return _entity_requirements(text, entities)

    figure_prerequisite: list[DecomposedRequirement] = []
    if re.search(r"(?:Figure|圖)\s*\d*", text, re.I) and re.search(
        r"(?:策略|方法|\([ab]\)|（[ab]）)", text, re.I
    ):
        figure_prerequisite.append(
            DecomposedRequirement(
                text="解析引用 Figure/圖示中的策略選項與對應關係",
                method="coordinated",
                confidence="medium",
            )
        )

    parts = _split_obligation_sentences(text)
    result: list[DecomposedRequirement] = []
    for part in parts:
        result.extend(_split_coordinated_part(part, block))
    if not result:
        result = [
            DecomposedRequirement(
                text=text,
                method=block.method,
                confidence=block.confidence,
            )
        ]
    return figure_prerequisite + result


def _split_obligation_sentences(text: str) -> list[str]:
    question_parts = [
        part.strip() for part in _QUESTION_BREAK.split(text) if part.strip()
    ]
    if len(question_parts) > 1:
        return question_parts
    continuation_parts = [
        part.strip() for part in _CONTINUATION.split(text) if part.strip()
    ]
    return continuation_parts or [text]


def _split_coordinated_part(
    text: str, block: QuestionBlock
) -> list[DecomposedRequirement]:
    cue = re.search(
        r"(?:並說明|并说明|並解釋|并解释|並簡述|并简述|以及|並回答|并回答)", text
    )
    if cue:
        prefix = text[: cue.start()].strip(" ：:，,；;")
        tail = text[cue.end() :].strip(" ：:，,；;")
        parts = [prefix] if prefix else []
        parts.extend(_split_coordination_tail(tail))
    else:
        parts = [text]

    method: DecompositionMethod = (
        "coordinated" if len(parts) > 1 or cue else block.method
    )
    confidence: DecompositionConfidence = (
        "medium" if method == "coordinated" else block.confidence
    )
    return [
        DecomposedRequirement(
            text=part.strip(" ：:，,；;。！？?!"), method=method, confidence=confidence
        )
        for part in parts
        if part.strip(" ：:，,；;。！？?!")
    ]


def _split_coordination_tail(text: str) -> list[str]:
    pieces = [piece.strip(" ：:，,；;。！？?!") for piece in _COORDINATION.split(text)]
    return [piece for piece in pieces if piece]


def _extract_entities(text: str) -> list[str]:
    candidates = _ENTITY_TOKEN.findall(text)
    entities: list[str] = []
    for candidate in candidates:
        if candidate.lower() in _ENTITY_STOPWORDS:
            continue
        if len(candidate) < 3 and "-" not in candidate:
            continue
        if not ("-" in candidate or any(char.isupper() for char in candidate[1:])):
            continue
        if candidate not in entities:
            entities.append(candidate)
    return entities


def _entity_requirements(text: str, entities: list[str]) -> list[DecomposedRequirement]:
    result = [
        DecomposedRequirement(
            text=f"針對 {entity}，回答題目中對該主體的要求。",
            method="entity_distributive",
            confidence="high",
        )
        for entity in entities
    ]
    if re.search(r"分類|classif", text, re.I):
        return result
    result.append(
        DecomposedRequirement(
            text="整合各主體的比較、選擇或 trade-off 結論。",
            method="entity_distributive",
            confidence="medium",
        )
    )
    return result


def _dedupe_requirements(
    requirements: list[DecomposedRequirement],
) -> list[DecomposedRequirement]:
    seen: set[str] = set()
    result: list[DecomposedRequirement] = []
    for requirement in requirements:
        normalized = " ".join(requirement.text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(
            DecomposedRequirement(
                text=normalized,
                method=requirement.method,
                confidence=requirement.confidence,
            )
        )
    return result


def _dedupe_constraints(
    constraints: list[ResponseConstraintDraft],
) -> list[ResponseConstraintDraft]:
    seen: set[tuple[ConstraintKind, str]] = set()
    result: list[ResponseConstraintDraft] = []
    for constraint in constraints:
        key = (constraint.kind, " ".join(constraint.text.split()))
        if key in seen:
            continue
        seen.add(key)
        result.append(ResponseConstraintDraft(text=key[1], kind=key[0]))
    return result


def _is_top_level_marker(text: str, start: int) -> bool:
    prefix = text[:start].rstrip()
    return not prefix or prefix[-1] in _BOUNDARY_CHARS


def _marker_value(match: re.Match[str]) -> int:
    arabic = match.group("arabic")
    if arabic is not None:
        return int(arabic)
    chinese = match.group("chinese") or match.group("chinese_parenthetical")
    if chinese == "十":
        return 10
    if len(chinese or "") == 2 and chinese and chinese[0] == "十":
        return 10 + _CHINESE_DIGITS[chinese[1]]
    return _CHINESE_DIGITS[chinese or ""]


__all__ = [
    "ConstraintKind",
    "DecompositionConfidence",
    "DecompositionMethod",
    "DecomposedRequirement",
    "QuestionBlock",
    "QuestionDecomposition",
    "ResponseConstraintDraft",
    "decompose_question",
    "split_top_level_blocks",
]
