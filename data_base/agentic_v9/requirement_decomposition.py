"""Deterministic, behavior-neutral question decomposition primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


DecompositionMethod = Literal[
    "numbered", "coordinated", "entity_distributive", "fallback"
]
DecompositionConfidence = Literal["high", "medium", "low"]


@dataclass(frozen=True, slots=True)
class QuestionBlock:
    """A top-level question block before obligation extraction."""

    text: str
    method: DecompositionMethod
    confidence: DecompositionConfidence


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
    "DecompositionConfidence",
    "DecompositionMethod",
    "QuestionBlock",
    "split_top_level_blocks",
]
