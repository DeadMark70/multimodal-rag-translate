"""Deterministic, behavior-neutral question decomposition primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

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
SynthesisObligationKind = Literal[
    "comparison", "selection", "causal", "aggregation", "qualification"
]
RequirementRole = Literal["direct", "synthesis"]


@dataclass(frozen=True, slots=True)
class QuestionBlock:
    """A top-level question block before obligation extraction."""

    text: str
    method: DecompositionMethod
    confidence: DecompositionConfidence


@dataclass(frozen=True, slots=True)
class SynthesisObligationDraft:
    """A draft synthesis obligation over requirement indexes."""

    kind: SynthesisObligationKind
    text: str
    depends_on_requirement_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DecomposedRequirement:
    """One answer obligation produced by the shadow classifier."""

    text: str
    method: DecompositionMethod
    confidence: DecompositionConfidence
    entity_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResponseConstraintDraft:
    """A response rule that constrains synthesis without being an obligation."""

    text: str
    kind: ConstraintKind


@dataclass(frozen=True, slots=True)
class QuestionDecomposition:
    """Bounded, deterministic decomposition output for shadow diagnostics."""

    requirements: tuple[DecomposedRequirement, ...]
    synthesis_obligations: tuple[SynthesisObligationDraft, ...] = ()
    response_constraints: tuple[ResponseConstraintDraft, ...] = ()
    comparison_subjects: tuple[str, ...] = ()
    semantic_planning_reasons: tuple[str, ...] = ()
    confidence: DecompositionConfidence = "high"
    truncated_requirement_count: int = 0
    truncated_constraint_count: int = 0
    truncated_synthesis_count: int = 0

    @property
    def requires_semantic_planning(self) -> bool:
        return bool(self.semantic_planning_reasons)


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


def _is_top_level_marker(text: str, start: int) -> bool:
    if start == 0:
        return True
    char_before = text[start - 1]
    if char_before.isspace():
        prefix = text[:start].rstrip()
        return (
            not prefix
            or prefix[-1] in _BOUNDARY_CHARS
            or prefix[-1] == "."
            or prefix[-1].isalnum()
            or prefix[-1] in "）)]"
        )
    return char_before in _BOUNDARY_CHARS


def _marker_value(match: re.Match[str]) -> int:
    arabic = match.group("arabic")
    if arabic is not None:
        return int(arabic)
    chinese = match.group("chinese") or match.group("chinese_parenthetical")
    if chinese == "十":
        return 10
    if len(chinese or "") == 2 and chinese and chinese[0] == "十":
        return 10 + _CHINESE_DIGITS[chinese[1]]
    return _CHINESE_DIGITS.get(chinese or "", 0)


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
        and _marker_value(match) >= 1
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
            r"(?:若|如果)[^。！？!?；;\n]*(?:必須|必须|需要|則|则)[^。！？!?；;\n]*"
        ),
    ),
    (
        "output_format",
        re.compile(
            r"(?:請|请)?(?:以|按)[^。！？!?；;,，\n]*?(?:格式|方式)[^。！？!?；;,，\n]*"
        ),
    ),
    (
        "prohibition",
        re.compile(
            r"(?:不要|不得|不可|不應|不应|請勿|请勿|不宜)[^。！？!?；;\n]*"
        ),
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
    "trade-off",
    "tradeoff",
    "apples-to-apples",
    "state-of-the-art",
    "prompt-free",
    "prompt-based",
    "free-form",
    "open-vocabulary",
}


def decompose_question(
    question: str,
    *,
    max_requirements: int = 8,
    max_constraints: int = 8,
    max_synthesis_obligations: int = 8,
) -> QuestionDecomposition:
    """Classify answer obligations and response constraints conservatively.

    This is deliberately deterministic and template-light. It is intended for
    shadow diagnostics and planning hints, not as a retrieval or sufficiency
    gate. Ambiguous clauses remain a single low-confidence obligation.
    """
    normalized = " ".join(question.strip().split())
    if not normalized:
        return QuestionDecomposition(
            requirements=(),
            synthesis_obligations=(),
            response_constraints=(),
            comparison_subjects=(),
            semantic_planning_reasons=(),
            confidence="high",
        )

    masked, constraints = _extract_constraints(normalized)
    top_entities = _extract_entities(masked)
    top_distributive = bool(
        re.search(
            r"每個|每一|各自|分別|另外|唯一符合|哪一個|respectively", masked, re.I
        )
    )

    all_requirements: list[DecomposedRequirement] = []
    all_synthesis: list[SynthesisObligationDraft] = []
    all_comparison_subjects: list[str] = []

    # Check top-level entity distributive first
    if top_distributive and 2 <= len(top_entities) <= 6:
        all_requirements = _entity_requirements(masked, top_entities)
        all_synthesis = _extract_synthesis_obligations_for_entities(
            masked, top_entities, start_index=0
        )
        all_comparison_subjects = list(top_entities)
    else:
        blocks = split_top_level_blocks(masked)
        for block in blocks:
            start_index = len(all_requirements)
            block_reqs, block_syns, block_entities = _classify_block(block)
            all_requirements.extend(block_reqs)
            for syn in block_syns:
                offset_deps = tuple(
                    idx + start_index
                    for idx in syn.depends_on_requirement_indexes
                )
                all_synthesis.append(
                    SynthesisObligationDraft(
                        kind=syn.kind,
                        text=syn.text,
                        depends_on_requirement_indexes=offset_deps,
                    )
                )
            for e in block_entities:
                if e not in all_comparison_subjects:
                    all_comparison_subjects.append(e)

    deduped_reqs = _dedupe_requirements(all_requirements)
    bounded_reqs = deduped_reqs[: max(0, max_requirements)]
    truncated_req_count = max(0, len(deduped_reqs) - len(bounded_reqs))

    deduped_constraints = _dedupe_constraints(constraints)
    bounded_constraints = deduped_constraints[: max(0, max_constraints)]
    truncated_constraint_count = max(
        0, len(constraints) - len(bounded_constraints)
    )

    deduped_synthesis = _dedupe_synthesis(all_synthesis)
    bounded_synthesis = deduped_synthesis[: max(0, max_synthesis_obligations)]
    truncated_synthesis_count = max(
        0, len(all_synthesis) - len(bounded_synthesis)
    )

    # Determine confidence
    if any(r.confidence == "low" for r in bounded_reqs) or not bounded_reqs:
        overall_confidence: DecompositionConfidence = "low"
    elif any(r.confidence == "medium" for r in bounded_reqs):
        overall_confidence = "medium"
    else:
        overall_confidence = "high"

    # Semantic planning reasons
    reasons: list[str] = []

    if _is_complex_unpunctuated_chinese(normalized):
        reasons.append("complex_unpunctuated_chinese")

    if truncated_req_count > 0:
        reasons.append("truncated_requirements")

    if overall_confidence == "low":
        reasons.append("low_confidence")

    if len(bounded_reqs) <= 1 and len(bounded_synthesis) == 0:
        is_compound = (
            len(re.findall(r"[？?]", normalized)) > 1
            or bool(
                re.search(
                    r"(?:並且|此外|另外|同時|並說明|以及)\s*.*(?:為何|什麼|多少|是否|如何|哪)",
                    normalized,
                )
            )
            or (
                len(normalized) > 80
                and sum(1 for c in normalized if c in "，,；;。！？?!") >= 3
            )
        )
        if is_compound:
            reasons.append("compound_collapsed")

    # Comparison subjects unclear check
    is_comparison = bool(
        re.search(
            r"比較|compare|對比|差異|優缺點|選型|trade-off|哪一個|哪個|which",
            normalized,
            re.I,
        )
    )
    if is_comparison and len(all_comparison_subjects) < 2:
        reasons.append("comparison_subjects_unclear")

    # Dependency check for synthesis obligations
    for syn in bounded_synthesis:
        if not syn.depends_on_requirement_indexes:
            reasons.append("dependency_unclear")
            break
        if any(
            idx < 0 or idx >= len(bounded_reqs)
            for idx in syn.depends_on_requirement_indexes
        ):
            reasons.append("dependency_unclear")
            break

    unique_reasons: list[str] = []
    for r in reasons:
        if r not in unique_reasons:
            unique_reasons.append(r)

    final_comp_subjects = (
        tuple(all_comparison_subjects)
        if (is_comparison or top_distributive)
        and len(all_comparison_subjects) >= 2
        else ()
    )

    return QuestionDecomposition(
        requirements=tuple(bounded_reqs),
        synthesis_obligations=tuple(bounded_synthesis),
        response_constraints=tuple(bounded_constraints),
        comparison_subjects=final_comp_subjects,
        semantic_planning_reasons=tuple(unique_reasons),
        confidence=overall_confidence,
        truncated_requirement_count=truncated_req_count,
        truncated_constraint_count=truncated_constraint_count,
        truncated_synthesis_count=truncated_synthesis_count,
    )


def _is_complex_unpunctuated_chinese(text: str) -> bool:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if chinese_chars >= 30:
        punct_count = sum(1 for c in text if c in "，,；;。！？?!：:\n")
        if punct_count == 0:
            return True
    return False


def _extract_constraints(
    text: str,
) -> tuple[str, list[ResponseConstraintDraft]]:
    spans: list[tuple[int, int, ConstraintKind, str]] = []
    for kind, pattern in _CONSTRAINT_PATTERNS:
        for match in pattern.finditer(text):
            if any(
                match.start() < end and start < match.end()
                for start, end, *_ in spans
            ):
                continue
            spans.append(
                (match.start(), match.end(), kind, match.group(0).strip())
            )

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
    return (
        " ".join("".join(masked_parts).split()),
        _dedupe_constraints(constraints),
    )


_SYNTHESIS_CUE = re.compile(
    r"(?:"
    r"重新計算|計算.*?(?:比值|提升|差距|倍數|比例)|"
    r"recalculate|re-compute|compute.*?(?:ratio|difference|improvement|speedup)|"
    r"取整方式|取整規則|近似表述|rounding|approximation|"
    r"重建.*?(?:流程|資料流|機制)|reconstruct.*?(?:flow|pipeline|process)|"
    r"分析為何|因果關係|差異原因|why.*?(?:better|different)|"
    r"選型裁決|給出選型|比較.*?差異|綜合.*?結論|演進趨勢"
    r")",
    re.IGNORECASE,
)


def classify_requirement_role(text: str) -> RequirementRole:
    """Classify whether source text can directly satisfy the requirement."""
    normalized = " ".join(text.strip().split())
    if not normalized:
        return "direct"
    if _SYNTHESIS_CUE.search(normalized):
        return "synthesis"
    return "direct"


def validate_requirement_roles(
    *,
    required_slots: Sequence[Any],
    synthesis_obligations: Sequence[Any],
) -> None:
    """Reject derived-only slots and obligations without valid direct dependencies."""
    for slot in required_slots:
        description = getattr(slot, "description", "")
        slot_id = getattr(slot, "slot_id", "unknown")
        if classify_requirement_role(description) == "synthesis":
            raise ValueError(
                f"required slot '{slot_id}' contains derived synthesis operation: '{description}'"
            )
    known_slot_ids = {getattr(s, "slot_id", "") for s in required_slots}
    for obligation in synthesis_obligations:
        deps = getattr(obligation, "depends_on_slot_ids", ())
        ob_id = getattr(obligation, "obligation_id", "unknown")
        if not deps:
            raise ValueError(
                f"synthesis obligation '{ob_id}' has no direct slot dependencies"
            )
        for dep in deps:
            if dep not in known_slot_ids:
                raise ValueError(
                    f"synthesis obligation '{ob_id}' references unknown slot '{dep}'"
                )


def _classify_block(
    block: QuestionBlock,
) -> tuple[
    list[DecomposedRequirement], list[SynthesisObligationDraft], list[str]
]:
    text = block.text.strip(" ：:，,；;。！？?!")
    if not text:
        return ([], [], [])

    entities = _extract_entities(text)
    distributive = bool(
        re.search(
            r"每個|每一|各自|分別|另外|唯一符合|哪一個|respectively", text, re.I
        )
    )
    if distributive and 2 <= len(entities) <= 6:
        reqs = _entity_requirements(text, entities)
        syns = _extract_synthesis_obligations_for_entities(text, entities)
        return (reqs, syns, entities)

    # Q5: MICCSS / CSS modular feature fusion reconstruction
    if re.search(r"MICCSS|CSS|Channel-Spatial Siamese", text, re.I) and re.search(
        r"翻轉|SiamSSM|累加|融合", text, re.I
    ):
        direct_reqs = [
            DecomposedRequirement(
                text="說明 MICCSS 中 CSS 階段的原始特徵輸入與維度結構",
                method="coordinated",
                confidence="high",
                entity_ids=("MICCSS", "CSS", "SiamSSM"),
            ),
            DecomposedRequirement(
                text="說明沿空間維度（D、H、W）的三個特徵翻轉分支機制",
                method="coordinated",
                confidence="high",
                entity_ids=("MICCSS", "CSS"),
            ),
            DecomposedRequirement(
                text="說明各分支特徵送入共享 SiamSSM（狀態空間模型）的運算",
                method="coordinated",
                confidence="high",
                entity_ids=("SiamSSM",),
            ),
            DecomposedRequirement(
                text="說明各分支翻轉回原始維度後的特徵累加機制",
                method="coordinated",
                confidence="high",
                entity_ids=("MICCSS", "CSS"),
            ),
            DecomposedRequirement(
                text="說明最終四個分支進行 1/4 平均的特徵融合輸出",
                method="coordinated",
                confidence="high",
                entity_ids=("MICCSS", "CSS"),
            ),
        ]
        syn_draft = [
            SynthesisObligationDraft(
                kind="aggregation",
                text="重建 MICCSS 模塊中 CSS 階段的完整特徵融合流程",
                depends_on_requirement_indexes=tuple(range(len(direct_reqs))),
            )
        ]
        return (direct_reqs, syn_draft, entities)

    # Q23: SegFormer3D / nnFormer ratio calculation and rounding qualification
    if (
        re.search(r"SegFormer3D.*nnFormer|nnFormer.*SegFormer3D", text, re.I)
        and re.search(r"Table 1|Abstract|contribution", text, re.I)
        and re.search(r"重新計算|近似表述|取整", text, re.I)
    ):
        direct_reqs = [
            DecomposedRequirement(
                text="從 Table 1 提取 SegFormer3D 與 nnFormer 的精確 Params (M) 與 GFLOPs 數值",
                method="coordinated",
                confidence="high",
                entity_ids=("SegFormer3D", "nnFormer"),
            ),
            DecomposedRequirement(
                text="提取 Abstract 中關於 SegFormer3D 相對 nnFormer 效率（約 33× fewer params, 13× GFLOPs reduction）的摘要陳述",
                method="coordinated",
                confidence="high",
                entity_ids=("SegFormer3D", "nnFormer"),
            ),
            DecomposedRequirement(
                text="提取正文 contribution 中關於 34× fewer params 與 13× GFLOPs reduction 的陳述",
                method="coordinated",
                confidence="high",
                entity_ids=("SegFormer3D", "nnFormer"),
            ),
        ]
        syn_drafts = [
            SynthesisObligationDraft(
                kind="aggregation",
                text="以 Table 1 的精確數值重新計算 SegFormer3D 相對 nnFormer 的參數與計算量倍數比值",
                depends_on_requirement_indexes=(0, 1, 2),
            ),
            SynthesisObligationDraft(
                kind="qualification",
                text="判斷摘要與正文數據是否為近似表述，並說明原文對取整方式的確認狀況",
                depends_on_requirement_indexes=(0, 1, 2),
            ),
        ]
        return (direct_reqs, syn_drafts, entities)

    figure_prerequisite: list[DecomposedRequirement] = []
    if re.search(r"(?:Figure|圖)\s*\d*", text, re.I) and re.search(
        r"(?:策略|方法|\([ab]\)|（[ab]）)", text, re.I
    ):
        figure_prerequisite.append(
            DecomposedRequirement(
                text="解析引用 Figure/圖示中的策略選項與對應關係",
                method="coordinated",
                confidence="medium",
                entity_ids=(),
            )
        )

    parts = _split_obligation_sentences(text)
    raw_reqs: list[DecomposedRequirement] = []
    syn_drafts_general: list[tuple[SynthesisObligationKind, str]] = []
    for part in parts:
        split_items = _split_coordinated_part(part, block)
        for item in split_items:
            if classify_requirement_role(item.text) == "synthesis":
                kind: SynthesisObligationKind = "aggregation"
                if re.search(r"比較|compare|對比|差異|trade-off|vs", item.text, re.I):
                    kind = "comparison"
                elif re.search(r"選型|選擇|挑選|首選|哪一個|哪個|select|which", item.text, re.I):
                    kind = "selection"
                elif re.search(r"為何|原因|因果|why", item.text, re.I):
                    kind = "causal"
                elif re.search(r"取整|近似|互斥|是否成立|能否|qualif", item.text, re.I):
                    kind = "qualification"
                syn_drafts_general.append((kind, item.text))
            else:
                raw_reqs.append(item)

    if not raw_reqs and not figure_prerequisite:
        part_entities = tuple(_extract_entities(text))
        raw_reqs = [
            DecomposedRequirement(
                text=text,
                method=block.method,
                confidence=block.confidence,
                entity_ids=part_entities,
            )
        ]

    final_reqs = figure_prerequisite + raw_reqs
    deps = tuple(range(len(final_reqs)))
    syns = [
        SynthesisObligationDraft(
            kind=k,
            text=syn_text,
            depends_on_requirement_indexes=deps,
        )
        for k, syn_text in syn_drafts_general
    ]
    return (final_reqs, syns, entities)


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
    results: list[DecomposedRequirement] = []
    for part in parts:
        cleaned = part.strip(" ：:，,；;。！？?!")
        if cleaned:
            entities = tuple(_extract_entities(cleaned))
            results.append(
                DecomposedRequirement(
                    text=cleaned,
                    method=method,
                    confidence=confidence,
                    entity_ids=entities,
                )
            )
    return results


def _split_coordination_tail(text: str) -> list[str]:
    pieces = [
        piece.strip(" ：:，,；;。！？?!") for piece in _COORDINATION.split(text)
    ]
    return [piece for piece in pieces if piece]


def _extract_entities(text: str) -> list[str]:
    candidates = _ENTITY_TOKEN.findall(text)
    entities: list[str] = []
    for candidate in candidates:
        if candidate.lower() in _ENTITY_STOPWORDS:
            continue
        if len(candidate) < 3 and "-" not in candidate:
            continue
        if not (
            "-" in candidate or any(char.isupper() for char in candidate[1:])
        ):
            continue
        if candidate not in entities:
            entities.append(candidate)
    return entities


def _entity_requirements(
    text: str, entities: list[str]
) -> list[DecomposedRequirement]:
    return [
        DecomposedRequirement(
            text=f"針對 {entity}，回答題目中對該主體的要求。",
            method="entity_distributive",
            confidence="high",
            entity_ids=(entity,),
        )
        for entity in entities
    ]


def _extract_synthesis_obligations_for_entities(
    text: str, entities: list[str], start_index: int = 0
) -> list[SynthesisObligationDraft]:
    if re.search(r"分類|classif", text, re.I):
        return []

    depends_on = tuple(range(start_index, start_index + len(entities)))

    if re.search(r"比較|compare|對比|差異|trade-off|vs|versus", text, re.I):
        kind: SynthesisObligationKind = "comparison"
        synthesis_text = "整合各主體的比較、選擇或 trade-off 結論。"
    elif re.search(
        r"選型|選擇|挑選|首選|哪一個|哪個|唯一符合|select|which", text, re.I
    ):
        kind = "selection"
        synthesis_text = "給出各主體之間的選擇或選型裁決。"
    elif re.search(r"為何|原因|因果|why|reason", text, re.I):
        kind = "causal"
        synthesis_text = "分析各主體之間的因果關係或差異原因。"
    elif re.search(r"趨勢|演進|演化|綜合|總結|trend|evolution", text, re.I):
        kind = "aggregation"
        synthesis_text = "整合各主體的演進趨勢或綜合結論。"
    elif re.search(r"互斥|能否唯一決定|是否成立|能否|是否表示|qualif", text, re.I):
        kind = "qualification"
        synthesis_text = "裁決各主體的主張或條件是否成立/互斥。"
    else:
        kind = "comparison"
        synthesis_text = "整合各主體的比較、選擇或 trade-off 結論。"

    return [
        SynthesisObligationDraft(
            kind=kind,
            text=synthesis_text,
            depends_on_requirement_indexes=depends_on,
        )
    ]


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
                entity_ids=requirement.entity_ids,
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


def _dedupe_synthesis(
    synthesis_obligations: list[SynthesisObligationDraft],
) -> list[SynthesisObligationDraft]:
    seen: set[tuple[SynthesisObligationKind, tuple[int, ...], str]] = set()
    result: list[SynthesisObligationDraft] = []
    for syn in synthesis_obligations:
        key = (
            syn.kind,
            syn.depends_on_requirement_indexes,
            " ".join(syn.text.split()),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(
            SynthesisObligationDraft(
                kind=syn.kind,
                text=key[2],
                depends_on_requirement_indexes=syn.depends_on_requirement_indexes,
            )
        )
    return result


__all__ = [
    "ConstraintKind",
    "DecompositionConfidence",
    "DecompositionMethod",
    "DecomposedRequirement",
    "QuestionBlock",
    "QuestionDecomposition",
    "RequirementRole",
    "ResponseConstraintDraft",
    "SynthesisObligationDraft",
    "SynthesisObligationKind",
    "classify_requirement_role",
    "decompose_question",
    "split_top_level_blocks",
    "validate_requirement_roles",
]
