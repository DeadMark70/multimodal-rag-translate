"""Conservative, calibrated prompt estimates for Agentic v9."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from data_base.agentic_v9.schemas import EvidencePacket


_TOKEN_UNITS = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|"
    r"[A-Za-z]+(?:['-][A-Za-z]+)*|"
    r"\d+(?:[.,]\d+)*|"
    r"[^\s]"
)
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")


@dataclass(frozen=True, slots=True)
class PromptTokenEstimate:
    """A complete final-prompt estimate, including its safety allowance."""

    instruction: int = 0
    question: int = 0
    contract: int = 0
    history: int = 0
    evidence: int = 0
    image: int = 0
    schema: int = 0
    safety_margin: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        components = (
            self.instruction,
            self.question,
            self.contract,
            self.history,
            self.evidence,
            self.image,
            self.schema,
            self.safety_margin,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in components
        ):
            raise ValueError("prompt token estimates must be non-negative integers")
        calculated = sum(components)
        if self.total_tokens not in (0, calculated):
            raise ValueError("total_tokens must equal all prompt components")
        object.__setattr__(self, "total_tokens", calculated)

    @property
    def fixed_overhead_tokens(self) -> int:
        """Prompt cost which evidence packing must reserve before selection."""
        return self.total_tokens - self.evidence

    def with_evidence(self, evidence_tokens: int) -> "PromptTokenEstimate":
        """Return the same prompt shape with a new whole-packet evidence cost."""
        return PromptTokenEstimate(
            instruction=self.instruction,
            question=self.question,
            contract=self.contract,
            history=self.history,
            evidence=evidence_tokens,
            image=self.image,
            schema=self.schema,
            safety_margin=self.safety_margin,
        )


@dataclass(frozen=True, slots=True)
class ProviderInputError:
    """One persisted comparison between a local estimate and provider usage."""

    estimated_input_tokens: int
    provider_input_tokens: int
    error_tokens: int
    error_ratio: float


class TokenEstimator:
    """Conservative final-prompt estimates with provider-error calibration.

    CJK characters, English words, numbers, LaTex and structural punctuation
    have distinct costs. Provider measurements are retained so the next prompt
    reserves observed under-count; repeated excessive under-counts fail closed.
    """

    def __init__(
        self,
        *,
        base_safety_margin_tokens: int = 8,
        excessive_error_ratio: float = 0.20,
        fail_closed_after_excessive_errors: int = 3,
    ) -> None:
        if base_safety_margin_tokens < 0:
            raise ValueError("base_safety_margin_tokens must be non-negative")
        if excessive_error_ratio <= 0:
            raise ValueError("excessive_error_ratio must be positive")
        if fail_closed_after_excessive_errors < 1:
            raise ValueError("fail_closed_after_excessive_errors must be positive")
        self._base_safety_margin_tokens = base_safety_margin_tokens
        self._excessive_error_ratio = excessive_error_ratio
        self._fail_closed_after_excessive_errors = fail_closed_after_excessive_errors
        self._provider_input_errors: list[ProviderInputError] = []
        self._calibrated_safety_margin_tokens = 0
        self._consecutive_excessive_errors = 0

    @property
    def provider_input_errors(self) -> tuple[ProviderInputError, ...]:
        """Immutable calibration observations suitable for trace persistence."""
        return tuple(self._provider_input_errors)

    @property
    def must_fail_closed(self) -> bool:
        """Whether calibration has crossed the configured safety threshold."""
        return self._consecutive_excessive_errors >= self._fail_closed_after_excessive_errors

    @property
    def safety_margin_tokens(self) -> int:
        """Fixed safety allowance plus the largest observed under-count."""
        return self._base_safety_margin_tokens + self._calibrated_safety_margin_tokens

    def record_provider_input_tokens(
        self, *, estimated_input_tokens: int, provider_input_tokens: int
    ) -> ProviderInputError:
        """Persist one provider measurement and update the safety calibration."""
        for name, value in (
            ("estimated_input_tokens", estimated_input_tokens),
            ("provider_input_tokens", provider_input_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        error_tokens = provider_input_tokens - estimated_input_tokens
        error_ratio = max(error_tokens, 0) / max(estimated_input_tokens, 1)
        observation = ProviderInputError(
            estimated_input_tokens=estimated_input_tokens,
            provider_input_tokens=provider_input_tokens,
            error_tokens=error_tokens,
            error_ratio=error_ratio,
        )
        self._provider_input_errors.append(observation)
        if error_ratio > 0:
            self._calibrated_safety_margin_tokens = max(
                self._calibrated_safety_margin_tokens,
                math.ceil(error_tokens * 1.25),
            )
        if error_ratio > self._excessive_error_ratio:
            self._consecutive_excessive_errors += 1
        else:
            self._consecutive_excessive_errors = 0
        return observation

    def calibration_state(self) -> dict[str, Any]:
        """Return serializable calibration state for run/event persistence."""
        return {
            "base_safety_margin_tokens": self._base_safety_margin_tokens,
            "excessive_error_ratio": self._excessive_error_ratio,
            "fail_closed_after_excessive_errors": self._fail_closed_after_excessive_errors,
            "calibrated_safety_margin_tokens": self._calibrated_safety_margin_tokens,
            "consecutive_excessive_errors": self._consecutive_excessive_errors,
            "provider_input_errors": [
                {
                    "estimated_input_tokens": item.estimated_input_tokens,
                    "provider_input_tokens": item.provider_input_tokens,
                    "error_tokens": item.error_tokens,
                    "error_ratio": item.error_ratio,
                }
                for item in self._provider_input_errors
            ],
        }

    def estimate_text(self, text: str) -> int:
        """Estimate mixed Chinese, English, number and LaTex prompt text."""
        if not text:
            return 0
        total = 0
        for unit in _TOKEN_UNITS.findall(text):
            if "\u3400" <= unit <= "\u9fff" or "\uf900" <= unit <= "\ufaff":
                total += 2
            elif unit[0].isalpha():
                total += 1 + math.ceil(len(unit) / 3)
            elif unit[0].isdigit():
                total += 1 + math.ceil(len(unit) / 2)
            else:
                total += 1
        total += 2 * len(_LATEX_COMMAND.findall(text))
        total += sum(text.count(mark) for mark in ("$", "^", "_"))
        return total

    def estimate_json(self, value: Any) -> int:
        """Estimate JSON with additional structural punctuation allowance."""
        if isinstance(value, str):
            rendered = value
        else:
            try:
                rendered = json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            except (TypeError, ValueError):
                rendered = repr(value)
        structure = sum(rendered.count(mark) for mark in "{}[],:\"")
        return self.estimate_text(rendered) + structure

    def estimate_table(self, table: str | Sequence[Sequence[object]]) -> int:
        """Estimate complete table cells and rows without splitting them."""
        if isinstance(table, str):
            rendered = table
            rows = [row for row in table.splitlines() if row.strip()]
            cells = sum(
                len([cell for cell in row.split("|") if cell.strip()]) for row in rows
            )
        else:
            rows = list(table)
            rendered = "\n".join(" | ".join(str(cell) for cell in row) for row in rows)
            cells = sum(len(row) for row in rows)
        return self.estimate_text(rendered) + (2 * len(rows)) + (2 * cells)

    def estimate_image(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        encoded_bytes: int | None = None,
    ) -> int:
        """Estimate image prompt cost from dimensions and encoded size."""
        for name, value in (
            ("width", width),
            ("height", height),
            ("encoded_bytes", encoded_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        pixels = (width or 256) * (height or 256)
        byte_cost = math.ceil((encoded_bytes or 0) / 512)
        return max(32, math.ceil(pixels / 4096) + byte_cost)

    def estimate_prompt(
        self,
        *,
        instruction: str = "",
        question: str = "",
        contract: object | None = None,
        history: Sequence[object] | None = None,
        evidence_tokens: int = 0,
        image_tokens: int = 0,
        schema: object | None = None,
        safety_margin_tokens: int | None = None,
    ) -> PromptTokenEstimate:
        """Return every fixed and variable component of a final prompt."""
        if evidence_tokens < 0 or image_tokens < 0:
            raise ValueError("evidence_tokens and image_tokens must be non-negative")
        instruction_tokens = self.estimate_text(instruction)
        question_tokens = self.estimate_text(question)
        contract_tokens = 0 if contract is None else self.estimate_json(_jsonable(contract))
        history_tokens = sum(
            self.estimate_json(_jsonable(item)) + 2 for item in (history or ())
        )
        schema_tokens = 0 if schema is None else self.estimate_json(_jsonable(schema))
        margin = self._base_safety_margin_tokens
        if safety_margin_tokens is not None:
            if isinstance(safety_margin_tokens, bool) or safety_margin_tokens < 0:
                raise ValueError("safety_margin_tokens must be a non-negative integer")
            margin = safety_margin_tokens
        return PromptTokenEstimate(
            instruction=instruction_tokens,
            question=question_tokens,
            contract=contract_tokens,
            history=history_tokens,
            evidence=evidence_tokens,
            image=image_tokens,
            schema=schema_tokens,
            safety_margin=margin + self._calibrated_safety_margin_tokens,
        )

    def estimate_packet(self, packet: EvidencePacket) -> int:
        """Return the estimate for one indivisible rendered evidence packet."""
        return self.estimate_text(render_evidence_packet(packet))


def _jsonable(value: object) -> object:
    """Normalize pydantic prompt contracts without importing their concrete types."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


def render_evidence_packet(packet: EvidencePacket) -> str:
    """Render provenance with the statement as one indivisible evidence unit."""
    slots = ",".join(packet.slot_ids)
    locator = _render_locator(packet)
    return (
        f"[{packet.evidence_id}] slots={slots} source={packet.source.doc_id} "
        f"locator={locator}\n{packet.statement}"
    )


def _render_locator(packet: EvidencePacket) -> str:
    locator = packet.locator
    parts: list[str] = []
    if locator.pdf_page_index is not None:
        parts.append(f"pdf_page={locator.pdf_page_index}")
    if locator.printed_page_label:
        parts.append(f"page={locator.printed_page_label}")
    if locator.section:
        parts.append(f"section={locator.section}")
    if locator.table_id:
        parts.append(f"table={locator.table_id}")
    if locator.figure_id:
        parts.append(f"figure={locator.figure_id}")
    if locator.bbox:
        parts.append("bbox=" + ",".join(str(value) for value in locator.bbox))
    return ";".join(parts)


def estimate_text_tokens(text: str) -> int:
    """Convenience entry point for callers that do not need an estimator object."""
    return TokenEstimator().estimate_text(text)


def estimate_evidence_packet_tokens(packet: EvidencePacket) -> int:
    """Convenience entry point for one atomic packet estimate."""
    return TokenEstimator().estimate_packet(packet)
