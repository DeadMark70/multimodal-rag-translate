"""Pydantic schemas for evaluation APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ThinkingControlType = Literal["none", "budget", "level"]
ThinkingLevel = Literal["minimal", "low", "medium", "high"]


def normalize_test_case_difficulty(value: Any) -> Any:
    """Normalize dataset difficulty labels while preserving missing values."""
    if value is None or not isinstance(value, str):
        return value
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    return normalized or None


class TestCase(BaseModel):
    """Single evaluation test case."""

    id: str = Field(..., min_length=1, max_length=128)
    question: str = Field(..., min_length=1, max_length=5000)
    ground_truth: str = Field(..., min_length=1)
    ground_truth_short: Optional[str] = Field(default=None, min_length=1)
    category: Optional[str] = None
    difficulty: Optional[str] = None
    source_docs: list[str] = Field(default_factory=list)
    requires_multi_doc_reasoning: bool = False
    test_objective: Optional[str] = None
    key_points: list[str] = Field(default_factory=list)
    ragas_focus: list[str] = Field(default_factory=list)
    question_version: Optional[str] = None
    required_modalities: list[str] = Field(default_factory=list)
    atomic_facts: list[dict[str, Any]] = Field(default_factory=list)
    expected_evidence: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("difficulty", mode="before")
    @classmethod
    def _normalize_difficulty(cls, value: Any) -> Any:
        return normalize_test_case_difficulty(value)


class GoldenDataset(BaseModel):
    """Golden dataset format compatible with bergen/golden_dataset.json."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    questions: list[TestCase] = Field(default_factory=list)


class ImportResult(BaseModel):
    """Summary after batch import."""

    imported: int = Field(..., ge=0)
    total: int = Field(..., ge=0)


class DeleteResult(BaseModel):
    """Delete operation summary."""

    deleted_id: str
    total: int = Field(..., ge=0)


class ThinkingCapability(BaseModel):
    """Model-specific thinking control metadata for evaluation UI/runtime."""

    supported: bool = False
    control_type: ThinkingControlType = "none"
    levels: list[ThinkingLevel] = Field(default_factory=list)
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    supports_disable: bool = False
    supports_dynamic: bool = False
    default_level: Optional[ThinkingLevel] = None
    default_budget: Optional[int] = None
    guidance: Optional[str] = None


class ModelConfig(BaseModel):
    """Saved model config preset for evaluation."""

    id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=100)
    model_name: str = Field(..., min_length=1, max_length=200)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1, le=100)
    max_input_tokens: int = Field(8192, ge=1)
    max_output_tokens: int = Field(8192, ge=1)
    thinking_mode: bool = False
    thinking_budget: Optional[int] = Field(default=8192, ge=-1, le=32768)
    thinking_level: Optional[ThinkingLevel] = None
    thinking_include_thoughts: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AvailableModel(BaseModel):
    """Available model metadata from Gemini API discovery."""

    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    input_token_limit: Optional[int] = None
    output_token_limit: Optional[int] = None
    supported_actions: list[str] = Field(default_factory=list)
    thinking: ThinkingCapability = Field(default_factory=ThinkingCapability)
