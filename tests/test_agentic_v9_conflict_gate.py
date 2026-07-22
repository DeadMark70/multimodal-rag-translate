"""Scope-aware evidence conflict checks for Agentic v9."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from data_base.agentic_v9.conflict_gate import (
    arbitrate_persisted_unresolved_candidates,
    detect_conflict_candidates,
)
from data_base.agentic_v9.schemas import (
    ConflictCandidate,
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    SourceLocator,
)


def _packet(
    evidence_id: str,
    value: str,
    *,
    slot_id: str = "score",
    dataset: str | None = "BraTS",
    split: str | None = "test",
    metric: str | None = "Dice",
    model_variant: str | None = "Model A",
    training_protocol: str | None = "Protocol 1",
    prompt_setting: str | None = "Prompt 1",
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=[slot_id],
        statement=f"The reported score is {value}.",
        support_type="direct",
        source=EvidenceSource(doc_id=f"doc-{evidence_id}", chunk_id=evidence_id),
        scope=EvidenceScope(
            dataset=dataset,
            split=split,
            metric=metric,
            model_variant=model_variant,
            training_protocol=training_protocol,
            prompt_setting=prompt_setting,
        ),
        locator=SourceLocator(pdf_page_index=1, table_id="table-1"),
        raw_value=Decimal(value),
        normalized_value=Decimal(value),
    )


def test_same_complete_scope_with_incompatible_values_is_an_unresolved_conflict() -> (
    None
):
    candidates = detect_conflict_candidates(
        [_packet("E1", "0.90"), _packet("E2", "0.82")]
    )

    assert len(candidates) == 1
    assert candidates[0].slot_id == "score"
    assert candidates[0].evidence_ids == ["E1", "E2"]
    assert candidates[0].scope_match == "same"
    assert candidates[0].unresolved is True


def test_different_dataset_is_not_a_conflict() -> None:
    candidates = detect_conflict_candidates(
        [_packet("E1", "0.90"), _packet("E2", "0.82", dataset="AMOS")]
    )

    assert candidates == []


def test_missing_scope_dimension_is_explicitly_scope_ambiguous() -> None:
    candidates = detect_conflict_candidates(
        [_packet("E1", "0.90"), _packet("E2", "0.82", prompt_setting=None)]
    )

    assert len(candidates) == 1
    assert candidates[0].scope_match == "unknown"
    assert "scope_ambiguous" in candidates[0].reason


def test_equal_normalized_values_are_not_a_conflict() -> None:
    candidates = detect_conflict_candidates(
        [_packet("E1", "0.90"), _packet("E2", "0.90")]
    )

    assert candidates == []


class _RecordingInvoker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def invoke(
        self,
        *,
        phase: str,
        purpose: str,
        messages: list[dict[str, object]],
    ) -> object:
        self.calls.append({"phase": phase, "purpose": purpose, "messages": messages})
        return SimpleNamespace(content="qualified")


@pytest.mark.asyncio
async def test_only_persisted_unresolved_candidates_trigger_one_arbitration_call() -> (
    None
):
    invoker = _RecordingInvoker()
    packets = [_packet("E1", "0.90"), _packet("E2", "0.82")]
    persisted = detect_conflict_candidates(packets)
    non_persisted = ConflictCandidate(
        candidate_id="unpersisted",
        slot_id="score",
        evidence_ids=["E1", "E2"],
        scope_match="same",
        reason="same scope",
    )

    result = await arbitrate_persisted_unresolved_candidates(
        persisted_candidates=[*persisted, non_persisted],
        persisted_candidate_ids={persisted[0].candidate_id},
        evidence_packets=packets,
        llm_invoker=invoker,
    )

    assert result.content == "qualified"
    assert len(invoker.calls) == 1
    assert invoker.calls[0]["phase"] == "conflict_arbitration"
    assert invoker.calls[0]["purpose"] == "conflict_arbitration"
    assert "E1" in str(invoker.calls[0]["messages"])
    assert "E2" in str(invoker.calls[0]["messages"])


@pytest.mark.asyncio
async def test_unpersisted_or_resolved_candidates_never_invoke_arbitration() -> None:
    invoker = _RecordingInvoker()
    packets = [_packet("E1", "0.90"), _packet("E2", "0.82")]
    candidate = detect_conflict_candidates(packets)[0]
    resolved = candidate.model_copy(update={"unresolved": False})

    result = await arbitrate_persisted_unresolved_candidates(
        persisted_candidates=[resolved],
        persisted_candidate_ids={candidate.candidate_id},
        evidence_packets=packets,
        llm_invoker=invoker,
    )

    assert result is None
    assert invoker.calls == []
