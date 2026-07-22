"""Deterministic-first selective CRAG tests for Agentic v9 retrieval."""

from __future__ import annotations

import asyncio

from data_base.agentic_v9.schemas import ResolvedSourceScope
from data_base.agentic_v9.selective_crag import (
    classify_retrieval_action,
    resolve_selective_crag,
)


def _scope() -> ResolvedSourceScope:
    return ResolvedSourceScope(authorized_doc_ids=["doc-authorized"])


def test_exact_authorized_locator_hit_passes_without_a_judge() -> None:
    decision = classify_retrieval_action(
        documents=[
            {
                "metadata": {"doc_id": "doc-authorized", "table_id": "Table 2"},
                "text": "Table 2 reports Dice 0.91.",
                "score": 0.96,
            }
        ],
        source_scope=_scope(),
        locator_hints=["Table 2"],
    )

    assert decision.action == "pass"
    assert decision.requires_llm_judge is False


def test_missing_or_wrong_scope_documents_correct_deterministically() -> None:
    missing = classify_retrieval_action(
        documents=[], source_scope=_scope(), locator_hints=["Table 2"]
    )
    wrong_scope = classify_retrieval_action(
        documents=[
            {"metadata": {"doc_id": "untrusted", "table_id": "Table 2"}, "score": 0.99}
        ],
        source_scope=_scope(),
        locator_hints=["Table 2"],
    )

    assert missing.action == "correct"
    assert missing.reason == "no_documents"
    assert wrong_scope.action == "correct"
    assert wrong_scope.reason == "no_authorized_documents"


def test_partial_or_conflicting_authorized_evidence_escalates_to_the_judge() -> None:
    partial = classify_retrieval_action(
        documents=[
            {
                "metadata": {"doc_id": "doc-authorized"},
                "text": "A related result is reported.",
                "score": 0.62,
            }
        ],
        source_scope=_scope(),
        locator_hints=["Table 2"],
    )
    conflicting = classify_retrieval_action(
        documents=[
            {"metadata": {"doc_id": "doc-authorized"}, "score": 0.92, "conflicting": True}
        ],
        source_scope=_scope(),
        locator_hints=[],
    )

    assert partial.action == "llm_judge"
    assert partial.requires_llm_judge is True
    assert conflicting.action == "llm_judge"


def test_low_score_evidence_corrects_conservatively_instead_of_failing_open() -> None:
    decision = classify_retrieval_action(
        documents=[
            {
                "metadata": {"doc_id": "doc-authorized", "table_id": "Table 2"},
                "text": "Table 2 reports Dice 0.91.",
                "score": 0.10,
            }
        ],
        source_scope=_scope(),
        locator_hints=["Table 2"],
    )

    assert decision.action == "correct"
    assert decision.reason == "low_relevance"


def test_judge_failure_or_missing_final_budget_corrects_conservatively() -> None:
    ambiguous = classify_retrieval_action(
        documents=[
            {"metadata": {"doc_id": "doc-authorized"}, "score": 0.62}
        ],
        source_scope=_scope(),
        locator_hints=["Table 2"],
    )

    async def failed_judge() -> bool:
        raise RuntimeError("provider unavailable")

    failed = asyncio.run(
        resolve_selective_crag(
            decision=ambiguous, judge=failed_judge, judge_budget_available=True
        )
    )
    protected = asyncio.run(
        resolve_selective_crag(
            decision=ambiguous, judge=failed_judge, judge_budget_available=False
        )
    )

    assert failed.action == "correct"
    assert failed.reason == "llm_judge_failed"
    assert protected.action == "correct"
    assert protected.reason == "final_budget_protected"
