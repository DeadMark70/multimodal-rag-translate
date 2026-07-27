"""Compatibility gate for the last known-good pre-atomic Agentic v9 planner."""

from __future__ import annotations

import pytest

from data_base.agentic_v9.route_planner import plan_query_contract
from data_base.agentic_v9.schemas import ResolvedSourceScope


@pytest.mark.asyncio
async def test_v9_uses_bounded_legacy_route_slots_without_atomic_hard_gates() -> None:
    scope = ResolvedSourceScope(
        requested_source_names=["Alpha.pdf", "Beta.pdf"],
        resolved_doc_ids=["doc-alpha", "doc-beta"],
        authorized_doc_ids=["doc-alpha", "doc-beta"],
        source_name_to_doc_ids={
            "Alpha.pdf": ["doc-alpha"],
            "Beta.pdf": ["doc-beta"],
        },
    )

    contract = await plan_query_contract(
        question="Compare Model-A versus Model-B using the two authorized papers.",
        resolved_source_scope=scope,
    )

    assert contract.route == "bounded_compare"
    assert [slot.slot_id for slot in contract.required_slots] == [
        "slot-1",
        "slot-2",
    ]
    assert contract.resolved_source_scope == scope
    assert contract.strategy_tier == "deterministic"
