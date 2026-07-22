"""Bounded, evidence-only repair task construction for Agentic v9."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from data_base.agentic_v9.schemas import AgenticV9Route, QueryContract, RetrievalTask
from data_base.agentic_v9.sufficiency_gate import SufficiencyEvaluation


ROUTE_REPAIR_CAPS: dict[AgenticV9Route, int] = {
    "single_lookup": 0,
    "bounded_compare": 1,
    "exact_structured": 1,
    "multi_document_exact": 2,
    "multi_hop": 1,
    "graph_relational": 1,
}
"""Frozen repair-round caps from the v9 evidence-first design."""

MAX_REPAIR_QUERIES_PER_ROUND = 2


class RepairPlan(BaseModel):
    """Serializable repair decision containing retrieval tasks, never answers."""

    model_config = ConfigDict(extra="forbid")

    repair_round_index: int = Field(ge=1)
    tasks: list[RetrievalTask] = Field(default_factory=list, max_length=2)
    stop_reason: str | None = None


def build_repair_plan(
    *,
    contract: QueryContract,
    sufficiency: SufficiencyEvaluation,
    query_id: str,
    repair_round_index: int,
    final_budget_available: bool,
) -> RepairPlan:
    """Compile at most two source-authorized retrieval repairs for missing slots.

    A repair is admitted only when the caller has retained a final-answer
    envelope.  The query itself intentionally excludes the original user
    question and prior generated text: it is assembled solely from the missing
    slot's entity, description, and locator contract.
    """
    normalized_query_id = query_id.strip()
    if not normalized_query_id:
        raise ValueError("query_id must not be empty")
    if repair_round_index < 1:
        raise ValueError("repair_round_index must be at least 1")
    scope = contract.resolved_source_scope
    if scope is None or not scope.authorized_doc_ids:
        raise ValueError("repair tasks require an authorized source scope")
    if not final_budget_available:
        return RepairPlan(
            repair_round_index=repair_round_index,
            stop_reason="final_budget_protected",
        )

    cap = min(ROUTE_REPAIR_CAPS[contract.route], contract.max_repair_rounds)
    if repair_round_index > cap:
        return RepairPlan(
            repair_round_index=repair_round_index,
            stop_reason="repair_round_cap_reached",
        )

    missing_slot_ids = set(sufficiency.repairable_slot_ids)
    missing_slots = [
        slot for slot in contract.required_slots if slot.required and slot.slot_id in missing_slot_ids
    ]
    if not missing_slots:
        return RepairPlan(
            repair_round_index=repair_round_index,
            stop_reason="no_repairable_slots",
        )

    tasks = [
        RetrievalTask(
            task_id=f"{normalized_query_id}:repair-{repair_round_index}:{slot.slot_id}",
            round_id=f"repair-{repair_round_index}",
            query_id=normalized_query_id,
            query=_repair_query(contract=contract, slot_id=slot.slot_id),
            target_slot_ids=[slot.slot_id],
            source_scope=scope,
            source_group_id=f"repair-{repair_round_index}",
            locator_hints=_unique(slot.locator_hints or contract.locator_hints),
            graph_policy=contract.graph_policy or "never",
            visual_required=contract.visual_required,
        )
        for slot in missing_slots[:MAX_REPAIR_QUERIES_PER_ROUND]
    ]
    return RepairPlan(repair_round_index=repair_round_index, tasks=tasks)


def _repair_query(*, contract: QueryContract, slot_id: str) -> str:
    slot = next(slot for slot in contract.required_slots if slot.slot_id == slot_id)
    entities = slot.entity_ids or contract.entities
    locators = slot.locator_hints or contract.locator_hints
    parts = [*entities, slot.description, *locators]
    query = " ".join(_unique(parts))
    if not query:
        raise ValueError(f"repair slot has no slot, entity, or locator content: {slot_id}")
    return query


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


__all__ = [
    "MAX_REPAIR_QUERIES_PER_ROUND",
    "ROUTE_REPAIR_CAPS",
    "RepairPlan",
    "build_repair_plan",
]
