"""Bounded, evidence-only repair task construction for Agentic v9."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from data_base.agentic_v9.schemas import (
    AgenticV9Route,
    ComparisonSubject,
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
    RetrievalTask,
)
from data_base.agentic_v9.slot_constraints import (
    authorized_doc_ids_for_slot,
    canonical_locator_set,
    canonical_term_set,
    display_locator_hints,
)
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
MAX_REPAIR_ROUNDS = 2


class RepairPlan(BaseModel):
    """Serializable repair decision containing retrieval tasks, never answers."""

    model_config = ConfigDict(extra="forbid")

    repair_round_index: int = Field(ge=1)
    tasks: list[RetrievalTask] = Field(default_factory=list, max_length=2)
    resulting_evidence_ids: list[str] = Field(default_factory=list)
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

    if contract.comparison_plan is not None:
        return _build_comparison_repair_plan(
            contract=contract,
            sufficiency=sufficiency,
            query_id=normalized_query_id,
            repair_round_index=repair_round_index,
            scope=scope,
        )

    cap = min(
        ROUTE_REPAIR_CAPS[contract.route],
        contract.max_repair_rounds,
        MAX_REPAIR_ROUNDS,
    )
    if repair_round_index > cap:
        return RepairPlan(
            repair_round_index=repair_round_index,
            stop_reason="repair_round_cap_reached",
        )

    not_found_slot_ids = {
        resolution.slot_id
        for resolution in sufficiency.slot_resolutions
        if resolution.status == "not_found"
    }
    missing_slot_ids = set(sufficiency.repairable_slot_ids).intersection(
        not_found_slot_ids
    )
    missing_slots = [
        slot for slot in contract.required_slots if slot.required and slot.slot_id in missing_slot_ids
    ]
    if not missing_slots:
        return RepairPlan(
            repair_round_index=repair_round_index,
            stop_reason="no_repairable_slots",
        )

    grouped_slots: dict[
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
        list[RequiredSlot],
    ] = {}
    for slot in missing_slots:
        authorized_doc_ids = authorized_doc_ids_for_slot(slot, scope)
        if not authorized_doc_ids:
            continue
        locators = _unique(slot.locator_hints or contract.locator_hints)
        terms = _unique(slot.entity_ids or contract.entities)
        key = (
            tuple(authorized_doc_ids),
            canonical_locator_set(locators),
            canonical_term_set(terms),
        )
        grouped_slots.setdefault(key, []).append(slot)

    tasks: list[RetrievalTask] = []
    for index, ((doc_ids, _locators, _terms), slots) in enumerate(
        grouped_slots.items(), start=1
    ):
        if len(tasks) >= MAX_REPAIR_QUERIES_PER_ROUND:
            break
        source_group_id = f"repair-{repair_round_index}:source-group-{index}"
        tasks.append(
            RetrievalTask(
                task_id=f"{normalized_query_id}:{source_group_id}",
                round_id=f"repair-{repair_round_index}",
                query_id=normalized_query_id,
                query=_repair_query_for_slots(contract=contract, slots=slots),
                target_slot_ids=[slot.slot_id for slot in slots],
                source_scope=_scope_for_docs(scope, list(doc_ids)),
                source_group_id=source_group_id,
                locator_hints=display_locator_hints(
                    hint
                    for slot in slots
                    for hint in (slot.locator_hints or contract.locator_hints)
                ),
                graph_policy=contract.graph_policy or "never",
                visual_required=any(
                    slot.visual_policy == "required" for slot in slots
                ),
            )
        )
    if not tasks:
        return RepairPlan(
            repair_round_index=repair_round_index,
            stop_reason="no_authorized_repair_groups",
        )
    return RepairPlan(repair_round_index=repair_round_index, tasks=tasks)


def _build_comparison_repair_plan(
    *,
    contract: QueryContract,
    sufficiency: SufficiencyEvaluation,
    query_id: str,
    repair_round_index: int,
    scope: ResolvedSourceScope,
) -> RepairPlan:
    """Target missing comparison subjects in one deterministic repair round."""
    if repair_round_index > 1 or contract.max_repair_rounds < 1:
        return RepairPlan(
            repair_round_index=repair_round_index,
            stop_reason="repair_round_cap_reached",
        )

    not_found_slot_ids = {
        resolution.slot_id
        for resolution in sufficiency.slot_resolutions
        if resolution.status == "not_found"
    }
    missing_slot_ids = set(sufficiency.repairable_slot_ids).intersection(
        not_found_slot_ids
    )
    slots_by_id = {
        slot.slot_id: slot
        for slot in contract.required_slots
        if slot.required and slot.slot_id in missing_slot_ids
    }
    subjects_by_slot_id = {
        f"comparison-subject:{subject.subject_id}": subject
        for subject in contract.comparison_plan.subjects
    }

    tasks: list[RetrievalTask] = []
    for slot_id, subject in subjects_by_slot_id.items():
        slot = slots_by_id.get(slot_id)
        if slot is None:
            continue
        source_group_id = (
            f"repair-{repair_round_index}:comparison:{subject.subject_id}"
        )
        tasks.append(
            RetrievalTask(
                task_id=f"{query_id}:{source_group_id}",
                round_id=f"repair-{repair_round_index}",
                query_id=query_id,
                query=_comparison_repair_query(
                    contract=contract,
                    slot=slot,
                    subject=subject,
                ),
                target_slot_ids=[slot.slot_id],
                source_scope=scope.model_copy(deep=True),
                source_group_id=source_group_id,
                subject_id=subject.subject_id,
                locator_hints=display_locator_hints(
                    slot.locator_hints or contract.locator_hints
                ),
                graph_policy=contract.graph_policy or "never",
                visual_required=(
                    contract.visual_required or slot.visual_policy == "required"
                ),
            )
        )
        # Comparison repair is one bounded corrective query for the first
        # missing subject in planner order. Remaining subjects stay explicit
        # in sufficiency and cap the final response at qualified_partial.
        break

    if not tasks:
        return RepairPlan(
            repair_round_index=repair_round_index,
            stop_reason="no_repairable_slots",
        )
    return RepairPlan(repair_round_index=repair_round_index, tasks=tasks)


def _comparison_repair_query(
    *,
    contract: QueryContract,
    slot: RequiredSlot,
    subject: ComparisonSubject,
) -> str:
    plan = contract.comparison_plan
    assert plan is not None
    return " ".join(
        _unique(
            [
                subject.display_name,
                *subject.aliases,
                subject.retrieval_query,
                *plan.dimensions,
                slot.description,
                *slot.locator_hints,
            ]
        )
    )


def _repair_query(*, contract: QueryContract, slot_id: str) -> str:
    slot = next(slot for slot in contract.required_slots if slot.slot_id == slot_id)
    return _repair_query_for_slots(contract=contract, slots=[slot])


def _repair_query_for_slots(
    *, contract: QueryContract, slots: list[RequiredSlot]
) -> str:
    parts: list[str] = []
    for slot in slots:
        parts.extend(slot.source_name_hints)
        parts.extend(slot.entity_ids or contract.entities)
        parts.append(slot.description)
        parts.extend(slot.locator_hints or contract.locator_hints)
    query = " ".join(_unique(parts))
    if not query:
        slot_ids = ", ".join(slot.slot_id for slot in slots)
        raise ValueError(
            f"repair slots have no slot, entity, or locator content: {slot_ids}"
        )
    return query


def _scope_for_docs(
    scope: ResolvedSourceScope, doc_ids: list[str]
) -> ResolvedSourceScope:
    authorized = [doc_id for doc_id in scope.authorized_doc_ids if doc_id in doc_ids]
    authorized_set = set(authorized)
    source_mapping = {
        name: [
            doc_id
            for doc_id in mapped_doc_ids
            if doc_id in authorized_set
        ]
        for name, mapped_doc_ids in scope.source_name_to_doc_ids.items()
        if any(doc_id in authorized_set for doc_id in mapped_doc_ids)
    }
    return ResolvedSourceScope(
        requested_doc_ids=[
            doc_id for doc_id in scope.requested_doc_ids if doc_id in authorized_set
        ],
        requested_source_names=[
            name for name in scope.requested_source_names if name in source_mapping
        ],
        resolved_doc_ids=[
            doc_id for doc_id in scope.resolved_doc_ids if doc_id in authorized_set
        ],
        authorized_doc_ids=authorized,
        source_name_to_doc_ids=source_mapping,
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


__all__ = [
    "MAX_REPAIR_QUERIES_PER_ROUND",
    "MAX_REPAIR_ROUNDS",
    "ROUTE_REPAIR_CAPS",
    "RepairPlan",
    "build_repair_plan",
]
