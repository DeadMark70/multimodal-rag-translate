"""Backward-compatible delegate to the atomic question contract planner."""

from __future__ import annotations

from typing import Any

from data_base.agentic_v9.contract_planner import QuestionContractPlanner
from data_base.agentic_v9.schemas import (
    LlmInvoker,
    QueryContract,
    ResolvedSourceScope,
)


class RoutePlanner:
    """Preserve the v1 planner call shape without duplicating planning logic."""

    def __init__(self, *, llm_invoker: LlmInvoker | None = None) -> None:
        self._planner = QuestionContractPlanner(llm_invoker=llm_invoker)

    async def plan(
        self,
        *,
        question: str,
        resolved_source_scope: ResolvedSourceScope,
        setup_policy: dict[str, Any] | None = None,
    ) -> QueryContract:
        contract = await self._planner.plan(
            question=question,
            authorized_source_names=resolved_source_scope.requested_source_names,
            authorized_source_doc_ids=resolved_source_scope.authorized_doc_ids,
            setup_policy=setup_policy or {},
        )
        return contract.model_copy(
            update={"resolved_source_scope": resolved_source_scope}
        )


async def plan_query_contract(
    *,
    question: str,
    resolved_source_scope: ResolvedSourceScope,
    llm_invoker: LlmInvoker | None = None,
    setup_policy: dict[str, Any] | None = None,
) -> QueryContract:
    return await RoutePlanner(llm_invoker=llm_invoker).plan(
        question=question,
        resolved_source_scope=resolved_source_scope,
        setup_policy=setup_policy,
    )


__all__ = ["RoutePlanner", "plan_query_contract"]
