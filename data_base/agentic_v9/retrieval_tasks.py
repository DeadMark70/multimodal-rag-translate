"""Compile bounded, evidence-only retrieval tasks from a v9 query contract."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from data_base.agentic_v9.schemas import (
    QueryContract,
    ResolvedSourceScope,
    RetrievalTask,
)


class RetrievalTaskPlan(BaseModel):
    """The typed retrieval-only work plan for a single v9 query."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)
    tasks: list[RetrievalTask] = Field(min_length=1)


class RetrievalTaskCompiler:
    """Create bounded retrieval work without generating an answer.

    The compiler does not infer source content.  It only partitions the
    resolver-owned authorized scope and makes round-two task dependencies
    explicit so downstream retrieval and repair remain bounded.
    """

    def compile(
        self,
        *,
        question: str,
        query_id: str,
        contract: QueryContract,
    ) -> RetrievalTaskPlan:
        """Compile one source-authorized ``QueryContract`` into typed tasks."""
        normalized_question = question.strip()
        normalized_query_id = query_id.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if not normalized_query_id:
            raise ValueError("query_id must not be empty")

        scope = contract.resolved_source_scope
        if scope is None or not scope.authorized_doc_ids:
            raise ValueError("retrieval tasks require an authorized source scope")

        if contract.route == "bounded_compare":
            tasks = self._compile_bounded_compare(
                question=normalized_question,
                query_id=normalized_query_id,
                contract=contract,
                scope=scope,
            )
        elif contract.route == "multi_document_exact":
            tasks = self._compile_source_groups(
                question=normalized_question,
                query_id=normalized_query_id,
                contract=contract,
                scope=scope,
            )
        elif contract.route == "multi_hop":
            tasks = self._compile_multi_hop(
                question=normalized_question,
                query_id=normalized_query_id,
                contract=contract,
                scope=scope,
            )
        else:
            tasks = [
                self._task(
                    task_id=f"{normalized_query_id}:round-1:source-group-1",
                    round_id="round-1",
                    query_id=normalized_query_id,
                    query=normalized_question,
                    target_slot_ids=self._slot_ids(contract),
                    scope=scope,
                    source_group_id="source-group-1",
                    contract=contract,
                )
            ]

        highest_round = max(_round_number(task.round_id) for task in tasks)
        if highest_round > contract.max_retrieval_rounds:
            raise ValueError("retrieval task plan exceeds the contract round budget")
        return RetrievalTaskPlan(
            query_id=normalized_query_id,
            contract_version=contract.contract_version,
            tasks=tasks,
        )

    def _compile_bounded_compare(
        self,
        *,
        question: str,
        query_id: str,
        contract: QueryContract,
        scope: ResolvedSourceScope,
    ) -> list[RetrievalTask]:
        entities = _unique(contract.entities)
        if len(entities) < 2:
            return [
                self._task(
                    task_id=f"{query_id}:round-1:source-group-1",
                    round_id="round-1",
                    query_id=query_id,
                    query=question,
                    target_slot_ids=self._slot_ids(contract),
                    scope=scope,
                    source_group_id="source-group-1",
                    contract=contract,
                )
            ]

        primary_slots, qualification_slots = self._partition_slots(contract)
        first, second = entities[:2]
        initial_tasks = [
            self._task(
                task_id=f"{query_id}:round-1:compare-a",
                round_id="round-1",
                query_id=query_id,
                query=f"{first}: {question}",
                target_slot_ids=primary_slots,
                scope=scope,
                source_group_id="compare-a",
                contract=contract,
            ),
            self._task(
                task_id=f"{query_id}:round-1:compare-b",
                round_id="round-1",
                query_id=query_id,
                query=f"{second}: {question}",
                target_slot_ids=primary_slots,
                scope=scope,
                source_group_id="compare-b",
                contract=contract,
            ),
        ]
        return [
            *initial_tasks,
            self._task(
                task_id=f"{query_id}:round-2:comparison-qualification",
                round_id="round-2",
                query_id=query_id,
                query=f"Compare bounded source scope for {first} and {second}: {question}",
                target_slot_ids=qualification_slots,
                scope=scope,
                source_group_id="comparison-qualification",
                contract=contract,
                depends_on_task_ids=[task.task_id for task in initial_tasks],
            ),
        ]

    def _compile_source_groups(
        self,
        *,
        question: str,
        query_id: str,
        contract: QueryContract,
        scope: ResolvedSourceScope,
    ) -> list[RetrievalTask]:
        primary_slots, qualification_slots = self._partition_slots(contract)
        initial_tasks: list[RetrievalTask] = []
        for index, doc_id in enumerate(sorted(scope.authorized_doc_ids), start=1):
            source_group_id = f"source-group-{index}"
            initial_tasks.append(
                self._task(
                    task_id=f"{query_id}:round-1:{source_group_id}",
                    round_id="round-1",
                    query_id=query_id,
                    query=f"{doc_id}: {question}",
                    target_slot_ids=primary_slots,
                    scope=_scope_for_doc(scope, doc_id),
                    source_group_id=source_group_id,
                    contract=contract,
                )
            )
        dependent_group_id = f"source-group-{len(initial_tasks) + 1}"
        return [
            *initial_tasks,
            self._task(
                task_id=f"{query_id}:round-2:{dependent_group_id}",
                round_id="round-2",
                query_id=query_id,
                query=f"Resolve cross-source qualifications: {question}",
                target_slot_ids=qualification_slots,
                scope=scope,
                source_group_id=dependent_group_id,
                contract=contract,
                depends_on_task_ids=[task.task_id for task in initial_tasks],
            ),
        ]

    def _compile_multi_hop(
        self,
        *,
        question: str,
        query_id: str,
        contract: QueryContract,
        scope: ResolvedSourceScope,
    ) -> list[RetrievalTask]:
        primary_slots, qualification_slots = self._partition_slots(contract)
        entities = _unique(contract.entities) or ["source-group-1"]
        initial_tasks = [
            self._task(
                task_id=f"{query_id}:round-1:hop-{index}",
                round_id="round-1",
                query_id=query_id,
                query=f"{entity}: {question}",
                target_slot_ids=primary_slots,
                scope=scope,
                source_group_id=f"hop-{index}",
                contract=contract,
            )
            for index, entity in enumerate(entities, start=1)
        ]
        return [
            *initial_tasks,
            self._task(
                task_id=f"{query_id}:round-2:hop-dependencies",
                round_id="round-2",
                query_id=query_id,
                query=f"Resolve source-bound relationships and qualifications: {question}",
                target_slot_ids=qualification_slots,
                scope=scope,
                source_group_id="hop-dependencies",
                contract=contract,
                depends_on_task_ids=[task.task_id for task in initial_tasks],
            ),
        ]

    def _task(
        self,
        *,
        task_id: str,
        round_id: str,
        query_id: str,
        query: str,
        target_slot_ids: list[str],
        scope: ResolvedSourceScope,
        source_group_id: str,
        contract: QueryContract,
        depends_on_task_ids: list[str] | None = None,
    ) -> RetrievalTask:
        return RetrievalTask(
            task_id=task_id,
            round_id=round_id,
            query_id=query_id,
            query=query,
            target_slot_ids=target_slot_ids,
            source_scope=scope,
            source_group_id=source_group_id,
            locator_hints=contract.locator_hints,
            graph_policy=contract.graph_policy or "never",
            visual_required=contract.visual_required,
            depends_on_task_ids=depends_on_task_ids or [],
        )

    def _slot_ids(self, contract: QueryContract) -> list[str]:
        return [slot.slot_id for slot in contract.required_slots] or ["slot-1"]

    def _partition_slots(self, contract: QueryContract) -> tuple[list[str], list[str]]:
        slot_ids = self._slot_ids(contract)
        return [slot_ids[0]], slot_ids[1:] or [slot_ids[0]]


def _scope_for_doc(scope: ResolvedSourceScope, doc_id: str) -> ResolvedSourceScope:
    """Narrow a resolver-owned scope without adding a source identifier."""
    return ResolvedSourceScope(
        requested_doc_ids=[doc_id]
        if doc_id in scope.requested_doc_ids
        else [],
        requested_source_names=[],
        resolved_doc_ids=[doc_id] if doc_id in scope.resolved_doc_ids else [],
        authorized_doc_ids=[doc_id],
    )


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _round_number(round_id: str) -> int:
    try:
        return int(round_id.removeprefix("round-"))
    except ValueError as error:
        raise ValueError(f"invalid retrieval round ID: {round_id}") from error


def compile_retrieval_tasks(
    *, question: str, query_id: str, contract: QueryContract
) -> RetrievalTaskPlan:
    """Convenience boundary for one deterministic retrieval task compilation."""
    return RetrievalTaskCompiler().compile(
        question=question,
        query_id=query_id,
        contract=contract,
    )


__all__ = [
    "RetrievalTaskCompiler",
    "RetrievalTaskPlan",
    "compile_retrieval_tasks",
]
