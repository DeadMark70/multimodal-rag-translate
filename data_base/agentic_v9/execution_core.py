"""Ordered, evidence-first orchestration for the versioned Agentic v9 path.

The core owns only transition order and cardinality assertions.  Retrieval,
budgeting, persistence, provider invocation, and all route policy remain in
typed adapter callables so this module cannot change the legacy v8 path.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, TypeVar

from data_base.agentic_v9.context_packer import PackedEvidenceContext
from data_base.agentic_v9.retrieval_tasks import RetrievalTaskCompiler
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    FinalAnswerResult,
    QueryContract,
    ResolvedSourceScope,
    RetrievalTask,
    SlotResolution,
    TaskRetrievalResult,
    V9ExecutionRequest,
    V9ExecutionResult,
)
from data_base.agentic_v9.sufficiency_gate import SufficiencyEvaluation


_T = TypeVar("_T")
_MaybeAwaitable = _T | Awaitable[_T]


@dataclass(frozen=True, slots=True)
class ConflictStageResult:
    """Final sufficiency/conflict output, including at most one arbitration."""

    sufficiency: SufficiencyEvaluation
    arbitration: Any | None = None
    arbitration_call_count: int = 0


ResolveScopeStage = Callable[[V9ExecutionRequest], _MaybeAwaitable[ResolvedSourceScope]]
PlanContractStage = Callable[
    [V9ExecutionRequest, ResolvedSourceScope], _MaybeAwaitable[QueryContract]
]
RetrieveStage = Callable[
    [tuple[RetrievalTask, ...]], _MaybeAwaitable[Sequence[TaskRetrievalResult]]
]
CandidateStage = Callable[
    [tuple[TaskRetrievalResult, ...], QueryContract],
    _MaybeAwaitable[Sequence[EvidencePacket]],
]
SufficiencyStage = Callable[
    [QueryContract, tuple[EvidencePacket, ...]], SufficiencyEvaluation
]
RepairStage = Callable[
    [QueryContract, SufficiencyEvaluation, str, int], Sequence[RetrievalTask]
]
ProseCuratorStage = Callable[
    [str, QueryContract, tuple[EvidencePacket, ...]],
    _MaybeAwaitable[Sequence[EvidencePacket]],
]
ConflictStage = Callable[
    [QueryContract, tuple[EvidencePacket, ...], SufficiencyEvaluation],
    _MaybeAwaitable[ConflictStageResult],
]
PackStage = Callable[
    [str, QueryContract, tuple[EvidencePacket, ...], SufficiencyEvaluation],
    _MaybeAwaitable[PackedEvidenceContext],
]
FinalStage = Callable[
    [
        str,
        QueryContract,
        PackedEvidenceContext,
        tuple[SlotResolution, ...],
        Any | None,
    ],
    _MaybeAwaitable[FinalAnswerResult],
]
DeterministicPartialStage = Callable[
    [QueryContract, SufficiencyEvaluation], FinalAnswerResult
]


@dataclass(frozen=True, slots=True)
class V9ExecutionStages:
    """Typed adapter boundary for all v9 work performed outside the core."""

    resolve_scope: ResolveScopeStage
    plan_contract: PlanContractStage
    retrieve: RetrieveStage
    deterministic_candidates: CandidateStage
    evaluate_sufficiency: SufficiencyStage
    plan_repair: RepairStage
    prose_curate: ProseCuratorStage
    resolve_conflicts: ConflictStage
    pack: PackStage
    generate_final: FinalStage
    deterministic_partial: DeterministicPartialStage


class V9ExecutionCore:
    """Run the fixed v9 evidence state machine without embedding policy.

    The only loop is bounded repair.  It returns more retrieval tasks, never a
    subtask answer.  Provider-backed prose curation, arbitration, and final
    generation are each represented by one stage and capped here.
    """

    def __init__(
        self,
        *,
        stages: V9ExecutionStages,
        retrieval_task_compiler: RetrievalTaskCompiler | None = None,
    ) -> None:
        self._stages = stages
        self._retrieval_task_compiler = retrieval_task_compiler or RetrievalTaskCompiler()

    async def execute(self, request: V9ExecutionRequest) -> V9ExecutionResult:
        """Execute the ordered evidence-first v9 flow for one request."""
        scope = await _resolve(self._stages.resolve_scope(request))
        contract = await _resolve(self._stages.plan_contract(request, scope))

        task_results: list[TaskRetrievalResult] = []
        evidence_packets: list[EvidencePacket] = []
        initial_tasks = self._initial_tasks(request, contract)
        if initial_tasks:
            await self._retrieve_candidates(
                initial_tasks, contract, task_results, evidence_packets
            )

        # Initial sufficiency owns the only repair decision; repair policy is
        # adapter-supplied, while the loop cap is a contract invariant.
        sufficiency = self._stages.evaluate_sufficiency(
            contract, tuple(evidence_packets)
        )
        repair_round = 0
        while repair_round < contract.max_repair_rounds:
            repair_tasks = tuple(
                self._stages.plan_repair(
                    contract, sufficiency, request.trace_id, repair_round + 1
                )
            )
            if not repair_tasks:
                break
            repair_round += 1
            await self._retrieve_candidates(
                repair_tasks, contract, task_results, evidence_packets
            )
            sufficiency = self._stages.evaluate_sufficiency(
                contract, tuple(evidence_packets)
            )

        # One final prose batch may curate packets; it cannot produce an answer.
        curated_packets = tuple(
            await _resolve(
                self._stages.prose_curate(
                    request.question, contract, tuple(evidence_packets)
                )
            )
        )
        prose_curator_call_count = 1
        assert prose_curator_call_count <= 1

        final_sufficiency = self._stages.evaluate_sufficiency(contract, curated_packets)
        conflict = await _resolve(
            self._stages.resolve_conflicts(contract, curated_packets, final_sufficiency)
        )
        assert 0 <= conflict.arbitration_call_count <= 1

        packed = await _resolve(
            self._stages.pack(
                request.question, contract, curated_packets, conflict.sufficiency
            )
        )
        final_generation_count = 0
        if conflict.sufficiency.report.answerable and _is_packable(packed):
            final_answer = await _resolve(
                self._stages.generate_final(
                    request.question,
                    contract,
                    packed,
                    conflict.sufficiency.slot_resolutions,
                    conflict.arbitration,
                )
            )
            final_generation_count = 1
        else:
            final_answer = self._stages.deterministic_partial(
                contract, conflict.sufficiency
            )

        assert final_generation_count <= 1
        assert final_answer.final_generation_count <= 1
        assert final_generation_count == final_answer.final_generation_count
        subtask_answer_count = 0
        assert subtask_answer_count == 0
        return V9ExecutionResult(
            trace_id=request.trace_id,
            task_results=task_results,
            final_answer=final_answer,
            sufficiency=conflict.sufficiency.report,
            metrics={
                "retrieval_query_count": len(task_results),
                "final_generation_count": final_generation_count,
                "subtask_answer_count": subtask_answer_count,
                "prose_curator_call_count": prose_curator_call_count,
                "arbitration_call_count": conflict.arbitration_call_count,
            },
        )

    def _initial_tasks(
        self, request: V9ExecutionRequest, contract: QueryContract
    ) -> tuple[RetrievalTask, ...]:
        scope = contract.resolved_source_scope
        if scope is None or not scope.authorized_doc_ids:
            return ()
        plan = self._retrieval_task_compiler.compile(
            question=request.question,
            query_id=request.trace_id,
            contract=contract,
        )
        return tuple(plan.tasks)

    async def _retrieve_candidates(
        self,
        tasks: tuple[RetrievalTask, ...],
        contract: QueryContract,
        task_results: list[TaskRetrievalResult],
        evidence_packets: list[EvidencePacket],
    ) -> None:
        results = tuple(await _resolve(self._stages.retrieve(tasks)))
        returned_task_ids = {result.task_id for result in results}
        expected_task_ids = {task.task_id for task in tasks}
        if returned_task_ids.difference(expected_task_ids):
            raise ValueError("retrieval returned a result for an undeclared task")
        task_results.extend(results)
        candidates = await _resolve(
            self._stages.deterministic_candidates(results, contract)
        )
        evidence_packets.extend(candidates)


async def _resolve(value: _MaybeAwaitable[_T]) -> _T:
    """Await async adapters while keeping deterministic stages lightweight."""
    if isawaitable(value):
        return await value
    return value


def _is_packable(packed: object) -> bool:
    """Accept the typed pack projection without coupling the core to policy."""
    return bool(getattr(packed, "is_packable", False))


__all__ = [
    "ConflictStageResult",
    "V9ExecutionCore",
    "V9ExecutionStages",
]
