"""State-machine contracts for the bounded Agentic v9 execution core."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from data_base.agentic_v9.execution_core import (
    ConflictStageResult,
    V9ExecutionCore,
    V9ExecutionStages,
)
from data_base.agentic_v9.execution_policy import (
    ExecutionDeadline,
    ExecutionCancellation,
    V9ExecutionPolicyRuntime,
)
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    ExecutionPolicy,
    FinalAnswerResult,
    QueryContract,
    RagRetrievalResult,
    RequiredSlot,
    ResolvedSourceScope,
    SlotResolution,
    SourceLocator,
    SufficiencyReport,
    TaskRetrievalResult,
    V9ExecutionRequest,
    V9RuntimeContext,
)
from data_base.agentic_v9.sufficiency_gate import SufficiencyEvaluation


def _request() -> V9ExecutionRequest:
    return V9ExecutionRequest(question="What is the score?", trace_id="trace-1")


def _contract() -> QueryContract:
    return QueryContract(
        route="exact_structured",
        intent="Report the source-bound score.",
        required_slots=[RequiredSlot(slot_id="score", description="reported score")],
        max_retrieval_rounds=1,
        max_repair_rounds=1,
        resolved_source_scope=ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
    )


def _packet() -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id="evidence-1",
        task_id="query-1:round-1:source-group-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["score"],
        statement="The source reports 0.91.",
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id="chunk-1"),
        scope=EvidenceScope(metric="score"),
        locator=SourceLocator(pdf_page_index=0),
        raw_value=Decimal("0.91"),
    )


def _complete_sufficiency() -> SufficiencyEvaluation:
    resolution = SlotResolution(
        slot_id="score", status="supported", evidence_ids=["evidence-1"]
    )
    return SufficiencyEvaluation(
        slot_resolutions=(resolution,),
        report=SufficiencyReport(
            evidence_complete=True,
            answerable=True,
            response_status="complete",
            supported_slot_ids=["score"],
        ),
    )


async def _event_sink(_: object) -> None:
    return None


def _runtime_context(
    *,
    deadline: ExecutionDeadline | None = None,
    cancellation: ExecutionCancellation | None = None,
) -> V9RuntimeContext:
    return V9RuntimeContext(
        cancellation_token=cancellation or ExecutionCancellation(),
        event_sink=_event_sink,
        budget_controller=object(),
        deadline=deadline or ExecutionDeadline(24.0),
        clock=datetime.now,
        llm_invoker=SimpleNamespace(),
    )


@pytest.mark.asyncio
async def test_core_runs_the_evidence_first_stages_in_order_and_enforces_call_caps() -> (
    None
):
    calls: list[str] = []
    packet = _packet()

    async def resolve_scope(_: V9ExecutionRequest) -> ResolvedSourceScope:
        calls.append("scope")
        return ResolvedSourceScope(authorized_doc_ids=["doc-1"])

    async def plan_contract(
        _: V9ExecutionRequest, __: ResolvedSourceScope
    ) -> QueryContract:
        calls.append("contract")
        return _contract()

    async def retrieve(tasks: tuple[object, ...]) -> tuple[TaskRetrievalResult, ...]:
        calls.append(f"retrieval:{len(tasks)}")
        return (
            TaskRetrievalResult(
                task_id=tasks[0].task_id,  # type: ignore[attr-defined]
                retrieval=RagRetrievalResult(retrieval_id="retrieval-1"),
            ),
        )

    async def deterministic_candidates(
        _: tuple[TaskRetrievalResult, ...], __: QueryContract
    ) -> tuple[EvidencePacket, ...]:
        calls.append("deterministic_candidates")
        return (packet,)

    def sufficiency(
        _: QueryContract, __: tuple[EvidencePacket, ...]
    ) -> SufficiencyEvaluation:
        calls.append("sufficiency")
        return _complete_sufficiency()

    def repair(
        _: QueryContract, __: SufficiencyEvaluation, ___: str, ____: int
    ) -> tuple[object, ...]:
        calls.append("repair")
        return ()

    async def prose_curate(
        _: str, __: QueryContract, packets: tuple[EvidencePacket, ...]
    ) -> tuple[EvidencePacket, ...]:
        calls.append("prose_curator")
        return packets

    async def conflict(
        _: QueryContract,
        __: tuple[EvidencePacket, ...],
        evaluation: SufficiencyEvaluation,
    ) -> ConflictStageResult:
        calls.append("conflict")
        return ConflictStageResult(sufficiency=evaluation)

    def pack(
        _: str,
        __: QueryContract,
        packets: tuple[EvidencePacket, ...],
        ___: SufficiencyEvaluation,
    ) -> object:
        calls.append("pack")
        return SimpleNamespace(packets=packets, is_packable=True)

    async def final(
        _: str,
        __: QueryContract,
        ___: object,
        ____: tuple[SlotResolution, ...],
        _____: object,
    ) -> FinalAnswerResult:
        calls.append("final")
        return FinalAnswerResult(
            response_status="complete",
            answer="0.91",
            used_evidence_ids=["evidence-1"],
            final_generation_count=1,
        )

    def deterministic_partial(
        _: QueryContract, evaluation: SufficiencyEvaluation
    ) -> FinalAnswerResult:
        calls.append("partial")
        return FinalAnswerResult(response_status=evaluation.report.response_status)

    core = V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=resolve_scope,
            plan_contract=plan_contract,
            retrieve=retrieve,
            deterministic_candidates=deterministic_candidates,
            evaluate_sufficiency=sufficiency,
            plan_repair=repair,
            prose_curate=prose_curate,
            resolve_conflicts=conflict,
            pack=pack,
            generate_final=final,
            deterministic_partial=deterministic_partial,
        )
    )

    result = await core.execute(_request(), runtime_context=_runtime_context())

    assert calls == [
        "scope",
        "contract",
        "retrieval:1",
        "deterministic_candidates",
        "sufficiency",
        "repair",
        "prose_curator",
        "sufficiency",
        "conflict",
        "pack",
        "final",
    ]
    assert result.metrics.subtask_answer_count == 0
    assert result.metrics.prose_curator_call_count == 1
    assert result.metrics.arbitration_call_count == 0
    assert result.metrics.final_generation_count == 1


@pytest.mark.asyncio
async def test_core_returns_a_deterministic_partial_without_final_generation_when_insufficient() -> (
    None
):
    calls: list[str] = []
    insufficient = SufficiencyEvaluation(
        slot_resolutions=(SlotResolution(slot_id="score", status="not_found"),),
        report=SufficiencyReport(
            evidence_complete=False,
            answerable=False,
            response_status="insufficient",
            not_found_slot_ids=["score"],
        ),
    )

    async def scope(_: V9ExecutionRequest) -> ResolvedSourceScope:
        return ResolvedSourceScope(authorized_doc_ids=["doc-1"])

    async def contract(_: V9ExecutionRequest, __: ResolvedSourceScope) -> QueryContract:
        return _contract()

    async def retrieve(tasks: tuple[object, ...]) -> tuple[TaskRetrievalResult, ...]:
        return (
            TaskRetrievalResult(
                task_id=tasks[0].task_id,  # type: ignore[attr-defined]
                retrieval=RagRetrievalResult(retrieval_id="retrieval-1"),
            ),
        )

    async def candidates(
        _: tuple[TaskRetrievalResult, ...], __: QueryContract
    ) -> tuple[EvidencePacket, ...]:
        return ()

    def evaluate(
        _: QueryContract, __: tuple[EvidencePacket, ...]
    ) -> SufficiencyEvaluation:
        return insufficient

    def repair(
        _: QueryContract, __: SufficiencyEvaluation, ___: str, ____: int
    ) -> tuple[object, ...]:
        return ()

    async def curator(
        _: str, __: QueryContract, packets: tuple[EvidencePacket, ...]
    ) -> tuple[EvidencePacket, ...]:
        return packets

    async def conflict(
        _: QueryContract,
        __: tuple[EvidencePacket, ...],
        evaluation: SufficiencyEvaluation,
    ) -> ConflictStageResult:
        return ConflictStageResult(sufficiency=evaluation)

    def pack(*_: object) -> object:
        calls.append("pack")
        return SimpleNamespace(packets=(), is_packable=False)

    async def final(*_: object) -> FinalAnswerResult:
        calls.append("final")
        raise AssertionError("final LLM must not run for zero supported slots")

    def partial(
        _: QueryContract, evaluation: SufficiencyEvaluation
    ) -> FinalAnswerResult:
        calls.append("partial")
        return FinalAnswerResult(response_status=evaluation.report.response_status)

    result = await V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=scope,
            plan_contract=contract,
            retrieve=retrieve,
            deterministic_candidates=candidates,
            evaluate_sufficiency=evaluate,
            plan_repair=repair,
            prose_curate=curator,
            resolve_conflicts=conflict,
            pack=pack,
            generate_final=final,
            deterministic_partial=partial,
        )
    ).execute(_request(), runtime_context=_runtime_context())

    assert calls == ["pack", "partial"]
    assert result.final_answer.response_status == "insufficient"
    assert result.metrics.final_generation_count == 0


@pytest.mark.asyncio
async def test_core_skips_repair_and_arbitration_when_only_final_reserve_remains() -> (
    None
):
    calls: list[str] = []
    initial = _complete_sufficiency()
    now = [0.0]

    async def scope(_: V9ExecutionRequest) -> ResolvedSourceScope:
        return ResolvedSourceScope(authorized_doc_ids=[])

    async def contract(_: V9ExecutionRequest, __: ResolvedSourceScope) -> QueryContract:
        now[0] = 10.0
        return _contract()

    def evaluate(
        _: QueryContract, __: tuple[EvidencePacket, ...]
    ) -> SufficiencyEvaluation:
        return initial

    def repair(*_: object) -> tuple[object, ...]:
        calls.append("repair")
        return ()

    async def curator(
        _: str, __: QueryContract, packets: tuple[EvidencePacket, ...]
    ) -> tuple[EvidencePacket, ...]:
        return packets

    async def conflict(*_: object) -> ConflictStageResult:
        calls.append("arbitration")
        return ConflictStageResult(sufficiency=initial, arbitration_call_count=1)

    def pack(*_: object) -> object:
        return SimpleNamespace(is_packable=True)

    async def final(*_: object) -> FinalAnswerResult:
        calls.append("final")
        return FinalAnswerResult(response_status="complete", final_generation_count=1)

    def partial(
        _: QueryContract, evaluation: SufficiencyEvaluation
    ) -> FinalAnswerResult:
        calls.append("partial")
        return FinalAnswerResult(response_status=evaluation.report.response_status)

    core = V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=scope,
            plan_contract=contract,
            retrieve=lambda _: (),
            deterministic_candidates=lambda *_: (),
            evaluate_sufficiency=evaluate,
            plan_repair=repair,
            prose_curate=curator,
            resolve_conflicts=conflict,
            pack=pack,
            generate_final=final,
            deterministic_partial=partial,
        ),
        runtime=V9ExecutionPolicyRuntime(ExecutionPolicy(total_deadline_s=24.0)),
    )

    result = await core.execute(
        _request(),
        runtime_context=_runtime_context(
            deadline=ExecutionDeadline(24.0, monotonic=lambda: now[0])
        ),
    )

    assert calls == ["final"]
    assert result.metrics.arbitration_call_count == 0


@pytest.mark.asyncio
async def test_core_requires_an_attempt_runtime_context_before_scope_resolution() -> (
    None
):
    calls: list[str] = []

    async def scope(_: V9ExecutionRequest) -> ResolvedSourceScope:
        calls.append("scope")
        return ResolvedSourceScope(authorized_doc_ids=[])

    async def contract(_: V9ExecutionRequest, __: ResolvedSourceScope) -> QueryContract:
        raise AssertionError("runtime context must be rejected before planning")

    core = V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=scope,
            plan_contract=contract,
            retrieve=lambda _: (),
            deterministic_candidates=lambda *_: (),
            evaluate_sufficiency=lambda *_: _complete_sufficiency(),
            plan_repair=lambda *_: (),
            prose_curate=lambda *_: (),
            resolve_conflicts=lambda *args: ConflictStageResult(sufficiency=args[-1]),
            pack=lambda *_: SimpleNamespace(is_packable=False),
            generate_final=lambda *_: FinalAnswerResult(response_status="complete"),
            deterministic_partial=lambda _, evaluation: FinalAnswerResult(
                response_status=evaluation.report.response_status
            ),
        )
    )

    with pytest.raises(ValueError, match="attempt runtime context"):
        await core.execute(_request())

    assert calls == []


@pytest.mark.asyncio
async def test_core_propagates_attempt_cancellation_to_an_inflight_scope() -> None:
    cancellation = ExecutionCancellation()
    started = asyncio.Event()

    async def scope(_: V9ExecutionRequest) -> ResolvedSourceScope:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    core = V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=scope,
            plan_contract=lambda *_: _contract(),
            retrieve=lambda _: (),
            deterministic_candidates=lambda *_: (),
            evaluate_sufficiency=lambda *_: _complete_sufficiency(),
            plan_repair=lambda *_: (),
            prose_curate=lambda *_: (),
            resolve_conflicts=lambda *args: ConflictStageResult(sufficiency=args[-1]),
            pack=lambda *_: SimpleNamespace(is_packable=False),
            generate_final=lambda *_: FinalAnswerResult(response_status="complete"),
            deterministic_partial=lambda _, evaluation: FinalAnswerResult(
                response_status=evaluation.report.response_status
            ),
        )
    )

    task = asyncio.create_task(
        core.execute(
            _request(), runtime_context=_runtime_context(cancellation=cancellation)
        )
    )
    await started.wait()
    cancellation.cancel("campaign_cancelled")

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.1)


@pytest.mark.asyncio
async def test_core_accepts_budgeted_final_fallback_without_count_assertion() -> None:
    calls: list[str] = []

    async def scope(_: V9ExecutionRequest) -> ResolvedSourceScope:
        return ResolvedSourceScope(authorized_doc_ids=[])

    async def contract(_: V9ExecutionRequest, __: ResolvedSourceScope) -> QueryContract:
        return QueryContract(
            route="exact_structured",
            intent="Report the source-bound score.",
            required_slots=[
                RequiredSlot(slot_id="score", description="reported score")
            ],
        )

    def fallback(*_: object) -> FinalAnswerResult:
        calls.append("final")
        return FinalAnswerResult(
            response_status="qualified_partial",
            answer="Final generation was unavailable; evidence is returned as a qualified partial.",
            final_generation_count=0,
        )

    result = await V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=scope,
            plan_contract=contract,
            retrieve=lambda _: (),
            deterministic_candidates=lambda *_: (),
            evaluate_sufficiency=lambda *_: _complete_sufficiency(),
            plan_repair=lambda *_: (),
            prose_curate=lambda _, __, packets: packets,
            resolve_conflicts=lambda *args: ConflictStageResult(sufficiency=args[-1]),
            pack=lambda *_: SimpleNamespace(is_packable=True),
            generate_final=fallback,
            deterministic_partial=lambda _, evaluation: FinalAnswerResult(
                response_status=evaluation.report.response_status
            ),
        )
    ).execute(_request(), runtime_context=_runtime_context())

    assert calls == ["final"]
    assert result.final_answer.final_generation_count == 0
    assert result.metrics.final_generation_count == 0


@pytest.mark.asyncio
async def test_core_returns_partial_without_packing_when_deadline_is_exhausted() -> (
    None
):
    calls: list[str] = []
    now = [0.0]
    deadline = ExecutionDeadline(24.0, monotonic=lambda: now[0])

    async def scope(_: V9ExecutionRequest) -> ResolvedSourceScope:
        return ResolvedSourceScope(authorized_doc_ids=[])

    async def contract(_: V9ExecutionRequest, __: ResolvedSourceScope) -> QueryContract:
        return QueryContract(
            route="exact_structured",
            intent="Report the source-bound score.",
            required_slots=[
                RequiredSlot(slot_id="score", description="reported score")
            ],
        )

    async def conflict(*args: object) -> ConflictStageResult:
        now[0] = 24.0
        return ConflictStageResult(sufficiency=args[-1])

    def partial(
        _: QueryContract, evaluation: SufficiencyEvaluation
    ) -> FinalAnswerResult:
        calls.append("partial")
        return FinalAnswerResult(response_status=evaluation.report.response_status)

    result = await V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=scope,
            plan_contract=contract,
            retrieve=lambda _: (),
            deterministic_candidates=lambda *_: (),
            evaluate_sufficiency=lambda *_: _complete_sufficiency(),
            plan_repair=lambda *_: (),
            prose_curate=lambda _, __, packets: packets,
            resolve_conflicts=conflict,
            pack=lambda *_: calls.append("pack"),
            generate_final=lambda *_: FinalAnswerResult(response_status="complete"),
            deterministic_partial=partial,
        )
    ).execute(_request(), runtime_context=_runtime_context(deadline=deadline))

    assert calls == ["partial"]
    assert result.final_answer.response_status == "complete"
    assert result.metrics.final_generation_count == 0
