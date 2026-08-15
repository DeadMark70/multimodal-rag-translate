"""State-machine contracts for the bounded Agentic v9 execution core."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from data_base.agentic_v9.execution_core import (
    ConflictStageResult,
    V9ExecutionCore,
    V9ExecutionStages,
    _prevent_response_status_upgrade,
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
    RetrievalTask,
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


def test_active_v2_allows_complete_response_when_sufficiency_is_complete() -> None:
    contract = QueryContract.model_validate(
        {**_contract().model_dump(), "contract_version": "2"}
    )
    complete = _complete_sufficiency().report

    result = _prevent_response_status_upgrade(
        FinalAnswerResult(
            response_status="complete",
            answer="All atomic slots supported.",
            final_generation_count=1,
        ),
        complete,
        contract,
    )

    assert result.response_status == "complete"
    assert contract.atomic_completeness is None
    assert contract.atomic_completeness_reason == "atomic_slot_matching_experimental"



def test_experimental_v2_policy_preserves_insufficient_without_usable_evidence() -> (
    None
):
    contract = QueryContract.model_validate(
        {**_contract().model_dump(), "contract_version": "2"}
    )
    insufficient = SufficiencyReport(
        evidence_complete=False,
        answerable=False,
        response_status="insufficient",
        not_found_slot_ids=["score"],
    )

    result = _prevent_response_status_upgrade(
        FinalAnswerResult(response_status="insufficient"),
        insufficient,
        contract,
    )

    assert result.response_status == "insufficient"


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
        deadline=deadline or ExecutionDeadline(64.0),
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
        ______: SufficiencyReport,
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
async def test_repair_candidate_is_not_supported_until_post_repair_qualification() -> None:
    contract = _contract().model_copy(update={"max_repair_rounds": 1})
    candidate_round = 0
    qualification_rounds = 0
    evaluated_ids: list[tuple[str, ...]] = []
    repair_rounds: list[int] = []

    async def retrieve(
        tasks: tuple[RetrievalTask, ...],
    ) -> tuple[TaskRetrievalResult, ...]:
        return tuple(
            TaskRetrievalResult(
                task_id=task.task_id,
                retrieval=RagRetrievalResult(retrieval_id=task.task_id),
            )
            for task in tasks
        )

    def candidates(*_: object) -> tuple[EvidencePacket, ...]:
        nonlocal candidate_round
        candidate_round += 1
        return (
            _packet().model_copy(
                update={
                    "evidence_id": f"raw-{candidate_round}",
                    "validation_status": "invalid",
                }
            ),
        )

    async def qualify(
        _: str, __: QueryContract, packets: tuple[EvidencePacket, ...]
    ) -> tuple[EvidencePacket, ...]:
        nonlocal qualification_rounds
        qualification_rounds += 1
        if qualification_rounds == 1:
            return ()
        return (
            _packet().model_copy(update={"evidence_id": "repair-qualified"}),
        )

    def evaluate(
        _: QueryContract, packets: tuple[EvidencePacket, ...]
    ) -> SufficiencyEvaluation:
        evaluated_ids.append(tuple(packet.evidence_id for packet in packets))
        if any(packet.evidence_id == "repair-qualified" for packet in packets):
            resolution = SlotResolution(
                slot_id="score",
                status="supported",
                evidence_ids=["repair-qualified"],
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
        return SufficiencyEvaluation(
            slot_resolutions=(SlotResolution(slot_id="score", status="not_found"),),
            report=SufficiencyReport(
                evidence_complete=False,
                answerable=False,
                response_status="insufficient",
                not_found_slot_ids=["score"],
            ),
            repairable_slot_ids=("score",),
        )

    def repair(
        _: QueryContract,
        __: SufficiencyEvaluation,
        query_id: str,
        round_index: int,
    ) -> tuple[RetrievalTask, ...]:
        repair_rounds.append(round_index)
        return (
            RetrievalTask(
                task_id=f"{query_id}:repair-{round_index}:score",
                round_id=f"repair-{round_index}",
                query_id=query_id,
                query="reported score",
                target_slot_ids=["score"],
                source_scope=ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
            ),
        )

    result = await V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=lambda _: ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
            plan_contract=lambda *_: contract,
            retrieve=retrieve,
            deterministic_candidates=candidates,
            evaluate_sufficiency=evaluate,
            plan_repair=repair,
            prose_curate=qualify,
            resolve_conflicts=lambda _contract, _packets, sufficiency: ConflictStageResult(
                sufficiency=sufficiency
            ),
            pack=lambda _, __, packets, ___: SimpleNamespace(
                packets=packets, is_packable=True
            ),
            generate_final=lambda *_: FinalAnswerResult(
                response_status="complete", final_generation_count=1
            ),
            deterministic_partial=lambda _contract, sufficiency: FinalAnswerResult(
                response_status=sufficiency.report.response_status
            ),
        )
    ).execute(_request(), runtime_context=_runtime_context())

    assert evaluated_ids == [(), ("repair-qualified",)]
    assert repair_rounds == [1]
    assert result.metrics.prose_curator_call_count == 2
    assert result.final_answer.response_status == "complete"


@pytest.mark.asyncio
async def test_no_evidence_skips_qualification_and_final_generation() -> None:
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
    result = await V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=lambda _: ResolvedSourceScope(authorized_doc_ids=[]),
            plan_contract=lambda *_: _contract().model_copy(
                update={"resolved_source_scope": ResolvedSourceScope(authorized_doc_ids=[])}
            ),
            retrieve=lambda _: (),
            deterministic_candidates=lambda *_: (),
            evaluate_sufficiency=lambda *_: insufficient,
            plan_repair=lambda *_: (),
            prose_curate=lambda *_: calls.append("qualify") or (),
            resolve_conflicts=lambda _contract, _packets, sufficiency: ConflictStageResult(
                sufficiency=sufficiency
            ),
            pack=lambda *_: SimpleNamespace(packets=(), is_packable=False),
            generate_final=lambda *_: calls.append("final")
            or FinalAnswerResult(response_status="complete", final_generation_count=1),
            deterministic_partial=lambda _contract, sufficiency: FinalAnswerResult(
                response_status=sufficiency.report.response_status
            ),
        )
    ).execute(_request(), runtime_context=_runtime_context())

    assert calls == []
    assert result.final_answer.response_status == "insufficient"
    assert result.metrics.prose_curator_call_count == 0
    assert result.metrics.final_generation_count == 0


@pytest.mark.asyncio
async def test_core_recomputes_sufficiency_after_each_of_at_most_two_repairs() -> None:
    contract = _contract().model_copy(update={"max_repair_rounds": 5})
    evaluation_calls: list[int] = []
    repair_rounds: list[int] = []
    retrieval_rounds: list[str] = []

    async def retrieve(
        tasks: tuple[RetrievalTask, ...],
    ) -> tuple[TaskRetrievalResult, ...]:
        retrieval_rounds.extend(task.round_id for task in tasks)
        return tuple(
            TaskRetrievalResult(
                task_id=task.task_id,
                retrieval=RagRetrievalResult(retrieval_id=f"retrieval:{task.task_id}"),
            )
            for task in tasks
        )

    def evaluate(
        _: QueryContract, packets: tuple[EvidencePacket, ...]
    ) -> SufficiencyEvaluation:
        evaluation_calls.append(len(packets))
        return SufficiencyEvaluation(
            slot_resolutions=(SlotResolution(slot_id="score", status="not_found"),),
            report=SufficiencyReport(
                evidence_complete=False,
                answerable=False,
                response_status="insufficient",
                not_found_slot_ids=["score"],
            ),
            repairable_slot_ids=("score",),
        )

    def repair(
        _: QueryContract,
        __: SufficiencyEvaluation,
        query_id: str,
        round_index: int,
    ) -> tuple[RetrievalTask, ...]:
        repair_rounds.append(round_index)
        return (
            RetrievalTask(
                task_id=f"{query_id}:repair-{round_index}:score",
                round_id=f"repair-{round_index}",
                query_id=query_id,
                query="reported score",
                target_slot_ids=["score"],
                source_scope=ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
            ),
        )

    stages = V9ExecutionStages(
        resolve_scope=lambda _: ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
        plan_contract=lambda *_: contract,
        retrieve=retrieve,
        deterministic_candidates=lambda *_: (),
        evaluate_sufficiency=evaluate,
        plan_repair=repair,
        prose_curate=lambda _question, _contract, packets: packets,
        resolve_conflicts=lambda _contract, _packets, sufficiency: ConflictStageResult(
            sufficiency=sufficiency
        ),
        pack=lambda *_: SimpleNamespace(packets=(), is_packable=False),
        generate_final=lambda *_: FinalAnswerResult(response_status="insufficient"),
        deterministic_partial=lambda _contract, sufficiency: FinalAnswerResult(
            response_status=sufficiency.report.response_status
        ),
    )

    await V9ExecutionCore(stages=stages).execute(
        _request(), runtime_context=_runtime_context()
    )

    assert repair_rounds == [1, 2]
    assert retrieval_rounds == ["round-1", "repair-1", "repair-2"]
    assert evaluation_calls == [0, 0, 0]


@pytest.mark.asyncio
async def test_core_does_not_request_repair_when_initial_sufficiency_is_terminal() -> (
    None
):
    repair_decisions: list[int] = []
    contract = _contract().model_copy(update={"max_repair_rounds": 2})
    stages = V9ExecutionStages(
        resolve_scope=lambda _: ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
        plan_contract=lambda *_: contract,
        retrieve=lambda tasks: tuple(
            TaskRetrievalResult(
                task_id=task.task_id,
                retrieval=RagRetrievalResult(retrieval_id=task.task_id),
            )
            for task in tasks
        ),
        deterministic_candidates=lambda *_: (_packet(),),
        evaluate_sufficiency=lambda *_: _complete_sufficiency(),
        plan_repair=lambda _contract, _evaluation, _query_id, round_index: (
            repair_decisions.append(round_index) or ()
        ),
        prose_curate=lambda _question, _contract, packets: packets,
        resolve_conflicts=lambda _contract, _packets, evaluation: ConflictStageResult(
            sufficiency=evaluation
        ),
        pack=lambda *_: SimpleNamespace(packets=(_packet(),), is_packable=True),
        generate_final=lambda *_: FinalAnswerResult(
            response_status="complete",
            used_evidence_ids=["evidence-1"],
            final_generation_count=1,
        ),
        deterministic_partial=lambda _contract, evaluation: FinalAnswerResult(
            response_status=evaluation.report.response_status
        ),
    )

    await V9ExecutionCore(stages=stages).execute(
        _request(), runtime_context=_runtime_context()
    )

    assert repair_decisions == []


@pytest.mark.asyncio
async def test_core_records_terminal_reserve_reason_after_executed_repair() -> None:
    now = [0.0]
    deadline = ExecutionDeadline(64.0, monotonic=lambda: now[0])
    contract = _contract().model_copy(update={"max_repair_rounds": 2})
    repair_decisions: list[int] = []
    terminal_reasons: list[str] = []
    evaluations = 0

    def evaluate(*_: object) -> SufficiencyEvaluation:
        nonlocal evaluations
        evaluations += 1
        if evaluations == 1:
            return SufficiencyEvaluation(
                slot_resolutions=(SlotResolution(slot_id="score", status="not_found"),),
                report=SufficiencyReport(
                    evidence_complete=False,
                    answerable=False,
                    response_status="insufficient",
                    not_found_slot_ids=["score"],
                ),
                repairable_slot_ids=("score",),
            )
        return SufficiencyEvaluation(
            slot_resolutions=(SlotResolution(slot_id="score", status="not_found"),),
            report=SufficiencyReport(
                evidence_complete=False,
                answerable=False,
                response_status="insufficient",
                not_found_slot_ids=["score"],
            ),
            repairable_slot_ids=("score",),
        )

    async def retrieve(
        tasks: tuple[RetrievalTask, ...],
    ) -> tuple[TaskRetrievalResult, ...]:
        if tasks[0].round_id == "repair-1":
            now[0] = 60.0
        return tuple(
            TaskRetrievalResult(
                task_id=task.task_id,
                retrieval=RagRetrievalResult(retrieval_id=task.task_id),
            )
            for task in tasks
        )

    def repair(
        _: QueryContract,
        __: SufficiencyEvaluation,
        query_id: str,
        round_index: int,
    ) -> tuple[RetrievalTask, ...]:
        repair_decisions.append(round_index)
        return (
            RetrievalTask(
                task_id=f"{query_id}:repair-{round_index}",
                round_id=f"repair-{round_index}",
                query_id=query_id,
                query="score",
                target_slot_ids=["score"],
                source_scope=ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
            ),
        )

    stages = V9ExecutionStages(
        resolve_scope=lambda _: ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
        plan_contract=lambda *_: contract,
        retrieve=retrieve,
        deterministic_candidates=lambda *_: (),
        evaluate_sufficiency=evaluate,
        plan_repair=repair,
        prose_curate=lambda _question, _contract, packets: packets,
        resolve_conflicts=lambda _contract, _packets, evaluation: ConflictStageResult(
            sufficiency=evaluation
        ),
        pack=lambda *_: SimpleNamespace(packets=(), is_packable=False),
        generate_final=lambda *_: FinalAnswerResult(response_status="insufficient"),
        deterministic_partial=lambda _contract, evaluation: FinalAnswerResult(
            response_status=evaluation.report.response_status
        ),
        record_repair_terminal=terminal_reasons.append,
    )

    await V9ExecutionCore(stages=stages).execute(
        _request(), runtime_context=_runtime_context(deadline=deadline)
    )

    assert repair_decisions == [1]
    assert terminal_reasons == ["final_budget_protected"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("repair_rounds", "expected_reason"),
    [
        pytest.param(1, "evidence_complete", id="one-round-final-success"),
        pytest.param(2, "evidence_complete", id="two-round-final-success"),
        pytest.param(1, "no_repairable_slots", id="final-no-repairable-slots"),
    ],
)
async def test_core_records_final_round_terminal_state_before_repair_cap(
    repair_rounds: int,
    expected_reason: str,
) -> None:
    contract = _contract().model_copy(update={"max_repair_rounds": repair_rounds})
    repair_decisions: list[int] = []
    terminal_reasons: list[str] = []
    evaluations = 0

    def evaluate(*_: object) -> SufficiencyEvaluation:
        nonlocal evaluations
        evaluations += 1
        if evaluations >= repair_rounds + 1:
            if expected_reason == "evidence_complete":
                return _complete_sufficiency()
            return SufficiencyEvaluation(
                slot_resolutions=(SlotResolution(slot_id="score", status="not_found"),),
                report=SufficiencyReport(
                    evidence_complete=False,
                    answerable=False,
                    response_status="insufficient",
                    not_found_slot_ids=["score"],
                ),
            )
        return SufficiencyEvaluation(
            slot_resolutions=(SlotResolution(slot_id="score", status="not_found"),),
            report=SufficiencyReport(
                evidence_complete=False,
                answerable=False,
                response_status="insufficient",
                not_found_slot_ids=["score"],
            ),
            repairable_slot_ids=("score",),
        )

    def repair(
        _: QueryContract,
        __: SufficiencyEvaluation,
        query_id: str,
        round_index: int,
    ) -> tuple[RetrievalTask, ...]:
        repair_decisions.append(round_index)
        return (
            RetrievalTask(
                task_id=f"{query_id}:repair-{round_index}:score",
                round_id=f"repair-{round_index}",
                query_id=query_id,
                query="reported score",
                target_slot_ids=["score"],
                source_scope=ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
            ),
        )

    stages = V9ExecutionStages(
        resolve_scope=lambda _: ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
        plan_contract=lambda *_: contract,
        retrieve=lambda tasks: tuple(
            TaskRetrievalResult(
                task_id=task.task_id,
                retrieval=RagRetrievalResult(retrieval_id=task.task_id),
            )
            for task in tasks
        ),
        deterministic_candidates=lambda *_: (),
        evaluate_sufficiency=evaluate,
        plan_repair=repair,
        prose_curate=lambda _question, _contract, packets: packets,
        resolve_conflicts=lambda _contract, _packets, evaluation: ConflictStageResult(
            sufficiency=evaluation
        ),
        pack=lambda *_: SimpleNamespace(packets=(_packet(),), is_packable=True),
        generate_final=lambda *_: FinalAnswerResult(
            response_status="complete",
            used_evidence_ids=["evidence-1"],
            final_generation_count=1,
        ),
        deterministic_partial=lambda _contract, evaluation: FinalAnswerResult(
            response_status=evaluation.report.response_status
        ),
        record_repair_terminal=terminal_reasons.append,
    )

    await V9ExecutionCore(stages=stages).execute(
        _request(), runtime_context=_runtime_context()
    )

    assert repair_decisions == list(range(1, repair_rounds + 1))
    assert terminal_reasons == [expected_reason]


@pytest.mark.asyncio
async def test_incomplete_sufficiency_cannot_be_upgraded_by_final_provider() -> None:
    contract = _contract().model_copy(
        update={
            "required_slots": [
                RequiredSlot(slot_id="score", description="reported score"),
                RequiredSlot(slot_id="source", description="source requirement"),
            ]
        }
    )
    degraded = SufficiencyEvaluation(
        slot_resolutions=(
            SlotResolution(
                slot_id="score",
                status="supported",
                evidence_ids=["evidence-1"],
            ),
            SlotResolution(
                slot_id="source",
                status="not_found",
                reason="Required source evidence was not found.",
            ),
        ),
        report=SufficiencyReport(
            evidence_complete=False,
            answerable=True,
            response_status="qualified_partial",
            supported_slot_ids=["score"],
            not_found_slot_ids=["source"],
        ),
    )
    received: dict[str, object] = {}

    async def final(
        _question: str,
        _contract: QueryContract,
        _packed: object,
        resolutions: tuple[SlotResolution, ...],
        _arbitration: object,
        report: SufficiencyReport,
    ) -> FinalAnswerResult:
        received["resolutions"] = resolutions
        received["report"] = report
        return FinalAnswerResult(
            response_status="complete",
            answer="Provider attempted an upgrade.",
            final_generation_count=1,
        )

    result = await V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=lambda _: ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
            plan_contract=lambda *_: contract,
            retrieve=lambda tasks: (
                TaskRetrievalResult(
                    task_id=tasks[0].task_id,
                    retrieval=RagRetrievalResult(retrieval_id="retrieval-1"),
                ),
            ),
            deterministic_candidates=lambda *_: (_packet(),),
            evaluate_sufficiency=lambda *_: degraded,
            plan_repair=lambda *_: (),
            prose_curate=lambda _, __, packets: packets,
            resolve_conflicts=lambda *args: ConflictStageResult(sufficiency=args[-1]),
            pack=lambda _, __, packets, ___: SimpleNamespace(
                packets=packets, is_packable=True
            ),
            generate_final=final,
            deterministic_partial=lambda _, evaluation: FinalAnswerResult(
                response_status=evaluation.report.response_status
            ),
        )
    ).execute(_request(), runtime_context=_runtime_context())

    assert received["resolutions"] == degraded.slot_resolutions
    assert received["report"] == degraded.report
    assert result.final_answer.response_status == "qualified_partial"
    assert result.sufficiency.response_status == "qualified_partial"


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
        now[0] = 32.1
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
        runtime=V9ExecutionPolicyRuntime(ExecutionPolicy(total_deadline_s=64.0)),
    )

    result = await core.execute(
        _request(),
        runtime_context=_runtime_context(
            deadline=ExecutionDeadline(64.0, monotonic=lambda: now[0])
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
    deadline = ExecutionDeadline(64.0, monotonic=lambda: now[0])

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
        now[0] = 64.0
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


@pytest.mark.asyncio
async def test_execution_core_uses_contract_planning_phase_when_requested() -> None:
    stage_calls: list[SimpleNamespace] = []
    runtime = V9ExecutionPolicyRuntime()
    original_run_llm = runtime.run_llm
    original_run_retrieval = runtime.run_retrieval

    async def spy_run_llm(
        operation: object,
        *,
        phase: str,
        cancellation: object = None,
        deadline: object = None,
    ) -> object:
        stage_calls.append(SimpleNamespace(kind="llm", phase=phase))
        return await original_run_llm(
            operation,  # type: ignore[arg-type]
            phase=phase,
            cancellation=cancellation,  # type: ignore[arg-type]
            deadline=deadline,  # type: ignore[arg-type]
        )

    async def spy_run_retrieval(
        operation: object,
        *,
        phase: str = "evidence_extract",
        cancellation: object = None,
        deadline: object = None,
    ) -> object:
        stage_calls.append(SimpleNamespace(kind="retrieval", phase=phase))
        return await original_run_retrieval(
            operation,  # type: ignore[arg-type]
            phase=phase,
            cancellation=cancellation,  # type: ignore[arg-type]
            deadline=deadline,  # type: ignore[arg-type]
        )

    runtime.run_llm = spy_run_llm  # type: ignore[method-assign]
    runtime.run_retrieval = spy_run_retrieval  # type: ignore[method-assign]

    core = V9ExecutionCore(
        stages=V9ExecutionStages(
            resolve_scope=lambda *_: ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
            plan_contract=lambda *_: _contract(),
            retrieve=lambda *_: (),
            deterministic_candidates=lambda *_: (),
            evaluate_sufficiency=lambda *_: _complete_sufficiency(),
            plan_repair=lambda *_: (),
            prose_curate=lambda *_: (),
            resolve_conflicts=lambda *_: ConflictStageResult(
                sufficiency=_complete_sufficiency()
            ),
            pack=lambda *_: SimpleNamespace(packets=(), is_packable=True),
            generate_final=lambda *_: FinalAnswerResult(
                response_status="complete", answer="0.91", final_generation_count=1
            ),
            deterministic_partial=lambda *_: FinalAnswerResult(
                response_status="complete"
            ),
        ),
        runtime=runtime,
    )

    request = V9ExecutionRequest(
        question="ambiguous question",
        trace_id="trace-1",
        contract_plan_requested=True,
    )
    await core.execute(request, runtime_context=_runtime_context())
    assert len(stage_calls) >= 2
    assert stage_calls[0].phase == "route_plan"
    assert stage_calls[1].phase == "contract_planning"

    stage_calls.clear()
    request_default = V9ExecutionRequest(
        question="exact question",
        trace_id="trace-2",
        contract_plan_requested=False,
    )
    await core.execute(request_default, runtime_context=_runtime_context())
    assert len(stage_calls) >= 2
    assert stage_calls[0].phase == "route_plan"
    assert stage_calls[1].phase == "route_plan"


@pytest.mark.asyncio
async def test_execution_core_qualifies_candidates_before_sufficiency_evaluation() -> None:
    order: list[str] = []
    raw_candidate = _packet().model_copy(update={"validation_status": "invalid"})
    qualified_packet = _packet().model_copy(update={"validation_status": "quote_bound"})

    async def candidate_fn(*_: Any) -> tuple[EvidencePacket, ...]:
        order.append("deterministic_candidates")
        return (raw_candidate,)

    async def qualify_fn(
        _: str, __: QueryContract, packets: tuple[EvidencePacket, ...]
    ) -> tuple[EvidencePacket, ...]:
        order.append("prose_curate")
        assert raw_candidate in packets
        return (qualified_packet,)

    def sufficiency_fn(
        _: QueryContract, packets: tuple[EvidencePacket, ...]
    ) -> SufficiencyEvaluation:
        order.append("evaluate_sufficiency")
        assert qualified_packet in packets
        assert raw_candidate not in packets
        return _complete_sufficiency()

    stages = V9ExecutionStages(
        resolve_scope=lambda _: ResolvedSourceScope(authorized_doc_ids=["doc-1"]),
        plan_contract=lambda *_: _contract(),
        retrieve=lambda tasks: (
            TaskRetrievalResult(
                task_id=tasks[0].task_id,
                retrieval=RagRetrievalResult(retrieval_id="ret-1"),
            ),
        ),
        deterministic_candidates=candidate_fn,
        evaluate_sufficiency=sufficiency_fn,
        plan_repair=lambda *_: (),
        prose_curate=qualify_fn,
        resolve_conflicts=lambda _c, _p, s: ConflictStageResult(sufficiency=s),
        pack=lambda *_: SimpleNamespace(packets=(qualified_packet,), is_packable=True),
        generate_final=lambda *_: FinalAnswerResult(
            response_status="complete", answer="0.91", final_generation_count=1
        ),
        deterministic_partial=lambda _c, s: FinalAnswerResult(
            response_status=s.report.response_status
        ),
    )

    result = await V9ExecutionCore(stages=stages).execute(
        _request(), runtime_context=_runtime_context()
    )

    assert order == ["deterministic_candidates", "prose_curate", "evaluate_sufficiency"]
    assert result.final_answer.response_status == "complete"

