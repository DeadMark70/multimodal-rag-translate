"""Concrete evidence-first adapter for Agentic v9 campaign executions.

The v9 core deliberately owns orchestration only.  This module is the
production evaluation adapter: it resolves a fail-closed source scope, runs
the typed core, and projects only cited evidence back into the existing
``RAGResult`` contract.  Durable materialization remains in the worker because
only the worker knows the promoted run and attempt identities.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document

from core.providers import get_llm
from data_base.RAG_QA_service import RAGResult, get_graph_evidence_bundle
from data_base.agentic_v9.budget_controller import RunBudgetController
from data_base.agentic_v9.budget_feasibility import (
    FeasibilityResult,
    FeasibilityStatus,
    validate_post_contract_feasibility,
    validate_pre_route_feasibility,
)
from data_base.agentic_v9.budgeted_llm import BudgetedLlmInvoker, LlmCallObserver
from data_base.agentic_v9.context_packer import (
    EvidenceContextPacker,
    PackedEvidenceContext,
)
from data_base.agentic_v9.contract_planner import QuestionContractPlanner
from data_base.agentic_v9.citation_renderer import render_verified_answer
from data_base.agentic_v9.execution_core import (
    ConflictStageResult,
    V9ExecutionCore,
    V9ExecutionStages,
)
from data_base.agentic_v9.execution_policy import (
    ExecutionCancellation,
    V9ExecutionPolicyRuntime,
)
from data_base.agentic_v9.final_answer import generate_final_answer
from data_base.agentic_v9.repair import (
    MAX_REPAIR_ROUNDS,
    ROUTE_REPAIR_CAPS,
    RepairPlan,
    build_repair_plan,
)
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
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
    V9ExecutionEvent,
    V9ExecutionRequest,
    V9RuntimeContext,
    UnresolvedRequirement,
)
from data_base.agentic_v9.sufficiency_gate import (
    SufficiencyEvaluation,
    evaluate_sufficiency,
)
from data_base.agentic_v9.slot_constraints import (
    authorized_doc_ids_for_slot,
    structured_locator_state,
)
from data_base.agentic_v9.asset_locator import VisualAssetCandidate
from data_base.agentic_v9.visual_asset_resolver import (
    VisualAssetResolutionDiagnostics,
    VisualAssetResolver,
)
from data_base.agentic_v9.visual_evidence_extractor import (
    VisualEvidenceExtractionResult,
    VisualEvidenceExtractor,
    visual_slots_requiring_extraction,
)
from data_base.document_metadata import get_document_id
from data_base.rag_filtering import filter_and_rerank_retrieval
from data_base.reranker import DocumentReranker
from data_base.rag_graph_locator import GraphSourceLocatorResult, locate_graph_sources
from data_base.rag_retrieval import retrieve_hybrid_documents
from data_base.vector_store_manager import get_user_retriever_async
from evaluation.agentic_campaign_adapter import used_evidence_documents
from evaluation.agentic_v9_admission import (
    DocumentReferenceResolver,
    build_v9_admission_contract,
)
from evaluation.observability import current_llm_call_observer
from evaluation.retrieval_profiles import AGENTIC_EVAL_PROFILE


@dataclass(frozen=True)
class V9RetrievalSelection:
    """Authorized retrieval documents with bounded, text-free diagnostics."""

    documents: tuple[Document, ...]
    diagnostics: dict[str, Any]


RetrievalAdapterResult = V9RetrievalSelection | list[Document]
RetrievalAdapter = Callable[[str, str, list[str]], Awaitable[RetrievalAdapterResult]]
GraphLocator = Callable[
    [str, str, list[Document], list[str], QueryContract],
    Awaitable[GraphSourceLocatorResult],
]
VisualExtractor = Callable[
    [Any, list[VisualAssetCandidate], str, RunBudgetController],
    Awaitable[VisualEvidenceExtractionResult],
]
ProviderFactory = Callable[[str], Any]


class _ConfigurationIncompatible(RuntimeError):
    def __init__(self, *, stage: str, feasibility: FeasibilityResult) -> None:
        super().__init__(feasibility.reason or "configuration_incompatible")
        self.stage = stage
        self.feasibility = feasibility


class AgenticV9CampaignRuntime:
    """Run one campaign item through the real typed Agentic v9 core."""

    def __init__(
        self,
        *,
        retrieve_documents: RetrievalAdapter | None = None,
        graph_locator: GraphLocator | None = None,
        visual_extractor: VisualExtractor | None = None,
        visual_asset_resolver: VisualAssetResolver | None = None,
        provider_factory: ProviderFactory | None = None,
        policy_runtime: V9ExecutionPolicyRuntime | None = None,
        document_reference_resolver: DocumentReferenceResolver | None = None,
        llm_call_observer: LlmCallObserver | None = None,
    ) -> None:
        self._retrieve_documents = retrieve_documents or _retrieve_documents
        self._graph_locator = graph_locator or _locate_graph_documents
        self._visual_extractor = visual_extractor or _extract_visual_evidence
        self._visual_asset_resolver = visual_asset_resolver or VisualAssetResolver()
        self._provider_factory = provider_factory or _provider_for_purpose
        self._policy_runtime = policy_runtime or V9ExecutionPolicyRuntime()
        self._llm_call_observer = llm_call_observer
        self._document_reference_resolver = (
            document_reference_resolver or _resolve_document_references
        )

    async def execute(
        self,
        *,
        question: str,
        user_id: str,
        authorized_doc_ids: list[str],
        setup_snapshot: dict[str, Any],
        trace_id: str,
    ) -> RAGResult:
        """Execute v9 or return a deterministic incompatible projection.

        Pre-route admission happens before the core (and thus before retrieval
        or provider work).  The contract adapter repeats admission immediately
        after planning, before the core can start retrieval.
        """
        pre_route = validate_pre_route_feasibility(
            setup_snapshot=setup_snapshot,
            remaining_token_budget=_pre_route_token_budget(setup_snapshot),
            remaining_llm_calls=_pre_route_llm_calls(setup_snapshot),
        )
        if pre_route.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE:
            return _configuration_incompatible_result(
                question=question,
                trace_id=trace_id,
                stage="pre_route",
                feasibility=pre_route,
            )

        admission = await build_v9_admission_contract(
            question=question,
            user_id=user_id,
            source_references=authorized_doc_ids,
            document_reference_resolver=self._document_reference_resolver,
            setup_policy=setup_snapshot,
        )
        source_scope = admission.source_scope
        runtime_contract = admission.contract
        budget_controller = RunBudgetController(
            max_llm_calls=_pre_route_llm_calls(setup_snapshot),
            runtime_token_budget=_pre_route_token_budget(setup_snapshot),
            setup_snapshot=setup_snapshot,
            final_input_tokens=_final_input_reserve(
                setup_snapshot, _pre_route_token_budget(setup_snapshot)
            ),
        )
        llm_call_observer = self._llm_call_observer or current_llm_call_observer()
        provider_name = str(setup_snapshot.get("provider") or "unknown")
        model_name = str(
            setup_snapshot.get("model_name") or setup_snapshot.get("model") or "unknown"
        )
        if (
            runtime_contract.route_decision is not None
            and runtime_contract.route_decision.decision_source == "safe_fallback"
            and runtime_contract.route_decision.fallback_reason == "planner_unavailable"
        ):
            runtime_contract = await QuestionContractPlanner(
                llm_invoker=BudgetedLlmInvoker(
                    controller=budget_controller,
                    provider_factory=self._provider_factory,
                    observer=llm_call_observer,
                    provider_name=provider_name,
                    model_name=model_name,
                    capture_policy=setup_snapshot.get("prompt_capture_policy"),
                )
            ).plan(
                question=question,
                authorized_source_names=list(source_scope.requested_source_names),
                authorized_source_doc_ids=list(source_scope.authorized_doc_ids),
                setup_policy=dict(setup_snapshot),
                authorized_source_name_to_doc_ids=dict(
                    source_scope.source_name_to_doc_ids
                ),
            )
            runtime_contract = runtime_contract.model_copy(
                update={"resolved_source_scope": source_scope}
            )
        request = V9ExecutionRequest(
            question=question,
            requested_doc_ids=list(source_scope.authorized_doc_ids),
            setup_snapshot=dict(setup_snapshot),
            trace_id=trace_id,
        )
        deadline = self._policy_runtime.start_deadline()
        cancellation = ExecutionCancellation()
        state: dict[str, Any] = {
            "contract": None,
            "pack": None,
            "repairs": [],
            "evidence_packets": [],
            "locator_diagnostics": [],
            "post_contract": None,
            "budget_controller": budget_controller,
            "tasks_by_id": {},
            "graph_execution": None,
            "visual_execution": None,
            "visual_packets": [],
            "visual_packets_emitted": False,
            "visual_resolution_diagnostics": None,
            "final_slot_resolutions": (),
            "retrieval_diagnostics": [],
        }

        async def resolve_scope(_: V9ExecutionRequest) -> ResolvedSourceScope:
            return source_scope

        async def plan_contract(
            _: V9ExecutionRequest, scope: ResolvedSourceScope
        ) -> QueryContract:
            # Route planning remains deterministic unless the planner has an
            # explicitly injected budgeted ambiguity invoker.  This prevents an
            # unreserved provider call while the contract budget is unknown.
            contract = runtime_contract
            post_contract = validate_post_contract_feasibility(
                contract=contract,
                setup_snapshot=setup_snapshot,
                remaining_token_budget=contract.runtime_token_budget,
                remaining_llm_calls=(
                    contract.max_llm_calls
                    - int(
                        bool(
                            contract.route_decision
                            and contract.route_decision.planner_call_used
                        )
                    )
                ),
                route_plan_used=bool(
                    contract.route_decision
                    and contract.route_decision.planner_call_used
                ),
            )
            state["contract"] = contract
            state["post_contract"] = post_contract
            state["graph_execution"] = _initial_graph_execution(contract)
            state["visual_execution"] = _initial_visual_execution(contract)
            if post_contract.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE:
                raise _ConfigurationIncompatible(
                    stage="post_contract", feasibility=post_contract
                )
            return contract

        async def retrieve(
            tasks: tuple[Any, ...],
        ) -> tuple[TaskRetrievalResult, ...]:
            results: list[TaskRetrievalResult] = []
            for task in tasks:
                state["tasks_by_id"][task.task_id] = task
                selection = _normalize_retrieval_selection(
                    await self._retrieve_documents(
                    user_id, task.query, list(task.source_scope.authorized_doc_ids)
                    ),
                    authorized_doc_ids=list(task.source_scope.authorized_doc_ids),
                )
                docs = list(selection.documents)
                state["retrieval_diagnostics"].append(
                    {"task_id": task.task_id, **selection.diagnostics}
                )
                if (
                    state["contract"].graph_policy == "required_locator"
                    and not state["graph_execution"]["attempted"]
                ):
                    try:
                        located = await self._graph_locator(
                            task.query,
                            user_id,
                            docs,
                            list(task.source_scope.authorized_doc_ids),
                            state["contract"],
                        )
                    except (
                        Exception
                    ) as error:  # Stage admitted; preserve partial answer.
                        state["graph_execution"] = _failed_required_stage(
                            policy="required_locator", error=error
                        )
                    else:
                        state["graph_execution"] = _graph_execution_projection(located)
                        docs = list(located.documents)
                chunks = [
                    _chunk_projection(document, index)
                    for index, document in enumerate(docs)
                ]
                results.append(
                    TaskRetrievalResult(
                        task_id=task.task_id,
                        retrieval=RagRetrievalResult(
                            retrieval_id=f"{trace_id}:{task.task_id}",
                            chunks=chunks,
                            diagnostics=dict(selection.diagnostics),
                        ),
                    )
                )
            text_packets = _evidence_packets_for_results(
                results=tuple(results),
                contract=state["contract"],
                trace_id=trace_id,
                tasks_by_id=state["tasks_by_id"],
                locator_diagnostics=state["locator_diagnostics"],
            )
            text_evaluation = evaluate_sufficiency(
                state["contract"],
                tuple(text_packets),
            )
            visual_slots = visual_slots_requiring_extraction(
                state["contract"],
                text_supported_slot_ids=set(text_evaluation.report.supported_slot_ids),
            )
            if (
                state["contract"].visual_requested
                and not visual_slots
                and not state["visual_execution"]["attempted"]
            ):
                state["visual_execution"] = {
                    **state["visual_execution"],
                    "state": "not_needed_text_satisfied",
                    "failure_reason": None,
                }
            if (
                state["contract"].visual_requested
                and visual_slots
                and not state["visual_execution"]["attempted"]
            ):
                controller = state["budget_controller"]
                assert isinstance(controller, RunBudgetController)
                visual_slot_ids = [slot.slot_id for slot in visual_slots]
                visual_task = tasks[0].model_copy(
                    update={
                        "target_slot_ids": visual_slot_ids,
                        "source_scope": (
                            state["contract"].resolved_source_scope
                            or tasks[0].source_scope
                        ),
                        "locator_hints": [],
                        "visual_required": True,
                    }
                )
                try:
                    resolution = self._visual_asset_resolver.resolve_task(
                        user_id=user_id,
                        task=visual_task,
                        slots=visual_slots,
                    )
                    state["visual_resolution_diagnostics"] = resolution.diagnostics
                    visual_result = await self._visual_extractor(
                        visual_task,
                        list(resolution.assets),
                        question,
                        controller,
                    )
                except Exception as error:  # Stage admitted; preserve partial answer.
                    state["visual_execution"] = _failed_required_stage(
                        policy=(
                            "visual_required"
                            if state["contract"].visual_required
                            else "visual_preferred"
                        ),
                        error=error,
                    )
                else:
                    visual_result = _bind_visual_result_to_contract(
                        result=visual_result,
                        contract=state["contract"],
                        allowed_slot_ids=set(visual_slot_ids),
                    )
                    state["visual_execution"] = _visual_execution_projection(
                        visual_result,
                        required=state["contract"].visual_required,
                        resolution_diagnostics=state["visual_resolution_diagnostics"],
                    )
                    state["visual_packets"].extend(visual_result.packets)
            return tuple(results)

        async def deterministic_candidates(
            results: tuple[TaskRetrievalResult, ...], contract: QueryContract
        ) -> tuple[EvidencePacket, ...]:
            packets = _evidence_packets_for_results(
                results=results,
                contract=contract,
                trace_id=trace_id,
                tasks_by_id=state["tasks_by_id"],
                locator_diagnostics=state["locator_diagnostics"],
            )
            if not state["visual_packets_emitted"]:
                packets.extend(state["visual_packets"])
                state["visual_packets_emitted"] = True
            _record_repair_evidence(
                repairs=state["repairs"],
                results=results,
                packets=packets,
            )
            state["evidence_packets"].extend(packets)
            return tuple(packets)

        def sufficiency(
            contract: QueryContract, packets: tuple[EvidencePacket, ...]
        ) -> SufficiencyEvaluation:
            evaluation = _apply_required_capability_constraints(
                contract=contract,
                evaluation=evaluate_sufficiency(contract, packets),
                graph_execution=state["graph_execution"],
                visual_execution=state["visual_execution"],
            )
            if (
                contract.slot_plan_status == "degraded"
                and evaluation.report.response_status == "complete"
            ):
                evaluation = evaluation.model_copy(
                    update={
                        "report": evaluation.report.model_copy(
                            update={
                                "evidence_complete": False,
                                "response_status": "qualified_partial",
                                "stop_reason": "slot_plan_degraded",
                            }
                        )
                    }
                )
            state["final_slot_resolutions"] = evaluation.slot_resolutions
            _record_repair_stop_reason(
                repairs=state["repairs"],
                contract=contract,
                evaluation=evaluation,
            )
            return evaluation

        def plan_repair(
            contract: QueryContract,
            evaluation: SufficiencyEvaluation,
            query_id: str,
            repair_round_index: int,
        ) -> Sequence[Any]:
            repair = build_repair_plan(
                contract=contract,
                sufficiency=evaluation,
                query_id=query_id,
                repair_round_index=repair_round_index,
                final_budget_available=self._policy_runtime.has_final_reserve(deadline),
            )
            state["repairs"].append(repair)
            return repair.tasks

        def record_repair_terminal(reason: str) -> None:
            if not state["repairs"] or not state["repairs"][-1].tasks:
                return
            current_reason = state["repairs"][-1].stop_reason
            precedence = {
                None: 0,
                "continue_repair": 0,
                "repair_round_cap_reached": 1,
                "deadline_exhausted": 2,
                "final_budget_protected": 2,
                "no_repairable_slots": 3,
                "evidence_complete": 4,
            }
            if precedence.get(reason, 0) < precedence.get(current_reason, 0):
                return
            state["repairs"][-1] = state["repairs"][-1].model_copy(
                update={"stop_reason": reason}
            )

        async def prose_curate(
            _: str, __: QueryContract, packets: tuple[EvidencePacket, ...]
        ) -> tuple[EvidencePacket, ...]:
            # Candidate extraction is deterministic and provenance-bound.  No
            # prose model is permitted to invent or promote evidence here.
            return packets

        async def resolve_conflicts(
            _: QueryContract,
            __: tuple[EvidencePacket, ...],
            evaluation: SufficiencyEvaluation,
        ) -> ConflictStageResult:
            return ConflictStageResult(sufficiency=evaluation)

        async def pack(
            _: str,
            contract: QueryContract,
            packets: tuple[EvidencePacket, ...],
            __: SufficiencyEvaluation,
        ) -> PackedEvidenceContext:
            setup_input = _setup_positive_int(
                setup_snapshot,
                "setup_max_input_tokens",
                "max_input_tokens",
                default=8192,
            )
            packer = EvidenceContextPacker(
                setup_input_ceiling=min(setup_input, contract.runtime_token_budget),
                remaining_runtime_tokens=contract.runtime_token_budget,
                final_output_reserve=min(
                    _setup_positive_int(
                        setup_snapshot,
                        "setup_max_output_tokens",
                        "max_output_tokens",
                        default=1,
                    ),
                    1536,
                ),
                thinking_token_reserve=_thinking_reserve(setup_snapshot),
                instruction="Answer only from the supplied evidence packets.",
                question=question,
                contract=contract,
            )
            packed = packer.pack(packets, required_slots=contract)
            state["pack"] = packed
            return packed

        async def generate_final(
            final_question: str,
            final_contract: QueryContract,
            packed: PackedEvidenceContext,
            resolutions: tuple[SlotResolution, ...],
            arbitration: Any | None,
            sufficiency_report: SufficiencyReport,
        ) -> FinalAnswerResult:
            controller = state["budget_controller"]
            assert isinstance(controller, RunBudgetController)
            if sufficiency_report.response_status == "complete":
                assert all(
                    resolution.status == "supported"
                    for resolution in resolutions
                    if any(
                        slot.slot_id == resolution.slot_id and slot.required
                        for slot in final_contract.required_slots
                    )
                )
            return await generate_final_answer(
                question=final_question,
                contract=final_contract,
                packed_packets=packed,
                slot_resolutions=resolutions,
                llm_invoker=BudgetedLlmInvoker(
                    controller=controller,
                    provider_factory=self._provider_factory,
                    observer=llm_call_observer,
                    provider_name=provider_name,
                    model_name=model_name,
                    capture_policy=setup_snapshot.get("prompt_capture_policy"),
                ),
                arbitration=arbitration,
                sufficiency_report=sufficiency_report,
            )

        def deterministic_partial(
            partial_contract: QueryContract, evaluation: SufficiencyEvaluation
        ) -> FinalAnswerResult:
            return FinalAnswerResult(
                response_status=evaluation.report.response_status,
                answer=render_verified_answer(
                    (),
                    (),
                    unresolved_requirements=_unresolved_requirements(
                        partial_contract, evaluation.slot_resolutions
                    ),
                ),
            )

        core = V9ExecutionCore(
            stages=V9ExecutionStages(
                resolve_scope=resolve_scope,
                plan_contract=plan_contract,
                retrieve=retrieve,
                deterministic_candidates=deterministic_candidates,
                evaluate_sufficiency=sufficiency,
                plan_repair=plan_repair,
                prose_curate=prose_curate,
                resolve_conflicts=resolve_conflicts,
                pack=pack,
                generate_final=generate_final,
                deterministic_partial=deterministic_partial,
                record_repair_terminal=record_repair_terminal,
            ),
            runtime=self._policy_runtime,
        )
        runtime_context = V9RuntimeContext(
            cancellation_token=cancellation,
            event_sink=_discard_event,
            budget_controller=state,
            deadline=deadline,
            clock=lambda: datetime.now(timezone.utc),
            llm_invoker=object(),
        )
        try:
            executed = await core.execute(request, runtime_context=runtime_context)
        except _ConfigurationIncompatible as error:
            return _configuration_incompatible_result(
                question=question,
                trace_id=trace_id,
                stage=error.stage,
                feasibility=error.feasibility,
                contract=state["contract"],
            )

        controller = state["budget_controller"]
        assert isinstance(controller, RunBudgetController)
        budget_snapshot = await controller.snapshot()
        metrics = executed.metrics.model_copy(
            update={
                "provider_attempt_count": budget_snapshot.provider_attempt_count,
                "reserved_tokens": budget_snapshot.reserved_tokens,
                "reconciled_tokens": budget_snapshot.reconciled_tokens,
            }
        )
        final = executed.final_answer or FinalAnswerResult(
            response_status="insufficient"
        )
        graph_execution = state["graph_execution"] or _initial_graph_execution(
            state["contract"]
        )
        visual_execution = state["visual_execution"] or _initial_visual_execution(
            state["contract"]
        )
        if (
            executed.sufficiency is not None
            and final.response_status == "complete"
            and executed.sufficiency.response_status != "complete"
        ):
            # Defensive invariant: capability constraints were already applied
            # before generation, so this branch must never be the primary guard.
            final = final.model_copy(
                update={"response_status": executed.sufficiency.response_status}
            )
        used_packets = [
            packet
            for packet in state["evidence_packets"]
            if packet.evidence_id in set(final.used_evidence_ids)
        ]
        documents = used_evidence_documents(used_packets, final)
        packed = state["pack"]
        trace = {
            "trace_id": trace_id,
            "mode": "agentic",
            "agentic_execution_version": "v9",
            "execution_profile": AGENTIC_EVAL_PROFILE,
            "response_status": final.response_status,
            "agentic_v9": {
                "schema_version": "1",
                "query_contract": state["contract"].model_dump(mode="json"),
                "evidence_packets": [
                    packet.model_dump(mode="json")
                    for packet in state["evidence_packets"]
                ],
                "locator_diagnostics": state["locator_diagnostics"],
                "retrieval_diagnostics": state["retrieval_diagnostics"],
                "slot_resolutions": [
                    resolution.model_dump(mode="json")
                    for resolution in state["final_slot_resolutions"]
                ],
                "sufficiency": executed.sufficiency.model_dump(mode="json")
                if executed.sufficiency
                else None,
                "context_pack": _context_pack_projection(packed),
                "graph_execution": graph_execution,
                "visual_execution": visual_execution,
                "budget_reservations": [
                    item.model_dump(mode="json")
                    for item in await controller.reservations()
                ],
                "repairs": [
                    repair.model_dump(mode="json") for repair in state["repairs"]
                ],
                "conflicts": [],
                "final_claims": [
                    claim.model_dump(mode="json") for claim in final.claims
                ],
                "metrics": metrics.model_dump(mode="json"),
                "completion": {"status": final.response_status},
            },
        }
        return RAGResult(
            answer=final.answer,
            source_doc_ids=[str(item.metadata["doc_id"]) for item in documents],
            documents=documents,
            usage={"total_tokens": budget_snapshot.total_tokens},
            agent_trace=trace,
        )


async def _retrieve_documents(
    user_id: str,
    question: str,
    authorized_doc_ids: list[str],
    *,
    rerank_with_scores: Callable[[str, list[Document], int], list[tuple[Document, float]]]
    | None = None,
    reranker_available: bool | None = None,
) -> V9RetrievalSelection:
    """Retrieve only within the source scope, without HyDE or query expansion."""
    if not authorized_doc_ids:
        return _retrieval_selection(
            authorized_doc_ids=authorized_doc_ids,
            pre_filter_documents=[],
            documents=[],
            reranking_available=False,
            fallback_reason="not_initialized",
        )
    retriever = await get_user_retriever_async(user_id, k=8, plain_mode=False)
    raw = await retrieve_hybrid_documents(
        question,
        retriever,
        enable_hyde=False,
        enable_multi_query=False,
    )
    available = (
        DocumentReranker.is_initialized()
        if reranker_available is None
        else reranker_available
    )
    try:
        filtered = await asyncio.wait_for(
            run_in_threadpool(
                filter_and_rerank_retrieval,
                question,
                raw,
                doc_ids=authorized_doc_ids,
                enable_reranking=True,
                reranker_available=available,
                rerank_with_scores=rerank_with_scores,
                strict_reranking=True,
            ),
            timeout=8.0,
        )
    except asyncio.TimeoutError:
        return _retrieval_selection(
            authorized_doc_ids=authorized_doc_ids,
            pre_filter_documents=list(raw.documents),
            documents=_authorized_documents(raw.documents, authorized_doc_ids),
            reranking_available=available,
            fallback_reason="timeout",
        )
    except Exception:
        return _retrieval_selection(
            authorized_doc_ids=authorized_doc_ids,
            pre_filter_documents=list(raw.documents),
            documents=_authorized_documents(raw.documents, authorized_doc_ids),
            reranking_available=available,
            fallback_reason="exception",
        )

    reranking = filtered.metadata["reranking"]
    return _retrieval_selection(
        authorized_doc_ids=authorized_doc_ids,
        pre_filter_documents=list(raw.documents),
        documents=list(filtered.documents),
        reranking_available=bool(reranking["available"]),
        fallback_reason=("not_initialized" if not reranking["available"] else None),
        pre_rerank_ranks=reranking["pre_rerank_ranks"],
        post_rerank_ranks=reranking["post_rerank_ranks"],
        post_filter_count=len(filtered.metadata["filtering"]["post_filter_ranks"]),
    )


def _normalize_retrieval_selection(
    result: RetrievalAdapterResult,
    *,
    authorized_doc_ids: list[str],
) -> V9RetrievalSelection:
    """Keep legacy document-list adapters source-safe at one compatibility boundary."""
    if isinstance(result, V9RetrievalSelection):
        source_filter = result.diagnostics.get("source_filter", {})
        reranking = result.diagnostics.get("reranking", {})
        documents = _authorized_documents(result.documents, authorized_doc_ids)
        return _retrieval_selection(
            authorized_doc_ids=authorized_doc_ids,
            pre_filter_documents=result.documents,
            documents=documents,
            reranking_available=bool(reranking.get("available")),
            fallback_reason=reranking.get("fallback_reason"),
            pre_rerank_ranks=reranking.get("pre_rerank_ranks"),
            post_rerank_ranks=reranking.get("post_rerank_ranks"),
            pre_filter_count=source_filter.get("pre_filter_count"),
            post_filter_count=(
                source_filter.get("post_filter_count")
                if len(documents) == len(result.documents)
                else len(documents)
            ),
        )
    return _retrieval_selection(
        authorized_doc_ids=authorized_doc_ids,
        pre_filter_documents=result,
        documents=_authorized_documents(result, authorized_doc_ids),
        reranking_available=False,
        fallback_reason="not_initialized",
    )


def _authorized_documents(
    documents: Sequence[Document], authorized_doc_ids: list[str]
) -> list[Document]:
    allowed = set(authorized_doc_ids)
    return [
        document
        for document in documents
        if get_document_id(document.metadata) in allowed
    ]


def _retrieval_selection(
    *,
    authorized_doc_ids: list[str],
    pre_filter_documents: Sequence[Document],
    documents: Sequence[Document],
    reranking_available: bool,
    fallback_reason: str | None,
    pre_rerank_ranks: list[dict[str, Any]] | None = None,
    post_rerank_ranks: list[dict[str, Any]] | None = None,
    pre_filter_count: int | None = None,
    post_filter_count: int | None = None,
) -> V9RetrievalSelection:
    selected_documents = tuple(documents)
    return V9RetrievalSelection(
        documents=selected_documents,
        diagnostics={
            "source_filter": {
                "authorized_doc_ids": list(authorized_doc_ids),
                "pre_filter_count": (
                    len(pre_filter_documents)
                    if pre_filter_count is None
                    else pre_filter_count
                ),
                "post_filter_count": (
                    len(documents) if post_filter_count is None else post_filter_count
                ),
            },
            "reranking": {
                "enabled": True,
                "available": reranking_available,
                "fallback_reason": fallback_reason,
                "pre_rerank_ranks": _diagnostic_rank_rows(
                    pre_rerank_ranks
                    if pre_rerank_ranks is not None
                    else _retrieval_rank_rows(documents)
                ),
                "post_rerank_ranks": _diagnostic_rank_rows(
                    post_rerank_ranks
                    if post_rerank_ranks is not None
                    else _retrieval_rank_rows(documents)
                ),
                "selected_count": len(selected_documents),
            },
        },
    )


def _retrieval_rank_rows(documents: Sequence[Document]) -> list[dict[str, Any]]:
    return [
        {"rank": rank, "metadata": _diagnostic_metadata(document.metadata), "score": None}
        for rank, document in enumerate(documents, start=1)
    ]


def _diagnostic_rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **{
                key: row[key]
                for key in ("rank", "pre_rerank_rank", "score")
                if key in row
            },
            "metadata": _diagnostic_metadata(dict(row.get("metadata") or {})),
        }
        for row in rows
    ]


def _diagnostic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep identifiers and locators while excluding retrieved chunk content."""
    allowed_keys = (
        "doc_id",
        "original_doc_uid",
        "chunk_id",
        "parent_id",
        "page_number",
        "pdf_page_index",
        "printed_page_label",
        "section",
        "table_id",
        "figure_id",
        "formula_id",
        "document_name",
        "source",
    )
    return {key: metadata[key] for key in allowed_keys if key in metadata}


async def _locate_graph_documents(
    question: str,
    user_id: str,
    vector_documents: list[Document],
    authorized_doc_ids: list[str],
    contract: QueryContract,
) -> GraphSourceLocatorResult:
    """Run the production graph boundary as a source locator, never context."""
    return await locate_graph_sources(
        question=question,
        user_id=user_id,
        vector_documents=vector_documents,
        requested_doc_ids=authorized_doc_ids,
        graph_execution_hints={
            "graph_evidence_mode": "locator_to_chunk",
            "graph_feature_flags": {
                "agentic_v9_required_locator": contract.graph_policy
                == "required_locator",
            },
        },
        required_modalities=[],
        evidence_mode="locator_to_chunk",
        bundle_locator=get_graph_evidence_bundle,
        search_mode="generic",
    )


def _apply_required_capability_constraints(
    *,
    contract: QueryContract,
    evaluation: SufficiencyEvaluation,
    graph_execution: dict[str, Any] | None,
    visual_execution: dict[str, Any] | None,
) -> SufficiencyEvaluation:
    """Downgrade affected required slots before packing or final generation."""
    required_slot_ids = {
        slot.slot_id for slot in contract.required_slots if slot.required
    }
    required_visual_slot_ids = {
        slot.slot_id
        for slot in contract.required_slots
        if slot.required and slot.visual_policy == "required"
    }
    if contract.visual_required and not required_visual_slot_ids:
        # Legacy v1 contracts expressed required visual evidence only through
        # the contract-level flag, so retain their all-required-slot behavior.
        required_visual_slot_ids = set(required_slot_ids)
    graph_failed = bool(
        contract.graph_policy == "required_locator"
        and (graph_execution or {}).get("state") != "executed"
    )
    visual_supported_slot_ids = set(
        (visual_execution or {}).get("supported_slot_ids") or ()
    )
    missing_required_visual_slot_ids = (
        required_visual_slot_ids - visual_supported_slot_ids
    )
    visual_failed = bool(missing_required_visual_slot_ids)
    if not graph_failed and not visual_failed:
        return evaluation

    adjusted: list[SlotResolution] = []
    for resolution in evaluation.slot_resolutions:
        if resolution.slot_id not in required_slot_ids:
            adjusted.append(resolution)
            continue
        if visual_failed and resolution.slot_id in missing_required_visual_slot_ids:
            reason = (visual_execution or {}).get("failure_reason") or (
                "Required visual evidence is unavailable."
            )
            adjusted.append(
                SlotResolution(
                    slot_id=resolution.slot_id,
                    status="explicitly_unavailable",
                    reason=reason,
                    resolution_stage="required_visual_capability",
                )
            )
            continue
        if not graph_failed:
            adjusted.append(resolution)
            continue
        graph_failure_reason = (graph_execution or {}).get("failure_reason")
        graph_stage_failed = (graph_execution or {}).get("fallback") == (
            "stage_execution_failed"
        )
        adjusted.append(
            SlotResolution(
                slot_id=resolution.slot_id,
                status=(
                    "explicitly_unavailable" if graph_stage_failed else "not_found"
                ),
                reason=graph_failure_reason
                or "Required graph source evidence was not found.",
                resolution_stage="required_graph_capability",
            )
        )

    adjusted_tuple = tuple(adjusted)
    required = tuple(
        resolution
        for resolution in adjusted_tuple
        if resolution.slot_id in required_slot_ids
    )
    supported = [
        resolution.slot_id
        for resolution in required
        if resolution.status == "supported"
    ]
    conflicted = [
        resolution.slot_id
        for resolution in required
        if resolution.status == "conflicted"
    ]
    unavailable = [
        resolution.slot_id
        for resolution in required
        if resolution.status == "explicitly_unavailable"
    ]
    not_found = [
        resolution.slot_id
        for resolution in required
        if resolution.status == "not_found"
    ]
    evidence_complete = bool(required_slot_ids) and len(supported) == len(
        required_slot_ids
    )
    answerable = bool(supported)
    report = SufficiencyReport(
        evidence_complete=evidence_complete,
        answerable=answerable,
        response_status=(
            "complete"
            if evidence_complete
            else "qualified_partial"
            if answerable
            else "insufficient"
        ),
        supported_slot_ids=supported,
        conflicted_slot_ids=conflicted,
        explicitly_unavailable_slot_ids=unavailable,
        not_found_slot_ids=not_found,
        stop_reason="required_capability_unavailable",
    )
    return SufficiencyEvaluation(
        slot_resolutions=adjusted_tuple,
        report=report,
        repairable_slot_ids=tuple(not_found),
        repair_stopped_slot_ids=tuple(unavailable),
    )


def _unresolved_requirements(
    contract: QueryContract,
    resolutions: Sequence[SlotResolution],
) -> tuple[UnresolvedRequirement, ...]:
    resolutions_by_slot = {resolution.slot_id: resolution for resolution in resolutions}
    rows: list[UnresolvedRequirement] = []
    for slot in contract.required_slots:
        if not slot.required:
            continue
        resolution = resolutions_by_slot.get(slot.slot_id)
        if resolution is not None and resolution.status == "supported":
            continue
        rows.append(
            UnresolvedRequirement(
                slot_id=slot.slot_id,
                reason=(
                    resolution.reason
                    if resolution is not None and resolution.reason
                    else "Required source-bound evidence was not found."
                ),
            )
        )
    return tuple(rows)


def _initial_graph_execution(contract: QueryContract | None) -> dict[str, Any]:
    policy = contract.graph_policy if contract is not None else "never"
    return {
        "policy": policy,
        "state": "not_requested" if policy == "never" else "not_triggered",
        "attempted": False,
        "failure_reason": None,
        "route": None,
        "path": None,
        "fallback": None,
        "latency_ms": None,
        "candidate_item_ids": [],
        "resolved_item_ids": [],
        "scope_approved_item_ids": [],
        "scored_item_ids": [],
        "packed_item_ids": [],
        "resolved_source_doc_ids": [],
        "resolved_source_chunk_ids": [],
    }


def _graph_execution_projection(
    located: GraphSourceLocatorResult,
) -> dict[str, Any]:
    resolved_documents = list(located.resolved_source_documents)
    has_eligible_source = bool(resolved_documents) and not located.fallback
    return {
        "policy": "required_locator",
        "state": "executed" if has_eligible_source else "required_but_not_satisfied",
        "attempted": True,
        "failure_reason": (
            None
            if has_eligible_source
            else (located.fallback or "no_eligible_graph_source_evidence")
        ),
        "route": located.route,
        "path": located.path,
        "fallback": located.fallback,
        "latency_ms": located.graph_latency_ms,
        "candidate_item_ids": list(located.candidate_item_ids),
        "resolved_item_ids": list(located.resolved_item_ids),
        "scope_approved_item_ids": list(located.scope_approved_item_ids),
        "scored_item_ids": list(located.scored_item_ids),
        "packed_item_ids": list(located.packed_item_ids),
        "resolved_source_doc_ids": list(located.resolved_source_doc_ids),
        "resolved_source_chunk_ids": list(located.resolved_source_chunk_ids),
    }


def _failed_required_stage(*, policy: str, error: Exception) -> dict[str, Any]:
    """Project an admitted stage failure without leaking provider internals."""
    required = policy in {"visual_required", "required_locator"}
    return {
        "policy": policy,
        "required": policy == "visual_required",
        "state": (
            "required_but_not_satisfied" if required else "attempted_without_evidence"
        ),
        "attempted": True,
        "failure_reason": f"{error.__class__.__name__}:stage_execution_failed",
        "route": None,
        "path": None,
        "fallback": "stage_execution_failed",
        "latency_ms": None,
        "candidate_item_ids": [],
        "resolved_item_ids": [],
        "scope_approved_item_ids": [],
        "scored_item_ids": [],
        "packed_item_ids": [],
        "resolved_source_doc_ids": [],
        "resolved_source_chunk_ids": [],
        "selected_asset_count": None,
        "dropped_asset_count": None,
        "evidence_packet_count": None,
        "supported_slot_ids": [],
    }


async def _extract_visual_evidence(
    task: Any,
    assets: list[VisualAssetCandidate],
    question: str,
    controller: RunBudgetController,
) -> VisualEvidenceExtractionResult:
    """Extract only manifest-resolved, source-bound visual evidence."""
    extractor = VisualEvidenceExtractor(
        BudgetedLlmInvoker(
            controller=controller,
            provider_factory=_provider_for_purpose,
            observer=current_llm_call_observer(),
        )
    )
    return await extractor.extract(
        task=task,
        assets=assets,
        question_fragment=question,
    )


def _requested_visual_slots(contract: QueryContract) -> list[RequiredSlot]:
    slots = [
        slot
        for slot in contract.required_slots
        if slot.visual_policy in {"preferred", "required"}
    ]
    if contract.visual_required and not slots:
        return [slot for slot in contract.required_slots if slot.required]
    return slots


def _bind_visual_result_to_contract(
    *,
    result: VisualEvidenceExtractionResult,
    contract: QueryContract,
    allowed_slot_ids: set[str] | None = None,
) -> VisualEvidenceExtractionResult:
    """Remove source/locator slot claims outside the exact query contract."""
    slots_by_id = {slot.slot_id: slot for slot in _requested_visual_slots(contract)}
    packets: list[EvidencePacket] = []
    for packet in result.packets:
        bound_slot_ids = [
            slot_id
            for slot_id in packet.slot_ids
            if (allowed_slot_ids is None or slot_id in allowed_slot_ids)
            if (slot := slots_by_id.get(slot_id)) is not None
            and _slot_accepts_visual_source(
                slot=slot,
                doc_id=packet.source.doc_id,
                figure_id=packet.locator.figure_id,
                table_id=packet.locator.table_id,
                formula_id=packet.locator.section,
            )
        ]
        if bound_slot_ids:
            packets.append(packet.model_copy(update={"slot_ids": bound_slot_ids}))
    return result.model_copy(update={"packets": tuple(packets)})


def _slot_accepts_visual_source(
    *,
    slot: RequiredSlot,
    doc_id: str,
    figure_id: object,
    table_id: object,
    formula_id: object,
) -> bool:
    if slot.authorized_source_doc_ids and doc_id not in slot.authorized_source_doc_ids:
        return False
    visual_hints = [
        hint
        for hint in slot.locator_hints
        if hint.strip()
        .casefold()
        .startswith(("figure", "fig.", "table", "formula", "equation"))
    ]
    if not visual_hints:
        return True
    return any(
        _visual_hint_matches(
            hint,
            figure_id=figure_id,
            table_id=table_id,
            formula_id=formula_id,
        )
        for hint in visual_hints
    )


def _visual_hint_matches(
    hint: str,
    *,
    figure_id: object,
    table_id: object,
    formula_id: object,
) -> bool:
    normalized_hint = _visual_locator_key(hint)
    prefix = hint.strip().casefold()
    if prefix.startswith(("figure", "fig.")):
        return _visual_identifier_matches(normalized_hint, "figure", figure_id)
    if prefix.startswith("table"):
        return _visual_identifier_matches(normalized_hint, "table", table_id)
    return _visual_identifier_matches(normalized_hint, "formula", formula_id)


def _visual_identifier_matches(
    normalized_hint: str,
    category: str,
    identifier: object,
) -> bool:
    if not isinstance(identifier, str) or not identifier.strip():
        return False
    identifier_key = _visual_locator_key(identifier)
    hint_suffix = normalized_hint.removeprefix(category)
    identifier_suffix = identifier_key.removeprefix(category)
    return not hint_suffix or hint_suffix == identifier_suffix


def _visual_locator_key(value: str) -> str:
    key = "".join(character for character in value.casefold() if character.isalnum())
    if key.startswith("fig") and not key.startswith("figure"):
        key = f"figure{key.removeprefix('fig')}"
    if key.startswith("equation"):
        key = f"formula{key.removeprefix('equation')}"
    return key


def _initial_visual_execution(contract: QueryContract | None) -> dict[str, Any]:
    required = bool(contract and contract.visual_required)
    requested = bool(contract and contract.visual_requested)
    return {
        "required": required,
        "state": "not_triggered" if requested else "not_requested",
        "attempted": False,
        "failure_reason": None,
        "selected_asset_count": None,
        "dropped_asset_count": None,
        "evidence_packet_count": None,
        "supported_slot_ids": [],
        "manifest_count": None,
        "authorized_count": None,
        "locator_match_count": None,
        "loaded_count": None,
        "selected_count": None,
        "dropped_count": None,
        "covered_slot_count": None,
    }


def _visual_execution_projection(
    result: VisualEvidenceExtractionResult,
    *,
    required: bool,
    resolution_diagnostics: VisualAssetResolutionDiagnostics | None = None,
) -> dict[str, Any]:
    packet_count = len(result.packets)
    supported_slot_ids = sorted(
        {slot_id for packet in result.packets for slot_id in packet.slot_ids}
    )
    diagnostics = (
        resolution_diagnostics.model_dump()
        if resolution_diagnostics is not None
        else {
            "manifest_count": None,
            "authorized_count": None,
            "locator_match_count": None,
            "loaded_count": None,
            "selected_count": None,
            "dropped_count": None,
            "evidence_packet_count": None,
            "covered_slot_count": None,
            "terminal_reason": None,
        }
    )
    diagnostics.update(
        {
            "evidence_packet_count": packet_count,
            "covered_slot_count": len(supported_slot_ids),
        }
    )
    return {
        **diagnostics,
        "required": required,
        "state": (
            "executed"
            if packet_count
            else "required_but_not_satisfied"
            if required
            else "attempted_without_evidence"
        ),
        "attempted": True,
        "failure_reason": (
            None
            if packet_count
            else (
                result.dropped_assets[0].reason
                if result.dropped_assets
                else diagnostics["terminal_reason"] or "no_eligible_visual_evidence"
            )
        ),
        "selected_asset_count": len(result.located_assets),
        "dropped_asset_count": len(result.dropped_assets),
        "evidence_packet_count": packet_count,
        "supported_slot_ids": supported_slot_ids,
    }


async def _resolve_document_references(
    user_id: str, references: list[str]
) -> dict[str, list[str]]:
    from data_base.repository import resolve_document_references

    return await resolve_document_references(user_id=user_id, references=references)


def _provider_for_purpose(_: str) -> Any:
    return get_llm("synthesizer")


def _chunk_projection(document: Document, index: int) -> dict[str, Any]:
    metadata = dict(document.metadata or {})
    doc_id = get_document_id(metadata)
    return {
        "doc_id": doc_id,
        "chunk_id": str(metadata.get("chunk_id") or f"chunk-{index + 1}"),
        "text": str(document.page_content or ""),
        "page_number": metadata.get("page_number"),
        "section": metadata.get("section"),
        "printed_page_label": metadata.get("printed_page_label"),
        "figure_id": metadata.get("figure_id"),
        "table_id": metadata.get("table_id"),
        "formula_id": metadata.get("formula_id"),
        "bbox": metadata.get("bbox"),
    }


def _evidence_packets_for_results(
    *,
    results: tuple[TaskRetrievalResult, ...],
    contract: QueryContract,
    trace_id: str,
    tasks_by_id: dict[str, RetrievalTask],
    locator_diagnostics: list[dict[str, Any]] | None = None,
) -> list[EvidencePacket]:
    packets: list[EvidencePacket] = []
    for task_result in results:
        task_id = task_result.task_id
        task = tasks_by_id.get(task_id)
        if task is None:
            continue
        for index, chunk in enumerate(task_result.retrieval.chunks):
            doc_id = chunk.get("doc_id")
            text = str(chunk.get("text") or "").strip()
            if not isinstance(doc_id, str) or not doc_id or not text:
                continue
            if doc_id not in task.source_scope.authorized_doc_ids:
                continue
            slot_ids = _slot_ids_supported_by_chunk(
                contract=contract,
                slot_ids=task.target_slot_ids,
                doc_id=doc_id,
                chunk=chunk,
                task_id=task_id,
                locator_diagnostics=locator_diagnostics,
            )
            if not slot_ids:
                continue
            digest = hashlib.sha256(
                f"{trace_id}:{task_id}:{doc_id}:{chunk.get('chunk_id')}:{index}".encode()
            ).hexdigest()[:24]
            page = chunk.get("page_number")
            locator_values = {
                "pdf_page_index": (
                    page if isinstance(page, int) and page >= 0 else None
                ),
                "printed_page_label": chunk.get("printed_page_label"),
                "section": (
                    str(chunk.get("formula_id"))
                    if chunk.get("formula_id")
                    else str(chunk.get("section") or "") or None
                ),
                "table_id": chunk.get("table_id"),
                "figure_id": chunk.get("figure_id"),
                "bbox": chunk.get("bbox"),
            }
            if not any(
                value is not None and value != "" for value in locator_values.values()
            ):
                locator_values["section"] = "retrieved_context"
            locator = SourceLocator(**locator_values)
            packets.append(
                EvidencePacket(
                    schema_version="1",
                    evidence_id=f"evidence:{digest}",
                    task_id=task_id,
                    round_id=task_id.split(":")[-2] if ":" in task_id else "round-1",
                    query_id=trace_id,
                    slot_ids=list(slot_ids),
                    statement=text,
                    support_type="direct",
                    source=EvidenceSource(
                        doc_id=doc_id, chunk_id=str(chunk.get("chunk_id") or index + 1)
                    ),
                    scope=EvidenceScope(),
                    locator=locator,
                    validation_status="deterministic_valid",
                )
            )
    return packets


def _record_repair_evidence(
    *,
    repairs: list[RepairPlan],
    results: tuple[TaskRetrievalResult, ...],
    packets: list[EvidencePacket],
) -> None:
    """Attach evidence produced by a repair round to its durable trace record."""
    result_task_ids = {result.task_id for result in results}
    for index in range(len(repairs) - 1, -1, -1):
        repair = repairs[index]
        repair_task_ids = {task.task_id for task in repair.tasks}
        if not result_task_ids.intersection(repair_task_ids):
            continue
        evidence_ids = [
            packet.evidence_id
            for packet in packets
            if packet.task_id in repair_task_ids
        ]
        repairs[index] = repair.model_copy(
            update={"resulting_evidence_ids": list(dict.fromkeys(evidence_ids))}
        )
        return


def _record_repair_stop_reason(
    *,
    repairs: list[RepairPlan],
    contract: QueryContract,
    evaluation: SufficiencyEvaluation,
) -> None:
    """Persist the post-sufficiency decision for the latest executed repair."""
    if not repairs or not repairs[-1].tasks:
        return
    repair = repairs[-1]
    if repair.stop_reason not in {None, "continue_repair"}:
        return
    if evaluation.report.evidence_complete:
        stop_reason = "evidence_complete"
    elif not evaluation.repairable_slot_ids:
        stop_reason = "no_repairable_slots"
    else:
        cap = min(
            contract.max_repair_rounds,
            ROUTE_REPAIR_CAPS[contract.route],
            MAX_REPAIR_ROUNDS,
        )
        stop_reason = (
            "repair_round_cap_reached"
            if repair.repair_round_index >= cap
            else "continue_repair"
        )
    repairs[-1] = repair.model_copy(update={"stop_reason": stop_reason})


def _slot_ids_supported_by_chunk(
    *,
    contract: QueryContract,
    slot_ids: list[str],
    doc_id: str,
    chunk: dict[str, Any],
    task_id: str = "",
    locator_diagnostics: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Bind a chunk independently to each compatible atomic slot."""
    slots_by_id = {slot.slot_id: slot for slot in contract.required_slots}
    scope = contract.resolved_source_scope
    if scope is None:
        return []
    authorized: list[str] = []
    for slot_id in slot_ids:
        slot = slots_by_id.get(slot_id)
        if slot is None:
            continue
        if doc_id not in authorized_doc_ids_for_slot(slot, scope):
            continue
        state = structured_locator_state(slot.locator_hints, chunk)
        accepted = state != "mismatched"
        if locator_diagnostics is not None:
            diagnostic = {
                "task_id": task_id,
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "slot_id": slot_id,
                "state": state,
                "accepted": accepted,
            }
            if diagnostic not in locator_diagnostics:
                locator_diagnostics.append(diagnostic)
        if not accepted:
            continue
        authorized.append(slot_id)
    return authorized


def _configuration_incompatible_result(
    *,
    question: str,
    trace_id: str,
    stage: str,
    feasibility: FeasibilityResult,
    contract: QueryContract | None = None,
) -> RAGResult:
    reason = feasibility.reason or "configuration_incompatible"
    return RAGResult(
        answer="Configuration is incompatible with the Agentic v9 execution policy.",
        source_doc_ids=[],
        documents=[],
        usage={"total_tokens": 0},
        agent_trace={
            "trace_id": trace_id,
            "mode": "agentic",
            "agentic_execution_version": "v9",
            "execution_profile": AGENTIC_EVAL_PROFILE,
            "response_status": "configuration_incompatible",
            "agentic_v9": {
                "schema_version": "1",
                "query_contract": contract.model_dump(mode="json")
                if contract
                else None,
                "evidence_packets": [],
                "slot_resolutions": [],
                "sufficiency": None,
                "context_pack": None,
                "budget_reservations": [],
                "repairs": [],
                "conflicts": [],
                "final_claims": [],
                "metrics": {},
                "configuration_incompatible": {
                    "stage": stage,
                    "reason": reason,
                    "reserved_tokens": feasibility.reserved_tokens,
                },
                "completion": {"status": "configuration_incompatible"},
            },
            "question": question,
        },
    )


def _setup_positive_int(snapshot: dict[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        value = snapshot.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return default


def _thinking_reserve(snapshot: dict[str, Any]) -> int:
    if not bool(snapshot.get("thinking_mode", snapshot.get("thinking_enabled", False))):
        return 0
    value = snapshot.get("thinking_token_reserve", snapshot.get("thinking_budget", 0))
    return value if isinstance(value, int) and value >= 0 else 0


def _pre_route_token_budget(snapshot: dict[str, Any]) -> int:
    return _setup_positive_int(snapshot, "runtime_token_budget", default=50_000)


def _pre_route_llm_calls(snapshot: dict[str, Any]) -> int:
    return _setup_positive_int(snapshot, "max_llm_calls", default=5)


def _final_input_reserve(snapshot: dict[str, Any], runtime_token_budget: int) -> int:
    return min(
        _setup_positive_int(
            snapshot, "setup_max_input_tokens", "max_input_tokens", default=8192
        ),
        8192,
        max(runtime_token_budget // 2, 1),
    )


def _context_pack_projection(
    packed: PackedEvidenceContext | None,
) -> dict[str, Any] | None:
    if packed is None:
        return None
    return {
        "packed_evidence_ids": [packet.evidence_id for packet in packed.packets],
        "dropped_evidence_ids": list(packed.dropped_packet_ids),
        "token_count": packed.estimated_input_tokens,
    }


async def _discard_event(_: V9ExecutionEvent) -> None:
    return None


__all__ = ["AgenticV9CampaignRuntime"]
