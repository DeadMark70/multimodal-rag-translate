"""Concrete evidence-first adapter for Agentic v9 campaign executions.

The v9 core deliberately owns orchestration only.  This module is the
production evaluation adapter: it resolves a fail-closed source scope, runs
the typed core, and projects only cited evidence back into the existing
``RAGResult`` contract.  Durable materialization remains in the worker because
only the worker knows the promoted run and attempt identities.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from typing import Any

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
from data_base.agentic_v9.budgeted_llm import BudgetedLlmInvoker
from data_base.agentic_v9.context_packer import EvidenceContextPacker, PackedEvidenceContext
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
from data_base.agentic_v9.repair import build_repair_plan
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    FinalAnswerResult,
    QueryContract,
    RagRetrievalResult,
    RequiredSlot,
    ResolvedSourceScope,
    SlotResolution,
    SourceLocator,
    SufficiencyReport,
    TaskRetrievalResult,
    V9ExecutionEvent,
    V9ExecutionRequest,
    V9RuntimeContext,
    UnresolvedRequirement,
)
from data_base.agentic_v9.sufficiency_gate import SufficiencyEvaluation, evaluate_sufficiency
from data_base.agentic_v9.asset_locator import VisualAssetCandidate
from data_base.agentic_v9.visual_evidence_extractor import (
    VisualEvidenceExtractionResult,
    VisualEvidenceExtractor,
)
from data_base.document_metadata import get_document_id
from data_base.rag_filtering import filter_and_rerank_retrieval
from data_base.rag_graph_locator import GraphSourceLocatorResult, locate_graph_sources
from data_base.rag_retrieval import retrieve_hybrid_documents
from data_base.vector_store_manager import get_user_retriever_async
from evaluation.agentic_campaign_adapter import used_evidence_documents
from evaluation.agentic_v9_admission import (
    DocumentReferenceResolver,
    build_v9_admission_contract,
)
from evaluation.retrieval_profiles import AGENTIC_EVAL_PROFILE


RetrievalAdapter = Callable[[str, str, list[str]], Awaitable[list[Document]]]
GraphLocator = Callable[
    [str, str, list[Document], list[str], QueryContract],
    Awaitable[GraphSourceLocatorResult],
]
VisualExtractor = Callable[
    [Any, list[Document], str, RunBudgetController],
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
        provider_factory: ProviderFactory | None = None,
        policy_runtime: V9ExecutionPolicyRuntime | None = None,
        document_reference_resolver: DocumentReferenceResolver | None = None,
    ) -> None:
        self._retrieve_documents = retrieve_documents or _retrieve_documents
        self._graph_locator = graph_locator or _locate_graph_documents
        self._visual_extractor = visual_extractor or _extract_visual_evidence
        self._provider_factory = provider_factory or _provider_for_purpose
        self._policy_runtime = policy_runtime or V9ExecutionPolicyRuntime()
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
        if (
            runtime_contract.route_decision is not None
            and runtime_contract.route_decision.decision_source == "safe_fallback"
            and runtime_contract.route_decision.fallback_reason
            == "planner_unavailable"
        ):
            runtime_contract = await QuestionContractPlanner(
                llm_invoker=BudgetedLlmInvoker(
                    controller=budget_controller,
                    provider_factory=self._provider_factory,
                )
            ).plan(
                question=question,
                authorized_source_names=list(
                    source_scope.requested_source_names
                ),
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
            "post_contract": None,
            "budget_controller": budget_controller,
            "task_slot_ids": {},
            "graph_execution": None,
            "visual_execution": None,
            "visual_packets": [],
            "visual_packets_emitted": False,
            "final_slot_resolutions": (),
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
            visual_documents: list[Document] = []
            for task in tasks:
                state["task_slot_ids"][task.task_id] = list(task.target_slot_ids)
                docs = await self._retrieve_documents(
                    user_id, task.query, list(task.source_scope.authorized_doc_ids)
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
                    except Exception as error:  # Stage admitted; preserve partial answer.
                        state["graph_execution"] = _failed_required_stage(
                            policy="required_locator", error=error
                        )
                    else:
                        state["graph_execution"] = _graph_execution_projection(located)
                        docs = list(located.documents)
                visual_documents.extend(
                    _visual_documents_for_contract(
                        documents=docs,
                        contract=state["contract"],
                    )
                )
                chunks = [_chunk_projection(document, index) for index, document in enumerate(docs)]
                results.append(
                    TaskRetrievalResult(
                        task_id=task.task_id,
                        retrieval=RagRetrievalResult(
                            retrieval_id=f"{trace_id}:{task.task_id}", chunks=chunks
                        ),
                    )
                )
            if (
                state["contract"].visual_requested
                and not state["visual_execution"]["attempted"]
            ):
                controller = state["budget_controller"]
                assert isinstance(controller, RunBudgetController)
                visual_slot_ids = [
                    slot.slot_id for slot in _requested_visual_slots(state["contract"])
                ]
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
                    visual_result = await self._visual_extractor(
                        visual_task,
                        _deduplicate_visual_documents(visual_documents),
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
                    )
                    state["visual_execution"] = _visual_execution_projection(
                        visual_result,
                        required=state["contract"].visual_required,
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
                task_slot_ids=state["task_slot_ids"],
            )
            if not state["visual_packets_emitted"]:
                packets.extend(state["visual_packets"])
                state["visual_packets_emitted"] = True
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
                setup_snapshot, "setup_max_input_tokens", "max_input_tokens", default=8192
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
        final = executed.final_answer or FinalAnswerResult(response_status="insufficient")
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
                "evidence_packets": [packet.model_dump(mode="json") for packet in state["evidence_packets"]],
                "slot_resolutions": [
                    resolution.model_dump(mode="json")
                    for resolution in state["final_slot_resolutions"]
                ],
                "sufficiency": executed.sufficiency.model_dump(mode="json") if executed.sufficiency else None,
                "context_pack": _context_pack_projection(packed),
                "graph_execution": graph_execution,
                "visual_execution": visual_execution,
                "budget_reservations": [
                    item.model_dump(mode="json") for item in await controller.reservations()
                ],
                "repairs": [repair.model_dump(mode="json") for repair in state["repairs"]],
                "conflicts": [],
                "final_claims": [claim.model_dump(mode="json") for claim in final.claims],
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
    user_id: str, question: str, authorized_doc_ids: list[str]
) -> list[Document]:
    """Retrieve only within the source scope, without HyDE or query expansion."""
    if not authorized_doc_ids:
        return []
    retriever = await get_user_retriever_async(user_id, k=8, plain_mode=False)
    raw = await retrieve_hybrid_documents(
        question,
        retriever,
        enable_hyde=False,
        enable_multi_query=False,
    )
    filtered = filter_and_rerank_retrieval(
        question,
        raw,
        doc_ids=authorized_doc_ids,
        enable_reranking=False,
    )
    return list(filtered.documents)


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
        if (
            visual_failed
            and resolution.slot_id in missing_required_visual_slot_ids
        ):
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
    resolutions_by_slot = {
        resolution.slot_id: resolution for resolution in resolutions
    }
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
            "required_but_not_satisfied"
            if required
            else "attempted_without_evidence"
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
        "selected_asset_count": 0,
        "dropped_asset_count": 0,
        "evidence_packet_count": 0,
        "supported_slot_ids": [],
    }


async def _extract_visual_evidence(
    task: Any,
    documents: list[Document],
    question: str,
    controller: RunBudgetController,
) -> VisualEvidenceExtractionResult:
    """Extract only selected, source-bound visual evidence from retrieved docs."""
    extractor = VisualEvidenceExtractor(
        BudgetedLlmInvoker(
            controller=controller,
            provider_factory=_provider_for_purpose,
        )
    )
    return await extractor.extract(
        task=task,
        assets=_visual_assets_from_documents(documents),
        question_fragment=question,
    )


def _visual_assets_from_documents(
    documents: list[Document],
) -> list[VisualAssetCandidate]:
    """Project only fully located page images supplied by retrieval metadata."""
    assets: list[VisualAssetCandidate] = []
    for index, document in enumerate(documents):
        metadata = dict(document.metadata or {})
        image_base64 = metadata.get("page_image_base64") or metadata.get("image_base64")
        page = metadata.get("page_number")
        width = metadata.get("page_width") or metadata.get("image_width")
        height = metadata.get("page_height") or metadata.get("image_height")
        doc_id = get_document_id(metadata)
        slot_ids = metadata.get("visual_slot_ids")
        if not (
            isinstance(image_base64, str)
            and image_base64
            and isinstance(page, int)
            and page >= 0
            and isinstance(width, int)
            and width > 0
            and isinstance(height, int)
            and height > 0
            and isinstance(doc_id, str)
            and doc_id
            and isinstance(slot_ids, list)
            and slot_ids
        ):
            continue
        assets.append(
            VisualAssetCandidate(
                asset_id=str(metadata.get("asset_id") or f"{doc_id}:page:{page}:{index}"),
                source=EvidenceSource(
                    doc_id=doc_id,
                    chunk_id=str(metadata.get("chunk_id") or index + 1),
                ),
                pdf_page_index=page,
                slot_ids=list(slot_ids),
                figure_id=metadata.get("figure_id"),
                table_id=metadata.get("table_id"),
                formula_id=metadata.get("formula_id"),
                bbox=metadata.get("bbox"),
                page_image_base64=image_base64,
                page_encoded_bytes=int(metadata.get("page_encoded_bytes") or 0),
                page_width=width,
                page_height=height,
            )
        )
    return assets


def _requested_visual_slots(contract: QueryContract) -> list[RequiredSlot]:
    slots = [
        slot
        for slot in contract.required_slots
        if slot.visual_policy in {"preferred", "required"}
    ]
    if contract.visual_required and not slots:
        return [slot for slot in contract.required_slots if slot.required]
    return slots


def _visual_documents_for_contract(
    *,
    documents: list[Document],
    contract: QueryContract,
) -> list[Document]:
    """Bind each retrieved visual candidate to compatible authorized slots."""
    visual_documents: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata or {})
        doc_id = get_document_id(metadata)
        if not isinstance(doc_id, str) or not doc_id:
            continue
        if not _has_visual_candidate_metadata(metadata):
            continue
        slot_ids = [
            slot.slot_id
            for slot in _requested_visual_slots(contract)
            if _slot_accepts_visual_source(
                slot=slot,
                doc_id=doc_id,
                figure_id=metadata.get("figure_id"),
                table_id=metadata.get("table_id"),
                formula_id=metadata.get("formula_id"),
            )
        ]
        if not slot_ids:
            continue
        metadata["visual_slot_ids"] = slot_ids
        visual_documents.append(
            Document(page_content=document.page_content, metadata=metadata)
        )
    return visual_documents


def _has_visual_candidate_metadata(metadata: dict[str, Any]) -> bool:
    image = metadata.get("page_image_base64") or metadata.get("image_base64")
    width = metadata.get("page_width") or metadata.get("image_width")
    height = metadata.get("page_height") or metadata.get("image_height")
    return bool(
        isinstance(image, str)
        and image
        and isinstance(metadata.get("page_number"), int)
        and isinstance(width, int)
        and width > 0
        and isinstance(height, int)
        and height > 0
    )


def _deduplicate_visual_documents(documents: list[Document]) -> list[Document]:
    deduplicated: list[Document] = []
    seen: set[tuple[object, ...]] = set()
    for document in documents:
        metadata = document.metadata or {}
        key = (
            metadata.get("asset_id"),
            get_document_id(metadata),
            metadata.get("page_number"),
            metadata.get("figure_id"),
            metadata.get("table_id"),
            metadata.get("formula_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(document)
    return deduplicated


def _bind_visual_result_to_contract(
    *,
    result: VisualEvidenceExtractionResult,
    contract: QueryContract,
) -> VisualEvidenceExtractionResult:
    """Remove source/locator slot claims outside the exact query contract."""
    slots_by_id = {
        slot.slot_id: slot for slot in _requested_visual_slots(contract)
    }
    packets: list[EvidencePacket] = []
    for packet in result.packets:
        bound_slot_ids = [
            slot_id
            for slot_id in packet.slot_ids
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
            packets.append(
                packet.model_copy(update={"slot_ids": bound_slot_ids})
            )
    return result.model_copy(update={"packets": tuple(packets)})


def _slot_accepts_visual_source(
    *,
    slot: RequiredSlot,
    doc_id: str,
    figure_id: object,
    table_id: object,
    formula_id: object,
) -> bool:
    if (
        slot.authorized_source_doc_ids
        and doc_id not in slot.authorized_source_doc_ids
    ):
        return False
    visual_hints = [
        hint
        for hint in slot.locator_hints
        if hint.strip().casefold().startswith(
            ("figure", "fig.", "table", "formula", "equation")
        )
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
    key = "".join(
        character for character in value.casefold() if character.isalnum()
    )
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
        "selected_asset_count": 0,
        "dropped_asset_count": 0,
        "evidence_packet_count": 0,
        "supported_slot_ids": [],
    }


def _visual_execution_projection(
    result: VisualEvidenceExtractionResult,
    *,
    required: bool,
) -> dict[str, Any]:
    packet_count = len(result.packets)
    supported_slot_ids = sorted(
        {
            slot_id
            for packet in result.packets
            for slot_id in packet.slot_ids
        }
    )
    return {
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
                else "no_eligible_visual_evidence"
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
    }


def _evidence_packets_for_results(
    *,
    results: tuple[TaskRetrievalResult, ...],
    contract: QueryContract,
    trace_id: str,
    task_slot_ids: dict[str, list[str]],
) -> list[EvidencePacket]:
    packets: list[EvidencePacket] = []
    for task_result in results:
        task_id = task_result.task_id
        for index, chunk in enumerate(task_result.retrieval.chunks):
            doc_id = chunk.get("doc_id")
            text = str(chunk.get("text") or "").strip()
            if not isinstance(doc_id, str) or not doc_id or not text:
                continue
            scope = contract.resolved_source_scope
            if scope is None or doc_id not in scope.authorized_doc_ids:
                continue
            slot_ids = task_slot_ids.get(
                task_id, [slot.slot_id for slot in contract.required_slots]
            )
            digest = hashlib.sha256(
                f"{trace_id}:{task_id}:{doc_id}:{chunk.get('chunk_id')}:{index}".encode()
            ).hexdigest()[:24]
            page = chunk.get("page_number")
            locator = (
                SourceLocator(pdf_page_index=page)
                if isinstance(page, int) and page >= 0
                else SourceLocator(section=str(chunk.get("section") or "retrieved_context"))
            )
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
                "query_contract": contract.model_dump(mode="json") if contract else None,
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


def _setup_positive_int(
    snapshot: dict[str, Any], *keys: str, default: int
) -> int:
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


def _context_pack_projection(packed: PackedEvidenceContext | None) -> dict[str, Any] | None:
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
