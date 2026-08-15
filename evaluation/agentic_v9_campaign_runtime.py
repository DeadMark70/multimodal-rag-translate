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
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
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
from data_base.agentic_v9.budgeted_llm import BudgetedLlmInvoker, LlmCallObserver
from data_base.agentic_v9.context_packer import (
    EvidenceContextPacker,
    FinalContextSelectionPolicy,
    PackedEvidenceContext,
)
from data_base.agentic_v9.contract_planner import (
    QuestionContractPlanner,
    atomic_contract_planner_response_schema,
)
from data_base.agentic_v9.comparison_context import (
    select_balanced_comparison_packets,
)
from data_base.agentic_v9.evidence_extractor import EvidenceExtractor
from data_base.agentic_v9.execution_core import (
    ConflictStageResult,
    V9ExecutionCore,
    V9ExecutionStages,
)
from data_base.agentic_v9.execution_policy import (
    ExecutionCancellation,
    V9ExecutionPolicyRuntime,
)
from data_base.agentic_v9.provider_boundary import (
    build_contract_planning_provider,
)
from data_base.agentic_v9.repair import build_repair_plan
from data_base.agentic_v9.requirement_shadow import build_requirement_shadow
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    FinalAnswerResult,
    FinalClaim,
    LlmInvoker,
    QueryContract,
    RagRetrievalResult,
    ResolvedSourceScope,
    SlotResolution,
    SourceLocator,
    TaskRetrievalResult,
    V9ExecutionEvent,
    V9ExecutionRequest,
    V9RuntimeContext,
)
from data_base.agentic_v9.sufficiency_gate import (
    SufficiencyEvaluation,
    evaluate_sufficiency,
)
from data_base.agentic_v9.asset_locator import VisualAssetCandidate
from data_base.agentic_v9.visual_evidence_extractor import (
    VisualEvidenceExtractionResult,
    VisualEvidenceExtractor,
)
from data_base.document_metadata import get_document_id
from data_base.rag_filtering import filter_and_rerank_retrieval
from data_base.rag_graph_locator import GraphSourceLocatorResult, locate_graph_sources
from data_base.rag_pipeline_schemas import (
    RagRetrievalResult as PipelineRetrievalResult,
)
from data_base.rag_retrieval import retrieve_hybrid_documents
from data_base.reranker import DocumentReranker
from data_base.vector_store_manager import get_user_retriever_async
from evaluation.agentic_campaign_adapter import used_evidence_documents
from evaluation.agentic_v9_admission import (
    DocumentReferenceResolver,
    OwnedDocumentIdsResolver,
    build_v9_admission_contract,
)
from evaluation.observability import current_llm_call_observer
from evaluation.retrieval_profiles import (
    AGENTIC_EVAL_PROFILE,
    AGENTIC_V9_CONTEXT_POLICY_VERSION,
    agentic_v9_execution_profile,
)

logger = logging.getLogger(__name__)

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

_REQUIREMENT_GUIDED_RUNTIME_ENV = "AGENTIC_V9_REQUIREMENT_GUIDED_RUNTIME"
_TRUE_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_FLAG_VALUES = frozenset({"0", "false", "no", "off"})
_MAX_REQUIREMENT_GUIDANCE_CHARS = 2048


def _optional_bool(value: Any) -> bool | None:
    """Parse explicit boolean values without treating arbitrary strings as true."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_FLAG_VALUES:
            return True
        if normalized in _FALSE_FLAG_VALUES:
            return False
    return None


def _resolve_requirement_guided_runtime(
    setup_snapshot: Mapping[str, Any],
) -> tuple[bool, str, str | None]:
    """Resolve the reversible advisory switch with explicit setup precedence."""

    if "requirement_guided_runtime" in setup_snapshot:
        value = _optional_bool(setup_snapshot.get("requirement_guided_runtime"))
        if value is not None:
            return value, "setup_snapshot", None
        return False, "setup_snapshot", "invalid_flag_value"

    environment_value = os.getenv(_REQUIREMENT_GUIDED_RUNTIME_ENV)
    if environment_value is not None:
        value = _optional_bool(environment_value)
        if value is not None:
            return value, "environment", None
        return False, "environment", "invalid_flag_value"

    return False, "default", None


def _initial_requirement_guidance(
    *, question: str, setup_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    """Build bounded advisory hints without making a provider call.

    The switch is intentionally advisory: it can enrich generic retrieval
    queries, but it never changes the typed contract, sufficiency result, or
    visual/graph capability policies.  Subject-specific comparison tasks are
    kept clean and are handled by the comparison planner's own query.
    """

    enabled, source, fallback_reason = _resolve_requirement_guided_runtime(
        setup_snapshot
    )
    state: dict[str, Any] = {
        "enabled": enabled,
        "mode": "advisory" if enabled else "off",
        "source": source,
        "candidate_requirement_ids": [],
        "applied_requirement_ids": [],
        "applied_task_ids": [],
        "applied_task_count": 0,
        "fallback_reason": fallback_reason,
        "_advisory_suffix": "",
    }
    if not enabled:
        return state

    try:
        shadow = build_requirement_shadow(question=question, documents=[])
    except Exception:  # noqa: BLE001 -- advisory diagnostics must be fail-soft
        logger.warning(
            "Agentic v9 requirement guidance projection failed", exc_info=True
        )
        state["fallback_reason"] = "guidance_projection_failed"
        return state

    candidates = [
        requirement
        for requirement in shadow.requirements
        if requirement.decomposition_confidence in {"medium", "high"}
    ]
    state["candidate_requirement_ids"] = [
        requirement.requirement_id for requirement in candidates
    ]
    if not candidates:
        state["fallback_reason"] = "no_medium_or_high_confidence_requirements"
        return state

    lines: list[str] = [
        "Advisory answer obligations (retrieval hints only; not evidence):"
    ]
    selected_ids: list[str] = []
    for requirement in candidates:
        line = f"- {requirement.text.strip()}"
        candidate_text = "\n".join([*lines, line])
        if len(candidate_text) > _MAX_REQUIREMENT_GUIDANCE_CHARS:
            break
        lines.append(line)
        selected_ids.append(requirement.requirement_id)
    if not selected_ids:
        state["fallback_reason"] = "guidance_budget_exhausted"
        return state

    state["applied_requirement_ids"] = selected_ids
    state["_advisory_suffix"] = "\n\n" + "\n".join(lines)
    return state


def _requirement_guided_query(
    task: Any, guidance: dict[str, Any]
) -> tuple[str, bool]:
    """Append advisory hints only to generic retrieval tasks."""

    suffix = str(guidance.get("_advisory_suffix") or "")
    if not guidance.get("enabled") or not suffix or task.subject_id is not None:
        return task.query, False
    task_id = str(task.task_id)
    applied_task_ids = guidance.setdefault("applied_task_ids", [])
    if task_id not in applied_task_ids:
        applied_task_ids.append(task_id)
    guidance["applied_task_count"] = len(applied_task_ids)
    return f"{task.query}{suffix}", True


def _requirement_guidance_projection(guidance: Mapping[str, Any]) -> dict[str, Any]:
    """Return the durable trace shape without exposing the private query suffix."""

    return {
        "enabled": bool(guidance.get("enabled")),
        "mode": str(guidance.get("mode") or "off"),
        "source": str(guidance.get("source") or "default"),
        "candidate_requirement_ids": list(
            guidance.get("candidate_requirement_ids") or []
        ),
        "applied_requirement_ids": list(
            guidance.get("applied_requirement_ids") or []
        ),
        "applied_task_ids": list(guidance.get("applied_task_ids") or []),
        "applied_task_count": int(guidance.get("applied_task_count") or 0),
        "fallback_reason": guidance.get("fallback_reason"),
    }


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
        owned_document_ids_resolver: OwnedDocumentIdsResolver | None = None,
        llm_call_observer: LlmCallObserver | None = None,
        comparison_specialization_enabled: bool = True,
    ) -> None:
        self._retrieve_documents = retrieve_documents or _retrieve_documents
        self._uses_default_retrieval = retrieve_documents is None
        self._uses_default_graph_locator = graph_locator is None
        self._graph_locator = graph_locator or _locate_graph_documents
        self._visual_extractor = visual_extractor
        self._provider_factory = provider_factory or _provider_for_purpose
        self._policy_runtime = policy_runtime or V9ExecutionPolicyRuntime()
        self._document_reference_resolver = (
            document_reference_resolver or _resolve_document_references
        )
        self._owned_document_ids_resolver = (
            owned_document_ids_resolver or _list_owned_document_ids
        )
        self._llm_call_observer = llm_call_observer
        self._comparison_specialization_enabled = comparison_specialization_enabled

    async def execute(
        self,
        *,
        question: str,
        user_id: str,
        authorized_doc_ids: list[str] | None,
        setup_snapshot: dict[str, Any],
        trace_id: str,
    ) -> RAGResult:
        """Execute v9 or return a deterministic incompatible projection.

        Pre-route admission happens before the core (and thus before retrieval
        or provider work).  The contract adapter repeats admission immediately
        after planning, before the core can start retrieval.
        """
        open_user_corpus = authorized_doc_ids is None
        execution_profile = agentic_v9_execution_profile(
            open_user_corpus=open_user_corpus
        )
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
                execution_profile=execution_profile,
            )

        admission = await build_v9_admission_contract(
            question=question,
            user_id=user_id,
            source_references=authorized_doc_ids,
            document_reference_resolver=self._document_reference_resolver,
            owned_document_ids_resolver=self._owned_document_ids_resolver,
            setup_policy=setup_snapshot,
        )
        source_scope = admission.source_scope
        runtime_contract = admission.contract
        preparation = QuestionContractPlanner.prepare(
            question=question,
            base_contract=runtime_contract,
        )
        request = V9ExecutionRequest(
            question=question,
            requested_doc_ids=list(source_scope.authorized_doc_ids),
            setup_snapshot=dict(setup_snapshot),
            trace_id=trace_id,
            contract_plan_requested=preparation.semantic_planning_requested,
        )
        deadline = self._policy_runtime.start_deadline()
        cancellation = ExecutionCancellation()
        llm_call_observer = self._llm_call_observer or current_llm_call_observer()
        state: dict[str, Any] = {
            "contract": None,
            "pack": None,
            "repairs": [],
            "evidence_packets": [],
            "quality_by_evidence_id": {},
            "post_contract": None,
            "budget_controller": None,
            "task_slot_ids": {},
            "task_subject_ids": {},
            "comparison_coverage_before_repair": None,
            "comparison_coverage_after_repair": None,
            "final_evidence_packets": None,
            "atomic_planner_call_count": 0,
            "atomic_planner_latency_ms": 0.0,
            "planner_diagnostics": None,
            "graph_execution": None,
            "visual_execution": None,
            "visual_packets": [],
            "visual_packets_emitted": False,
            "retrieval_diagnostics": [],
            "requirement_shadow_documents": [],
        }

        async def resolve_scope(_: V9ExecutionRequest) -> ResolvedSourceScope:
            return source_scope

        async def plan_contract(
            _: V9ExecutionRequest, scope: ResolvedSourceScope
        ) -> QueryContract:
            planner_admitted = False
            if preparation.semantic_planning_requested:
                post_contract = validate_post_contract_feasibility(
                    contract=runtime_contract,
                    setup_snapshot=setup_snapshot,
                    remaining_token_budget=runtime_contract.runtime_token_budget,
                    remaining_llm_calls=runtime_contract.max_llm_calls,
                    route_plan_used=False,
                    contract_plan_requested=True,
                    evidence_qualification_provider_calls=1,
                )
                if post_contract.status is FeasibilityStatus.FEASIBLE:
                    planner_admitted = True
                else:
                    post_contract = validate_post_contract_feasibility(
                        contract=runtime_contract,
                        setup_snapshot=setup_snapshot,
                        remaining_token_budget=runtime_contract.runtime_token_budget,
                        remaining_llm_calls=runtime_contract.max_llm_calls,
                        route_plan_used=False,
                        contract_plan_requested=False,
                        evidence_qualification_provider_calls=1,
                    )
            else:
                post_contract = validate_post_contract_feasibility(
                    contract=runtime_contract,
                    setup_snapshot=setup_snapshot,
                    remaining_token_budget=runtime_contract.runtime_token_budget,
                    remaining_llm_calls=runtime_contract.max_llm_calls,
                    route_plan_used=False,
                    contract_plan_requested=False,
                    evidence_qualification_provider_calls=1,
                )

            state["post_contract"] = post_contract
            if post_contract.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE:
                state["contract"] = runtime_contract
                raise _ConfigurationIncompatible(
                    stage="post_contract", feasibility=post_contract
                )

            state["budget_controller"] = RunBudgetController(
                max_llm_calls=runtime_contract.max_llm_calls,
                runtime_token_budget=runtime_contract.runtime_token_budget,
                setup_snapshot=setup_snapshot,
                final_input_tokens=_final_input_reserve(
                    setup_snapshot, runtime_contract.runtime_token_budget
                ),
            )

            controller = state["budget_controller"]
            assert isinstance(controller, RunBudgetController)
            invoker = BudgetedLlmInvoker(
                controller=controller,
                provider_factory=self._provider_factory,
                observer=llm_call_observer,
                provider_name=str(
                    setup_snapshot.get("provider") or "unknown"
                ),
                model_name=str(
                    setup_snapshot.get("model_name") or "unknown"
                ),
            )
            planner = QuestionContractPlanner(llm_invoker=invoker)
            outcome = await planner.plan(
                question=question,
                base_contract=runtime_contract,
                preparation=preparation,
                allow_semantic_planning=planner_admitted,
            )
            contract = outcome.contract
            state["contract"] = contract
            state["atomic_planner_call_count"] = outcome.planner_call_count
            state["atomic_planner_latency_ms"] = outcome.latency_ms
            state["planner_diagnostics"] = outcome.planner_diagnostics
            state["graph_execution"] = _initial_graph_execution(contract)
            state["visual_execution"] = _initial_visual_execution(contract)
            return contract

        async def retrieve(
            tasks: tuple[Any, ...],
        ) -> tuple[TaskRetrievalResult, ...]:
            results: list[TaskRetrievalResult] = []
            for task in tasks:
                state["task_slot_ids"][task.task_id] = list(task.target_slot_ids)
                state["task_subject_ids"][task.task_id] = task.subject_id
                source_scope = list(task.source_scope.authorized_doc_ids)
                retrieval_query = task.query
                if self._uses_default_retrieval:
                    docs = await self._retrieve_documents(
                        user_id,
                        retrieval_query,
                        source_scope,
                        diversify_rerank_candidates=_requires_diverse_rerank_candidates(
                            state["contract"].route
                        ),
                    )
                else:
                    docs = await self._retrieve_documents(
                        user_id, retrieval_query, source_scope
                    )
                pre_subject_limit_count = len(docs)
                if task.subject_id is not None:
                    docs = docs[:2]
                state["retrieval_diagnostics"].append(
                    _retrieval_diagnostic_projection(
                        task.task_id,
                        docs,
                        subject_id=task.subject_id,
                        query=retrieval_query,
                        pre_subject_limit_count=pre_subject_limit_count,
                    )
                )
                if (
                    state["contract"].graph_policy == "required_locator"
                    and not state["graph_execution"]["attempted"]
                ):
                    try:
                        if self._uses_default_graph_locator:
                            controller = state["budget_controller"]
                            assert isinstance(controller, RunBudgetController)
                            located = await _locate_graph_documents(
                                task.query,
                                user_id,
                                docs,
                                list(task.source_scope.authorized_doc_ids),
                                state["contract"],
                                llm_invoker=BudgetedLlmInvoker(
                                    controller=controller,
                                    provider_factory=self._provider_factory,
                                    observer=llm_call_observer,
                                    provider_name=str(
                                        setup_snapshot.get("provider") or "unknown"
                                    ),
                                    model_name=str(
                                        setup_snapshot.get("model_name") or "unknown"
                                    ),
                                ),
                            )
                        else:
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
                if (
                    state["contract"].visual_required
                    and not state["visual_execution"]["attempted"]
                ):
                    controller = state["budget_controller"]
                    assert isinstance(controller, RunBudgetController)
                    try:
                        if self._visual_extractor is not None:
                            visual_result = await self._visual_extractor(
                                task, docs, question, controller
                            )
                        else:
                            visual_result = await _extract_visual_evidence(
                                task,
                                docs,
                                question,
                                controller,
                                observer=llm_call_observer,
                                provider_factory=self._provider_factory,
                                provider_name=str(
                                    setup_snapshot.get("provider") or "unknown"
                                ),
                                model_name=str(
                                    setup_snapshot.get("model_name") or "unknown"
                                ),
                            )
                    except (
                        Exception
                    ) as error:  # Stage admitted; preserve partial answer.
                        state["visual_execution"] = _failed_required_stage(
                            policy="visual_required", error=error
                        )
                    else:
                        state["visual_execution"] = _visual_execution_projection(
                            visual_result
                        )
                        state["visual_packets"].extend(visual_result.packets)
                state["requirement_shadow_documents"].extend(docs)
                chunks = [
                    _chunk_projection(document, index, task_id=task.task_id)
                    for index, document in enumerate(docs)
                ]
                results.append(
                    TaskRetrievalResult(
                        task_id=task.task_id,
                        retrieval=RagRetrievalResult(
                            retrieval_id=f"{trace_id}:{task.task_id}", chunks=chunks
                        ),
                    )
                )
            return tuple(results)

        async def deterministic_candidates(
            results: tuple[TaskRetrievalResult, ...], contract: QueryContract
        ) -> tuple[EvidencePacket, ...]:
            projection = _evidence_packets_for_results(
                results=results,
                contract=contract,
                trace_id=trace_id,
                task_slot_ids=state["task_slot_ids"],
            )
            packets = projection.packets
            state["quality_by_evidence_id"].update(
                projection.quality_by_evidence_id
            )
            if not state["visual_packets_emitted"]:
                packets.extend(state["visual_packets"])
                state["visual_packets_emitted"] = True
            state["evidence_packets"].extend(packets)
            return tuple(packets)

        def sufficiency(
            contract: QueryContract, packets: tuple[EvidencePacket, ...]
        ) -> SufficiencyEvaluation:
            effective_packets = (
                select_balanced_comparison_packets(
                    packets,
                    plan=contract.comparison_plan,
                    quality_by_evidence_id=state["quality_by_evidence_id"],
                )
                if contract.comparison_plan is not None
                else packets
            )
            if contract.comparison_plan is not None:
                coverage = _comparison_subject_coverage(
                    contract, effective_packets
                )
                if state["comparison_coverage_before_repair"] is None:
                    state["comparison_coverage_before_repair"] = coverage
                state["comparison_coverage_after_repair"] = coverage
            return evaluate_sufficiency(contract, effective_packets)

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
            curate_question: str,
            contract: QueryContract,
            packets: tuple[EvidencePacket, ...],
        ) -> tuple[EvidencePacket, ...]:
            controller = state["budget_controller"]
            assert isinstance(controller, RunBudgetController)
            invoker = BudgetedLlmInvoker(
                controller=controller,
                provider_factory=self._provider_factory,
                observer=llm_call_observer,
                provider_name=str(setup_snapshot.get("provider") or "unknown"),
                model_name=str(setup_snapshot.get("model_name") or "unknown"),
            )
            extractor = EvidenceExtractor(budgeted_invoker=invoker)
            extracted = await extractor.extract(
                contract,
                packets,
                repairs_complete=True,
                question=curate_question or question,
            )
            for packet in extracted:
                if packet.evidence_id not in state["quality_by_evidence_id"]:
                    for cand_id, score in list(state["quality_by_evidence_id"].items()):
                        if cand_id in packet.evidence_id:
                            state["quality_by_evidence_id"][packet.evidence_id] = score
                            break
            if contract.comparison_plan is not None:
                selected = select_balanced_comparison_packets(
                    extracted,
                    plan=contract.comparison_plan,
                    quality_by_evidence_id=state["quality_by_evidence_id"],
                )
            else:
                selected = extracted
            state["final_evidence_packets"] = selected
            return tuple(selected)

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
            selected_packets = (
                select_balanced_comparison_packets(
                    packets,
                    plan=contract.comparison_plan,
                    quality_by_evidence_id=state["quality_by_evidence_id"],
                )
                if contract.comparison_plan is not None
                else packets
            )
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
            packed = packer.pack(
                selected_packets,
                required_slots=contract,
                quality_by_evidence_id=state["quality_by_evidence_id"],
                selection_policy=FinalContextSelectionPolicy(
                    version="soft_final_pack_r1"
                ),
            )
            state["pack"] = packed
            return packed

        async def generate_final(
            _: str,
            __: QueryContract,
            packed: PackedEvidenceContext,
            resolutions: tuple[SlotResolution, ...],
            ___: Any | None,
            ____: Any,
        ) -> FinalAnswerResult:
            controller = state["budget_controller"]
            assert isinstance(controller, RunBudgetController)
            response = await BudgetedLlmInvoker(
                controller=controller,
                provider_factory=self._provider_factory,
                observer=llm_call_observer,
                provider_name=str(setup_snapshot.get("provider") or "unknown"),
                model_name=str(setup_snapshot.get("model_name") or "unknown"),
            ).invoke(
                phase="final_answer",
                purpose="agentic_v9_final_answer",
                messages=[
                    {
                        "role": "system",
                        "content": "Use only supplied evidence. Cite no source not present.",
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nEvidence:\n{packed.rendered_text}",
                    },
                ],
            )
            if isinstance(response, FinalAnswerResult):
                return response
            answer = _response_text(response)
            used_ids = [packet.evidence_id for packet in packed.packets]
            claims = [
                FinalClaim(
                    claim_id=f"claim:{trace_id}",
                    statement=answer or "Evidence-backed answer unavailable.",
                    support_type="direct",
                    evidence_ids=used_ids,
                )
            ]
            return FinalAnswerResult(
                response_status="complete",
                answer=answer,
                claims=claims,
                used_evidence_ids=used_ids,
                final_generation_count=1,
            )

        def deterministic_partial(
            _: QueryContract, evaluation: SufficiencyEvaluation
        ) -> FinalAnswerResult:
            return FinalAnswerResult(
                response_status=evaluation.report.response_status,
                answer="Evidence was insufficient for a fully supported answer.",
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
                execution_profile=execution_profile,
            )

        controller = state["budget_controller"]
        assert isinstance(controller, RunBudgetController)
        budget_snapshot = await controller.snapshot()
        metrics = executed.metrics.model_copy(
            update={
                "provider_attempt_count": budget_snapshot.provider_attempt_count,
                "reserved_tokens": budget_snapshot.reserved_tokens,
                "reconciled_tokens": budget_snapshot.reconciled_tokens,
                "atomic_planner_call_count": state.get("atomic_planner_call_count", 0),
                "comparison_planner_call_count": 0,
                "slot_binding_method": "task_target_inherited",
                "semantic_qualification": "not_enabled",
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
            (
                state["contract"].graph_policy == "required_locator"
                and _required_capability_execution_failed(graph_execution)
            )
            or (
                state["contract"].visual_required
                and _required_capability_execution_failed(visual_execution)
            )
        ) and final.response_status == "complete":
            # A capability that actually ran and failed remains fail-closed.
            # No eligible input is a capability gap, not evidence that the
            # text-backed answer itself is incomplete.
            final = final.model_copy(update={"response_status": "qualified_partial"})
        final_evidence_packets = state["final_evidence_packets"]
        if final_evidence_packets is None:
            final_evidence_packets = tuple(state["evidence_packets"])
        all_packets = {
            packet.evidence_id: packet
            for packet in (*state["evidence_packets"], *final_evidence_packets)
        }
        used_packets = [
            all_packets[evidence_id]
            for evidence_id in final.used_evidence_ids
            if evidence_id in all_packets
        ]
        documents = used_evidence_documents(used_packets, final)
        packed = state["pack"]
        comparison = _comparison_trace_projection(
            contract=state["contract"],
            retrieval_diagnostics=state["retrieval_diagnostics"],
            coverage_before=state["comparison_coverage_before_repair"],
            coverage_after=state["comparison_coverage_after_repair"],
            repairs=state["repairs"],
            packed=packed,
            final_status=final.response_status,
        )
        try:
            requirement_shadow = build_requirement_shadow(
                question=question,
                documents=state["requirement_shadow_documents"],
            ).model_dump(mode="json")
        except Exception:
            logger.warning(
                "Agentic v9 requirement shadow projection failed",
                exc_info=True,
            )
            requirement_shadow = {
                "schema_version": "shadow_requirements_v2",
                "behavior_influence": False,
                "status": "unavailable",
                "reason": "diagnostic_projection_failed",
            }
        v9_trace = {
            "schema_version": "1",
            "retrieval_scope": {
                "policy": (
                    "open_user_corpus"
                    if open_user_corpus
                    else "explicit_source_scope"
                ),
                "expected_sources_used_at_runtime": False,
            },
            "query_contract": state["contract"].model_dump(mode="json"),
            "requirement_shadow": requirement_shadow,
            "retrieval_diagnostics": state["retrieval_diagnostics"],
            "evidence_packets": [
                packet.model_dump(mode="json")
                for packet in state["evidence_packets"]
            ],
            "slot_resolutions": [
                resolution.model_dump(mode="json")
                for resolution in (
                    executed.sufficiency
                    and evaluate_sufficiency(
                        state["contract"], final_evidence_packets
                    ).slot_resolutions
                    or ()
                )
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
            "planner_diagnostics": state["planner_diagnostics"].model_dump(
                mode="json"
            )
            if state["planner_diagnostics"] is not None
            else None,
            "completion": {"status": final.response_status},
        }
        if comparison is not None:
            v9_trace["comparison"] = comparison
        trace = {
            "trace_id": trace_id,
            "mode": "agentic",
            "agentic_execution_version": "v9",
            "execution_profile": execution_profile,
            "context_policy_version": AGENTIC_V9_CONTEXT_POLICY_VERSION,
            "response_status": final.response_status,
            "agentic_v9": v9_trace,
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
    diversify_rerank_candidates: bool = False,
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
    if not DocumentReranker.is_initialized():
        fallback = filter_and_rerank_retrieval(
            question,
            raw,
            doc_ids=authorized_doc_ids,
            enable_reranking=True,
            reranker_available=False,
            target_k=4,
            max_candidates=8,
            diversify_rerank_candidates=diversify_rerank_candidates,
        )
        return _annotate_rerank_selection(
            fallback,
            status="fallback",
            fallback_reason="reranker_unavailable",
        )

    try:
        selection = await asyncio.to_thread(
            filter_and_rerank_retrieval,
            question,
            raw,
            doc_ids=authorized_doc_ids,
            enable_reranking=True,
            target_k=4,
            max_candidates=8,
            diversify_rerank_candidates=diversify_rerank_candidates,
            strict_reranking=True,
        )
    except Exception as error:  # noqa: BLE001 -- bounded fail-soft stage boundary
        logger.warning(
            "Agentic v9 reranking failed; using Hybrid top 4 (%s)",
            type(error).__name__,
        )
        fallback = filter_and_rerank_retrieval(
            question,
            raw,
            doc_ids=authorized_doc_ids,
            enable_reranking=True,
            reranker_available=False,
            target_k=4,
            max_candidates=8,
        )
        return _annotate_rerank_selection(
            fallback,
            status="fallback",
            fallback_reason="reranker_error",
        )

    scored = any(
        row.get("score") is not None
        for row in selection.metadata["reranking"]["post_rerank_ranks"]
    )
    return _annotate_rerank_selection(
        selection,
        status="executed" if scored else "fallback",
        fallback_reason=None if scored else "reranker_empty_result",
    )


def _annotate_rerank_selection(
    selection: PipelineRetrievalResult,
    *,
    status: str,
    fallback_reason: str | None,
) -> list[Document]:
    reranking = dict(selection.metadata.get("reranking") or {})
    rows = list(reranking.get("post_rerank_ranks") or [])
    candidate_count = int(reranking.get("candidate_count") or 0)
    candidate_diversification = _candidate_diversification_projection(
        reranking.get("candidate_diversification")
    )
    selected_count = len(selection.documents)
    annotated: list[Document] = []
    for post_rank, document in enumerate(selection.documents, start=1):
        row = rows[post_rank - 1] if post_rank <= len(rows) else {}
        annotation: dict[str, Any] = {
            "status": status,
            "fallback_reason": fallback_reason,
            "candidate_count": candidate_count,
            "selected_count": selected_count,
            "pre_rerank_rank": int(row.get("pre_rerank_rank") or post_rank),
            "post_rerank_rank": post_rank,
            "rerank_score": (
                row.get("score") if status == "executed" else None
            ),
        }
        if candidate_diversification is not None:
            annotation["candidate_diversification"] = candidate_diversification
        annotated.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **dict(document.metadata),
                    "agentic_v9_reranking": annotation,
                },
            )
        )
    return annotated


def _retrieval_diagnostic_projection(
    task_id: str,
    documents: Sequence[Document],
    *,
    subject_id: str | None = None,
    query: str | None = None,
    pre_subject_limit_count: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    status = "not_instrumented"
    fallback_reason: str | None = None
    candidate_count = len(documents)
    candidate_diversification: dict[str, Any] | None = None
    for index, document in enumerate(documents):
        metadata = dict(document.metadata)
        reranking = metadata.get("agentic_v9_reranking")
        if not isinstance(reranking, dict):
            continue
        status = str(reranking.get("status") or status)
        fallback_reason = (
            str(reranking["fallback_reason"])
            if reranking.get("fallback_reason")
            else None
        )
        candidate_count = int(reranking.get("candidate_count") or candidate_count)
        if candidate_diversification is None:
            candidate_diversification = _candidate_diversification_projection(
                reranking.get("candidate_diversification")
            )
        rows.append(
            {
                "doc_id": get_document_id(metadata),
                "chunk_id": _project_chunk_id(
                    metadata, index, task_id=task_id
                ),
                "content_hash": hashlib.sha256(
                    document.page_content.encode("utf-8")
                ).hexdigest(),
                "pre_rerank_rank": reranking.get("pre_rerank_rank"),
                "post_rerank_rank": reranking.get("post_rerank_rank"),
                "rerank_score": reranking.get("rerank_score"),
            }
        )
    projection = {
        "task_id": task_id,
        "status": status,
        "fallback_reason": fallback_reason,
        "candidate_count": candidate_count,
        "selected_count": len(documents),
        "selected": rows,
    }
    if subject_id is not None:
        normalized_query = " ".join((query or "").split())
        projection.update(
            {
                "subject_id": subject_id,
                "query_hash": (
                    "sha256:"
                    + hashlib.sha256(
                        normalized_query.encode("utf-8")
                    ).hexdigest()
                ),
                "query_preview": normalized_query[:160],
                "pre_subject_limit_count": (
                    int(pre_subject_limit_count)
                    if pre_subject_limit_count is not None
                    else len(documents)
                ),
            }
        )
    if candidate_diversification is not None:
        projection["candidate_diversification"] = candidate_diversification
    return projection


def _comparison_subject_coverage(
    contract: QueryContract,
    packets: Sequence[EvidencePacket],
) -> dict[str, list[str]]:
    plan = contract.comparison_plan
    if plan is None:
        return {"covered": [], "missing": []}
    covered_slot_ids = {
        slot_id
        for packet in packets
        for slot_id in packet.slot_ids
    }
    covered = [
        subject.subject_id
        for subject in plan.subjects
        if any(slot_id in covered_slot_ids for slot_id in subject.evidence_slot_ids)
        or f"comparison-subject:{subject.subject_id}" in covered_slot_ids
    ]
    return {
        "covered": covered,
        "missing": [
            subject.subject_id
            for subject in plan.subjects
            if subject.subject_id not in covered
        ],
    }


def _comparison_trace_projection(
    *,
    contract: QueryContract,
    retrieval_diagnostics: Sequence[Mapping[str, Any]],
    coverage_before: Mapping[str, Any] | None,
    coverage_after: Mapping[str, Any] | None,
    repairs: Sequence[Any],
    packed: PackedEvidenceContext | None,
    final_status: str,
) -> dict[str, Any] | None:
    plan = contract.comparison_plan
    if plan is None:
        return None

    before = coverage_before or {"covered": [], "missing": []}
    after = coverage_after or before
    packed_packets = tuple(packed.packets) if packed is not None else ()
    final_evidence = [
        {
            "evidence_id": packet.evidence_id,
            "doc_id": packet.source.doc_id,
            "chunk_id": packet.source.chunk_id,
            "subject_ids": [
                subject.subject_id
                for subject in plan.subjects
                if any(
                    slot_id in packet.slot_ids
                    for slot_id in subject.evidence_slot_ids
                )
                or f"comparison-subject:{subject.subject_id}" in packet.slot_ids
            ],
        }
        for packet in packed_packets
    ]
    packed_slot_ids = {
        slot_id
        for packet in packed_packets
        for slot_id in packet.slot_ids
    }
    final_subject_ids = [
        subject.subject_id
        for subject in plan.subjects
        if any(
            slot_id in packed_slot_ids
            for slot_id in subject.evidence_slot_ids
        )
        or f"comparison-subject:{subject.subject_id}" in packed_slot_ids
    ]

    task_diagnostics = [
        dict(row)
        for row in retrieval_diagnostics
        if isinstance(row.get("subject_id"), str)
    ]
    return {
        "planner_status": (
            "planned"
            if contract.slot_plan_status in {"planned", "complete"}
            or contract.comparison_plan is not None
            else "fallback"
            if contract.slot_plan_status == "fallback"
            else "not_requested"
        ),
        "planner_latency_ms": 0.0,
        "planner_fallback_reason": contract.slot_plan_fallback_reason,
        "fallback_stage": None,
        "validation_issues": [],
        "is_comparison": True,
        "subjects": [
            subject.model_dump(mode="json")
            for subject in plan.subjects
        ],
        "dimensions": list(plan.dimensions),
        "task_diagnostics": task_diagnostics,
        "coverage_before_repair": list(before.get("covered") or []),
        "missing_before_repair": list(before.get("missing") or []),
        "repair_executed": any(
            bool(getattr(repair, "tasks", ())) for repair in repairs
        ),
        "coverage_after_repair": list(after.get("covered") or []),
        "missing_after_repair": list(after.get("missing") or []),
        "final_status": final_status,
        "final_evidence_subjects": final_subject_ids,
        "final_evidence_count": len(packed_packets),
        "final_evidence": final_evidence,
    }


def _candidate_diversification_projection(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None

    def ordered_ids(field: str) -> list[str]:
        raw_ids = value.get(field)
        if not isinstance(raw_ids, list):
            return []
        return list(dict.fromkeys(item for item in raw_ids if isinstance(item, str) and item))

    policy = value.get("policy")
    if not isinstance(policy, str) or not policy:
        return None
    return {
        "policy": policy,
        "enabled": bool(value.get("enabled")),
        "applied": bool(value.get("applied")),
        "retrieved_doc_ids": ordered_ids("retrieved_doc_ids"),
        "candidate_doc_ids": ordered_ids("candidate_doc_ids"),
        "represented_doc_ids_before_tail": ordered_ids(
            "represented_doc_ids_before_tail"
        ),
        "admitted_doc_ids": ordered_ids("admitted_doc_ids"),
    }


async def _locate_graph_documents(
    question: str,
    user_id: str,
    vector_documents: list[Document],
    authorized_doc_ids: list[str],
    contract: QueryContract,
    *,
    llm_invoker: LlmInvoker | None = None,
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
        llm_invoker=llm_invoker,
    )


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


def _required_capability_execution_failed(execution: dict[str, Any]) -> bool:
    """Downgrade only an attempted capability failure, never unavailable input."""
    if execution.get("state") == "executed" or not execution.get("attempted"):
        return False
    return execution.get("failure_reason") not in {
        "no_eligible_graph_source_evidence",
        "no_source_bound_graph_evidence",
        "no_eligible_visual_evidence",
    }


def _failed_required_stage(*, policy: str, error: Exception) -> dict[str, Any]:
    """Project an admitted stage failure without leaking provider internals."""
    return {
        "policy": policy,
        "required": policy == "visual_required",
        "state": "required_but_not_satisfied",
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
    }


async def _extract_visual_evidence(
    task: Any,
    documents: list[Document],
    question: str,
    controller: RunBudgetController,
    *,
    observer: LlmCallObserver | None = None,
    provider_factory: ProviderFactory | None = None,
    provider_name: str = "unknown",
    model_name: str = "unknown",
) -> VisualEvidenceExtractionResult:
    """Extract only selected, source-bound visual evidence from retrieved docs."""
    extractor = VisualEvidenceExtractor(
        BudgetedLlmInvoker(
            controller=controller,
            provider_factory=provider_factory or _provider_for_purpose,
            observer=observer,
            provider_name=provider_name,
            model_name=model_name,
        )
    )
    return await extractor.extract(
        task=task,
        assets=_visual_assets_from_documents(documents, task),
        question_fragment=question,
    )


def _visual_assets_from_documents(
    documents: list[Document], task: Any
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
        ):
            continue
        assets.append(
            VisualAssetCandidate(
                asset_id=str(
                    metadata.get("asset_id") or f"{doc_id}:page:{page}:{index}"
                ),
                source=EvidenceSource(
                    doc_id=doc_id,
                    chunk_id=str(metadata.get("chunk_id") or index + 1),
                ),
                pdf_page_index=page,
                slot_ids=list(task.target_slot_ids),
                figure_id=metadata.get("figure_id"),
                table_id=metadata.get("table_id"),
                bbox=metadata.get("bbox"),
                page_image_base64=image_base64,
                page_encoded_bytes=int(metadata.get("page_encoded_bytes") or 0),
                page_width=width,
                page_height=height,
            )
        )
    return assets


def _initial_visual_execution(contract: QueryContract | None) -> dict[str, Any]:
    required = bool(contract and contract.visual_required)
    return {
        "required": required,
        "state": "not_triggered" if required else "not_requested",
        "attempted": False,
        "failure_reason": None,
        "selected_asset_count": 0,
        "dropped_asset_count": 0,
        "evidence_packet_count": 0,
    }


def _visual_execution_projection(
    result: VisualEvidenceExtractionResult,
) -> dict[str, Any]:
    packet_count = len(result.packets)
    return {
        "required": True,
        "state": "executed" if packet_count else "required_but_not_satisfied",
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
    }


async def _resolve_document_references(
    user_id: str, references: list[str]
) -> dict[str, list[str]]:
    from data_base.repository import resolve_document_references

    return await resolve_document_references(user_id=user_id, references=references)


async def _list_owned_document_ids(user_id: str) -> list[str]:
    from data_base.repository import list_owned_document_ids

    return await list_owned_document_ids(user_id=user_id)


def _provider_for_purpose(purpose: str) -> Any:
    if purpose == "atomic_contract_planning":
        return build_contract_planning_provider(
            response_schema=atomic_contract_planner_response_schema()
        )
    return get_llm("synthesizer")


def _project_chunk_id(
    metadata: dict[str, Any], index: int, *, task_id: str | None = None
) -> str:
    explicit_chunk_id = metadata.get("chunk_id")
    if explicit_chunk_id not in (None, ""):
        return str(explicit_chunk_id)
    fallback = f"chunk-{index + 1}"
    return f"{task_id}:{fallback}" if task_id else fallback


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if isfinite(numeric) else None


def _chunk_reranking_projection(
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    value = metadata.get("agentic_v9_reranking")
    if not isinstance(value, Mapping):
        return None
    return {
        "status": str(value.get("status") or "not_instrumented"),
        "post_rerank_rank": _positive_int_or_none(value.get("post_rerank_rank")),
        "rerank_score": _finite_float_or_none(value.get("rerank_score")),
    }


def _chunk_projection(
    document: Document, index: int, *, task_id: str | None = None
) -> dict[str, Any]:
    metadata = dict(document.metadata or {})
    doc_id = get_document_id(metadata)
    projection = {
        "doc_id": doc_id,
        "chunk_id": _project_chunk_id(metadata, index, task_id=task_id),
        "text": str(document.page_content or ""),
        "page_number": metadata.get("page_number"),
        "section": metadata.get("section"),
    }
    reranking = _chunk_reranking_projection(metadata)
    if reranking is not None:
        projection["reranking"] = reranking
    for field in ("asset_id", "figure_id", "table_id"):
        value = metadata.get(field)
        if isinstance(value, str) and value:
            projection[field] = value
    return projection


def _packet_quality(chunk: Mapping[str, Any], fallback_index: int) -> float:
    reranking = chunk.get("reranking")
    if isinstance(reranking, Mapping):
        rank = _positive_int_or_none(reranking.get("post_rerank_rank"))
        if rank is not None:
            return 1.0 / rank
    return 1.0 / (fallback_index + 1)


@dataclass(frozen=True, slots=True)
class EvidencePacketProjection:
    packets: list[EvidencePacket]
    quality_by_evidence_id: dict[str, float]


def _evidence_packets_for_results(
    *,
    results: tuple[TaskRetrievalResult, ...],
    contract: QueryContract,
    trace_id: str,
    task_slot_ids: dict[str, list[str]],
) -> EvidencePacketProjection:
    packets: list[EvidencePacket] = []
    quality_by_evidence_id: dict[str, float] = {}
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
            locator_fields: dict[str, Any] = {}
            if isinstance(page, int) and page >= 0:
                locator_fields["pdf_page_index"] = page
            else:
                locator_fields["section"] = str(
                    chunk.get("section") or "retrieved_context"
                )
            for field in ("figure_id", "table_id"):
                value = chunk.get(field)
                if isinstance(value, str) and value:
                    locator_fields[field] = value
            evidence_id = f"evidence:{digest}"
            source_fields: dict[str, Any] = {
                "doc_id": doc_id,
                "chunk_id": str(chunk.get("chunk_id") or index + 1),
            }
            asset_id = chunk.get("asset_id")
            if isinstance(asset_id, str) and asset_id:
                source_fields["asset_id"] = asset_id
            packets.append(
                EvidencePacket(
                    schema_version="1",
                    evidence_id=evidence_id,
                    task_id=task_id,
                    round_id=task_id.split(":")[-2] if ":" in task_id else "round-1",
                    query_id=trace_id,
                    slot_ids=list(slot_ids),
                    statement=text,
                    support_type="direct",
                    source=EvidenceSource(**source_fields),
                    scope=EvidenceScope(),
                    locator=SourceLocator(**locator_fields),
                    validation_status="invalid",
                )
            )
            quality_by_evidence_id[evidence_id] = _packet_quality(chunk, index)
    return EvidencePacketProjection(
        packets=packets,
        quality_by_evidence_id=quality_by_evidence_id,
    )


def _requires_diverse_rerank_candidates(route: str) -> bool:
    """Enable the recall-only treatment for routes that span source claims."""
    return route in {
        "bounded_compare",
        "multi_hop",
        "multi_document_exact",
        "graph_relational",
    }


def _configuration_incompatible_result(
    *,
    question: str,
    trace_id: str,
    stage: str,
    feasibility: FeasibilityResult,
    contract: QueryContract | None = None,
    execution_profile: str = AGENTIC_EVAL_PROFILE,
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
            "execution_profile": execution_profile,
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


def _response_text(response: Any) -> str:
    content = (
        response.get("content")
        if isinstance(response, dict)
        else getattr(response, "content", response)
    )
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "").strip()


def _context_pack_projection(
    packed: PackedEvidenceContext | None,
) -> dict[str, Any] | None:
    if packed is None:
        return None
    return {
        "packed_evidence_ids": [packet.evidence_id for packet in packed.packets],
        "dropped_evidence_ids": list(packed.dropped_packet_ids),
        "token_count": packed.estimated_input_tokens,
        "selection_policy_version": packed.selection_policy_version,
        "candidate_count": len(packed.selection_decisions),
        "selection_decisions": [
            {
                "evidence_id": decision.evidence_id,
                "selected": decision.selected,
                "base_quality": decision.base_quality,
                "source_bonus": decision.source_bonus,
                "redundancy_penalty": decision.redundancy_penalty,
                "visual_penalty": decision.visual_penalty,
                "utility": decision.utility,
                "reason": decision.reason,
            }
            for decision in packed.selection_decisions
        ],
    }


async def _discard_event(_: V9ExecutionEvent) -> None:
    return None


__all__ = ["AgenticV9CampaignRuntime"]
