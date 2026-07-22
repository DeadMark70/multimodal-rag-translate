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
from data_base.RAG_QA_service import RAGResult
from data_base.agentic_v9.budget_controller import RunBudgetController
from data_base.agentic_v9.budget_feasibility import (
    FeasibilityResult,
    FeasibilityStatus,
    validate_post_contract_feasibility,
    validate_pre_route_feasibility,
)
from data_base.agentic_v9.budgeted_llm import BudgetedLlmInvoker
from data_base.agentic_v9.context_packer import EvidenceContextPacker, PackedEvidenceContext
from data_base.agentic_v9.execution_core import (
    ConflictStageResult,
    V9ExecutionCore,
    V9ExecutionStages,
)
from data_base.agentic_v9.execution_policy import (
    ExecutionCancellation,
    V9ExecutionPolicyRuntime,
)
from data_base.agentic_v9.repair import build_repair_plan
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    FinalAnswerResult,
    FinalClaim,
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
from data_base.agentic_v9.sufficiency_gate import SufficiencyEvaluation, evaluate_sufficiency
from data_base.document_metadata import get_document_id
from data_base.rag_filtering import filter_and_rerank_retrieval
from data_base.rag_retrieval import retrieve_hybrid_documents
from data_base.vector_store_manager import get_user_retriever_async
from evaluation.agentic_campaign_adapter import used_evidence_documents
from evaluation.agentic_v9_admission import (
    DocumentReferenceResolver,
    build_v9_admission_contract,
)
from evaluation.retrieval_profiles import AGENTIC_EVAL_PROFILE


RetrievalAdapter = Callable[[str, str, list[str]], Awaitable[list[Document]]]
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
        provider_factory: ProviderFactory | None = None,
        policy_runtime: V9ExecutionPolicyRuntime | None = None,
        document_reference_resolver: DocumentReferenceResolver | None = None,
    ) -> None:
        self._retrieve_documents = retrieve_documents or _retrieve_documents
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
        )
        source_scope = admission.source_scope
        runtime_contract = admission.contract
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
            "budget_controller": None,
            "task_slot_ids": {},
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
                remaining_llm_calls=contract.max_llm_calls,
                route_plan_used=False,
            )
            state["contract"] = contract
            state["post_contract"] = post_contract
            if post_contract.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE:
                raise _ConfigurationIncompatible(
                    stage="post_contract", feasibility=post_contract
                )
            state["budget_controller"] = RunBudgetController(
                max_llm_calls=contract.max_llm_calls,
                runtime_token_budget=contract.runtime_token_budget,
                setup_snapshot=setup_snapshot,
                final_input_tokens=_final_input_reserve(
                    setup_snapshot, contract.runtime_token_budget
                ),
            )
            return contract

        async def retrieve(
            tasks: tuple[Any, ...],
        ) -> tuple[TaskRetrievalResult, ...]:
            results: list[TaskRetrievalResult] = []
            for task in tasks:
                state["task_slot_ids"][task.task_id] = list(task.target_slot_ids)
                docs = await self._retrieve_documents(
                    user_id, task.query, list(task.source_scope.authorized_doc_ids)
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
            state["evidence_packets"].extend(packets)
            return tuple(packets)

        def sufficiency(
            contract: QueryContract, packets: tuple[EvidencePacket, ...]
        ) -> SufficiencyEvaluation:
            return evaluate_sufficiency(contract, packets)

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
            _: str,
            __: QueryContract,
            packed: PackedEvidenceContext,
            resolutions: tuple[SlotResolution, ...],
            ___: Any | None,
        ) -> FinalAnswerResult:
            controller = state["budget_controller"]
            assert isinstance(controller, RunBudgetController)
            response = await BudgetedLlmInvoker(
                controller=controller,
                provider_factory=self._provider_factory,
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
                    for resolution in (executed.sufficiency and evaluate_sufficiency(state["contract"], state["evidence_packets"]).slot_resolutions or ())
                ],
                "sufficiency": executed.sufficiency.model_dump(mode="json") if executed.sufficiency else None,
                "context_pack": _context_pack_projection(packed),
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
    return _setup_positive_int(snapshot, "max_llm_calls", default=3)


def _final_input_reserve(snapshot: dict[str, Any], runtime_token_budget: int) -> int:
    return min(
        _setup_positive_int(
            snapshot, "setup_max_input_tokens", "max_input_tokens", default=8192
        ),
        8192,
        max(runtime_token_budget // 2, 1),
    )


def _response_text(response: Any) -> str:
    content = response.get("content") if isinstance(response, dict) else getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content).strip()
    return str(content or "").strip()


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
