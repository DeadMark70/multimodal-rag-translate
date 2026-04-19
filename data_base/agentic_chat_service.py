"""Chat-facing agentic benchmark streaming service."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Optional

from agents.planner import (
    classify_question_intent,
    classify_question_intent_semantic,
    required_coverage_for_intent,
)
from core.errors import AppError
from data_base.RAG_QA_service import RAGResult, rag_answer_question
from data_base.repository import persist_research_conversation
from data_base.schemas_agentic_chat import AgenticBenchmarkStreamRequest
from data_base.schemas_deep_research import (
    EditableSubTask,
    ExecutePlanRequest,
    ExecutePlanResponse,
    SubTaskExecutionResult,
)
from data_base.sse_events import (
    ErrorData,
    SSEEventType,
    format_sse_event,
)
from evaluation.agentic_evaluation_service import (
    AGENTIC_EVAL_PROFILE,
    AGENTIC_IMAGE_ANALYSIS_ENABLED,
    AgenticEvaluationService,
    _append_trace_step,
    _drilldown_iterations_for_strategy,
    _finalize_trace_payload,
    _micro_route_for_task,
    _normalize_tool_calls,
    _route_profile_for_task,
    _strategy_config_from_complexity,
    _strategy_tier_for_intent,
    _subtask_limit_for_strategy,
)
from evaluation.retry import run_with_retry

logger = logging.getLogger(__name__)

_TASK_STAGE_LABELS = {
    "query_expansion": "正在擴展查詢",
    "retrieval": "正在檢索文件",
    "reranking": "正在重排序結果",
    "crag_correction": "正在校正檢索結果",
    "graph_context": "正在分析圖譜上下文",
    "answer_generation": "正在生成回答",
}


class StreamingAgenticEvaluationService(AgenticEvaluationService):
    """Agentic benchmark runtime that emits chat SSE events while executing."""

    def __init__(
        self,
        *,
        emit_event,
        max_concurrent_tasks: int = 3,
        default_max_iterations: int = 2,
    ) -> None:
        super().__init__(
            max_concurrent_tasks=max_concurrent_tasks,
            default_max_iterations=default_max_iterations,
        )
        self._emit_event = emit_event
        self.trace_steps: list[dict[str, Any]] = []
        self.trace_summary: str = "Agentic trace unavailable"

    async def _emit(self, event_type: SSEEventType, payload: Any) -> None:
        await self._emit_event(format_sse_event(event_type, payload))

    async def _record_trace_step(self, **kwargs: Any) -> None:
        _append_trace_step(self.trace_steps, **kwargs)
        await self._emit(
            SSEEventType.TRACE_STEP,
            self.trace_steps[-1],
        )

    async def _execute_tasks(
        self,
        tasks: list[EditableSubTask],
        user_id: str,
        doc_ids: Optional[list[str]],
        enable_reranking: bool,
        iteration: int,
        enable_deep_image_analysis: bool = False,
    ) -> list[SubTaskExecutionResult]:
        effective_intent = self._active_question_intent or "enumeration_definition"
        effective_tier = self._active_strategy_tier or "tier_1_detail_lookup"
        stage_hint = "verification" if iteration > 0 else "exploration"

        results: list[SubTaskExecutionResult] = []
        for task in tasks:
            predicted_micro_route = _micro_route_for_task(
                question_intent=effective_intent,
                task_type=task.task_type,
                task_question=task.question,
            )
            if self._semantic_router_mode == "active":
                route_profile = _route_profile_for_task(
                    strategy_tier=effective_tier,
                    question_intent=effective_intent,
                    task_type=task.task_type,
                    task_question=task.question,
                    iteration=iteration,
                    micro_route=predicted_micro_route,
                )
                applied_micro_route = predicted_micro_route
            else:
                route_profile = _route_profile_for_task(
                    strategy_tier=effective_tier,
                    question_intent=effective_intent,
                    task_type=task.task_type,
                    task_question=task.question,
                    iteration=iteration,
                    micro_route=None,
                )
                applied_micro_route = "broad_context_rag"
            start_event = (
                SSEEventType.DRILLDOWN_TASK_START
                if iteration > 0
                else SSEEventType.TASK_START
            )
            await self._emit(
                start_event,
                {
                    "id": task.id,
                    "question": task.question,
                    "task_type": task.task_type,
                    "iteration": iteration,
                    "route_profile": route_profile,
                    "micro_route": predicted_micro_route,
                    "strategy_tier": effective_tier,
                },
            )

            kwargs = self._route_kwargs(
                route_profile=route_profile,
                micro_route=applied_micro_route,
                enable_reranking=enable_reranking,
                enable_visual_verification=enable_deep_image_analysis,
                task_type=task.task_type,
                stage_hint=stage_hint,
            )

            async def emit_task_phase(
                stage: str,
                details: Optional[dict[str, Any]] = None,
            ) -> None:
                await self._emit(
                    SSEEventType.TASK_PHASE_UPDATE,
                    {
                        "id": task.id,
                        "iteration": iteration,
                        "stage": stage,
                        "label": _TASK_STAGE_LABELS.get(stage),
                        "details": details,
                    },
                )

            try:
                result = await rag_answer_question(
                    question=task.question,
                    user_id=user_id,
                    doc_ids=doc_ids,
                    progress_callback=emit_task_phase,
                    **kwargs,
                )
                assert isinstance(result, RAGResult)
                contexts = [document.page_content for document in result.documents]
                evidence_units = self._build_evidence_units(
                    result_id=task.id,
                    question=task.question,
                    iteration=iteration,
                    route_profile=route_profile,
                    contexts=contexts,
                    source_doc_ids=list(result.source_doc_ids),
                )
                sub_result = SubTaskExecutionResult(
                    id=task.id,
                    question=task.question,
                    answer=result.answer,
                    sources=list(result.source_doc_ids),
                    contexts=contexts,
                    is_drilldown=iteration > 0,
                    iteration=iteration,
                    usage=dict(result.usage or {"total_tokens": 0}),
                    thought_process=result.thought_process,
                    tool_calls=list(result.tool_calls or []),
                    strategy_tier=effective_tier,
                    route_profile=route_profile,
                    micro_route=predicted_micro_route,
                    evidence_units=evidence_units,
                    visual_verification_meta=dict(result.visual_verification_meta or {}),
                )
            except Exception as exc:  # noqa: BLE001
                sub_result = SubTaskExecutionResult(
                    id=task.id,
                    question=task.question,
                    answer=f"無法回答此問題: {str(exc)[:160]}",
                    sources=[],
                    contexts=[],
                    is_drilldown=iteration > 0,
                    iteration=iteration,
                    strategy_tier=effective_tier,
                    route_profile=route_profile,
                    micro_route=predicted_micro_route,
                    visual_verification_meta={},
                )

            results.append(sub_result)
            done_event = (
                SSEEventType.DRILLDOWN_TASK_DONE
                if iteration > 0
                else SSEEventType.TASK_DONE
            )
            await self._emit(
                done_event,
                {
                    "id": sub_result.id,
                    "question": sub_result.question,
                    "answer": sub_result.answer,
                    "sources": sub_result.sources,
                    "contexts": sub_result.contexts,
                    "iteration": sub_result.iteration,
                    "route_profile": sub_result.route_profile,
                    "micro_route": sub_result.micro_route,
                    "usage": sub_result.usage,
                    "tool_calls": sub_result.tool_calls,
                },
            )
            normalized_tool_calls = _normalize_tool_calls(sub_result.tool_calls)
            await self._record_trace_step(
                phase="drilldown" if sub_result.iteration > 0 else "execution",
                step_type="sub_task_execution",
                title=f"Step {sub_result.id}",
                input_preview=sub_result.question,
                output_preview=sub_result.answer[:420],
                raw_text=sub_result.thought_process,
                tool_calls=normalized_tool_calls,
                token_usage=dict(sub_result.usage or {}),
                metadata={
                    "iteration": sub_result.iteration,
                    "is_drilldown": sub_result.is_drilldown,
                    "strategy_tier": sub_result.strategy_tier,
                    "route_profile": sub_result.route_profile,
                    "micro_route": sub_result.micro_route,
                    "source_count": len(sub_result.sources),
                    "context_count": len(sub_result.contexts),
                },
            )

        await self._emit(
            SSEEventType.EVALUATION_UPDATE,
            {
                "iteration": iteration,
                "stage": "task_batch_complete",
                "task_count": len(results),
                "coverage_status": self._coverage_status(results),
                "coverage_gaps": self._coverage_gaps(results),
            },
        )
        return results

    async def _drill_down_loop(
        self,
        original_question: str,
        current_results: list[SubTaskExecutionResult],
        user_id: str,
        doc_ids: Optional[list[str]],
        enable_reranking: bool,
        max_iterations: int,
        enable_deep_image_analysis: bool = False,
    ) -> int:
        if max_iterations <= 0:
            return 0

        await self._emit(
            SSEEventType.EVALUATION_UPDATE,
            {
                "iteration": 0,
                "stage": "drilldown_start",
                "gate_pass": False,
                "coverage_gaps": self._coverage_gaps(current_results),
                "details": {"max_iterations": max_iterations},
            },
        )
        performed = await super()._drill_down_loop(
            original_question=original_question,
            current_results=current_results,
            user_id=user_id,
            doc_ids=doc_ids,
            enable_reranking=enable_reranking,
            max_iterations=max_iterations,
            enable_deep_image_analysis=enable_deep_image_analysis,
        )
        await self._emit(
            SSEEventType.EVALUATION_UPDATE,
            {
                "iteration": performed,
                "stage": "drilldown_complete",
                "gate_pass": True,
                "coverage_gaps": self._coverage_gaps(current_results),
                "details": {
                    "total_results": len(current_results),
                    "tier_shift": self._tier_shift,
                    "pruned_followups": self._pruned_followups_total,
                },
            },
        )
        return performed

    async def _synthesize_execution_results(
        self,
        *,
        original_question: str,
        all_results: list[SubTaskExecutionResult],
        total_iterations: int,
    ) -> ExecutePlanResponse:
        await self._emit(
            SSEEventType.SYNTHESIS_START,
            {"total_tasks": len(all_results)},
        )
        result = await super()._synthesize_execution_results(
            original_question=original_question,
            all_results=all_results,
            total_iterations=total_iterations,
        )
        self.trace_summary = result.summary or "Agentic research completed"
        await self._emit(
            SSEEventType.EVALUATION_UPDATE,
            {
                "iteration": total_iterations,
                "stage": "critic",
                "details": dict(result.critic_summary),
            },
        )
        await self._record_trace_step(
            phase="synthesis",
            step_type="report_synthesis",
            title="Synthesize final report",
            input_preview=f"{len(result.sub_tasks)} sub-tasks",
            output_preview=(result.summary or "")[:420],
            raw_text=result.detailed_answer,
            token_usage={
                "total_tokens": sum(
                    int(sub.usage.get("total_tokens", 0))
                    for sub in result.sub_tasks
                )
            },
            metadata={
                "source_count": len(result.all_sources),
                "total_iterations": result.total_iterations,
                "question_intent": self._active_question_intent,
                "strategy_tier": self._active_strategy_tier,
                "coverage_gaps": self._coverage_gaps(result.sub_tasks),
                "subtask_coverage_status": self._coverage_status(result.sub_tasks),
                "critic_summary": dict(result.critic_summary),
            },
        )
        return result

    def build_trace_payload(
        self,
        *,
        question_id: str,
        question: str,
        run_number: int,
        result: ExecutePlanResponse,
    ) -> dict[str, Any]:
        flattened_tool_calls: list[dict[str, Any]] = []
        route_profiles: list[str] = []
        visual_verification_attempted = False
        visual_tool_call_count = 0
        visual_force_fallback_used = False

        for sub in result.sub_tasks:
            flattened_tool_calls.extend(sub.tool_calls)
            if sub.route_profile:
                route_profiles.append(sub.route_profile)
            visual_meta = dict(sub.visual_verification_meta or {})
            visual_verification_attempted = (
                visual_verification_attempted
                or bool(visual_meta.get("visual_verification_attempted"))
            )
            visual_tool_call_count += int(visual_meta.get("visual_tool_call_count", 0) or 0)
            visual_force_fallback_used = (
                visual_force_fallback_used
                or bool(visual_meta.get("visual_force_fallback_used"))
            )

        dominant_route_profile = route_profiles[0] if route_profiles else None
        coverage_status = self._coverage_status(result.sub_tasks)
        coverage_gaps = self._coverage_gaps(result.sub_tasks)
        supported_claim_count = int(result.critic_summary.get("supported_claim_count", 0))
        unsupported_claim_count = int(result.critic_summary.get("unsupported_claim_count", 0))

        return _finalize_trace_payload(
            question_id=question_id,
            question=question,
            run_number=run_number,
            steps=self.trace_steps,
            summary=self.trace_summary,
            trace_status="completed",
            execution_profile=self.execution_profile,
            question_intent=self._active_question_intent,
            strategy_tier=self._active_strategy_tier,
            route_profile=dominant_route_profile,
            required_coverage=list(self._required_coverage),
            coverage_gaps=coverage_gaps,
            subtask_coverage_status=coverage_status,
            claims=list(result.claims),
            supported_claim_count=supported_claim_count,
            unsupported_claim_count=unsupported_claim_count,
            visual_verification_attempted=visual_verification_attempted,
            visual_tool_call_count=visual_tool_call_count,
            visual_force_fallback_used=visual_force_fallback_used,
            classifier_decision=dict(self._classifier_decision),
            complexity_score=int(self._classifier_decision.get("complexity_score", 0) or 0) or None,
            tier_shift=self._tier_shift,
            pruned_followups=self._pruned_followups_total,
            semantic_gate_score=self._last_semantic_gate_score,
        )


class AgenticChatService:
    """Streaming orchestration for chat-facing agentic benchmark execution."""

    async def execute_stream(
        self,
        *,
        request: AgenticBenchmarkStreamRequest,
        user_id: str,
    ):
        queue: asyncio.Queue[dict[str, str] | None] = asyncio.Queue()

        async def emit_event(event: dict[str, str]) -> None:
            await queue.put(event)

        async def run_pipeline() -> None:
            service = StreamingAgenticEvaluationService(emit_event=emit_event, max_concurrent_tasks=3)
            question_intent = classify_question_intent(request.question)
            semantic_complexity_score = 3
            classifier_decision: dict[str, Any] = {
                "intent": question_intent,
                "complexity_score": semantic_complexity_score,
                "confidence": 0.55,
                "rationale": "heuristic default decision",
                "source": "heuristic",
            }

            if service._semantic_router_mode in {"shadow", "active"}:
                semantic_decision = await classify_question_intent_semantic(request.question)
                classifier_decision = semantic_decision.model_dump(mode="json")
                semantic_complexity_score = semantic_decision.complexity_score

            if service._semantic_router_mode == "active":
                question_intent = str(classifier_decision.get("intent") or question_intent)
                strategy_tier, subtask_limit, max_drilldown_iterations = _strategy_config_from_complexity(
                    semantic_complexity_score
                )
            else:
                strategy_tier = _strategy_tier_for_intent(question_intent)
                subtask_limit = _subtask_limit_for_strategy(
                    strategy_tier=strategy_tier,
                    question_intent=question_intent,
                )
                max_drilldown_iterations = _drilldown_iterations_for_strategy(
                    strategy_tier=strategy_tier,
                    question_intent=question_intent,
                )

            service._active_question_intent = question_intent
            service._active_strategy_tier = strategy_tier
            service._active_subtask_limit = subtask_limit
            service._active_max_iterations = max_drilldown_iterations
            service._required_coverage = required_coverage_for_intent(question_intent)
            service._classifier_decision = classifier_decision
            service._tier_shift = "keep"
            service._pruned_followups_total = 0
            service._last_semantic_gate_score = None

            try:
                plan_response = await run_with_retry(
                    service.generate_agentic_plan,
                    question=request.question,
                    user_id=user_id,
                    question_intent=question_intent,
                    strategy_tier=strategy_tier,
                )
                await emit_event(
                    format_sse_event(
                        SSEEventType.PLAN_READY,
                        {
                            "original_question": request.question,
                            "estimated_complexity": plan_response.estimated_complexity,
                            "task_count": len(plan_response.sub_tasks),
                            "enabled_count": len(plan_response.sub_tasks),
                            "question_intent": question_intent,
                            "strategy_tier": strategy_tier,
                            "semantic_router_mode": service._semantic_router_mode,
                            "classifier_decision": dict(classifier_decision),
                            "complexity_score": semantic_complexity_score,
                            "subtask_limit": subtask_limit,
                            "max_iterations": max_drilldown_iterations,
                            "sub_tasks": [task.model_dump(mode="json") for task in plan_response.sub_tasks],
                        },
                    )
                )
                planning_text = "\n".join(
                    f"{task.id}. [{task.task_type}] {task.question}"
                    for task in plan_response.sub_tasks
                )
                await service._record_trace_step(
                    phase="planning",
                    step_type="plan_generation",
                    title="Generate benchmark plan",
                    input_preview=request.question,
                    output_preview=f"{len(plan_response.sub_tasks)} tasks / {plan_response.estimated_complexity}",
                    raw_text=planning_text,
                    metadata={
                        "estimated_complexity": plan_response.estimated_complexity,
                        "question_intent": question_intent,
                        "strategy_tier": strategy_tier,
                        "semantic_router_mode": service._semantic_router_mode,
                        "classifier_decision": dict(classifier_decision),
                        "complexity_score": semantic_complexity_score,
                        "subtask_limit": subtask_limit,
                        "max_drilldown_iterations": max_drilldown_iterations,
                        "required_coverage": list(service._required_coverage),
                        "sub_tasks": [task.model_dump(mode="json") for task in plan_response.sub_tasks],
                    },
                )

                execute_request = ExecutePlanRequest(
                    original_question=request.question,
                    sub_tasks=plan_response.sub_tasks,
                    doc_ids=request.doc_ids,
                    enable_reranking=request.enable_reranking,
                    enable_drilldown=max_drilldown_iterations > 0,
                    max_iterations=max(max_drilldown_iterations, 1),
                    enable_deep_image_analysis=request.enable_deep_image_analysis
                    if request.enable_deep_image_analysis is not None
                    else AGENTIC_IMAGE_ANALYSIS_ENABLED,
                    conversation_id=request.conversation_id,
                )
                result = await run_with_retry(
                    service.run_execute_plan,
                    request=execute_request,
                    user_id=user_id,
                )

                trace_payload = service.build_trace_payload(
                    question_id=f"chat-{request.conversation_id or 'adhoc'}",
                    question=request.question,
                    run_number=1,
                    result=result,
                )
                await self._persist_result(
                    request=request,
                    user_id=user_id,
                    result=result,
                    trace_payload=trace_payload,
                )
                await emit_event(
                    format_sse_event(
                        SSEEventType.COMPLETE,
                        {
                            "result": result.model_dump(mode="json"),
                            "agent_trace": trace_payload,
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Agentic benchmark chat stream failed: %s", exc, exc_info=True)
                await emit_event(
                    format_sse_event(
                        SSEEventType.ERROR,
                        ErrorData(message="Agentic benchmark stream failed"),
                    )
                )
            finally:
                service._active_question_intent = None
                service._active_strategy_tier = None
                service._active_subtask_limit = None
                service._active_max_iterations = None
                service._required_coverage = []
                service._classifier_decision = {}
                await queue.put(None)

        runner = asyncio.create_task(run_pipeline())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not runner.done():
                runner.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await runner

    async def _persist_result(
        self,
        *,
        request: AgenticBenchmarkStreamRequest,
        user_id: str,
        result: ExecutePlanResponse,
        trace_payload: dict[str, Any],
    ) -> None:
        if not request.conversation_id:
            return

        try:
            metadata_payload: dict[str, Any] = {
                "research_engine": "agentic_benchmark",
                "engine": "agentic_benchmark",
                "execution_profile": AGENTIC_EVAL_PROFILE,
                "original_question": request.question,
                "result": result.model_dump(mode="json"),
                "agent_trace": trace_payload,
            }
            await persist_research_conversation(
                conversation_id=request.conversation_id,
                user_id=user_id,
                title=request.question[:100] if request.question else None,
                metadata=metadata_payload,
            )
        except AppError as exc:
            logger.error(
                "Failed to persist agentic benchmark chat result: %s",
                exc,
                exc_info=True,
            )


_agentic_chat_service: AgenticChatService | None = None


def get_agentic_chat_service() -> AgenticChatService:
    global _agentic_chat_service
    if _agentic_chat_service is None:
        _agentic_chat_service = AgenticChatService()
    return _agentic_chat_service
