"""Evaluation-only Agentic RAG service forked from user-facing Deep Research wrappers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

from agents.planner import TaskPlanner
from data_base.RAG_QA_service import RAGResult
from data_base.research_execution_core import ResearchExecutionCore
from data_base.schemas_deep_research import EditableSubTask, ExecutePlanRequest, ResearchPlanResponse
from evaluation.retry import run_with_retry
from evaluation.trace_schemas import AgentTraceToolCall

AGENTIC_EVAL_PROFILE = "agentic_eval_v2"
LEGACY_SHARED_PROFILE = "legacy_shared"
AGENTIC_INITIAL_SUBTASKS = 3
AGENTIC_MAX_DRILLDOWN_ITERATIONS = 1
AGENTIC_IMAGE_ANALYSIS_ENABLED = True


class AgentTraceCaptureError(RuntimeError):
    """Exception wrapper that carries a partial agent trace."""

    def __init__(self, message: str, agent_trace: dict[str, Any]) -> None:
        super().__init__(message)
        self.agent_trace = agent_trace


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preview(text: str | None, limit: int = 280) -> str | None:
    if not text:
        return None
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        payload = dict(call or {})
        status = "failed" if payload.get("success") is False or payload.get("status") == "failed" else "completed"
        action = str(payload.get("action") or payload.get("name") or payload.get("tool") or f"tool_{index + 1}")
        normalized.append(
            AgentTraceToolCall(
                index=index,
                action=action,
                status=status,
                payload=payload,
                result_preview=_preview(
                    str(payload.get("result") or payload.get("output") or payload.get("error") or "")
                ),
            ).model_dump(mode="json")
        )
    return normalized


def _append_trace_step(
    steps: list[dict[str, Any]],
    *,
    phase: str,
    step_type: str,
    title: str,
    status: str = "completed",
    input_preview: str | None = None,
    output_preview: str | None = None,
    raw_text: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    token_usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    timestamp = _utc_now_iso()
    steps.append(
        {
            "step_id": f"{phase}-{len(steps) + 1}",
            "phase": phase,
            "step_type": step_type,
            "title": title,
            "status": status,
            "started_at": timestamp,
            "completed_at": timestamp,
            "input_preview": input_preview,
            "output_preview": output_preview,
            "raw_text": raw_text,
            "tool_calls": tool_calls or [],
            "token_usage": token_usage or {},
            "metadata": metadata or {},
        }
    )


def _finalize_trace_payload(
    *,
    question_id: str,
    question: str,
    run_number: int,
    steps: list[dict[str, Any]],
    summary: str,
    trace_status: str,
    execution_profile: str,
) -> dict[str, Any]:
    tool_call_count = sum(len(step.get("tool_calls", [])) for step in steps)
    total_tokens = sum(int(step.get("token_usage", {}).get("total_tokens", 0) or 0) for step in steps)
    return {
        "trace_id": str(uuid4()),
        "question_id": question_id,
        "question": question,
        "mode": "agentic",
        "run_number": run_number,
        "execution_profile": execution_profile,
        "trace_status": trace_status,
        "summary": summary,
        "step_count": len(steps),
        "tool_call_count": tool_call_count,
        "total_tokens": total_tokens,
        "created_at": _utc_now_iso(),
        "steps": steps,
    }


class AgenticEvaluationService(ResearchExecutionCore):
    """Evaluation-only agentic execution wrapper with stable profiling metadata."""

    execution_profile = AGENTIC_EVAL_PROFILE

    async def generate_agentic_plan(
        self,
        *,
        question: str,
        user_id: str,
    ) -> ResearchPlanResponse:
        """Generate the dedicated evaluation baseline plan for agentic RAG."""
        planner = TaskPlanner(
            max_subtasks=AGENTIC_INITIAL_SUBTASKS,
            enable_graph_planning=True,
        )
        plan = await planner.plan(question)
        editable_tasks = [
            EditableSubTask(
                id=task.id,
                question=task.question,
                task_type=task.task_type,
                enabled=True,
            )
            for task in plan.sub_tasks[:AGENTIC_INITIAL_SUBTASKS]
        ]
        return ResearchPlanResponse(
            status="waiting_confirmation",
            original_question=question,
            sub_tasks=editable_tasks,
            estimated_complexity=plan.estimated_complexity,
            doc_ids=None,
        )

    async def run_case(
        self,
        *,
        question_id: str,
        question: str,
        user_id: str,
        run_number: int,
    ) -> RAGResult:
        trace_steps: list[dict[str, Any]] = []
        trace_summary = "Agentic trace unavailable"

        try:
            plan_response = await run_with_retry(
                self.generate_agentic_plan,
                question=question,
                user_id=user_id,
            )
            planning_text = "\n".join(
                f"{task.id}. [{task.task_type}] {task.question}"
                for task in plan_response.sub_tasks
            )
            _append_trace_step(
                trace_steps,
                phase="planning",
                step_type="plan_generation",
                title="Generate research plan",
                input_preview=question,
                output_preview=f"{len(plan_response.sub_tasks)} tasks / {plan_response.estimated_complexity}",
                raw_text=planning_text,
                metadata={
                    "estimated_complexity": plan_response.estimated_complexity,
                    "sub_tasks": [task.model_dump(mode="json") for task in plan_response.sub_tasks],
                },
            )

            request = ExecutePlanRequest(
                original_question=question,
                sub_tasks=plan_response.sub_tasks,
                doc_ids=None,
                enable_reranking=True,
                enable_drilldown=True,
                max_iterations=AGENTIC_MAX_DRILLDOWN_ITERATIONS,
                enable_deep_image_analysis=AGENTIC_IMAGE_ANALYSIS_ENABLED,
            )
            result_response = await run_with_retry(
                self.run_execute_plan,
                request=request,
                user_id=user_id,
            )

            aggregated_docs: list[Document] = []
            seen_contexts: set[str] = set()
            total_tokens = 0
            flattened_tool_calls: list[dict[str, Any]] = []
            seen_iterations: set[int] = set()

            for sub_result in result_response.sub_tasks:
                total_tokens += sub_result.usage.get("total_tokens", 0)
                for context in sub_result.contexts:
                    if context not in seen_contexts:
                        aggregated_docs.append(Document(page_content=context))
                        seen_contexts.add(context)

                if sub_result.iteration > 0 and sub_result.iteration not in seen_iterations:
                    seen_iterations.add(sub_result.iteration)
                    drilldown_count = len(
                        [item for item in result_response.sub_tasks if item.iteration == sub_result.iteration]
                    )
                    _append_trace_step(
                        trace_steps,
                        phase="drilldown",
                        step_type="drilldown_iteration",
                        title=f"Drill-down iteration {sub_result.iteration}",
                        input_preview=question,
                        output_preview=f"{drilldown_count} follow-up tasks",
                        metadata={"iteration": sub_result.iteration, "task_count": drilldown_count},
                    )

                normalized_tool_calls = _normalize_tool_calls(sub_result.tool_calls)
                flattened_tool_calls.extend(sub_result.tool_calls)
                _append_trace_step(
                    trace_steps,
                    phase="drilldown" if sub_result.iteration > 0 else "execution",
                    step_type="sub_task_execution",
                    title=f"Step {sub_result.id}",
                    input_preview=sub_result.question,
                    output_preview=_preview(sub_result.answer, limit=420),
                    raw_text=sub_result.thought_process,
                    tool_calls=normalized_tool_calls,
                    token_usage=dict(sub_result.usage or {}),
                    metadata={
                        "iteration": sub_result.iteration,
                        "is_drilldown": sub_result.is_drilldown,
                        "source_count": len(sub_result.sources),
                        "context_count": len(sub_result.contexts),
                        "sources": list(sub_result.sources),
                    },
                )

            trace_summary = result_response.summary or "Agentic research completed"
            _append_trace_step(
                trace_steps,
                phase="synthesis",
                step_type="report_synthesis",
                title="Synthesize final report",
                input_preview=f"{len(result_response.sub_tasks)} sub-tasks",
                output_preview=_preview(result_response.summary, limit=420),
                raw_text=result_response.detailed_answer,
                token_usage={"total_tokens": total_tokens},
                metadata={
                    "source_count": len(result_response.all_sources),
                    "total_iterations": result_response.total_iterations,
                },
            )

            return RAGResult(
                answer=result_response.detailed_answer,
                source_doc_ids=result_response.all_sources,
                documents=aggregated_docs,
                usage={"total_tokens": total_tokens},
                thought_process=result_response.summary,
                tool_calls=flattened_tool_calls,
                agent_trace=_finalize_trace_payload(
                    question_id=question_id,
                    question=question,
                    run_number=run_number,
                    steps=trace_steps,
                    summary=trace_summary,
                    trace_status="completed",
                    execution_profile=self.execution_profile,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _append_trace_step(
                trace_steps,
                phase="synthesis" if trace_steps else "planning",
                step_type="agentic_failure",
                title="Agentic run failed",
                status="failed",
                input_preview=question,
                output_preview=_preview(str(exc)),
                raw_text=str(exc),
            )
            raise AgentTraceCaptureError(
                str(exc),
                agent_trace=_finalize_trace_payload(
                    question_id=question_id,
                    question=question,
                    run_number=run_number,
                    steps=trace_steps,
                    summary=_preview(str(exc), limit=420) or trace_summary,
                    trace_status="failed" if len(trace_steps) <= 1 else "partial",
                    execution_profile=self.execution_profile,
                ),
            ) from exc




