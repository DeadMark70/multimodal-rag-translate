"""Reusable benchmark execution helpers for evaluation campaigns."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from langchain_core.documents import Document

from core.llm_factory import llm_runtime_override
from data_base.RAG_QA_service import RAGResult, rag_answer_question
from data_base.deep_research_service import DeepResearchService
from data_base.schemas_deep_research import ExecutePlanRequest
from evaluation.retry import run_with_retry
from evaluation.schemas import TestCase
from evaluation.trace_schemas import AgentTraceToolCall

RAG_MODES: dict[str, dict[str, Any]] = {
    "naive": {
        "enable_reranking": False,
        "enable_hyde": False,
        "enable_multi_query": False,
        "enable_graph_rag": False,
        "enable_visual_verification": False,
    },
    "advanced": {
        "enable_reranking": True,
        "enable_hyde": True,
        "enable_multi_query": True,
        "enable_graph_rag": False,
        "enable_visual_verification": False,
    },
    "graph": {
        "enable_reranking": True,
        "enable_hyde": True,
        "enable_multi_query": True,
        "enable_graph_rag": True,
        "graph_search_mode": "hybrid",
        "enable_visual_verification": False,
    },
    "agentic": {
        "enable_reranking": True,
        "enable_hyde": True,
        "enable_multi_query": True,
        "enable_graph_rag": True,
        "graph_search_mode": "hybrid",
        "enable_visual_verification": True,
    },
}


@dataclass
class BenchmarkExecutionResult:
    """Normalized result payload consumed by campaign persistence."""

    question_id: str
    question: str
    ground_truth: str
    mode: str
    answer: str
    contexts: list[str]
    source_doc_ids: list[str]
    expected_sources: list[str]
    latency_ms: float
    token_usage: dict[str, int]
    category: Optional[str]
    difficulty: Optional[str]
    error_message: Optional[str] = None
    agent_trace: Optional[dict[str, Any]] = None


class AgentTraceCaptureError(RuntimeError):
    """Exception wrapper that carries a partial agent trace."""

    def __init__(self, message: str, agent_trace: dict[str, Any]) -> None:
        super().__init__(message)
        self.agent_trace = agent_trace


def _runtime_overrides(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": model_config.get("model_name"),
        "temperature": model_config.get("temperature"),
        "top_p": model_config.get("top_p"),
        "top_k": model_config.get("top_k"),
        "max_output_tokens": model_config.get("max_output_tokens"),
    }


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
    tool_calls: Optional[list[dict[str, Any]]] = None,
    token_usage: Optional[dict[str, int]] = None,
    metadata: Optional[dict[str, Any]] = None,
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
) -> dict[str, Any]:
    tool_call_count = sum(len(step.get("tool_calls", [])) for step in steps)
    total_tokens = sum(int(step.get("token_usage", {}).get("total_tokens", 0) or 0) for step in steps)
    return {
        "trace_id": str(uuid4()),
        "question_id": question_id,
        "question": question,
        "mode": "agentic",
        "run_number": run_number,
        "trace_status": trace_status,
        "summary": summary,
        "step_count": len(steps),
        "tool_call_count": tool_call_count,
        "total_tokens": total_tokens,
        "created_at": _utc_now_iso(),
        "steps": steps,
    }


async def run_campaign_case(
    *,
    test_case: TestCase,
    user_id: str,
    mode: str,
    model_config: dict[str, Any],
    run_number: int = 1,
) -> BenchmarkExecutionResult:
    """Execute one test case under one RAG mode."""
    if mode not in RAG_MODES:
        raise ValueError(f"Unsupported RAG mode: {mode}")

    with llm_runtime_override(**_runtime_overrides(model_config)):
        start_time = time.perf_counter()
        if mode == "agentic":
            result = await _run_agentic_case(
                question_id=test_case.id,
                question=test_case.question,
                user_id=user_id,
                run_number=run_number,
            )
        else:
            rag_result = await run_with_retry(
                rag_answer_question,
                question=test_case.question,
                user_id=user_id,
                return_docs=True,
                **RAG_MODES[mode],
            )
            assert isinstance(rag_result, RAGResult)
            result = rag_result
        latency_ms = (time.perf_counter() - start_time) * 1000

    contexts = _extract_contexts(result.documents)
    return BenchmarkExecutionResult(
        question_id=test_case.id,
        question=test_case.question,
        ground_truth=test_case.ground_truth,
        mode=mode,
        answer=result.answer,
        contexts=contexts,
        source_doc_ids=list(result.source_doc_ids),
        expected_sources=list(test_case.source_docs),
        latency_ms=latency_ms,
        token_usage=dict(result.usage or {}),
        category=test_case.category,
        difficulty=test_case.difficulty,
        agent_trace=result.agent_trace,
    )


async def _run_agentic_case(
    *,
    question_id: str,
    question: str,
    user_id: str,
    run_number: int,
) -> RAGResult:
    service = DeepResearchService(max_concurrent_tasks=3)
    trace_steps: list[dict[str, Any]] = []
    trace_summary = "Agentic trace unavailable"
    try:
        plan_response = await run_with_retry(
            service.generate_plan,
            question=question,
            user_id=user_id,
            doc_ids=None,
            enable_graph_planning=True,
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
            max_iterations=2,
            enable_deep_image_analysis=True,
        )
        result_response = await run_with_retry(
            service.execute_plan,
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
            ),
        ) from exc


def _extract_contexts(documents: list[Document]) -> list[str]:
    contexts: list[str] = []
    for document in documents:
        if hasattr(document, "page_content"):
            contexts.append(document.page_content[:500])
    return contexts
