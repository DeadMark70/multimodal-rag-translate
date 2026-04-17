"""Shared execution core for Deep Research-style plan-and-execute workflows."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, List, Optional

from agents.evaluator import RAGEvaluator
from agents.planner import SubTask, TaskPlanner
from agents.synthesizer import SubTaskResult, synthesize_results
from core.llm_factory import get_llm
from data_base.RAG_QA_service import rag_answer_question
from data_base.schemas_deep_research import (
    AtomicFact,
    EditableSubTask,
    ExecutePlanRequest,
    ExecutePlanResponse,
    ResearchPlanResponse,
    SubTaskExecutionResult,
)
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

_FACT_STATE_PROMPT = """你是研究助理。請把下列子任務結果轉成 1-3 條可驗證的原子事實。
僅輸出 JSON 陣列，不要額外文字。

格式:
[
  {{"claim": "具體事實", "source_doc_ids": ["doc-1"]}}
]

規則:
1. claim 必須是單一、可被文獻支持的陳述。
2. source_doc_ids 只能使用已提供的來源 doc id；若無法判定可留空陣列。
3. 不要輸出推測語句。

問題:
{question}

已知來源:
{source_doc_ids}

回答內容:
{answer}
"""

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+")


class ResearchExecutionCore:
    """Shared task planning and execution primitives for research workflows."""

    def __init__(
        self,
        max_concurrent_tasks: int = 3,
        default_max_iterations: int = 2,
    ) -> None:
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_max_iterations = default_max_iterations
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)

    @staticmethod
    def _graph_execution_hints(
        *,
        stage_hint: str,
        task_type: str,
    ) -> dict[str, object]:
        """Build generic-mode graph routing hints for research execution."""
        return {
            "stage_hint": stage_hint,
            "task_type_hint": task_type,
            "prefer_global": stage_hint == "exploration" or task_type == "graph_analysis",
            "prefer_local": stage_hint == "verification" and task_type != "graph_analysis",
        }

    @staticmethod
    def _normalize_claim(claim: str) -> str:
        return " ".join(str(claim).strip().split())

    @staticmethod
    def _response_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return "" if content is None else str(content)

    @staticmethod
    def _parse_json_payload(raw_text: str) -> Any | None:
        text = (raw_text or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        code_match = _JSON_BLOCK_RE.search(text)
        if code_match:
            candidate = code_match.group(1).strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
        return None

    def _normalize_atomic_facts(
        self,
        payload: Any,
        *,
        fallback_source_ids: list[str],
    ) -> list[AtomicFact]:
        candidates: Any = payload
        if isinstance(payload, dict):
            for key in ("facts", "atomic_facts", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    candidates = value
                    break

        if not isinstance(candidates, list):
            return []

        facts: list[AtomicFact] = []
        seen_claims: set[str] = set()
        for item in candidates:
            if not isinstance(item, dict):
                continue
            claim = self._normalize_claim(str(item.get("claim") or ""))
            if not claim:
                continue
            claim_key = claim.lower()
            if claim_key in seen_claims:
                continue
            raw_source_ids = item.get("source_doc_ids")
            source_ids = [
                str(source).strip()
                for source in (raw_source_ids or [])
                if str(source).strip()
            ] if isinstance(raw_source_ids, list) else []
            if fallback_source_ids:
                source_ids = [source for source in source_ids if source in fallback_source_ids] or fallback_source_ids[:1]
            facts.append(AtomicFact(claim=claim, source_doc_ids=source_ids))
            seen_claims.add(claim_key)
            if len(facts) >= 3:
                break
        return facts

    def _fallback_atomic_facts(self, result: SubTaskExecutionResult) -> list[AtomicFact]:
        answer = (result.answer or "").strip()
        if not answer:
            return []
        sentences = [
            self._normalize_claim(chunk)
            for chunk in _SENTENCE_SPLIT_RE.split(answer)
            if chunk and chunk.strip()
        ]
        if not sentences:
            return []
        claim = max(sentences[:3], key=len)[:220]
        source_ids = [str(source) for source in result.sources if str(source).strip()]
        return [AtomicFact(claim=claim, source_doc_ids=source_ids[:2])]

    async def _extract_atomic_facts(
        self,
        result: SubTaskExecutionResult,
    ) -> list[AtomicFact]:
        if result.atomic_facts:
            return result.atomic_facts
        if not result.answer:
            return []

        source_ids = [str(source) for source in result.sources if str(source).strip()]
        try:
            llm = get_llm("planner")
            prompt = _FACT_STATE_PROMPT.format(
                question=result.question,
                source_doc_ids=", ".join(source_ids) if source_ids else "(none)",
                answer=result.answer[:2400],
            )
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            parsed = self._parse_json_payload(self._response_text(getattr(response, "content", "")))
            facts = self._normalize_atomic_facts(
                parsed,
                fallback_source_ids=source_ids,
            )
            if facts:
                return facts
        except Exception as exc:  # noqa: BLE001
            logger.debug("Atomic fact extraction fallback for task %s: %s", result.id, exc)

        return self._fallback_atomic_facts(result)

    def _merge_atomic_facts(
        self,
        current: list[AtomicFact],
        additions: list[AtomicFact],
    ) -> list[AtomicFact]:
        merged: dict[str, AtomicFact] = {}
        for fact in current:
            key = self._normalize_claim(fact.claim).lower()
            if not key:
                continue
            merged[key] = AtomicFact(
                claim=self._normalize_claim(fact.claim),
                source_doc_ids=list(dict.fromkeys(fact.source_doc_ids)),
            )
        for fact in additions:
            key = self._normalize_claim(fact.claim).lower()
            if not key:
                continue
            if key in merged:
                merged[key].source_doc_ids = list(
                    dict.fromkeys([*merged[key].source_doc_ids, *fact.source_doc_ids])
                )
            else:
                merged[key] = AtomicFact(
                    claim=self._normalize_claim(fact.claim),
                    source_doc_ids=list(dict.fromkeys(fact.source_doc_ids)),
                )
        return list(merged.values())

    async def _refresh_fact_state(
        self,
        results: list[SubTaskExecutionResult],
        seed: Optional[list[AtomicFact]] = None,
    ) -> list[AtomicFact]:
        fact_state = list(seed or [])
        for result in results:
            if not result.atomic_facts:
                result.atomic_facts = await self._extract_atomic_facts(result)
            fact_state = self._merge_atomic_facts(fact_state, result.atomic_facts)
        return fact_state

    async def generate_plan(
        self,
        question: str,
        user_id: str,
        doc_ids: Optional[List[str]] = None,
        enable_graph_planning: bool = False,
    ) -> ResearchPlanResponse:
        """Generate a user-editable research plan."""
        logger.info("Generating research plan for user %s: %s...", user_id, question[:50])

        planner = TaskPlanner(
            max_subtasks=5,
            enable_graph_planning=enable_graph_planning,
        )
        plan = await planner.plan(question)

        editable_tasks = [
            EditableSubTask(
                id=task.id,
                question=task.question,
                task_type=task.task_type,
                enabled=True,
            )
            for task in plan.sub_tasks
        ]

        logger.info("Generated plan with %s sub-tasks", len(editable_tasks))

        return ResearchPlanResponse(
            status="waiting_confirmation",
            original_question=question,
            sub_tasks=editable_tasks,
            estimated_complexity=plan.estimated_complexity,
            doc_ids=doc_ids,
        )

    async def run_execute_plan(
        self,
        request: ExecutePlanRequest,
        user_id: str,
    ) -> ExecutePlanResponse:
        """Run a confirmed research plan without persistence concerns."""
        logger.info(
            "Executing research plan for user %s: %s tasks, max_iter=%s",
            user_id,
            len(request.sub_tasks),
            request.max_iterations,
        )

        enabled_tasks = [task for task in request.sub_tasks if task.enabled]
        if not enabled_tasks:
            logger.warning("No enabled tasks in plan")
            return ExecutePlanResponse(
                question=request.original_question,
                summary="沒有啟用的子任務。",
                detailed_answer="請至少啟用一個子任務後重試。",
                sub_tasks=[],
                all_sources=[],
                confidence=0.0,
                total_iterations=0,
            )

        all_results: List[SubTaskExecutionResult] = []
        current_results = await self._execute_tasks(
            tasks=enabled_tasks,
            user_id=user_id,
            doc_ids=request.doc_ids,
            enable_reranking=request.enable_reranking,
            iteration=0,
            enable_deep_image_analysis=request.enable_deep_image_analysis,
        )
        all_results.extend(current_results)

        total_iterations = 0
        should_skip = self._should_skip_drilldown(all_results, current_iteration=0)
        if request.enable_drilldown and request.max_iterations > 0 and not should_skip:
            total_iterations = await self._drill_down_loop(
                original_question=request.original_question,
                current_results=all_results,
                user_id=user_id,
                doc_ids=request.doc_ids,
                enable_reranking=request.enable_reranking,
                max_iterations=request.max_iterations,
                enable_deep_image_analysis=request.enable_deep_image_analysis,
            )

        return await self._synthesize_execution_results(
            original_question=request.original_question,
            all_results=all_results,
            total_iterations=total_iterations,
        )

    async def _synthesize_execution_results(
        self,
        *,
        original_question: str,
        all_results: List[SubTaskExecutionResult],
        total_iterations: int,
    ) -> ExecutePlanResponse:
        synthesizer_results = [
            SubTaskResult(
                task_id=result.id,
                question=result.question,
                answer=result.answer,
                sources=result.sources,
                confidence=1.0 if result.answer else 0.0,
            )
            for result in all_results
        ]

        report = await synthesize_results(
            original_question=original_question,
            sub_results=synthesizer_results,
            enabled=len(synthesizer_results) > 1,
            use_academic_template=False,
        )

        all_sources = list(set(src for result in all_results for src in result.sources))
        fact_state = await self._refresh_fact_state(all_results)
        logger.info(
            "Research complete: %s tasks, %s drill-down iterations, %s sources",
            len(all_results),
            total_iterations,
            len(all_sources),
        )

        return ExecutePlanResponse(
            question=original_question,
            summary=report.summary,
            detailed_answer=report.detailed_answer,
            sub_tasks=all_results,
            all_sources=all_sources,
            confidence=report.confidence,
            total_iterations=total_iterations,
            fact_state=fact_state,
        )

    async def _execute_tasks(
        self,
        tasks: List[EditableSubTask],
        user_id: str,
        doc_ids: Optional[List[str]],
        enable_reranking: bool,
        iteration: int,
        enable_deep_image_analysis: bool = False,
    ) -> List[SubTaskExecutionResult]:
        async def execute_single(task: EditableSubTask) -> SubTaskExecutionResult:
            async with self._semaphore:
                try:
                    result = await rag_answer_question(
                        question=task.question,
                        user_id=user_id,
                        doc_ids=doc_ids,
                        enable_reranking=enable_reranking,
                        enable_crag=True,
                        enable_graph_rag=True,
                        graph_search_mode="generic",
                        graph_execution_hints=self._graph_execution_hints(
                            stage_hint="exploration",
                            task_type=task.task_type,
                        ),
                        enable_visual_verification=enable_deep_image_analysis,
                        return_docs=True,
                    )

                    if hasattr(result, "answer"):
                        answer = result.answer
                        sources = result.source_doc_ids
                        contexts = [document.page_content for document in result.documents]
                        usage = result.usage or {"total_tokens": 0}
                        thought_process = result.thought_process
                        tool_calls = result.tool_calls
                    else:
                        answer, sources = result
                        contexts = []
                        usage = {"total_tokens": 0}
                        thought_process = None
                        tool_calls = []

                    return SubTaskExecutionResult(
                        id=task.id,
                        question=task.question,
                        answer=answer,
                        sources=sources,
                        contexts=contexts,
                        is_drilldown=iteration > 0,
                        iteration=iteration,
                        usage=usage,
                        thought_process=thought_process,
                        tool_calls=tool_calls,
                    )
                except (RuntimeError, ValueError) as exc:
                    logger.warning("Task %s failed: %s", task.id, exc)
                    return SubTaskExecutionResult(
                        id=task.id,
                        question=task.question,
                        answer=f"無法回答此問題: {str(exc)[:100]}",
                        sources=[],
                        contexts=[],
                        is_drilldown=iteration > 0,
                        iteration=iteration,
                    )

        results = await asyncio.gather(*[execute_single(task) for task in tasks])
        return list(results)

    async def _drill_down_loop(
        self,
        original_question: str,
        current_results: List[SubTaskExecutionResult],
        user_id: str,
        doc_ids: Optional[List[str]],
        enable_reranking: bool,
        max_iterations: int,
        enable_deep_image_analysis: bool = False,
    ) -> int:
        planner = TaskPlanner(max_subtasks=3, enable_graph_planning=False)
        evaluator = RAGEvaluator()

        min_accuracy_score = 6.0
        max_retries_per_task = 2
        fact_state = await self._refresh_fact_state(current_results)

        for iteration in range(1, max_iterations + 1):
            logger.info("Drill-down iteration %s/%s", iteration, max_iterations)

            fact_state = await self._refresh_fact_state(current_results, fact_state)
            findings_summary = self._build_findings_summary(
                current_results,
                fact_state=fact_state,
            )
            coverage_gap_resolver = getattr(self, "_coverage_gaps", None)
            coverage_gaps = (
                coverage_gap_resolver(current_results)
                if callable(coverage_gap_resolver)
                else None
            )
            followup_tasks = await planner.create_followup_tasks(
                original_question=original_question,
                current_findings=findings_summary,
                existing_tasks=[
                    SubTask(id=result.id, question=result.question)
                    for result in current_results
                ],
                question_intent=getattr(self, "_active_question_intent", None),
                coverage_gaps=coverage_gaps,
            )

            if not followup_tasks:
                logger.info("No knowledge gaps found at iteration %s", iteration)
                return iteration - 1

            logger.info("Found %s follow-up tasks", len(followup_tasks))
            max_id = max(result.id for result in current_results)

            for index, task in enumerate(followup_tasks):
                task_id = max_id + index + 1
                current_question = task.question
                retry_count = 0

                while retry_count <= max_retries_per_task:
                    try:
                        use_graph = task.task_type == "graph_analysis"
                        result = await rag_answer_question(
                            question=current_question,
                            user_id=user_id,
                            doc_ids=doc_ids,
                            enable_reranking=enable_reranking,
                            enable_crag=True,
                            enable_graph_rag=use_graph,
                            graph_search_mode="generic" if use_graph else "auto",
                            graph_execution_hints=self._graph_execution_hints(
                                stage_hint="verification",
                                task_type=task.task_type,
                            ),
                            return_docs=True,
                            enable_visual_verification=enable_deep_image_analysis,
                        )

                        if hasattr(result, "answer"):
                            answer = result.answer
                            sources = result.source_doc_ids
                            documents = result.documents
                            contexts = [document.page_content for document in documents]
                            usage = result.usage or {"total_tokens": 0}
                            thought_process = result.thought_process
                            tool_calls = result.tool_calls
                        else:
                            answer, sources = result
                            documents = []
                            contexts = []
                            usage = {"total_tokens": 0}
                            thought_process = None
                            tool_calls = []
                    except (RuntimeError, ValueError) as exc:
                        logger.warning("Task %s failed: %s", task_id, exc)
                        answer = f"無法回答此問題: {str(exc)[:100]}"
                        sources = []
                        documents = []
                        contexts = []
                        usage = {"total_tokens": 0}
                        thought_process = None
                        tool_calls = []

                    if documents and retry_count < max_retries_per_task:
                        evaluation = await evaluator.evaluate_detailed(
                            question=current_question,
                            documents=documents,
                            answer=answer,
                        )

                        if evaluation.accuracy < min_accuracy_score:
                            logger.info(
                                "Task %s low accuracy (%.1f/10), reason: %s...",
                                task_id,
                                evaluation.accuracy,
                                evaluation.reason[:50],
                            )
                            retry_hint = evaluation.suggestion or evaluation.reason
                            refined_query = await planner.refine_query_from_evaluation(
                                original_question=current_question,
                                evaluation_reason=retry_hint,
                                failed_answer=answer,
                            )
                            if refined_query != current_question:
                                logger.info("Smart retry #%s with refined query", retry_count + 1)
                                current_question = refined_query
                                retry_count += 1
                                continue
                            logger.info("Could not refine query, accepting current answer")

                    sub_result = SubTaskExecutionResult(
                        id=task_id,
                        question=task.question,
                        answer=answer,
                        sources=sources,
                        contexts=contexts,
                        is_drilldown=True,
                        iteration=iteration,
                        usage=usage,
                        thought_process=thought_process,
                        tool_calls=tool_calls,
                    )
                    sub_result.atomic_facts = await self._extract_atomic_facts(sub_result)
                    current_results.append(sub_result)
                    fact_state = self._merge_atomic_facts(fact_state, sub_result.atomic_facts)
                    if retry_count > 0:
                        logger.info("Task %s accepted after %s retry(s)", task_id, retry_count)
                    break

        return max_iterations

    def _build_findings_summary(
        self,
        results: List[SubTaskExecutionResult],
        fact_state: Optional[List[AtomicFact]] = None,
    ) -> str:
        if fact_state:
            lines: list[str] = ["Structured Fact State:"]
            for index, fact in enumerate(fact_state, start=1):
                source_label = ", ".join(fact.source_doc_ids) if fact.source_doc_ids else "unknown"
                lines.append(f"- [{index}] {fact.claim} (sources: {source_label})")
            lines.append("")
            lines.append("Task Coverage Snapshot:")
            for result in results:
                lines.append(f"- Q{result.id}: {result.question}")
            return "\n".join(lines)

        lines: list[str] = []
        for result in results:
            answer_preview = result.answer[:300] + "..." if len(result.answer) > 300 else result.answer
            lines.append(f"【問題 {result.id}】{result.question}")
            lines.append(f"【回答】{answer_preview}")
            lines.append("")
        return "\n".join(lines)

    def _should_skip_drilldown(
        self,
        results: List[SubTaskExecutionResult],
        min_answer_length: int = 200,
        min_complete_ratio: float = 0.8,
        current_iteration: int = -1,
    ) -> bool:
        if current_iteration == 0:
            logger.info("Phase 6 Forced Drill-down: iteration 0 requires at least one drill-down")
            return False
        if not results:
            return False

        failure_markers = [
            "無法回答",
            "找不到",
            "沒有相關",
            "抱歉",
            "無法找到",
            "unable to answer",
            "not found",
            "no relevant",
            "sorry",
            "無法確定",
            "資料不足",
            "沒有足夠",
        ]

        complete_count = 0
        for result in results:
            answer_lower = result.answer.lower()
            has_failure = any(marker in answer_lower for marker in failure_markers)
            is_long_enough = len(result.answer) >= min_answer_length
            if not has_failure and is_long_enough:
                complete_count += 1

        complete_ratio = complete_count / len(results)
        has_quantitative_data = any(
            "dsc" in result.answer.lower() and any(char.isdigit() for char in result.answer)
            for result in results
        )
        if has_quantitative_data and complete_ratio > 0.5:
            logger.info(
                "Self-Stop Triggered: Found quantitative data (DSC) and incomplete ratio > 0.5"
            )
            return True

        should_skip = complete_ratio >= min_complete_ratio
        if should_skip:
            logger.info(
                "Smart termination: %s/%s answers complete (%.0f%%), skipping drill-down",
                complete_count,
                len(results),
                complete_ratio * 100,
            )
        return should_skip

    async def _execute_single_task(
        self,
        task: EditableSubTask,
        user_id: str,
        doc_ids: Optional[List[str]],
        enable_reranking: bool,
        iteration: int,
        enable_deep_image_analysis: bool = False,
    ) -> SubTaskExecutionResult:
        try:
            use_graph = task.task_type == "graph_analysis"
            result = await rag_answer_question(
                question=task.question,
                user_id=user_id,
                doc_ids=doc_ids,
                enable_reranking=enable_reranking,
                enable_crag=True,
                enable_graph_rag=use_graph,
                graph_search_mode="generic" if use_graph else "auto",
                graph_execution_hints=self._graph_execution_hints(
                    stage_hint="exploration" if iteration == 0 else "verification",
                    task_type=task.task_type,
                ),
                return_docs=True,
                enable_visual_verification=enable_deep_image_analysis,
            )

            if hasattr(result, "answer"):
                answer = result.answer
                sources = result.source_doc_ids
                contexts = [document.page_content for document in result.documents]
                usage = result.usage or {"total_tokens": 0}
                thought_process = result.thought_process
                tool_calls = result.tool_calls
            else:
                answer, sources = result
                contexts = []
                usage = {"total_tokens": 0}
                thought_process = None
                tool_calls = []

            return SubTaskExecutionResult(
                id=task.id,
                question=task.question,
                answer=answer,
                sources=sources,
                contexts=contexts,
                is_drilldown=iteration > 0,
                iteration=iteration,
                usage=usage,
                thought_process=thought_process,
                tool_calls=tool_calls,
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning("Task %s failed: %s", task.id, exc)
            return SubTaskExecutionResult(
                id=task.id,
                question=task.question,
                answer=f"無法回答此問題: {str(exc)[:100]}",
                sources=[],
                contexts=[],
                is_drilldown=iteration > 0,
                iteration=iteration,
            )

