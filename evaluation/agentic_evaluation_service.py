"""Evaluation-only Agentic RAG service forked from user-facing Deep Research wrappers."""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from agents.planner import (
    SemanticIntentDecision,
    SubTask,
    QuestionIntent,
    TaskPlanner,
    classify_question_intent,
    classify_question_intent_semantic,
    required_coverage_for_intent,
)
from agents.synthesizer import SubTaskResult, synthesize_results
from core.providers import get_llm
from data_base.indexing_service import DEFAULT_PRODUCTION_INDEXING_PROFILE
from data_base.RAG_QA_service import RAGResult, rag_answer_question
from data_base.research_execution_core import ResearchExecutionCore
from data_base.schemas_deep_research import (
    AtomicFact,
    EditableSubTask,
    ExecutePlanRequest,
    ExecutePlanResponse,
    ResearchPlanResponse,
    SubTaskExecutionResult,
)
from evaluation.retry import run_with_retry
from evaluation.trace_schemas import AgentTraceToolCall

StrategyTier = Literal[
    "tier_1_detail_lookup",
    "tier_2_structured_compare",
    "tier_3_multi_hop_analysis",
]
RouteProfile = Literal[
    "hybrid_exact",
    "hybrid_compare",
    "graph_global",
    "visual_verify",
    "generic_graph",
]
MicroRoute = Literal[
    "direct_point_access",
    "broad_context_rag",
    "visual_evidence_path",
]
SemanticRouterMode = Literal["off", "shadow", "active"]

AGENTIC_EVAL_PROFILE = f"agentic_eval_v7_semantic_router_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
LEGACY_SHARED_PROFILE = "legacy_shared"
AGENTIC_INITIAL_SUBTASKS = 3
AGENTIC_FIGURE_FLOW_INITIAL_SUBTASKS = 2
AGENTIC_IMAGE_ANALYSIS_ENABLED = True
_SEMANTIC_ROUTER_MODE_RAW = str(os.getenv("AGENTIC_SEMANTIC_ROUTER_MODE", "active") or "active").strip().lower()
AGENTIC_SEMANTIC_ROUTER_MODE: SemanticRouterMode = (
    _SEMANTIC_ROUTER_MODE_RAW if _SEMANTIC_ROUTER_MODE_RAW in {"off", "shadow", "active"} else "active"
)
_CLAIM_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(\.\d+)?\s*(%|x|m|k|g|ms|s|fps|gb|mb)?\b", re.IGNORECASE)
_BENCHMARK_NUMERIC_KEYWORDS = (
    "dice",
    "score",
    "metric",
    "flops",
    "params",
    "param",
    "accuracy",
    "auc",
    "f1",
    "iou",
    "miou",
    "latency",
    "throughput",
    "fps",
    "指標",
    "數值",
    "參數",
    "效能",
)
_BENCHMARK_NUMERIC_EXCLUSION_PHRASES = (
    "dice supervision",
    "pseudo-label",
    "cross-entropy",
    "監督機制",
    "損失函數",
    "loss function",
)
_BENCHMARK_GRAPH_ROUTE_KEYWORDS = (
    "cross-document",
    "across",
    "relation",
    "relationship",
    "trend",
    "arbitration",
    "裁決",
    "跨文件",
    "跨文獻",
    "關係",
    "關聯",
    "趨勢",
)
_FIGURE_FLOW_ANCHOR_BLOCKLIST = (
    "overall architecture",
    "general architecture",
    "high-level architecture",
    "總體架構",
    "整體架構",
    "overview",
)
_DIRECT_POINT_KEYWORDS = (
    "table",
    "tab.",
    "figure",
    "fig.",
    "value",
    "exact",
    "number",
    "數值",
    "數字",
    "精確",
)
_VISUAL_PATH_KEYWORDS = (
    "figure",
    "fig.",
    "image",
    "diagram",
    "flow",
    "pipeline",
    "架構圖",
    "流程",
    "圖表",
)
_DUPLICATE_FOLLOWUP_PROMPT = """You are checking whether a follow-up research task is redundant.

Follow-up task:
{task}

Known facts:
{facts}

Return JSON only:
{{"duplicate": true, "reason": "short reason"}}
"""


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


def _initial_subtask_limit(question_intent: QuestionIntent) -> int:
    if question_intent == "figure_flow":
        return AGENTIC_FIGURE_FLOW_INITIAL_SUBTASKS
    return AGENTIC_INITIAL_SUBTASKS


def _strategy_config_from_complexity(complexity_score: int) -> tuple[StrategyTier, int, int]:
    normalized = max(1, min(5, int(complexity_score)))
    if normalized == 1:
        return "tier_1_detail_lookup", 1, 0
    if normalized == 2:
        return "tier_1_detail_lookup", 1, 1
    if normalized == 3:
        return "tier_2_structured_compare", 2, 1
    if normalized == 4:
        return "tier_3_multi_hop_analysis", 3, 1
    return "tier_3_multi_hop_analysis", 4, 2


def _strategy_tier_for_intent(question_intent: QuestionIntent) -> StrategyTier:
    if question_intent == "benchmark_data":
        return "tier_3_multi_hop_analysis"
    if question_intent in {"comparison_disambiguation", "figure_flow"}:
        return "tier_2_structured_compare"
    return "tier_1_detail_lookup"


def _subtask_limit_for_strategy(
    *,
    strategy_tier: StrategyTier,
    question_intent: QuestionIntent,
) -> int:
    if strategy_tier == "tier_1_detail_lookup":
        return 1
    if strategy_tier == "tier_2_structured_compare":
        return AGENTIC_FIGURE_FLOW_INITIAL_SUBTASKS if question_intent == "figure_flow" else 2
    return AGENTIC_INITIAL_SUBTASKS


def _drilldown_iterations_for_strategy(
    *,
    strategy_tier: StrategyTier,
    question_intent: QuestionIntent,
) -> int:
    if question_intent == "figure_flow":
        return 0
    if strategy_tier == "tier_1_detail_lookup":
        return 0
    return 1


def _followup_cap_for_strategy(strategy_tier: StrategyTier) -> int:
    if strategy_tier == "tier_1_detail_lookup":
        return 2
    if strategy_tier == "tier_2_structured_compare":
        return 1
    return 2


def _semantic_router_mode() -> SemanticRouterMode:
    return AGENTIC_SEMANTIC_ROUTER_MODE


def _is_numeric_benchmark_subtask(*, task_type: str, task_question: str) -> bool:
    question_lower = task_question.lower()
    has_numeric_keyword = any(keyword in question_lower for keyword in _BENCHMARK_NUMERIC_KEYWORDS)
    has_numeric_token = bool(_NUMERIC_TOKEN_RE.search(question_lower))
    has_methodology_exclusion = any(
        phrase in question_lower for phrase in _BENCHMARK_NUMERIC_EXCLUSION_PHRASES
    )

    # Require both metric keyword and numeric context. Terms like "3D" or
    # "Dice supervision" alone should not trigger numeric benchmark routing.
    if not (has_numeric_keyword and has_numeric_token):
        return False
    if has_methodology_exclusion:
        return False
    return True


def _needs_graph_route_for_benchmark(*, task_type: str, task_question: str) -> bool:
    if task_type == "graph_analysis":
        return True
    question_lower = task_question.lower()
    return any(keyword in question_lower for keyword in _BENCHMARK_GRAPH_ROUTE_KEYWORDS)


def _micro_route_for_task(
    *,
    question_intent: QuestionIntent,
    task_type: str,
    task_question: str,
) -> MicroRoute:
    lowered = task_question.lower()
    if question_intent == "figure_flow" or any(token in lowered for token in _VISUAL_PATH_KEYWORDS):
        return "visual_evidence_path"
    if any(token in lowered for token in _DIRECT_POINT_KEYWORDS) or _is_numeric_benchmark_subtask(
        task_type=task_type,
        task_question=task_question,
    ):
        return "direct_point_access"
    return "broad_context_rag"


def _retrieval_policy_for_micro_route(micro_route: MicroRoute) -> dict[str, int]:
    if micro_route == "direct_point_access":
        return {"retrieval_k": 6, "target_k": 4}
    if micro_route == "visual_evidence_path":
        return {"target_k": 8}
    return {"target_k": 8}


def _route_profile_for_task(
    *,
    strategy_tier: StrategyTier,
    question_intent: QuestionIntent,
    task_type: str,
    task_question: str,
    iteration: int,
    micro_route: Optional[MicroRoute] = None,
) -> RouteProfile:
    if micro_route == "visual_evidence_path":
        return "visual_verify"
    if micro_route == "direct_point_access":
        return "hybrid_exact"
    if micro_route == "broad_context_rag":
        if _needs_graph_route_for_benchmark(task_type=task_type, task_question=task_question):
            return "generic_graph"
        if task_type == "graph_analysis":
            return "graph_global"
        if strategy_tier == "tier_3_multi_hop_analysis":
            return "generic_graph"
        return "hybrid_compare"

    if question_intent == "benchmark_data":
        is_numeric = _is_numeric_benchmark_subtask(
            task_type=task_type,
            task_question=task_question,
        )
        if is_numeric and _needs_graph_route_for_benchmark(
            task_type=task_type,
            task_question=task_question,
        ):
            return "generic_graph"
        return "hybrid_compare"
    if question_intent == "figure_flow":
        return "visual_verify"
    if task_type == "graph_analysis":
        return "graph_global"
    if strategy_tier == "tier_1_detail_lookup":
        return "hybrid_exact"
    if strategy_tier == "tier_2_structured_compare":
        return "hybrid_compare"
    return "generic_graph"


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _claim_candidates(answer: str, limit: int = 6) -> list[str]:
    if not answer:
        return []
    chunks = [part.strip() for part in _CLAIM_SPLIT_RE.split(answer) if part.strip()]
    if len(chunks) <= limit:
        return chunks
    return chunks[:limit]


def _support_level(overlap: int) -> str:
    if overlap >= 8:
        return "strong"
    if overlap >= 3:
        return "medium"
    return "weak"


def _coverage_keywords() -> dict[str, tuple[str, ...]]:
    return {
        "direct_difference": (
            "difference",
            "compare",
            "versus",
            "vs",
            "差異",
            "不同",
            "區別",
        ),
        "effective_strategy_or_evidence": (
            "lora",
            "adapter",
            "fine-tun",
            "strategy",
            "effective",
            "evidence",
            "decoder",
            "domain gap",
            "微調",
            "策略",
            "證據",
            "解碼器",
        ),
        "confusion_or_limitation": (
            "do not",
            "not confuse",
            "distinguish",
            "limitation",
            "caveat",
            "domain gap",
            "混淆",
            "限制",
            "不要",
            "不是",
            "區分",
        ),
        "ordered_flow": (
            "flow",
            "order",
            "sequence",
            "資料流",
            "順序",
            "concat",
            "gn",
            "gelu",
            "residual",
        ),
        "component_or_branch": (
            "branch",
            "parallel",
            "component",
            "block",
            "layer",
            "分支",
            "並行",
            "組件",
            "卷積",
        ),
        "comparative_metric": (
            "dice",
            "score",
            "metric",
            "performance",
            "improvement",
            "數據",
            "數值",
            "指標",
            "效能",
        ),
        "baseline_or_setting": (
            "baseline",
            "benchmark",
            "dataset",
            "setting",
            "compared",
            "基準",
            "資料集",
            "設定",
        ),
        "explicit_list": (
            "list",
            "include",
            "which",
            "列出",
            "包含",
            "哪些",
            "哪兩個",
        ),
        "short_characteristics": (
            "feature",
            "characteristic",
            "brief",
            "特點",
            "特徵",
            "簡述",
        ),
    }


def _is_figure_flow_auxiliary_task(*, original_question: str, candidate_question: str) -> bool:
    candidate_lower = candidate_question.strip().lower()
    original_lower = original_question.strip().lower()
    if not candidate_lower or candidate_lower == original_lower:
        return False
    if any(blocked in candidate_lower for blocked in _FIGURE_FLOW_ANCHOR_BLOCKLIST):
        return False

    candidate_tokens = _tokenize(candidate_lower)
    original_tokens = _tokenize(original_lower)
    overlap = len(candidate_tokens & original_tokens)
    has_flow_signal = any(
        keyword in candidate_lower
        for keyword in (
            "flow",
            "order",
            "sequence",
            "branch",
            "step",
            "flip",
            "accumul",
            "siam",
            "流程",
            "順序",
            "分支",
            "步驟",
            "翻轉",
            "累加",
            "機制",
        )
    )
    return has_flow_signal and overlap >= 2


def _finalize_trace_payload(
    *,
    question_id: str,
    question: str,
    run_number: int,
    steps: list[dict[str, Any]],
    summary: str,
    trace_status: str,
    execution_profile: str,
    question_intent: Optional[QuestionIntent] = None,
    strategy_tier: Optional[StrategyTier] = None,
    route_profile: Optional[RouteProfile] = None,
    required_coverage: Optional[list[str]] = None,
    coverage_gaps: Optional[list[str]] = None,
    subtask_coverage_status: Optional[dict[str, bool]] = None,
    claims: Optional[list[dict[str, Any]]] = None,
    supported_claim_count: int = 0,
    unsupported_claim_count: int = 0,
    visual_verification_attempted: bool = False,
    visual_tool_call_count: int = 0,
    visual_force_fallback_used: bool = False,
    classifier_decision: Optional[dict[str, Any]] = None,
    complexity_score: Optional[int] = None,
    tier_shift: Optional[str] = None,
    pruned_followups: int = 0,
    semantic_gate_score: Optional[float] = None,
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
        "question_intent": question_intent,
        "strategy_tier": strategy_tier,
        "route_profile": route_profile,
        "required_coverage": required_coverage or [],
        "coverage_gaps": coverage_gaps or [],
        "subtask_coverage_status": subtask_coverage_status or {},
        "claims": claims or [],
        "supported_claim_count": supported_claim_count,
        "unsupported_claim_count": unsupported_claim_count,
        "visual_verification_attempted": visual_verification_attempted,
        "visual_tool_call_count": visual_tool_call_count,
        "visual_force_fallback_used": visual_force_fallback_used,
        "classifier_decision": classifier_decision or {},
        "complexity_score": complexity_score,
        "tier_shift": tier_shift,
        "pruned_followups": pruned_followups,
        "semantic_gate_score": semantic_gate_score,
        "steps": steps,
    }


class AgenticEvaluationService(ResearchExecutionCore):
    """Evaluation-only agentic execution wrapper with stable profiling metadata."""

    execution_profile = AGENTIC_EVAL_PROFILE

    def __init__(
        self,
        max_concurrent_tasks: int = 3,
        default_max_iterations: int = 2,
    ) -> None:
        super().__init__(
            max_concurrent_tasks=max_concurrent_tasks,
            default_max_iterations=default_max_iterations,
        )
        self._active_question_intent: Optional[QuestionIntent] = None
        self._active_strategy_tier: Optional[StrategyTier] = None
        self._required_coverage: list[str] = []
        self._semantic_router_mode: SemanticRouterMode = _semantic_router_mode()
        self._active_subtask_limit: Optional[int] = None
        self._active_max_iterations: Optional[int] = None
        self._classifier_decision: dict[str, Any] = {}
        self._tier_shift: str = "keep"
        self._pruned_followups_total: int = 0
        self._last_semantic_gate_score: Optional[float] = None

    def _coverage_status(
        self,
        results: list[SubTaskExecutionResult],
    ) -> dict[str, bool]:
        if not self._required_coverage:
            return {}

        combined = "\n".join(
            f"{result.question}\n{result.answer}" for result in results
        ).lower()
        keyword_map = _coverage_keywords()
        status: dict[str, bool] = {}
        for coverage_key in self._required_coverage:
            keywords = keyword_map.get(coverage_key, ())
            status[coverage_key] = any(keyword in combined for keyword in keywords)
        return status

    def _coverage_gaps(
        self,
        results: list[SubTaskExecutionResult],
    ) -> list[str]:
        status = self._coverage_status(results)
        return [key for key, covered in status.items() if not covered]

    def _tier_rank(self, tier: StrategyTier) -> int:
        if tier == "tier_1_detail_lookup":
            return 1
        if tier == "tier_2_structured_compare":
            return 2
        return 3

    def _tier_from_rank(self, rank: int) -> StrategyTier:
        if rank <= 1:
            return "tier_1_detail_lookup"
        if rank == 2:
            return "tier_2_structured_compare"
        return "tier_3_multi_hop_analysis"

    def _semantic_overlap_score(self, text: str, fact_state: list[AtomicFact]) -> float:
        question_tokens = _tokenize(text)
        if not question_tokens or not fact_state:
            return 0.0
        best = 0.0
        for fact in fact_state:
            fact_tokens = _tokenize(fact.claim)
            if not fact_tokens:
                continue
            overlap = len(question_tokens & fact_tokens) / max(1, len(question_tokens | fact_tokens))
            if overlap > best:
                best = overlap
        return best

    async def _is_duplicate_followup_via_llm(
        self,
        *,
        question: str,
        fact_state: list[AtomicFact],
    ) -> bool:
        if not fact_state:
            return False
        facts_text = "\n".join(f"- {fact.claim}" for fact in fact_state[:6])
        prompt = _DUPLICATE_FOLLOWUP_PROMPT.format(task=question, facts=facts_text)
        try:
            llm = get_llm("planner")
            response = await asyncio.wait_for(
                llm.ainvoke([HumanMessage(content=prompt)]),
                timeout=0.35,
            )
            content = str(getattr(response, "content", "")).strip()
            lowered = content.lower()
            if '"duplicate": true' in lowered:
                return True
            if "```" in content:
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
                if match and '"duplicate"' in match.group(1).lower():
                    return '"duplicate": true' in match.group(1).lower()
            return False
        except Exception:
            return False

    def _adjust_strategy_after_exploration(
        self,
        *,
        gate_meta: dict[str, Any],
    ) -> str:
        if not self._active_strategy_tier:
            self._tier_shift = "keep"
            return "keep"
        current_rank = self._tier_rank(self._active_strategy_tier)
        coverage_gaps = list(gate_meta.get("coverage_gaps") or [])
        support_ratio = float(gate_meta.get("claim_support_ratio", 0.0) or 0.0)
        semantic_score = float(gate_meta.get("semantic_gate_score", 0.0) or 0.0)

        if not coverage_gaps and support_ratio >= 0.65 and semantic_score >= 0.70 and current_rank > 1:
            self._active_strategy_tier = self._tier_from_rank(current_rank - 1)
            self._active_max_iterations = _drilldown_iterations_for_strategy(
                strategy_tier=self._active_strategy_tier,
                question_intent=self._active_question_intent or "general_research",
            )
            self._tier_shift = "downshift"
            return "downshift"
        if coverage_gaps and (support_ratio < 0.35 or semantic_score < 0.45) and current_rank < 3:
            self._active_strategy_tier = self._tier_from_rank(current_rank + 1)
            self._active_max_iterations = _drilldown_iterations_for_strategy(
                strategy_tier=self._active_strategy_tier,
                question_intent=self._active_question_intent or "general_research",
            )
            self._tier_shift = "upshift"
            return "upshift"
        self._tier_shift = "keep"
        return "keep"

    def _route_kwargs(
        self,
        *,
        route_profile: RouteProfile,
        micro_route: MicroRoute,
        enable_reranking: bool,
        enable_visual_verification: bool,
        task_type: str,
        stage_hint: str,
    ) -> dict[str, Any]:
        retrieval_policy = _retrieval_policy_for_micro_route(micro_route)
        kwargs: dict[str, Any] = {
            "enable_reranking": enable_reranking,
            "enable_crag": True,
            "plain_mode": False,
            "return_docs": True,
            "enable_visual_verification": enable_visual_verification,
            "mode_hints": {
                "question_intent": self._active_question_intent,
                "strategy_tier": self._active_strategy_tier,
                "task_type": task_type,
                "stage_hint": stage_hint,
                "route_profile": route_profile,
                "micro_route": micro_route,
                "retrieval_policy": retrieval_policy,
            },
        }
        if route_profile == "hybrid_exact":
            kwargs.update(
                {
                    "enable_reranking": False,
                    "enable_hyde": False,
                    "enable_multi_query": False,
                    "enable_graph_rag": False,
                }
            )
        elif route_profile == "hybrid_compare":
            kwargs.update(
                {
                    "enable_hyde": True,
                    "enable_multi_query": True,
                    "enable_graph_rag": False,
                }
            )
        elif route_profile == "graph_global":
            kwargs.update(
                {
                    "enable_hyde": False,
                    "enable_multi_query": False,
                    "enable_graph_rag": True,
                    "graph_search_mode": "generic",
                    "graph_execution_hints": self._graph_execution_hints(
                        stage_hint=stage_hint,
                        task_type=task_type,
                    ),
                }
            )
        elif route_profile == "visual_verify":
            kwargs.update(
                {
                    "enable_hyde": False,
                    "enable_multi_query": True,
                    "enable_graph_rag": False,
                    "enable_visual_verification": True,
                }
            )
        else:  # generic_graph
            kwargs.update(
                {
                    "enable_hyde": True,
                    "enable_multi_query": True,
                    "enable_graph_rag": True,
                    "graph_search_mode": "generic",
                    "graph_execution_hints": self._graph_execution_hints(
                        stage_hint=stage_hint,
                        task_type=task_type,
                    ),
                }
            )
        return kwargs

    def _build_evidence_units(
        self,
        *,
        result_id: int,
        question: str,
        iteration: int,
        route_profile: RouteProfile,
        contexts: list[str],
        source_doc_ids: list[str],
    ) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        primary_source = source_doc_ids[0] if source_doc_ids else None
        for index, context in enumerate(contexts):
            units.append(
                {
                    "evidence_id": f"t{result_id}-c{index + 1}",
                    "task_id": result_id,
                    "question": question,
                    "iteration": iteration,
                    "route_profile": route_profile,
                    "source_doc_id": primary_source,
                    "modality": "text",
                    "text": context,
                    "retrieval_score": None,
                    "provenance": {
                        "task_id": result_id,
                        "context_index": index,
                        "iteration": iteration,
                    },
                }
            )
        return units

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

        async def execute_single(task: EditableSubTask) -> SubTaskExecutionResult:
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
                # Shadow/off mode logs prediction but keeps retrieval behavior stable.
                applied_micro_route = "broad_context_rag"
            kwargs = self._route_kwargs(
                route_profile=route_profile,
                micro_route=applied_micro_route,
                enable_reranking=enable_reranking,
                enable_visual_verification=enable_deep_image_analysis,
                task_type=task.task_type,
                stage_hint=stage_hint,
            )
            try:
                result = await rag_answer_question(
                    question=task.question,
                    user_id=user_id,
                    doc_ids=doc_ids,
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
                return SubTaskExecutionResult(
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
                return SubTaskExecutionResult(
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

        results = await asyncio.gather(*[execute_single(task) for task in tasks])
        return list(results)

    def _evidence_index(
        self,
        sub_tasks: list[SubTaskExecutionResult],
    ) -> list[dict[str, Any]]:
        indexed: list[dict[str, Any]] = []
        for result in sub_tasks:
            if result.evidence_units:
                indexed.extend(result.evidence_units)
                continue
            indexed.extend(
                self._build_evidence_units(
                    result_id=result.id,
                    question=result.question,
                    iteration=result.iteration,
                    route_profile=(result.route_profile or "hybrid_exact"),
                    contexts=result.contexts,
                    source_doc_ids=result.sources,
                )
            )
        return indexed

    def _claim_evidence_map(
        self,
        *,
        answer: str,
        evidence_units: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, int]:
        claims: list[dict[str, Any]] = []
        if not answer:
            return claims, 0, 0

        candidates = _claim_candidates(answer)
        if not candidates:
            return claims, 0, 0

        evidence_token_index: list[tuple[dict[str, Any], set[str]]] = []
        for unit in evidence_units:
            text = str(unit.get("text") or "")
            evidence_token_index.append((unit, _tokenize(text)))

        supported_count = 0
        unsupported_count = 0
        for idx, claim_text in enumerate(candidates, start=1):
            claim_tokens = _tokenize(claim_text)
            ranked: list[tuple[int, dict[str, Any]]] = []
            for unit, tokens in evidence_token_index:
                overlap = len(claim_tokens & tokens)
                if overlap > 0:
                    ranked.append((overlap, unit))
            ranked.sort(key=lambda item: item[0], reverse=True)
            top_units = ranked[:2]
            evidence_ids = [str(unit.get("evidence_id")) for _, unit in top_units]
            if evidence_ids:
                supported_count += 1
            else:
                unsupported_count += 1
            claims.append(
                {
                    "claim_id": f"C{idx}",
                    "text": claim_text,
                    "evidence_ids": evidence_ids,
                    "support_level": _support_level(top_units[0][0]) if top_units else "none",
                    "supported": bool(evidence_ids),
                }
            )
        return claims, supported_count, unsupported_count

    async def generate_agentic_plan(
        self,
        *,
        question: str,
        user_id: str,
        question_intent: Optional[QuestionIntent] = None,
        strategy_tier: Optional[StrategyTier] = None,
    ) -> ResearchPlanResponse:
        """Generate the dedicated evaluation baseline plan for agentic RAG."""
        effective_intent = question_intent or classify_question_intent(question)
        effective_strategy_tier = strategy_tier or _strategy_tier_for_intent(effective_intent)
        subtask_limit = self._active_subtask_limit or _subtask_limit_for_strategy(
            strategy_tier=effective_strategy_tier,
            question_intent=effective_intent,
        )
        if effective_strategy_tier == "tier_1_detail_lookup":
            editable_tasks = [
                EditableSubTask(
                    id=1,
                    question=question,
                    task_type="rag",
                    enabled=True,
                )
            ]
            return ResearchPlanResponse(
                status="waiting_confirmation",
                original_question=question,
                sub_tasks=editable_tasks,
                estimated_complexity="simple",
                doc_ids=None,
            )
        planner = TaskPlanner(
            max_subtasks=subtask_limit,
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
            for task in plan.sub_tasks[:subtask_limit]
        ]
        if effective_intent == "figure_flow":
            anchored_tasks: list[EditableSubTask] = [
                EditableSubTask(
                    id=1,
                    question=question,
                    task_type="rag",
                    enabled=True,
                )
            ]
            auxiliary_candidates = [
                task
                for task in editable_tasks
                if _is_figure_flow_auxiliary_task(
                    original_question=question,
                    candidate_question=task.question,
                )
            ]
            if auxiliary_candidates:
                aux = auxiliary_candidates[0]
                anchored_tasks.append(
                    EditableSubTask(
                        id=2,
                        question=aux.question,
                        task_type=aux.task_type,
                        enabled=True,
                    )
                )
            editable_tasks = anchored_tasks
        return ResearchPlanResponse(
            status="waiting_confirmation",
            original_question=question,
            sub_tasks=editable_tasks,
            estimated_complexity=plan.estimated_complexity,
            doc_ids=None,
        )

    async def _synthesize_execution_results(
        self,
        *,
        original_question: str,
        all_results: list[SubTaskExecutionResult],
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
            enabled=True,
            use_academic_template=False,
            question_intent=self._active_question_intent,
            force_llm_for_single=True,
            enable_conflict_arbitration=True,
        )

        evidence_units = self._evidence_index(all_results)
        claims, supported_claim_count, unsupported_claim_count = self._claim_evidence_map(
            answer=report.detailed_answer,
            evidence_units=evidence_units,
        )
        if supported_claim_count > 0 and unsupported_claim_count > supported_claim_count:
            supported_claims = [claim["text"] for claim in claims if claim.get("supported")]
            report.detailed_answer = "\n".join(f"- {claim}" for claim in supported_claims)

        all_sources = list(set(src for result in all_results for src in result.sources))
        fact_state = await self._refresh_fact_state(all_results)
        return ExecutePlanResponse(
            question=original_question,
            summary=report.summary,
            detailed_answer=report.detailed_answer,
            sub_tasks=all_results,
            all_sources=all_sources,
            confidence=report.confidence,
            total_iterations=total_iterations,
            claims=claims,
            critic_summary={
                "supported_claim_count": supported_claim_count,
                "unsupported_claim_count": unsupported_claim_count,
                "stop_reason": (
                    "all_claims_supported"
                    if unsupported_claim_count == 0
                    else "unsupported_claims_present"
                ),
            },
            fact_state=fact_state,
        )

    def _retrieval_quality_gate(
        self,
        results: list[SubTaskExecutionResult],
        *,
        min_answer_length: int = 200,
        min_complete_ratio: float = 0.67,
    ) -> tuple[bool, dict[str, Any]]:
        """Semantic-aware quality gate for corrective drill-down."""
        if not results:
            return False, {"reason": "no_results"}

        coverage_gaps = self._coverage_gaps(results)
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
        contextless_count = 0
        supported_claims = 0
        total_claims = 0
        relevance_hits = 0
        for result in results:
            answer_lower = result.answer.lower()
            has_failure = any(marker in answer_lower for marker in failure_markers)
            is_long_enough = len(result.answer) >= min_answer_length
            if not has_failure and is_long_enough:
                complete_count += 1
            if not result.contexts and not result.sources:
                contextless_count += 1

            answer_claims = _claim_candidates(result.answer, limit=4)
            total_claims += len(answer_claims)
            context_tokens = _tokenize("\n".join(result.contexts))
            for claim in answer_claims:
                overlap = len(_tokenize(claim) & context_tokens)
                if overlap > 0:
                    supported_claims += 1

            question_tokens = _tokenize(result.question)
            if question_tokens and context_tokens:
                overlap_ratio = len(question_tokens & context_tokens) / max(
                    1,
                    len(question_tokens),
                )
                if overlap_ratio >= 0.2:
                    relevance_hits += 1

        complete_ratio = complete_count / len(results)
        claim_support_ratio = supported_claims / max(1, total_claims)
        relevance_ratio = relevance_hits / len(results)
        coverage_score = 1.0 if not self._required_coverage else 1.0 - (
            len(coverage_gaps) / max(1, len(self._required_coverage))
        )
        semantic_gate_score = (
            (0.45 * complete_ratio)
            + (0.30 * claim_support_ratio)
            + (0.15 * relevance_ratio)
            + (0.10 * coverage_score)
        )
        self._last_semantic_gate_score = semantic_gate_score
        gate_pass = (
            not coverage_gaps
            and contextless_count == 0
            and complete_ratio >= min_complete_ratio
            and claim_support_ratio >= 0.5
            and relevance_ratio >= 0.5
        )
        return gate_pass, {
            "coverage_gaps": coverage_gaps,
            "contextless_count": contextless_count,
            "complete_ratio": complete_ratio,
            "claim_support_ratio": claim_support_ratio,
            "context_relevance_ratio": relevance_ratio,
            "coverage_score": coverage_score,
            "semantic_gate_score": semantic_gate_score,
        }

    def _is_gap_targeted_followup(self, *, question: str, coverage_gaps: list[str]) -> bool:
        if not coverage_gaps:
            return True
        lowered = question.lower()
        keyword_map = _coverage_keywords()
        for gap in coverage_gaps:
            if any(keyword in lowered for keyword in keyword_map.get(gap, ())):
                return True
        return False

    def _should_skip_drilldown(
        self,
        results: list[SubTaskExecutionResult],
        min_answer_length: int = 200,
        min_complete_ratio: float = 0.67,
        current_iteration: int = -1,
    ) -> bool:
        if self._active_strategy_tier == "tier_1_detail_lookup" and (self._active_max_iterations or 0) <= 0:
            return True
        gate_pass, _gate_meta = self._retrieval_quality_gate(
            results,
            min_answer_length=min_answer_length,
            min_complete_ratio=min_complete_ratio,
        )
        return gate_pass

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

        effective_max_iterations = max_iterations
        followup_cap = _followup_cap_for_strategy(self._active_strategy_tier or "tier_1_detail_lookup")
        planner = TaskPlanner(max_subtasks=followup_cap, enable_graph_planning=False)
        fact_state: list[AtomicFact] = await self._refresh_fact_state(current_results)

        iteration = 1
        while iteration <= effective_max_iterations:
            gate_pass, gate_meta = self._retrieval_quality_gate(current_results)
            if iteration == 1:
                shift = self._adjust_strategy_after_exploration(gate_meta=gate_meta)
                followup_cap = _followup_cap_for_strategy(self._active_strategy_tier or "tier_1_detail_lookup")
                planner = TaskPlanner(max_subtasks=followup_cap, enable_graph_planning=False)
                if shift == "upshift" and self._active_strategy_tier:
                    effective_max_iterations = max(
                        effective_max_iterations,
                        _drilldown_iterations_for_strategy(
                            strategy_tier=self._active_strategy_tier,
                            question_intent=self._active_question_intent or "general_research",
                        ),
                    )
            if gate_pass:
                return iteration - 1

            coverage_gaps = list(gate_meta.get("coverage_gaps") or self._coverage_gaps(current_results))
            if not coverage_gaps:
                return iteration - 1

            fact_state = await self._refresh_fact_state(current_results, fact_state)
            findings_summary = self._build_findings_summary(
                current_results,
                fact_state=fact_state,
            )
            followup_tasks = await planner.create_followup_tasks(
                original_question=original_question,
                current_findings=findings_summary,
                existing_tasks=[
                    SubTask(id=result.id, question=result.question, task_type="rag")
                    for result in current_results
                ],
                question_intent=self._active_question_intent,
                coverage_gaps=coverage_gaps,
            )
            candidate_followups = list(followup_tasks[: max(1, followup_cap * 2)])
            targeted_followups: list[SubTask] = []
            for task in candidate_followups:
                gap_targeted = self._is_gap_targeted_followup(
                    question=task.question,
                    coverage_gaps=coverage_gaps,
                )
                overlap = self._semantic_overlap_score(task.question, fact_state)
                if overlap >= 0.55 and not gap_targeted:
                    if overlap >= 0.72 and await self._is_duplicate_followup_via_llm(
                        question=task.question,
                        fact_state=fact_state,
                    ):
                        self._pruned_followups_total += 1
                        continue
                    self._pruned_followups_total += 1
                    continue
                if gap_targeted or overlap < 0.55:
                    targeted_followups.append(task)
            if not targeted_followups:
                return iteration - 1

            max_id = max((result.id for result in current_results), default=0)
            editable_tasks = [
                EditableSubTask(
                    id=max_id + offset + 1,
                    question=task.question,
                    task_type=task.task_type,
                    enabled=True,
                )
                for offset, task in enumerate(targeted_followups[:followup_cap])
            ]
            executed = await self._execute_tasks(
                tasks=editable_tasks,
                user_id=user_id,
                doc_ids=doc_ids,
                enable_reranking=enable_reranking,
                iteration=iteration,
                enable_deep_image_analysis=enable_deep_image_analysis,
            )
            current_results.extend(executed)
            fact_state = await self._refresh_fact_state(executed, fact_state)

            if self._should_skip_drilldown(current_results, current_iteration=iteration):
                return iteration
            iteration += 1

        return effective_max_iterations

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
        question_intent = classify_question_intent(question)
        semantic_decision = SemanticIntentDecision(
            intent=question_intent,
            complexity_score=3,
            confidence=0.55,
            rationale="heuristic default decision",
            source="heuristic",
        )
        if self._semantic_router_mode in {"shadow", "active"}:
            semantic_decision = await classify_question_intent_semantic(question)

        if self._semantic_router_mode == "active":
            question_intent = semantic_decision.intent
            strategy_tier, subtask_limit, max_drilldown_iterations = _strategy_config_from_complexity(
                semantic_decision.complexity_score
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

        required_coverage = required_coverage_for_intent(question_intent)
        self._active_question_intent = question_intent
        self._active_strategy_tier = strategy_tier
        self._active_subtask_limit = subtask_limit
        self._active_max_iterations = max_drilldown_iterations
        self._required_coverage = required_coverage
        self._classifier_decision = semantic_decision.model_dump(mode="json")
        self._tier_shift = "keep"
        self._pruned_followups_total = 0
        self._last_semantic_gate_score = None

        try:
            plan_response = await run_with_retry(
                self.generate_agentic_plan,
                question=question,
                user_id=user_id,
                question_intent=question_intent,
                strategy_tier=strategy_tier,
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
                    "question_intent": question_intent,
                    "strategy_tier": strategy_tier,
                    "semantic_router_mode": self._semantic_router_mode,
                    "classifier_decision": dict(self._classifier_decision),
                    "complexity_score": semantic_decision.complexity_score,
                    "subtask_limit": subtask_limit,
                    "max_drilldown_iterations": max_drilldown_iterations,
                    "required_coverage": list(required_coverage),
                    "sub_tasks": [task.model_dump(mode="json") for task in plan_response.sub_tasks],
                },
            )

            request = ExecutePlanRequest(
                original_question=question,
                sub_tasks=plan_response.sub_tasks,
                doc_ids=None,
                enable_reranking=True,
                enable_drilldown=max_drilldown_iterations > 0,
                max_iterations=max(max_drilldown_iterations, 1),
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
            route_profiles: list[str] = []
            visual_verification_attempted = False
            visual_tool_call_count = 0
            visual_force_fallback_used = False

            for sub_result in result_response.sub_tasks:
                total_tokens += sub_result.usage.get("total_tokens", 0)
                if sub_result.route_profile:
                    route_profiles.append(sub_result.route_profile)
                for context in sub_result.contexts:
                    if context not in seen_contexts:
                        aggregated_docs.append(
                            Document(
                                page_content=context,
                                metadata={
                                    "task_id": sub_result.id,
                                    "iteration": sub_result.iteration,
                                    "question": sub_result.question,
                                    "route_profile": sub_result.route_profile,
                                },
                            )
                        )
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
                        metadata={
                            "iteration": sub_result.iteration,
                            "task_count": drilldown_count,
                            "coverage_gaps": self._coverage_gaps(result_response.sub_tasks),
                        },
                    )

                normalized_tool_calls = _normalize_tool_calls(sub_result.tool_calls)
                flattened_tool_calls.extend(sub_result.tool_calls)
                visual_meta = dict(sub_result.visual_verification_meta or {})
                subtask_visual_attempted = bool(
                    visual_meta.get("visual_verification_attempted")
                )
                subtask_visual_tool_call_count = int(
                    visual_meta.get("visual_tool_call_count", 0) or 0
                )
                subtask_visual_force_fallback_used = bool(
                    visual_meta.get("visual_force_fallback_used")
                )
                visual_verification_attempted = (
                    visual_verification_attempted or subtask_visual_attempted
                )
                visual_tool_call_count += subtask_visual_tool_call_count
                visual_force_fallback_used = (
                    visual_force_fallback_used or subtask_visual_force_fallback_used
                )
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
                        "strategy_tier": sub_result.strategy_tier,
                        "route_profile": sub_result.route_profile,
                        "micro_route": sub_result.micro_route,
                        "evidence_count": len(sub_result.evidence_units),
                        "source_count": len(sub_result.sources),
                        "context_count": len(sub_result.contexts),
                        "visual_verification_attempted": subtask_visual_attempted,
                        "visual_tool_call_count": subtask_visual_tool_call_count,
                        "visual_force_fallback_used": subtask_visual_force_fallback_used,
                        "sources": list(sub_result.sources),
                    },
                )

            coverage_status = self._coverage_status(result_response.sub_tasks)
            coverage_gaps = self._coverage_gaps(result_response.sub_tasks)
            supported_claim_count = int(result_response.critic_summary.get("supported_claim_count", 0))
            unsupported_claim_count = int(result_response.critic_summary.get("unsupported_claim_count", 0))
            dominant_route_profile = route_profiles[0] if route_profiles else None
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
                    "question_intent": question_intent,
                    "strategy_tier": strategy_tier,
                    "route_profile": dominant_route_profile,
                    "supported_claim_count": supported_claim_count,
                    "unsupported_claim_count": unsupported_claim_count,
                    "required_coverage": list(required_coverage),
                    "coverage_gaps": coverage_gaps,
                    "subtask_coverage_status": coverage_status,
                    "critic_summary": dict(result_response.critic_summary),
                    "classifier_decision": dict(self._classifier_decision),
                    "complexity_score": semantic_decision.complexity_score,
                    "tier_shift": self._tier_shift,
                    "pruned_followups": self._pruned_followups_total,
                    "semantic_gate_score": self._last_semantic_gate_score,
                    "visual_verification_attempted": visual_verification_attempted,
                    "visual_tool_call_count": visual_tool_call_count,
                    "visual_force_fallback_used": visual_force_fallback_used,
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
                    question_intent=question_intent,
                    strategy_tier=strategy_tier,
                    route_profile=dominant_route_profile,
                    required_coverage=list(required_coverage),
                    coverage_gaps=coverage_gaps,
                    subtask_coverage_status=coverage_status,
                    claims=list(result_response.claims),
                    supported_claim_count=supported_claim_count,
                    unsupported_claim_count=unsupported_claim_count,
                    visual_verification_attempted=visual_verification_attempted,
                    visual_tool_call_count=visual_tool_call_count,
                    visual_force_fallback_used=visual_force_fallback_used,
                    classifier_decision=dict(self._classifier_decision),
                    complexity_score=semantic_decision.complexity_score,
                    tier_shift=self._tier_shift,
                    pruned_followups=self._pruned_followups_total,
                    semantic_gate_score=self._last_semantic_gate_score,
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
                metadata={
                    "question_intent": question_intent,
                    "strategy_tier": strategy_tier,
                    "classifier_decision": dict(self._classifier_decision),
                    "complexity_score": semantic_decision.complexity_score,
                    "required_coverage": list(required_coverage),
                    "coverage_gaps": self._coverage_gaps([]),
                },
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
                    question_intent=question_intent,
                    strategy_tier=strategy_tier,
                    route_profile=None,
                    required_coverage=list(required_coverage),
                    coverage_gaps=self._coverage_gaps([]),
                    subtask_coverage_status={},
                    claims=[],
                    supported_claim_count=0,
                    unsupported_claim_count=0,
                    visual_verification_attempted=False,
                    visual_tool_call_count=0,
                    visual_force_fallback_used=False,
                    classifier_decision=dict(self._classifier_decision),
                    complexity_score=semantic_decision.complexity_score,
                    tier_shift=self._tier_shift,
                    pruned_followups=self._pruned_followups_total,
                    semantic_gate_score=self._last_semantic_gate_score,
                ),
            ) from exc
        finally:
            self._active_question_intent = None
            self._active_strategy_tier = None
            self._active_subtask_limit = None
            self._active_max_iterations = None
            self._required_coverage = []
            self._classifier_decision = {}


