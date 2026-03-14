"""
GraphRAG Generic Mode routing and evidence utilities.

Provides:
- lightweight query routing with heuristic fast-path and LLM fallback
- structured graph evidence records for local/global search
- budget-aware evidence merging for compact graph context construction
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from langchain_core.messages import HumanMessage

from core.providers import get_llm

logger = logging.getLogger(__name__)

QueryKind = Literal["fact", "relation", "summary"]
RoutePath = Literal["local-first", "global-first", "blended"]
StageHint = Literal["exploration", "verification"]
TaskTypeHint = Literal["rag", "graph_analysis"]
EvidenceType = Literal[
    "local_node",
    "local_edge",
    "community_summary",
    "community_answer",
]

_SUMMARY_KEYWORDS = (
    "summary",
    "summarize",
    "overview",
    "trend",
    "trends",
    "global",
    "整體",
    "總結",
    "概述",
    "趨勢",
    "綜合",
)
_RELATION_KEYWORDS = (
    "relationship",
    "relation",
    "compare",
    "comparison",
    "difference",
    "versus",
    "vs",
    "across",
    "between",
    "關係",
    "關聯",
    "比較",
    "對比",
    "差異",
    "跨文件",
    "連結",
)
_FACT_KEYWORDS = (
    "what",
    "which",
    "when",
    "where",
    "score",
    "metric",
    "accuracy",
    "f1",
    "dsc",
    "number",
    "數據",
    "分數",
    "指標",
    "多少",
    "哪個",
    "什麼",
)

_ROUTER_PROMPT = """你是一個圖譜檢索路由器。請為問題選擇檢索類型與路徑。

問題：{question}
Hints:
- stage_hint: {stage_hint}
- task_type_hint: {task_type_hint}
- prefer_global: {prefer_global}
- prefer_local: {prefer_local}
- has_communities: {has_communities}

規則：
1. fact = 需要精確數據、定義、單點事實
2. relation = 需要跨實體比較、關聯、對比
3. summary = 需要整體趨勢、總結、全域概覽
4. local-first = 先抓實體與關係證據
5. global-first = 先抓社群摘要
6. blended = 需要 local + global，但仍要控制上下文

只輸出 JSON：
{{"query_kind":"fact|relation|summary","path":"local-first|global-first|blended","budget":"tight|balanced|wide"}}
"""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _keyword_hits(question: str, keywords: tuple[str, ...]) -> int:
    normalized = _normalize(question)
    return sum(1 for keyword in keywords if keyword in normalized)


def estimate_token_count(text: str) -> int:
    """Rough token estimate used for budget control."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass(slots=True)
class GraphQueryHints:
    """Execution-layer hints for generic graph routing."""

    stage_hint: Optional[StageHint] = None
    task_type_hint: Optional[TaskTypeHint] = None
    prefer_global: bool = False
    prefer_local: bool = False


@dataclass(slots=True)
class GraphRouteDecision:
    """Resolved generic graph route."""

    query_kind: QueryKind
    path: RoutePath
    hops: int = 2
    max_nodes: int = 10
    max_communities: int = 2
    token_budget: int = 900


@dataclass(slots=True)
class GraphEvidence:
    """Structured graph evidence unit for budget-aware merging."""

    evidence_id: str
    evidence_type: EvidenceType
    text: str
    score: float
    token_estimate: int
    metadata: dict[str, object] = field(default_factory=dict)


class GenericGraphRouter:
    """Adaptive router for generic graph search mode."""

    def _fast_path(
        self,
        question: str,
        *,
        has_communities: bool,
        hints: Optional[GraphQueryHints],
    ) -> Optional[GraphRouteDecision]:
        hints = hints or GraphQueryHints()
        normalized = _normalize(question)
        summary_hits = _keyword_hits(normalized, _SUMMARY_KEYWORDS)
        relation_hits = _keyword_hits(normalized, _RELATION_KEYWORDS)
        fact_hits = _keyword_hits(normalized, _FACT_KEYWORDS)

        if hints.stage_hint == "verification" or hints.prefer_local:
            return GraphRouteDecision(
                query_kind="fact" if fact_hits >= relation_hits else "relation",
                path="local-first" if not has_communities else "blended",
                hops=2,
                max_nodes=8,
                max_communities=1,
                token_budget=720,
            )

        if hints.stage_hint == "exploration" or hints.prefer_global:
            return GraphRouteDecision(
                query_kind="summary" if summary_hits >= relation_hits else "relation",
                path="global-first" if has_communities else "local-first",
                hops=1,
                max_nodes=8,
                max_communities=3,
                token_budget=980,
            )

        if hints.task_type_hint == "graph_analysis":
            return GraphRouteDecision(
                query_kind="relation" if relation_hits >= summary_hits else "summary",
                path="blended" if has_communities else "local-first",
                hops=2,
                max_nodes=12,
                max_communities=3,
                token_budget=960,
            )

        if len(normalized) <= 24 and fact_hits and not relation_hits and not summary_hits:
            return GraphRouteDecision(
                query_kind="fact",
                path="local-first",
                hops=2,
                max_nodes=8,
                max_communities=1,
                token_budget=700,
            )

        if summary_hits and summary_hits >= relation_hits:
            return GraphRouteDecision(
                query_kind="summary",
                path="global-first" if has_communities else "local-first",
                hops=1,
                max_nodes=8,
                max_communities=3,
                token_budget=1000,
            )

        if relation_hits:
            return GraphRouteDecision(
                query_kind="relation",
                path="blended" if has_communities else "local-first",
                hops=2,
                max_nodes=12,
                max_communities=2,
                token_budget=920,
            )

        if fact_hits:
            return GraphRouteDecision(
                query_kind="fact",
                path="local-first",
                hops=2,
                max_nodes=10,
                max_communities=1,
                token_budget=760,
            )

        return None

    async def _llm_route(
        self,
        question: str,
        *,
        has_communities: bool,
        hints: Optional[GraphQueryHints],
    ) -> GraphRouteDecision:
        hints = hints or GraphQueryHints()
        try:
            llm = get_llm("graph_extraction")
            response = await llm.ainvoke(
                [
                    HumanMessage(
                        content=_ROUTER_PROMPT.format(
                            question=question,
                            stage_hint=hints.stage_hint or "none",
                            task_type_hint=hints.task_type_hint or "none",
                            prefer_global=str(hints.prefer_global).lower(),
                            prefer_local=str(hints.prefer_local).lower(),
                            has_communities=str(has_communities).lower(),
                        )
                    )
                ]
            )
            payload = json.loads(re.search(r"\{[\s\S]*\}", response.content).group(0))
            budget_name = payload.get("budget", "balanced")
            token_budget = {"tight": 700, "balanced": 900, "wide": 1100}.get(
                budget_name,
                900,
            )
            decision = GraphRouteDecision(
                query_kind=payload.get("query_kind", "fact"),
                path=payload.get("path", "local-first"),
                token_budget=token_budget,
            )
            return self._finalize(decision, has_communities=has_communities, hints=hints)
        except Exception as exc:
            logger.warning("Generic router LLM fallback failed: %s", exc)
            fallback = GraphRouteDecision(
                query_kind="relation" if has_communities else "fact",
                path="blended" if has_communities else "local-first",
            )
            return self._finalize(fallback, has_communities=has_communities, hints=hints)

    def _finalize(
        self,
        decision: GraphRouteDecision,
        *,
        has_communities: bool,
        hints: Optional[GraphQueryHints],
    ) -> GraphRouteDecision:
        hints = hints or GraphQueryHints()
        if not has_communities and decision.path == "global-first":
            decision.path = "local-first"
        elif not has_communities and decision.path == "blended":
            decision.path = "local-first"

        if decision.query_kind == "summary":
            decision.hops = 1
            decision.max_nodes = min(decision.max_nodes, 8)
            decision.max_communities = max(decision.max_communities, 2)
        elif decision.query_kind == "fact":
            decision.hops = max(decision.hops, 2)
            decision.max_nodes = min(max(decision.max_nodes, 8), 10)
            decision.max_communities = 1
        else:
            decision.hops = max(decision.hops, 2)
            decision.max_nodes = min(max(decision.max_nodes, 10), 14)
            decision.max_communities = min(max(decision.max_communities, 2), 3)

        if hints.stage_hint == "verification":
            decision.token_budget = min(decision.token_budget, 780)
        elif hints.stage_hint == "exploration":
            decision.token_budget = max(decision.token_budget, 980)

        return decision

    async def route(
        self,
        question: str,
        *,
        has_communities: bool,
        hints: Optional[GraphQueryHints] = None,
    ) -> GraphRouteDecision:
        fast_path = self._fast_path(
            question,
            has_communities=has_communities,
            hints=hints,
        )
        if fast_path is not None:
            return self._finalize(fast_path, has_communities=has_communities, hints=hints)
        return await self._llm_route(question, has_communities=has_communities, hints=hints)


def merge_graph_evidence(
    *,
    local_evidence: list[GraphEvidence],
    global_evidence: list[GraphEvidence],
    token_budget: int,
) -> tuple[str, list[GraphEvidence]]:
    """Merge local/global evidence into a compact graph context."""
    type_priority = {
        "local_edge": 4,
        "local_node": 3,
        "community_answer": 2,
        "community_summary": 1,
    }

    merged: list[GraphEvidence] = []
    seen_texts: set[str] = set()
    spent_tokens = 0

    candidates = sorted(
        [*local_evidence, *global_evidence],
        key=lambda item: (
            item.score,
            type_priority.get(item.evidence_type, 0),
            -item.token_estimate,
        ),
        reverse=True,
    )

    for item in candidates:
        normalized_text = _normalize(item.text)
        if not normalized_text or normalized_text in seen_texts:
            continue
        if spent_tokens + item.token_estimate > token_budget and merged:
            continue
        merged.append(item)
        seen_texts.add(normalized_text)
        spent_tokens += item.token_estimate

    if not merged:
        return "", []

    sections = {
        "local_edge": [],
        "local_node": [],
        "community_answer": [],
        "community_summary": [],
    }
    for item in merged:
        sections.setdefault(item.evidence_type, []).append(item.text)

    lines = ["=== Graph Evidence ==="]
    if sections["local_edge"]:
        lines.append("關係證據：")
        lines.extend(f"- {text}" for text in sections["local_edge"])
    if sections["local_node"]:
        lines.append("實體證據：")
        lines.extend(f"- {text}" for text in sections["local_node"])
    if sections["community_answer"]:
        lines.append("社群答案：")
        lines.extend(f"- {text}" for text in sections["community_answer"])
    if sections["community_summary"]:
        lines.append("社群摘要：")
        lines.extend(f"- {text}" for text in sections["community_summary"])

    return "\n".join(lines), merged
