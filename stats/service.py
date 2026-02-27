"""Service layer for dashboard statistics aggregation."""

from __future__ import annotations

from datetime import datetime, timedelta

from stats.repository import list_query_logs
from stats.schemas import DashboardStats, DocumentStat


async def build_dashboard_stats(*, user_id: str) -> DashboardStats:
    """Builds dashboard statistics from query logs."""
    logs = await list_query_logs(user_id=user_id)
    total = len(logs)

    if total == 0:
        return DashboardStats(
            total_queries=0,
            accuracy_rate=0.0,
            grounded_count=0,
            hallucinated_count=0,
            uncertain_count=0,
            avg_confidence=0.0,
            queries_last_7_days=[0] * 7,
            top_documents=[],
        )

    grounded = sum(1 for log in logs if log.get("faithfulness") == "grounded")
    hallucinated = sum(1 for log in logs if log.get("faithfulness") == "hallucinated")
    uncertain = total - grounded - hallucinated

    evaluated = grounded + hallucinated
    accuracy_rate = grounded / evaluated if evaluated > 0 else 0.0

    confidences = [log.get("confidence") for log in logs if log.get("confidence")]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    now = datetime.utcnow()
    queries_last_7_days: list[int] = []
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_end = day_start + timedelta(days=1)
        count = sum(
            1
            for log in logs
            if log.get("created_at")
            and day_start.isoformat() <= log["created_at"] < day_end.isoformat()
        )
        queries_last_7_days.append(count)

    doc_counts: dict[str, int] = {}
    for log in logs:
        doc_ids = log.get("doc_ids") or []
        for doc_id in doc_ids:
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    top_documents = [
        DocumentStat(doc_id=doc_id, query_count=count)
        for doc_id, count in sorted(
            doc_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
    ]

    return DashboardStats(
        total_queries=total,
        accuracy_rate=accuracy_rate,
        grounded_count=grounded,
        hallucinated_count=hallucinated,
        uncertain_count=uncertain,
        avg_confidence=avg_confidence,
        queries_last_7_days=queries_last_7_days,
        top_documents=top_documents,
    )
