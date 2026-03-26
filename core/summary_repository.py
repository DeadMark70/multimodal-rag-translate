"""Repository helpers for document summary persistence."""

from __future__ import annotations

from core.supabase_repository import execute_supabase_operation


async def update_document_summary_record(*, doc_id: str, summary: str) -> None:
    """Updates executive summary by document id."""
    await execute_supabase_operation(
        operation="update_document_summary_record",
        failure_message="Failed to update executive summary",
        handler=lambda client: client.table("documents")
            .update({"executive_summary": summary})
            .eq("id", doc_id)
            .execute(),
    )


async def get_document_summary_record(*, doc_id: str) -> str | None:
    """Returns executive summary by document id."""
    result = await execute_supabase_operation(
        operation="get_document_summary_record",
        failure_message="Failed to get executive summary",
        handler=lambda client: client.table("documents")
            .select("executive_summary")
            .eq("id", doc_id)
            .single()
            .execute(),
    )
    return result.data.get("executive_summary") if result.data else None
