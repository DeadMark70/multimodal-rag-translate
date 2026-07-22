"""Persistence helpers for RAG domain Supabase access."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from core.supabase_repository import execute_supabase_operation


async def insert_chat_log(
    *,
    user_id: str,
    question: str,
    answer: str,
) -> None:
    """Inserts one row into chat_logs."""
    payload = {"user_id": user_id, "question": question, "answer": answer}
    await execute_supabase_operation(
        operation="insert_chat_log",
        failure_message="Failed to save chat log",
        handler=lambda client: client.table("chat_logs").insert(payload).execute(),
    )


async def insert_query_log(
    *,
    user_id: str,
    question: str,
    answer: str | None,
    has_history: bool,
    faithfulness: str | None,
    confidence: float | None,
    response_time_ms: int | None,
    doc_ids: list[str] | None,
) -> None:
    """Inserts one row into query_logs."""
    payload = {
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "has_history": has_history,
        "faithfulness": faithfulness,
        "confidence": confidence,
        "response_time_ms": response_time_ms,
        "doc_ids": doc_ids,
    }
    await execute_supabase_operation(
        operation="insert_query_log",
        failure_message="Failed to save query log",
        handler=lambda client: client.table("query_logs").insert(payload).execute(),
    )


async def fetch_document_filenames(doc_ids: list[str]) -> dict[str, str]:
    """Returns a map of document id to file_name."""
    if not doc_ids:
        return {}

    response = await execute_supabase_operation(
        operation="fetch_document_filenames",
        failure_message="Failed to fetch document metadata",
        handler=lambda client: client.table("documents")
            .select("id, file_name")
            .in_("id", doc_ids)
            .execute(),
    )

    name_map: dict[str, str] = {}
    for row in response.data or []:
        doc_id = row.get("id")
        if not doc_id:
            continue
        name_map[doc_id] = row.get("file_name") or f"文件-{doc_id[:8]}"
    return name_map


async def resolve_document_references(
    user_id: str, references: list[str]
) -> dict[str, list[str]]:
    """Resolve owned canonical IDs from immutable IDs or stored file names.

    Evaluation test cases created before canonical document IDs were persisted
    may hold ``file_name`` values.  Keep this resolution fail-closed: duplicate
    file names remain multiple candidates for ``SourceScopeResolver`` to reject
    instead of selecting an arbitrary document.
    """
    unique_references = list(dict.fromkeys(value for value in references if value))
    if not unique_references:
        return {}

    async def select(field: str, values: list[str]):
        return await execute_supabase_operation(
            operation=f"resolve_document_references_by_{field}",
            failure_message="Failed to resolve evaluation source references",
            handler=lambda client: client.table("documents")
            .select("id, file_name")
            .eq("user_id", user_id)
            .in_(field, values)
            .execute(),
        )

    canonical_ids = [
        reference for reference in unique_references if _is_canonical_document_id(reference)
    ]
    responses = []
    if canonical_ids:
        responses.append(await select("id", canonical_ids))
    responses.append(await select("file_name", unique_references))
    resolved: dict[str, set[str]] = {reference: set() for reference in unique_references}
    for response in responses:
        for row in response.data or []:
            document_id = row.get("id")
            if not isinstance(document_id, str) or not document_id:
                continue
            if document_id in resolved:
                resolved[document_id].add(document_id)
            file_name = row.get("file_name")
            if isinstance(file_name, str) and file_name in resolved:
                resolved[file_name].add(document_id)
    return {
        reference: sorted(document_ids)
        for reference, document_ids in resolved.items()
        if document_ids
    }


def _is_canonical_document_id(value: str) -> bool:
    """Return whether a value is safe to send to the UUID ``documents.id`` field."""
    try:
        UUID(value)
    except (TypeError, ValueError, AttributeError):
        return False
    return True


async def persist_research_conversation(
    *,
    conversation_id: str,
    user_id: str,
    title: str | None,
    metadata: dict,
) -> None:
    """Persists deep-research result payload to conversations metadata."""
    existing_response = await execute_supabase_operation(
        operation="load_existing_research_conversation",
        failure_message="Failed to load existing research conversation",
        handler=lambda client: client.table("conversations")
            .select("metadata")
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute(),
    )

    existing_metadata = {}
    if existing_response.data:
        existing_row = existing_response.data[0] or {}
        existing_value = existing_row.get("metadata")
        if isinstance(existing_value, dict):
            existing_metadata = existing_value

    payload = {
        "metadata": {
            **existing_metadata,
            **metadata,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if title:
        payload["title"] = title

    await execute_supabase_operation(
        operation="persist_research_conversation",
        failure_message="Failed to persist research conversation",
        handler=lambda client: client.table("conversations")
            .update(payload)
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .execute(),
    )
