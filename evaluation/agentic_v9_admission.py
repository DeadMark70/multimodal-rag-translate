"""Shared source-scope and contract admission for Agentic v9 campaigns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass

from data_base.agentic_v9.route_planner import plan_query_contract
from data_base.agentic_v9.schemas import QueryContract, ResolvedSourceScope
from data_base.agentic_v9.source_scope_resolver import SourceScopeResolver
from data_base.repository import resolve_document_references


DocumentReferenceResolver = Callable[
    [str, list[str]], Awaitable[Mapping[str, str | Iterable[str]]]
]


@dataclass(frozen=True, slots=True)
class V9AdmissionContract:
    """The exact source scope and deterministic contract used by v9."""

    source_scope: ResolvedSourceScope
    contract: QueryContract


async def build_v9_admission_contract(
    *,
    question: str,
    user_id: str,
    source_references: list[str],
    document_reference_resolver: DocumentReferenceResolver = resolve_document_references,
) -> V9AdmissionContract:
    """Resolve test-case references before creating the real runtime contract."""
    references = list(dict.fromkeys(value for value in source_references if value))
    name_to_doc_ids = await document_reference_resolver(user_id, references)
    authorized_doc_ids = sorted(
        {
            document_id
            for document_ids in name_to_doc_ids.values()
            for document_id in (
                [document_ids] if isinstance(document_ids, str) else document_ids
            )
            if isinstance(document_id, str) and document_id
        }
    )
    source_scope = SourceScopeResolver(name_to_doc_ids).resolve(
        requested_source_names=references,
        requested_doc_ids=authorized_doc_ids,
        authorized_doc_ids=authorized_doc_ids,
    )
    contract = await plan_query_contract(
        question=question,
        resolved_source_scope=source_scope,
    )
    return V9AdmissionContract(source_scope=source_scope, contract=contract)


__all__ = [
    "DocumentReferenceResolver",
    "V9AdmissionContract",
    "build_v9_admission_contract",
]
