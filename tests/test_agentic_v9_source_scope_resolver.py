"""Focused authorization contracts for the Agentic v9 source scope resolver."""

from __future__ import annotations

from data_base.agentic_v9 import SourceScopeResolver


def _resolver() -> SourceScopeResolver:
    return SourceScopeResolver(
        {
            "paper-a.pdf": "doc-a",
            "paper-b.pdf": "doc-b",
            "duplicate.pdf": ("doc-a", "doc-b"),
        }
    )


def test_resolver_intersects_resolved_names_authorization_and_requested_ids() -> None:
    scope = _resolver().resolve(
        requested_source_names=["PAPER-A.PDF"],
        requested_doc_ids=["doc-a", "doc-b"],
        authorized_doc_ids=["doc-a", "doc-b"],
    )

    assert scope.resolved_doc_ids == ["doc-a"]
    assert scope.authorized_doc_ids == ["doc-a"]
    assert scope.rejected_source_names == []


def test_resolver_uses_requested_ids_as_the_explicit_scope_without_names() -> None:
    scope = _resolver().resolve(
        requested_source_names=[],
        requested_doc_ids=["doc-b", "doc-a", "doc-b"],
        authorized_doc_ids=["doc-a", "doc-b", "doc-c"],
    )

    assert scope.resolved_doc_ids == []
    assert scope.authorized_doc_ids == ["doc-a", "doc-b"]


def test_resolver_fails_closed_when_a_name_is_ambiguous() -> None:
    scope = _resolver().resolve(
        requested_source_names=["paper-a.pdf", "duplicate.pdf"],
        requested_doc_ids=["doc-a"],
        authorized_doc_ids=["doc-a", "doc-b"],
    )

    assert scope.resolved_doc_ids == []
    assert scope.authorized_doc_ids == []
    assert scope.rejected_source_names == ["duplicate.pdf"]


def test_resolver_fails_closed_when_any_requested_source_is_unauthorized() -> None:
    scope = _resolver().resolve(
        requested_source_names=["paper-a.pdf", "paper-b.pdf"],
        requested_doc_ids=[],
        authorized_doc_ids=["doc-a"],
    )

    assert scope.resolved_doc_ids == ["doc-a", "doc-b"]
    assert scope.authorized_doc_ids == []
    assert scope.rejected_source_names == ["paper-b.pdf"]


def test_resolver_fails_closed_for_unknown_names_and_unauthorized_document_ids() -> None:
    unknown_name_scope = _resolver().resolve(
        requested_source_names=["missing.pdf"],
        requested_doc_ids=[],
        authorized_doc_ids=["doc-a"],
    )
    unauthorized_id_scope = _resolver().resolve(
        requested_source_names=[],
        requested_doc_ids=["doc-a", "doc-b"],
        authorized_doc_ids=["doc-a"],
    )

    assert unknown_name_scope.authorized_doc_ids == []
    assert unknown_name_scope.rejected_source_names == ["missing.pdf"]
    assert unauthorized_id_scope.authorized_doc_ids == []
