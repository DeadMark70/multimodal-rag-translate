"""Fail-closed source-name resolution for the Agentic v9 retrieval boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import unicodedata

from data_base.agentic_v9.schemas import ResolvedSourceScope


class SourceScopeResolver:
    """Resolve requested source names without expanding an authorized document scope."""

    def __init__(
        self, source_name_to_doc_ids: Mapping[str, str | Iterable[str]]
    ) -> None:
        self._source_name_to_doc_ids = _build_name_index(source_name_to_doc_ids)

    def resolve(
        self,
        *,
        requested_source_names: Iterable[str] | None,
        requested_doc_ids: Iterable[str] | None,
        authorized_doc_ids: Iterable[str] | None,
    ) -> ResolvedSourceScope:
        """Return only IDs selected by the request and allowed by authorization.

        Any unknown, ambiguous, or unauthorized request element empties the
        effective scope.  This prevents a partially resolved request from
        silently broadening retrieval to documents the caller did not name.
        """
        names = _unique_values(requested_source_names)
        requested_ids = _unique_values(requested_doc_ids)
        authorized_ids = set(_unique_values(authorized_doc_ids))

        resolved_by_name: dict[str, str] = {}
        rejected_names: list[str] = []
        for name in names:
            matches = self._source_name_to_doc_ids.get(_normalize_name(name), ())
            if len(matches) != 1:
                rejected_names.append(name)
                continue
            resolved_by_name[name] = matches[0]

        if rejected_names:
            return ResolvedSourceScope(
                requested_doc_ids=requested_ids,
                requested_source_names=names,
                rejected_source_names=rejected_names,
            )

        resolved_ids = sorted(set(resolved_by_name.values()))
        candidates = set(resolved_ids if names else requested_ids)
        if names and requested_ids:
            candidates.intersection_update(requested_ids)

        unauthorized_names = sorted(
            name
            for name, doc_id in resolved_by_name.items()
            if doc_id not in authorized_ids
        )
        unauthorized_requested_ids = set(requested_ids).difference(authorized_ids)
        if unauthorized_names or unauthorized_requested_ids:
            return ResolvedSourceScope(
                requested_doc_ids=requested_ids,
                requested_source_names=names,
                resolved_doc_ids=resolved_ids,
                rejected_source_names=unauthorized_names,
            )

        return ResolvedSourceScope(
            requested_doc_ids=requested_ids,
            requested_source_names=names,
            resolved_doc_ids=resolved_ids,
            authorized_doc_ids=sorted(candidates.intersection(authorized_ids)),
        )


def _build_name_index(
    source_name_to_doc_ids: Mapping[str, str | Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """Normalize names and retain every target so aliases cannot hide ambiguity."""
    index: dict[str, set[str]] = {}
    for name, value in source_name_to_doc_ids.items():
        doc_ids = (value,) if isinstance(value, str) else tuple(value)
        normalized_name = _normalize_name(name)
        targets = index.setdefault(normalized_name, set())
        targets.update(_unique_values(doc_ids))
    return {name: tuple(sorted(doc_ids)) for name, doc_ids in index.items()}


def _unique_values(values: Iterable[str] | None) -> list[str]:
    """Return stable, non-empty string identifiers without coercing other types."""
    if values is None:
        return []
    return list(dict.fromkeys(value for value in values if value))


def _normalize_name(name: str) -> str:
    """Apply a deterministic display-name comparison key."""
    return unicodedata.normalize("NFKC", name).strip().casefold()


__all__ = ["SourceScopeResolver"]
