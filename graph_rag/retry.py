"""Shared transient-failure classification for GraphRAG extraction."""

from __future__ import annotations

import httpx


def is_retryable_graph_error(exc: Exception) -> bool:
    """Return whether an extraction failure can reasonably succeed on retry."""
    status_code = getattr(exc, "status_code", None)
    return (
        isinstance(exc, (TimeoutError, httpx.TransportError))
        or status_code in {408, 429}
        or isinstance(status_code, int) and 500 <= status_code <= 599
    )
