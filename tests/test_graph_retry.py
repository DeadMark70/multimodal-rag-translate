"""Retry classification for transient GraphRAG provider failures."""

from graph_rag.retry import is_retryable_graph_error


class ProviderError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_provider_timeout_and_server_errors_are_retryable() -> None:
    assert is_retryable_graph_error(ProviderError(408)) is True
    assert is_retryable_graph_error(ProviderError(503)) is True


def test_provider_bad_request_is_not_retryable() -> None:
    assert is_retryable_graph_error(ProviderError(400)) is False
