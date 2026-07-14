"""Classify evaluation failures without exposing provider error details."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
from google.api_core import exceptions as google_exceptions
from pydantic import ValidationError


@dataclass(frozen=True)
class ErrorDecision:
    error_type: str
    retryable: bool
    retry_after_seconds: float | None
    safe_message: str


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield each distinct explicit cause or implicit context, starting at *exc*."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        next_error = current.__cause__ or current.__context__
        current = next_error if isinstance(next_error, BaseException) else None


def _as_status_code(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _status_code(exc: BaseException) -> int | None:
    direct_status = _as_status_code(getattr(exc, "status_code", None))
    if direct_status is not None:
        return direct_status
    response = getattr(exc, "response", None)
    return _as_status_code(getattr(response, "status_code", None))


def _retry_after_seconds(exc: BaseException) -> float | None:
    for attribute in ("retry_after_seconds", "retry_after"):
        value = getattr(exc, attribute, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    response = getattr(exc, "response", None)
    headers: Any = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _decision(
    error_type: str,
    retryable: bool,
    safe_message: str,
    retry_after_seconds: float | None,
) -> ErrorDecision:
    return ErrorDecision(
        error_type=error_type,
        retryable=retryable,
        retry_after_seconds=retry_after_seconds if retryable else None,
        safe_message=safe_message,
    )


def classify_evaluation_error(exc: BaseException) -> ErrorDecision:
    """Return retry guidance using exception types and status codes, never raw details."""

    chain = tuple(_exception_chain(exc))
    retry_after = next(
        (
            delay
            for current in chain
            if (delay := _retry_after_seconds(current)) is not None
        ),
        None,
    )
    status_codes = {status for current in chain if (status := _status_code(current)) is not None}

    if any(isinstance(current, ModuleNotFoundError) for current in chain):
        return _decision(
            "missing_dependency",
            False,
            "A required evaluation dependency is unavailable.",
            retry_after,
        )
    if 401 in status_codes or 403 in status_codes:
        return _decision(
            "authentication",
            False,
            "Provider authentication or authorization failed.",
            retry_after,
        )
    if any(isinstance(current, (ValidationError, ValueError)) for current in chain):
        return _decision(
            "invalid_configuration",
            False,
            "Evaluation configuration or dataset input is invalid.",
            retry_after,
        )
    if any(
        isinstance(current, (KeyboardInterrupt, SystemExit, InterruptedError))
        for current in chain
    ):
        return _decision(
            "interrupted",
            True,
            "Evaluation processing was interrupted.",
            retry_after,
        )
    if any(isinstance(current, google_exceptions.ResourceExhausted) for current in chain):
        return _decision(
            "rate_limit",
            True,
            "The evaluation provider rate limit was reached.",
            retry_after,
        )
    if any(isinstance(current, google_exceptions.ServiceUnavailable) for current in chain):
        return _decision(
            "service_unavailable",
            True,
            "The evaluation provider is temporarily unavailable.",
            retry_after,
        )
    if any(isinstance(current, (httpx.TimeoutException, TimeoutError)) for current in chain):
        return _decision(
            "timeout",
            True,
            "The evaluation provider request timed out.",
            retry_after,
        )
    if 429 in status_codes:
        return _decision(
            "rate_limit",
            True,
            "The evaluation provider rate limit was reached.",
            retry_after,
        )
    if 408 in status_codes:
        return _decision(
            "timeout",
            True,
            "The evaluation provider request timed out.",
            retry_after,
        )
    if any(500 <= status <= 599 for status in status_codes):
        return _decision(
            "server_error",
            True,
            "The evaluation provider encountered a temporary server error.",
            retry_after,
        )
    if any(isinstance(current, httpx.TransportError) for current in chain):
        return _decision(
            "transport",
            True,
            "A network transport error occurred during evaluation.",
            retry_after,
        )
    if status_codes:
        return _decision(
            "provider_error",
            False,
            "The evaluation provider rejected the request.",
            retry_after,
        )
    return _decision(
        "unknown",
        False,
        "An unexpected evaluation error occurred.",
        retry_after,
    )


def retry_delay_seconds(
    attempt_number: int,
    retry_after_seconds: float | None,
) -> float:
    if retry_after_seconds is not None:
        return max(0.0, min(retry_after_seconds, 900.0))
    exponent = max(0, attempt_number - 1)
    return min(2.0**exponent, 60.0)
