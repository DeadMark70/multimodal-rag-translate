"""Shared retry helpers for evaluation workflows."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from google.api_core import exceptions as google_exceptions
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
RetryCallback = Callable[[int, BaseException], Awaitable[None]]


async def run_with_retry(
    operation: Callable[..., Awaitable[_T]],
    *args,
    on_retry: RetryCallback | None = None,
    **kwargs,
) -> _T:
    """Retry known transient Gemini API failures with exponential backoff."""

    async def before_sleep(state) -> None:  # noqa: ANN001
        error = state.outcome.exception() if state.outcome else None
        if not isinstance(error, BaseException):
            error = RuntimeError("Retry scheduled without an exception")
        logger.warning("Evaluation retry %s after %s", state.attempt_number, error)
        if on_retry is not None:
            await on_retry(state.attempt_number, error)

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(
            (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
            )
        ),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
        before_sleep=before_sleep,
    ):
        with attempt:
            return await operation(*args, **kwargs)
    raise RuntimeError("Retry loop exited unexpectedly")


@dataclass
class RateBudget:
    """Simple sliding-window RPM limiter used per campaign."""

    rpm_limit: int
    _timestamps: deque[float] = field(default_factory=deque)

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] >= 60:
                self._timestamps.popleft()
            if len(self._timestamps) < self.rpm_limit:
                self._timestamps.append(now)
                return
            await _sleep(max(0, 60 - (now - self._timestamps[0])))


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
