"""Shared retry helpers for evaluation workflows."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from google.api_core import exceptions as google_exceptions
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


async def run_with_retry(
    operation: Callable[..., Awaitable[_T]],
    *args,
    **kwargs,
) -> _T:
    """Retry known transient Gemini API failures with exponential backoff."""
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
        before_sleep=lambda state: logger.warning(
            "Evaluation retry %s after %s",
            state.attempt_number,
            state.outcome.exception() if state.outcome else "unknown error",
        ),
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
        while len(self._timestamps) >= self.rpm_limit:
            window_age = time.monotonic() - self._timestamps[0]
            if window_age >= 60:
                self._timestamps.popleft()
                continue
            await _sleep(60 - window_age)
        self._timestamps.append(time.monotonic())


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)

