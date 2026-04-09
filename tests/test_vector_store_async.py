from __future__ import annotations

import asyncio
from threading import Event

import pytest

from data_base import vector_store_manager


@pytest.mark.asyncio
async def test_user_locked_vector_store_call_serializes_same_user_requests() -> None:
    started: list[str] = []
    release_first = Event()

    def blocking_call(label: str) -> str:
        started.append(label)
        if label == "first":
            release_first.wait(timeout=0.5)
        return label

    first_task = asyncio.create_task(
        vector_store_manager._run_user_locked_vector_store_call(
            "user-1",
            blocking_call,
            "first",
        )
    )

    await asyncio.sleep(0.05)

    second_task = asyncio.create_task(
        vector_store_manager._run_user_locked_vector_store_call(
            "user-1",
            blocking_call,
            "second",
        )
    )

    await asyncio.sleep(0.05)
    assert started == ["first"]

    release_first.set()
    results = await asyncio.gather(first_task, second_task)
    assert results == ["first", "second"]
    assert started == ["first", "second"]


@pytest.mark.asyncio
async def test_user_locked_vector_store_call_allows_parallel_different_users() -> None:
    started: list[str] = []
    release_first = Event()

    def blocking_call(label: str, hold: bool = False) -> str:
        started.append(label)
        if hold:
            release_first.wait(timeout=0.5)
        return label

    first_task = asyncio.create_task(
        vector_store_manager._run_user_locked_vector_store_call(
            "user-1",
            blocking_call,
            "first",
            True,
        )
    )

    await asyncio.sleep(0.05)

    second_task = asyncio.create_task(
        vector_store_manager._run_user_locked_vector_store_call(
            "user-2",
            blocking_call,
            "second",
            False,
        )
    )

    await asyncio.sleep(0.05)
    assert started == ["first", "second"]

    release_first.set()
    results = await asyncio.gather(first_task, second_task)
    assert results == ["first", "second"]
