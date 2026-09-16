"""Committed writes invalidate summaries; stale builds never become current."""

import asyncio
from typing import Literal
from unittest.mock import AsyncMock

from fastapi import HTTPException
from pydantic import BaseModel
import pytest
import pytest_asyncio

from evaluation import analysis_cache as cache
from evaluation.postgres import close_db, connect_db


class Summary(BaseModel):
    value: int
    analysis_status: Literal["ready", "updating"] = "ready"
    analysis_updated_at: str | None = None


@pytest_asyncio.fixture
async def campaign():
    async with connect_db() as conn:
        await conn.execute(
            "INSERT INTO campaigns(id,user_id,status,config_json,created_at,updated_at) "
            "VALUES ('cache-campaign','owner','completed','{}','2026-09-16','2026-09-16')"
        )
    yield "cache-campaign"
    await cache.stop_refreshes()


async def read(campaign, loader, user_id="owner"):
    return await cache.read_analysis(user_id=user_id, campaign_id=campaign,
                                    kind="summary", model=Summary, loader=loader)


@pytest.mark.asyncio
async def test_cache_persists_and_enforces_owner(campaign):
    loader = AsyncMock(return_value=Summary(value=1))
    first = await read(campaign, loader)
    assert first.analysis_status == "ready"
    await close_db()
    assert await read(campaign, loader) == first
    loader.assert_awaited_once()
    with pytest.raises(HTTPException) as error:
        await read(campaign, loader, "another-owner")
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_commits_refresh_and_rollbacks_keep_cache(campaign):
    loader = AsyncMock(return_value=Summary(value=1))
    await read(campaign, loader)
    async with connect_db() as conn:
        await conn.execute("UPDATE campaigns SET name='rolled back' WHERE id=%s", (campaign,))
        await conn.rollback()
    assert (await read(campaign, loader)).analysis_status == "ready"
    loader.assert_awaited_once()

    async with connect_db() as conn:
        await conn.execute("UPDATE campaigns SET name='committed' WHERE id=%s", (campaign,))
    started, release = asyncio.Event(), asyncio.Event()

    async def refresh():
        started.set()
        await release.wait()
        return Summary(value=2)

    old = await read(campaign, refresh)
    assert old.value == 1 and old.analysis_status == "updating"
    await started.wait()
    task = cache._tasks[(campaign, "summary")]
    release.set()
    await task
    latest = await read(campaign, loader)
    assert latest.value == 2 and latest.analysis_status == "ready"
    loader.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_write_rejects_cold_build_and_retries(campaign):
    async def racing_loader():
        async with connect_db() as conn:
            await conn.execute("UPDATE campaigns SET name='changed' WHERE id=%s", (campaign,))
        return Summary(value=1)

    with pytest.raises(HTTPException) as error:
        await read(campaign, racing_loader)
    assert error.value.status_code == 503
    assert (await read(campaign, AsyncMock(return_value=Summary(value=2)))).value == 2


@pytest.mark.asyncio
async def test_simultaneous_cold_reads_share_one_build(campaign):
    async def build():
        await asyncio.sleep(0.03)
        return Summary(value=5)

    loader = AsyncMock(side_effect=build)
    results = await asyncio.gather(*(read(campaign, loader) for _ in range(8)))
    assert all(result.value == 5 for result in results)
    loader.assert_awaited_once()
