"""Persistent page summaries, invalidated by committed source changes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TypeVar

from pydantic import BaseModel
from fastapi import HTTPException

from evaluation.postgres import connect_db

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
FORMAT_VERSION = 5
_tasks: dict[tuple[str, str], asyncio.Task] = {}


async def _read(user_id: str, campaign_id: str, kind: str) -> dict:
    async with connect_db() as connection:
        row = await (await connection.execute(
            "SELECT COALESCE(state.revision, 0) AS current_revision, cache.* "
            "FROM campaigns c LEFT JOIN evaluation_analysis_state state ON state.campaign_id=c.id "
            "LEFT JOIN evaluation_analysis_cache cache ON cache.campaign_id=c.id AND cache.kind=%s "
            "WHERE c.id=%s AND c.user_id=%s", (kind, campaign_id, user_id)
        )).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return row


async def build_analysis(
    user_id: str, campaign_id: str, kind: str, loader: Callable[[], Awaitable[T]],
) -> T | None:
    # Compute without holding a pooled connection. Per-process tasks coalesce
    # duplicate requests; revision-checked publication is safe across processes.
    before = await _read(user_id, campaign_id, kind)
    result = await loader()
    built_at = datetime.now(timezone.utc).isoformat()
    async with connect_db() as connection:
        # Revision equality also rejects a build that mixed reads across writes.
        await connection.execute(
            "INSERT INTO evaluation_analysis_cache(campaign_id,kind,revision,format_version,payload,built_at) "
            "SELECT %s,%s,%s,%s,%s,%s WHERE "
            "COALESCE((SELECT revision FROM evaluation_analysis_state WHERE campaign_id=%s),0)=%s "
            "ON CONFLICT(campaign_id,kind) DO UPDATE SET "
            "revision=excluded.revision,format_version=excluded.format_version,"
            "payload=excluded.payload,built_at=excluded.built_at "
            "WHERE evaluation_analysis_cache.revision<=excluded.revision",
            (campaign_id, kind, before["current_revision"], FORMAT_VERSION,
             result.model_dump_json(), built_at, campaign_id, before["current_revision"]),
        )
    return result


def _schedule(user_id: str, campaign_id: str, kind: str, loader: Callable) -> asyncio.Task:
    key = (campaign_id, kind)
    task = _tasks.get(key)
    if task is None or task.done():
        task = asyncio.create_task(build_analysis(user_id, campaign_id, kind, loader))
        _tasks[key] = task

        def finished(done: asyncio.Task) -> None:
            if _tasks.get(key) is done:
                _tasks.pop(key, None)
            if not done.cancelled() and done.exception() is not None:
                logger.error("Analysis refresh failed for %s/%s", campaign_id, kind,
                             exc_info=done.exception())

        task.add_done_callback(finished)
    return task


async def read_analysis(
    *, user_id: str, campaign_id: str, kind: str,
    model: type[T], loader: Callable[[], Awaitable[T]],
) -> T:
    row = await _read(user_id, campaign_id, kind)
    valid = row["payload"] is not None and row["format_version"] == FORMAT_VERSION
    stale = not valid or row["revision"] != row["current_revision"]
    if stale:
        task = _schedule(user_id, campaign_id, kind, loader)
        if not valid:
            # First access fills the cache once; subsequent changes refresh in
            # the background. A deployment can warm these before opening traffic.
            await asyncio.shield(task)
            row = await _read(user_id, campaign_id, kind)
            if row["payload"] is None or row["format_version"] != FORMAT_VERSION:
                raise HTTPException(503, detail="Analysis updating", headers={"Retry-After": "2"})
    result = model.model_validate_json(row["payload"])
    if "analysis_status" in model.model_fields:
        result = result.model_copy(update={
            "analysis_status": "ready" if row["revision"] == row["current_revision"] else "updating",
            "analysis_updated_at": row["built_at"],
        })
    return result


async def refresh_loop() -> None:
    """Refresh only pages previously requested; no new queue infrastructure."""
    from evaluation.research_analytics import ResearchAnalyticsService
    service = ResearchAnalyticsService()
    methods = {"summary": service.get_summary, "questions": service.get_question_comparison,
               "behavior": service.get_agent_behavior}
    while True:
        try:
            async with connect_db() as connection:
                rows = await (await connection.execute(
                    "SELECT c.id,c.user_id,cache.kind FROM evaluation_analysis_cache cache "
                    "JOIN campaigns c ON c.id=cache.campaign_id "
                    "JOIN evaluation_analysis_state state ON state.campaign_id=c.id "
                    "WHERE cache.revision<>state.revision OR cache.format_version<>%s "
                    "ORDER BY cache.built_at LIMIT 8", (FORMAT_VERSION,)
                )).fetchall()
            for row in rows:
                method = methods.get(row["kind"])
                if method is not None:
                    await asyncio.shield(_schedule(
                        row["user_id"], row["id"], row["kind"],
                        lambda row=row, method=method: method(user_id=row["user_id"], campaign_id=row["id"]),
                    ))
        except Exception:
            logger.exception("Unable to refresh evaluation analyses")
        await asyncio.sleep(5)


async def stop_refreshes() -> None:
    tasks = list(_tasks.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    _tasks.clear()
