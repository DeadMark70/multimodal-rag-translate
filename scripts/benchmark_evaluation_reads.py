"""Compare legacy/full-campaign and scoped reads using a read-only SQLite source.

EVALUATION_DATABASE_URL must point to an imported, isolated PostgreSQL copy.
No evaluation/model work is started. Only derived PostgreSQL caches are written.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

import aiosqlite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(value) -> str:
    payload = value.model_dump(mode="json")
    payload.pop("analysis_updated_at", None)
    payload.pop("analysis_status", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


async def measure(call, repeats: int) -> tuple[dict, object]:
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        result = await call()
        elapsed.append(round((perf_counter() - start) * 1000, 2))
    return {"milliseconds": elapsed, "bytes": len(result.model_dump_json().encode()),
            "sha256": digest(result)}, result


def difference_paths(left, right, path: str = "") -> list[str]:
    """Report structural differences without printing user content."""
    if left == right:
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        return [item for key in left.keys() | right.keys()
                for item in difference_paths(left.get(key), right.get(key), path + "/" + key)]
    if isinstance(left, list) and isinstance(right, list):
        if sorted(json.dumps(x, sort_keys=True) for x in left) == sorted(json.dumps(x, sort_keys=True) for x in right):
            return [path + ": order_only"]
        return [path + ": differing_list"]
    return [path]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--campaign", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not args.sqlite.is_file():
        raise ValueError("Missing SQLite source")
    from evaluation import db, accounting_store, observability_storage, analytics, job_store
    from evaluation.postgres import close_db, force_init_db
    from evaluation.research_analytics import ResearchAnalyticsService, _project_interactive_run_observability
    from evaluation.analysis_cache import read_analysis
    from evaluation.accounting_schemas import CampaignResearchSummaryResponse
    await force_init_db()
    uri = args.sqlite.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as source:
        if not args.campaign:
            args.campaign = [r[0] for r in source.execute(
                "SELECT campaign_id FROM evaluation_v9_attempt_materializations "
                "GROUP BY campaign_id ORDER BY SUM(length(trace_json)) DESC LIMIT 3"
            )]
        targets = []
        for cid in args.campaign:
            row = source.execute(
                "SELECT c.user_id,r.id FROM campaigns c JOIN campaign_results r ON r.campaign_id=c.id "
                "WHERE c.id=? AND r.status='completed' ORDER BY r.created_at DESC LIMIT 1", (cid,)
            ).fetchone()
            if row:
                targets.append((cid, row[0], row[1]))

    @asynccontextmanager
    async def readonly():
        async with aiosqlite.connect(uri, uri=True) as connection:
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA query_only=ON")
            yield connection

    async def initialized():
        pass

    modules = (db, accounting_store, observability_storage, analytics, job_store)
    originals = [(module, module.connect_db, module.init_db) for module in modules]
    report = {"source_bytes": args.sqlite.stat().st_size, "campaigns": []}
    try:
        for cid, uid, rid in targets:
            service = ResearchAnalyticsService()
            for module in modules:
                module.connect_db, module.init_db = readonly, initialized

            async def legacy_run():
                result = await service._results.get(user_id=uid, campaign_id=cid, result_id=rid)
                full = await service._load_campaign_run_observability(campaign_id=cid, results=[result])
                return _project_interactive_run_observability(full[rid])

            item = {"campaign_id": cid}
            item["sqlite_full_campaign_run"], old_run = await measure(legacy_run, args.repeats)
            item["sqlite_scoped_run"], scoped = await measure(
                lambda: service.get_run_observability(user_id=uid, campaign_id=cid, run_id=rid), args.repeats
            )
            item["sqlite_summary"], old_summary = await measure(
                lambda: service.get_summary(user_id=uid, campaign_id=cid), args.repeats
            )
            for module, connect, init in originals:
                module.connect_db, module.init_db = connect, init
            item["postgres_scoped_run"], pg_run = await measure(
                lambda: service.get_run_observability(user_id=uid, campaign_id=cid, run_id=rid), args.repeats
            )
            item["postgres_summary"], pg_summary = await measure(
                lambda: service.get_summary(user_id=uid, campaign_id=cid), args.repeats
            )
            async def cache():
                return await read_analysis(
                    user_id=uid, campaign_id=cid, kind="summary", model=CampaignResearchSummaryResponse,
                    loader=lambda: service.get_summary(user_id=uid, campaign_id=cid),
                )
            await cache()
            item["postgres_cached_summary"], cached = await measure(cache, args.repeats)
            item["run_equivalent"] = digest(old_run) == digest(scoped) == digest(pg_run)
            item["summary_equivalent"] = digest(old_summary) == digest(pg_summary) == digest(cached)
            if not item["run_equivalent"]:
                item["run_differences"] = difference_paths(old_run.model_dump(mode="json"), pg_run.model_dump(mode="json"))
            report["campaigns"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not all(r["run_equivalent"] and r["summary_equivalent"] for r in report["campaigns"]):
            raise RuntimeError("Output equivalence failed; inspect the benchmark report")
    finally:
        for module, connect, init in originals:
            module.connect_db, module.init_db = connect, init
        await close_db()


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
