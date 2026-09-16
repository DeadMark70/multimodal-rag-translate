"""Compare full-campaign, scoped and cached reads on a PostgreSQL test copy.

EVALUATION_DATABASE_URL must point to an imported, isolated PostgreSQL copy.
No evaluation/model work is started. Only derived PostgreSQL caches are written.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter


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
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--campaign", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    from evaluation.postgres import close_db, connect_db, init_db
    from evaluation.research_analytics import ResearchAnalyticsService, _project_interactive_run_observability
    from evaluation.analysis_cache import read_analysis
    from evaluation.accounting_schemas import CampaignResearchSummaryResponse
    await init_db()
    async with connect_db() as source:
        if not args.campaign:
            args.campaign = [r["campaign_id"] for r in await (await source.execute(
                "SELECT campaign_id FROM evaluation_v9_attempt_materializations "
                "GROUP BY campaign_id ORDER BY SUM(length(trace_json)) DESC LIMIT 3"
            )).fetchall()]
        targets = []
        for cid in args.campaign:
            row = await (await source.execute(
                "SELECT c.user_id,r.id FROM campaigns c JOIN campaign_results r ON r.campaign_id=c.id "
                "WHERE c.id=%s AND r.status='completed' ORDER BY r.created_at DESC LIMIT 1", (cid,)
            )).fetchone()
            if row:
                targets.append((cid, row["user_id"], row["id"]))

    report = {"campaigns": []}
    try:
        for cid, uid, rid in targets:
            service = ResearchAnalyticsService()
            async def full_campaign_run():
                result = await service._results.get(user_id=uid, campaign_id=cid, result_id=rid)
                full = await service._load_campaign_run_observability(campaign_id=cid, results=[result])
                return _project_interactive_run_observability(full[rid])

            item = {"campaign_id": cid}
            item["postgres_full_campaign_run"], full_run = await measure(full_campaign_run, args.repeats)
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
            item["run_equivalent"] = digest(full_run) == digest(pg_run)
            item["summary_equivalent"] = digest(pg_summary) == digest(cached)
            if not item["run_equivalent"]:
                item["run_differences"] = difference_paths(full_run.model_dump(mode="json"), pg_run.model_dump(mode="json"))
            report["campaigns"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not all(r["run_equivalent"] and r["summary_equivalent"] for r in report["campaigns"]):
            raise RuntimeError("Output equivalence failed; inspect the benchmark report")
    finally:
        await close_db()


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
