"""Separate connections must not claim the same item or recover live work."""

import asyncio
from datetime import timedelta

import pytest

from tests.test_evaluation_job_store import store, fixed_now, _spec  # noqa: F401
from evaluation.job_store import EvaluationJobStore


@pytest.mark.asyncio
async def test_parallel_claims_and_stale_recovery(store, fixed_now):  # noqa: F811
    await store.create_job_with_items(
        user_id="user-a", campaign_id="cmp-1", job_type="initial", selection={},
        config_snapshot={}, items=[_spec(logical_key=f"execution:Q{i}:naive:1:none") for i in range(4)],
    )
    claims = await asyncio.gather(*(
        EvaluationJobStore().claim_ready_items(limit=1, now=fixed_now) for _ in range(4)
    ))
    claims = [claim for batch in claims for claim in batch]
    assert len(claims) == 4
    assert len({claim.job_item_id for claim in claims}) == 4
    at = fixed_now + timedelta(minutes=6)
    live = claims[0].attempt_id
    await store.heartbeat_attempt(live, at=at)
    assert await store.recover_interrupted_attempts(
        at=at, stale_before=at-timedelta(minutes=5),
    ) == 3
    assert await store.recover_interrupted_attempts(at=at, attempt_ids=[]) == 0
    assert await store.recover_interrupted_attempts(at=at, attempt_ids=[live]) == 1
