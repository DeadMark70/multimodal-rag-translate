from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from shutil import rmtree

import pytest
import pytest_asyncio

from core.llm_usage_context import emit_direct_usage
import evaluation.db as evaluation_db
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.error_policy import ErrorDecision
from evaluation.job_schemas import ClaimedEvaluationWork, RagasAttemptOutput
from evaluation.ragas_worker import RagasBatchWorker
from tenacity import wait_none


class FakeStore:
    def __init__(self) -> None:
        self.completed: list[tuple[ClaimedEvaluationWork, RagasAttemptOutput]] = []
        self.failed: list[tuple[ClaimedEvaluationWork, ErrorDecision]] = []

    async def complete_ragas_attempt(
        self, claim: ClaimedEvaluationWork, output: RagasAttemptOutput
    ) -> int:
        self.completed.append((claim, output))
        return len(output.scores)

    async def fail_attempt(
        self,
        claim: ClaimedEvaluationWork,
        decision: ErrorDecision,
        *,
        next_retry_at=None,
    ) -> None:
        self.failed.append((claim, decision))


TEST_PRICE_SNAPSHOT = {
    "snapshot_id": "ragas-test-v1",
    "currency": "USD",
    "usd_to_twd": None,
    "models": {
        "test-evaluator": {
            "input_per_1m_usd": 500.0,
            "output_per_1m_usd": 1000.0,
            "reasoning_per_1m_usd": 0.0,
        }
    },
}


@pytest_asyncio.fixture
async def accounting_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> EvaluationAccountingStore:
    database_path = tmp_path / "evaluation.db"
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", database_path)
    await evaluation_db.force_init_db()
    async with evaluation_db.connect_db() as connection:
        now = datetime.now(UTC).isoformat()
        await connection.execute(
            """INSERT INTO campaigns (id, user_id, name, status, config_json, created_at, updated_at)
               VALUES ('campaign-1', 'user-1', NULL, 'pending', '{}', ?, ?)""",
            (now, now),
        )
        await connection.execute(
            """INSERT INTO campaigns (id, user_id, name, status, config_json, created_at, updated_at)
               VALUES ('campaign-2', 'user-1', NULL, 'pending', '{}', ?, ?)""",
            (now, now),
        )
        for index in range(3):
            await connection.execute(
                """INSERT INTO campaign_results (
                       id, campaign_id, user_id, question_id, question, ground_truth,
                       mode, run_number, answer, contexts_json, source_doc_ids_json,
                       expected_sources_json, token_usage_json, status, created_at
                   ) VALUES (?, 'campaign-1', 'user-1', ?, 'Question', 'Ground truth',
                             'naive', ?, 'Answer', '[]', '[]', '[]', '{}', 'completed', ?)""",
                (f"result-{index}", f"Q-{index}", index, now),
            )
        await connection.commit()
    try:
        yield EvaluationAccountingStore()
    finally:
        for path in (
            database_path,
            database_path.with_suffix(".db-shm"),
            database_path.with_suffix(".db-wal"),
        ):
            path.unlink(missing_ok=True)
        rmtree(database_path.parent, ignore_errors=True)


def _claim(index: int, *, metric: str = "faithfulness") -> ClaimedEvaluationWork:
    return ClaimedEvaluationWork(
        job_id=f"job-{index}",
        job_item_id=f"item-{index}",
        work_item_id=f"work-{index}",
        attempt_id=f"attempt-{index}",
        input_snapshot={
            "user_id": "user-1",
            "campaign_id": "campaign-1",
            "campaign_result_id": f"result-{index}",
            "metric_name": metric,
            "evaluation_signature": "signature-v1",
            "result": {
                "id": f"result-{index}",
                "question_id": f"Q-{index}",
                "question": "Question",
                "answer": "Answer",
                "contexts": ["context"],
                "ground_truth": "Ground truth",
                "ground_truth_short": "Short ground truth",
                "context_policy_version": "v1",
            },
        },
        logical_key=f"ragas:result-{index}:{metric}:signature-v1",
        work_type="ragas_metric",
    )


class FakeEvaluator:
    def __init__(self, results: list[object]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    async def evaluate_metric_batch(
        self, metric_name, rows, evaluator_llm, evaluator_embeddings
    ):  # noqa: ANN001
        self.calls.append((metric_name, len(rows)))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class AccountingFakeEvaluator(FakeEvaluator):
    async def evaluate_metric_batch(
        self, metric_name, rows, evaluator_llm, evaluator_embeddings
    ):  # noqa: ANN001
        failed = isinstance(self.results[0], BaseException)
        await emit_direct_usage(
            purpose="ragas_evaluator",
            provider="google",
            model_name="test-evaluator",
            raw_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            status="failed" if failed else "completed",
        )
        return await super().evaluate_metric_batch(
            metric_name, rows, evaluator_llm, evaluator_embeddings
        )


class PromotionStore(FakeStore):
    def __init__(self, promoted_counts: list[int]) -> None:
        super().__init__()
        self.promoted_counts = promoted_counts

    async def complete_ragas_attempt(
        self, claim: ClaimedEvaluationWork, output: RagasAttemptOutput
    ) -> int:
        await super().complete_ragas_attempt(claim, output)
        return self.promoted_counts.pop(0)


class UnconfirmedPromotionStore(FakeStore):
    async def complete_ragas_attempt(
        self, claim: ClaimedEvaluationWork, output: RagasAttemptOutput
    ) -> None:
        await super().complete_ragas_attempt(claim, output)


@pytest.mark.asyncio
async def test_ragas_batch_records_one_shared_scope_for_all_claims(
    accounting_store: EvaluationAccountingStore,
) -> None:
    evaluator = AccountingFakeEvaluator(results=[[0.8, 0.7, 0.9]])
    worker = RagasBatchWorker(
        store=FakeStore(),
        evaluator=evaluator,
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )

    await worker.execute([_claim(index) for index in range(3)])

    scopes = await accounting_store.list_campaign_scopes("campaign-1")
    ragas_scopes = [scope for scope in scopes if scope.scope_type == "ragas_batch"]
    assert len(ragas_scopes) == 1
    assert len(ragas_scopes[0].targets) == 3
    assert ragas_scopes[0].metric_name == "faithfulness"
    assert ragas_scopes[0].status == "completed"
    assert all(target.is_official for target in ragas_scopes[0].targets)
    events = await accounting_store.list_campaign_events("campaign-1")
    assert [event.run_id for event in events] == [None]
    assert {target.campaign_result_id for target in ragas_scopes[0].targets} == {
        "result-0",
        "result-1",
        "result-2",
    }


@pytest.mark.asyncio
async def test_ragas_retry_cost_stays_operational_overhead(
    accounting_store: EvaluationAccountingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from google.api_core.exceptions import ServiceUnavailable

    monkeypatch.setattr("evaluation.retry.wait_exponential", lambda **_: wait_none())
    evaluator = AccountingFakeEvaluator(results=[ServiceUnavailable("retry"), [0.8]])
    worker = RagasBatchWorker(
        store=FakeStore(),
        evaluator=evaluator,
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )

    await worker.execute([_claim(0)])

    events = await accounting_store.list_campaign_events("campaign-1")
    assert len(events) == 2
    assert [event.status for event in events] == ["failed", "success"]
    assert sum(event.estimated_cost_usd or 0 for event in events) == pytest.approx(0.02)
    assert all(
        event.phase == "ragas_scoring" and event.run_id is None for event in events
    )
    scope = (await accounting_store.list_campaign_scopes("campaign-1"))[0]
    assert scope.retry_count == 1


@pytest.mark.asyncio
async def test_exhausted_ragas_retries_persist_only_scheduled_retries(
    accounting_store: EvaluationAccountingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from google.api_core.exceptions import ServiceUnavailable

    monkeypatch.setattr("evaluation.retry.wait_exponential", lambda **_: wait_none())
    worker = RagasBatchWorker(
        store=FakeStore(),
        evaluator=AccountingFakeEvaluator(results=[ServiceUnavailable("retry")] * 5),
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )

    await worker.execute([_claim(0)])

    scope = (await accounting_store.list_campaign_scopes("campaign-1"))[0]
    assert scope.status == "failed"
    assert scope.retry_count == 4


@pytest.mark.asyncio
async def test_cancelled_ragas_batch_retains_events_without_official_targets(
    accounting_store: EvaluationAccountingStore,
) -> None:
    evaluator = AccountingFakeEvaluator(results=[asyncio.CancelledError()])
    worker = RagasBatchWorker(
        store=FakeStore(),
        evaluator=evaluator,
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )

    with pytest.raises(asyncio.CancelledError):
        await worker.execute([_claim(0)])

    scope = (await accounting_store.list_campaign_scopes("campaign-1"))[0]
    assert scope.status == "cancelled"
    assert not any(target.is_official for target in scope.targets)
    assert len(await accounting_store.list_campaign_events("campaign-1")) == 1


@pytest.mark.asyncio
async def test_ragas_only_marks_targets_with_promoted_scores(
    accounting_store: EvaluationAccountingStore,
) -> None:
    store = PromotionStore([1, 0])
    worker = RagasBatchWorker(
        store=store,
        evaluator=AccountingFakeEvaluator(results=[[0.8, 0.7]]),
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )

    await worker.execute([_claim(0), _claim(1)])

    targets = (await accounting_store.list_campaign_scopes("campaign-1"))[0].targets
    assert [target.is_official for target in targets] == [True, False]


@pytest.mark.asyncio
async def test_ragas_missing_scores_leave_targets_unofficial(
    accounting_store: EvaluationAccountingStore,
) -> None:
    evaluator = AccountingFakeEvaluator(results=[[None]])
    evaluator.allows_missing_metric_values = True
    worker = RagasBatchWorker(
        store=PromotionStore([0]),
        evaluator=evaluator,
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )

    await worker.execute([_claim(0)])

    assert (
        not (await accounting_store.list_campaign_scopes("campaign-1"))[0]
        .targets[0]
        .is_official
    )


@pytest.mark.asyncio
async def test_ragas_unconfirmed_promotion_leaves_target_unofficial(
    accounting_store: EvaluationAccountingStore,
) -> None:
    worker = RagasBatchWorker(
        store=UnconfirmedPromotionStore(),
        evaluator=AccountingFakeEvaluator(results=[[0.8]]),
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )

    await worker.execute([_claim(0)])

    assert (
        not (await accounting_store.list_campaign_scopes("campaign-1"))[0]
        .targets[0]
        .is_official
    )


@pytest.mark.asyncio
async def test_ragas_batches_do_not_share_accounting_scopes_across_campaigns(
    accounting_store: EvaluationAccountingStore,
) -> None:
    base = _claim(1)
    second = base.model_copy(
        update={
            "input_snapshot": dict(
                base.model_dump(mode="json")["input_snapshot"],
                campaign_id="campaign-2",
            )
        }
    )
    worker = RagasBatchWorker(
        store=PromotionStore([1, 1]),
        evaluator=AccountingFakeEvaluator(results=[[0.8], [0.7]]),
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )

    await worker.execute([_claim(0), second])

    assert len(await accounting_store.list_campaign_scopes("campaign-1")) == 1
    assert len(await accounting_store.list_campaign_scopes("campaign-2")) == 1


@pytest.mark.asyncio
async def test_completed_metric_checkpoints_survive_later_batch_failure() -> None:
    store = FakeStore()
    evaluator = FakeEvaluator(
        [[0.8, 0.7, 0.9, 0.6], RuntimeError("later batch failed")]
    )
    worker = RagasBatchWorker(
        store=store, evaluator=evaluator, batch_size=4, parallel_batches=2
    )

    await worker.execute([_claim(i) for i in range(4)])
    await worker.execute([_claim(4)])

    assert len(store.completed) == 4
    assert len(store.failed) == 1
    assert store.failed[-1][1].error_type == "unknown"


@pytest.mark.asyncio
async def test_missing_dependency_is_failure_not_completed_empty_scores() -> None:
    store = FakeStore()
    evaluator = FakeEvaluator([ModuleNotFoundError("ragas")])
    worker = RagasBatchWorker(store=store, evaluator=evaluator)

    await worker.execute([_claim(1)])

    assert store.completed == []
    assert len(store.failed) == 1
    assert store.failed[0][1].error_type == "missing_dependency"


@pytest.mark.asyncio
async def test_missing_metric_value_is_failure_not_zero_score() -> None:
    store = FakeStore()
    evaluator = FakeEvaluator([[None]])
    worker = RagasBatchWorker(store=store, evaluator=evaluator)

    await worker.execute([_claim(1)])

    assert store.completed == []
    assert len(store.failed) == 1
    assert store.failed[0][1].error_type == "invalid_configuration"


@pytest.mark.asyncio
async def test_compatible_claims_are_chunked_and_run_with_bounded_parallelism() -> None:
    store = FakeStore()
    evaluator = FakeEvaluator([[0.5] * 4, [0.6] * 4, [0.7] * 2])
    worker = RagasBatchWorker(
        store=store, evaluator=evaluator, batch_size=4, parallel_batches=2
    )

    await worker.execute([_claim(i) for i in range(10)])

    assert evaluator.calls == [
        ("faithfulness", 4),
        ("faithfulness", 4),
        ("faithfulness", 2),
    ]
    assert len(store.completed) == 10


@pytest.mark.asyncio
async def test_batch_group_key_batches_distinct_result_signatures_and_preserves_identity() -> (
    None
):
    store = FakeStore()
    evaluator = FakeEvaluator([[0.5] * 4])
    worker = RagasBatchWorker(store=store, evaluator=evaluator, batch_size=4)
    claims = []
    for index in range(4):
        base = _claim(index)
        snapshot = base.model_dump(mode="json")["input_snapshot"]
        claim = base.model_copy(
            update={
                "input_snapshot": dict(
                    snapshot,
                    evaluation_signature=f"result-signature-{index}",
                    batch_group_key="shared-batch-key",
                ),
            }
        )
        claims.append(claim)

    await worker.execute(claims)

    assert evaluator.calls == [("faithfulness", 4)]
    signatures = {
        output.scores[0]["evaluation_signature"] for _, output in store.completed
    }
    assert signatures == {f"result-signature-{index}" for index in range(4)}


@pytest.mark.asyncio
async def test_provider_parallelism_is_bounded_across_compatible_groups() -> None:
    store = FakeStore()

    class ConcurrentEvaluator:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def evaluate_metric_batch(
            self, metric_name, rows, evaluator_llm, evaluator_embeddings
        ):  # noqa: ANN001
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return [0.5] * len(rows)

    evaluator = ConcurrentEvaluator()
    worker = RagasBatchWorker(
        store=store, evaluator=evaluator, batch_size=4, parallel_batches=2
    )
    claims = []
    for group in ("batch-group-a", "batch-group-b"):
        for index in range(8):
            base = _claim(index + (0 if group.endswith("a") else 100))
            snapshot = base.model_dump(mode="json")["input_snapshot"]
            claims.append(
                base.model_copy(
                    update={
                        "input_snapshot": dict(snapshot, batch_group_key=group),
                    }
                )
            )

    await worker.execute(claims)

    assert evaluator.max_active == 2
    assert len(store.completed) == len(claims)
