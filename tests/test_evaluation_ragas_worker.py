from __future__ import annotations

import asyncio

import pytest

from evaluation.error_policy import ErrorDecision
from evaluation.job_schemas import ClaimedEvaluationWork, RagasAttemptOutput
from evaluation.ragas_worker import RagasBatchWorker


class FakeStore:
    def __init__(self) -> None:
        self.completed: list[tuple[ClaimedEvaluationWork, RagasAttemptOutput]] = []
        self.failed: list[tuple[ClaimedEvaluationWork, ErrorDecision]] = []

    async def complete_ragas_attempt(
        self, claim: ClaimedEvaluationWork, output: RagasAttemptOutput
    ) -> None:
        self.completed.append((claim, output))

    async def fail_attempt(
        self,
        claim: ClaimedEvaluationWork,
        decision: ErrorDecision,
        *,
        next_retry_at=None,
    ) -> None:
        self.failed.append((claim, decision))


def _claim(index: int, *, metric: str = "faithfulness") -> ClaimedEvaluationWork:
    return ClaimedEvaluationWork(
        job_id=f"job-{index}",
        job_item_id=f"item-{index}",
        work_item_id=f"work-{index}",
        attempt_id=f"attempt-{index}",
        input_snapshot={
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

    async def evaluate_metric_batch(self, metric_name, rows, evaluator_llm, evaluator_embeddings):  # noqa: ANN001
        self.calls.append((metric_name, len(rows)))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.asyncio
async def test_completed_metric_checkpoints_survive_later_batch_failure() -> None:
    store = FakeStore()
    evaluator = FakeEvaluator([[0.8, 0.7, 0.9, 0.6], RuntimeError("later batch failed")])
    worker = RagasBatchWorker(store=store, evaluator=evaluator, batch_size=4, parallel_batches=2)

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
    worker = RagasBatchWorker(store=store, evaluator=evaluator, batch_size=4, parallel_batches=2)

    await worker.execute([_claim(i) for i in range(10)])

    assert evaluator.calls == [("faithfulness", 4), ("faithfulness", 4), ("faithfulness", 2)]
    assert len(store.completed) == 10


@pytest.mark.asyncio
async def test_batch_group_key_batches_distinct_result_signatures_and_preserves_identity() -> None:
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

        async def evaluate_metric_batch(self, metric_name, rows, evaluator_llm, evaluator_embeddings):  # noqa: ANN001
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
