from __future__ import annotations

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
