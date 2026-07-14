"""Durable, checkpointed RAGAS metric batch execution."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from typing import Any

from evaluation.error_policy import classify_evaluation_error
from evaluation.job_schemas import ClaimedEvaluationWork, RagasAttemptOutput
from evaluation.job_store import EvaluationJobStore
from evaluation.retry import run_with_retry


class RagasBatchWorker:
    """Evaluate compatible metric claims in bounded batches.

    A metric call is the only unit sent to the provider.  Each successful value
    is checkpointed through the ledger independently, so an interruption or a
    later provider failure cannot erase earlier metric values.
    """

    def __init__(
        self,
        *,
        store: EvaluationJobStore | Any | None = None,
        evaluator: Any,
        evaluator_llm: Any = None,
        evaluator_embeddings: Any = None,
        campaign_repository: Any | None = None,
        batch_size: int = 4,
        parallel_batches: int = 2,
    ) -> None:
        self._store = store or EvaluationJobStore()
        self._evaluator = evaluator
        self._evaluator_llm = evaluator_llm
        self._evaluator_embeddings = evaluator_embeddings
        self._campaign_repository = campaign_repository
        self._batch_size = max(1, min(8, batch_size))
        self._parallel_batches = max(1, min(8, parallel_batches))

    async def execute(self, claims: list[ClaimedEvaluationWork]) -> None:
        """Run claims grouped by metric/signature and persist every result."""
        if not claims:
            return

        if self._evaluator_llm is None and self._evaluator_embeddings is None:
            prepare = getattr(self._evaluator, "evaluator_handles", None)
            if callable(prepare):
                try:
                    self._evaluator_llm, self._evaluator_embeddings = await prepare()
                except asyncio.CancelledError:
                    await self._cancel_claims(claims)
                    await self._derive_campaign_state(claims)
                    raise
                except Exception as exc:  # noqa: BLE001
                    decision = classify_evaluation_error(exc)
                    await asyncio.gather(
                        *(self._fail_claim(claim, decision) for claim in claims)
                    )
                    await self._derive_campaign_state(claims)
                    return

        groups: dict[tuple[str, str | None, int, int], list[ClaimedEvaluationWork]] = {}
        for claim in claims:
            metric_name, signature = self._metric_and_signature(claim)
            batch_group_key = self._batch_group_key(claim, signature)
            batch_size, parallel_batches = self._batch_config(claim)
            groups.setdefault(
                (metric_name, batch_group_key, batch_size, parallel_batches), []
            ).append(claim)

        # A single worker-wide semaphore bounds provider calls across all
        # compatibility groups.  Per-group semaphores multiply concurrency
        # when multiple groups are present, so the aggregate cap stays at two
        # calls regardless of grouping.
        global_semaphore = asyncio.Semaphore(2)

        async def run_chunk(
            metric_name: str,
            batch_group_key: str | None,
            chunk: list[ClaimedEvaluationWork],
        ) -> None:
            async with global_semaphore:
                await self._execute_chunk(metric_name, batch_group_key, chunk)

        tasks: list[asyncio.Task[None]] = []
        for (metric_name, batch_group_key, batch_size, parallel_batches), group in groups.items():
            for offset in range(0, len(group), batch_size):
                tasks.append(
                    asyncio.create_task(
                        run_chunk(
                            metric_name,
                            batch_group_key,
                            group[offset : offset + batch_size],
                        )
                    )
                )
        if tasks:
            await asyncio.gather(*tasks)

    async def _execute_chunk(
        self,
        metric_name: str,
        batch_group_key: str | None,
        claims: list[ClaimedEvaluationWork],
    ) -> None:
        rows = [self._row_for_claim(claim) for claim in claims]
        try:
            values = await run_with_retry(
                self._evaluator.evaluate_metric_batch,
                metric_name,
                rows,
                self._evaluator_llm,
                self._evaluator_embeddings,
            )
            if not isinstance(values, list) or len(values) != len(claims):
                raise ValueError(
                    f"RAGAS metric {metric_name!r} returned {len(values) if isinstance(values, list) else 'non-list'} values for {len(claims)} rows"
                )
            normalized = [self._score_value(value) for value in values]
        except asyncio.CancelledError:
            await self._cancel_claims(claims)
            await self._derive_campaign_state(claims)
            raise
        except Exception as exc:  # noqa: BLE001
            decision = classify_evaluation_error(exc)
            await asyncio.gather(
                *(
                    self._fail_claim(claim, decision)
                    for claim in claims
                )
            )
            await self._derive_campaign_state(claims)
            return

        # Promote each value separately.  This is intentionally outside one
        # transaction: a later promotion failure must not roll back checkpoints.
        try:
            for claim, value in zip(claims, normalized, strict=True):
                score = {
                    "campaign_result_id": self._result_id(claim),
                    "metric_name": metric_name,
                    "metric_value": value,
                    "evaluation_signature": claim.input_snapshot.get(
                        "evaluation_signature"
                    ),
                    "details": {
                        "evaluator_model": getattr(self._evaluator, "evaluator_model", None),
                        "question_id": self._row_for_claim(claim).question_id,
                        "invalid_metric": False,
                        "batch_group_key": batch_group_key,
                    },
                }
                try:
                    await self._store.complete_ragas_attempt(
                        claim, RagasAttemptOutput(scores=[score])
                    )
                except Exception as exc:  # noqa: BLE001
                    await self._fail_claim(claim, classify_evaluation_error(exc))
        except asyncio.CancelledError:
            await self._cancel_claims(claims)
            await self._derive_campaign_state(claims)
            raise
        await self._derive_campaign_state(claims)

    async def _derive_campaign_state(self, claims: list[ClaimedEvaluationWork]) -> None:
        if self._campaign_repository is None or not claims:
            return
        campaigns = {
            (
                str(claim.input_snapshot.get("user_id")),
                str(claim.input_snapshot.get("campaign_id")),
                claim.job_id,
            )
            for claim in claims
            if claim.input_snapshot.get("user_id") and claim.input_snapshot.get("campaign_id")
        }
        for user_id, campaign_id, job_id in campaigns:
            try:
                await self._campaign_repository.derive_ragas_state(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    job_id=job_id,
                )
            except TypeError:
                # Keep compatibility with lightweight repository doubles and
                # older adapters that have not added the optional job filter.
                await self._campaign_repository.derive_ragas_state(
                    user_id=user_id,
                    campaign_id=campaign_id,
                )

    async def _cancel_claims(self, claims: list[ClaimedEvaluationWork]) -> None:
        cancel = getattr(self._store, "cancel_attempt", None)
        if not callable(cancel):
            return
        await asyncio.gather(
            *(
                cancel(claim, safe_message="Evaluation batch was cancelled.")
                for claim in claims
            ),
            return_exceptions=True,
        )

    async def _fail_claim(self, claim: ClaimedEvaluationWork, decision: Any) -> None:
        try:
            await self._store.fail_attempt(claim, decision, next_retry_at=None)
        except ValueError:
            # Cancellation/recovery may have already terminalized the claim.
            return

    @staticmethod
    def _score_value(value: Any) -> float:
        if isinstance(value, bool) or value is None:
            raise ValueError("RAGAS returned a missing metric value")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("RAGAS returned a non-numeric metric value") from exc
        if not math.isfinite(normalized):
            raise ValueError("RAGAS returned a non-finite metric value")
        return normalized

    @staticmethod
    def _metric_and_signature(
        claim: ClaimedEvaluationWork,
    ) -> tuple[str, str | None]:
        snapshot = claim.input_snapshot
        metric = snapshot.get("metric_name")
        signature = snapshot.get("evaluation_signature")
        if not metric:
            parts = (claim.logical_key or "").split(":")
            metric = parts[2] if len(parts) > 2 else "faithfulness"
            if signature is None and len(parts) > 3:
                signature = parts[3]
        return str(metric), str(signature) if signature is not None else None

    @staticmethod
    def _batch_group_key(
        claim: ClaimedEvaluationWork, signature: str | None
    ) -> str | None:
        raw = claim.input_snapshot.get("batch_group_key")
        return str(raw) if raw is not None else signature

    def _batch_config(self, claim: ClaimedEvaluationWork) -> tuple[int, int]:
        snapshot = claim.input_snapshot
        try:
            batch_size = int(snapshot.get("ragas_batch_size") or self._batch_size)
        except (TypeError, ValueError):
            batch_size = self._batch_size
        try:
            parallel_batches = int(
                snapshot.get("ragas_parallel_batches") or self._parallel_batches
            )
        except (TypeError, ValueError):
            parallel_batches = self._parallel_batches
        return max(1, min(4, batch_size)), max(1, min(2, parallel_batches))

    @staticmethod
    def _result_id(claim: ClaimedEvaluationWork) -> str:
        snapshot = claim.input_snapshot
        return str(snapshot.get("campaign_result_id") or snapshot.get("result_id") or snapshot.get("result", {}).get("id"))

    @classmethod
    def _row_for_claim(cls, claim: ClaimedEvaluationWork) -> Any:
        snapshot = claim.input_snapshot
        raw = snapshot.get("result") or snapshot.get("result_snapshot") or snapshot.get("row")
        if isinstance(raw, dict):
            payload = dict(raw)
        else:
            payload = {}
        payload.setdefault("id", cls._result_id(claim))
        payload.setdefault("question_id", str(snapshot.get("question_id") or payload["id"]))
        return SimpleNamespace(**payload)
