"""Durable, checkpointed RAGAS metric batch execution."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from core.llm_usage_context import llm_accounting_phase, llm_accounting_scope
from evaluation.accounting_runtime import (
    EvaluationAccountingSink,
    start_ragas_batch_scope,
)
from evaluation.accounting_schemas import AccountingScopeTarget
from evaluation.accounting_store import EvaluationAccountingStore
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
        accounting_store: EvaluationAccountingStore | None = None,
        price_snapshot: dict[str, Any] | None = None,
        batch_size: int = 4,
        parallel_batches: int = 2,
    ) -> None:
        self._store = store or EvaluationJobStore()
        self._evaluator = evaluator
        self._evaluator_llm = evaluator_llm
        self._evaluator_embeddings = evaluator_embeddings
        self._campaign_repository = campaign_repository
        self._accounting_store = accounting_store
        self._accounting_sink = (
            EvaluationAccountingSink(
                store=accounting_store, price_snapshot=price_snapshot
            )
            if accounting_store is not None
            else None
        )
        self._batch_size = max(1, min(8, batch_size))
        self._parallel_batches = max(1, min(8, parallel_batches))

    async def execute(self, claims: list[ClaimedEvaluationWork]) -> None:
        """Run claims grouped by metric/signature and persist every result."""
        if not claims:
            return

        # Keep older evaluator adapters usable while they migrate to the
        # checkpoint-oriented metric API.  The legacy campaign call owns its
        # own score persistence; the ledger still terminalizes each claimed
        # item only after that call succeeds, so a provider failure remains a
        # failed attempt and is never promoted as a score.
        if not callable(getattr(self._evaluator, "evaluate_metric_batch", None)):
            legacy_evaluate = getattr(self._evaluator, "evaluate_campaign", None)
            if callable(legacy_evaluate):
                await self._execute_legacy_campaigns(claims, legacy_evaluate)
                return
            raise RuntimeError(
                "RAGAS evaluator does not provide a supported evaluation API"
            )

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

        groups: dict[
            tuple[str, str, str | None, int, int], list[ClaimedEvaluationWork]
        ] = {}
        for claim in claims:
            metric_name, signature = self._metric_and_signature(claim)
            batch_group_key = self._batch_group_key(claim, signature)
            batch_size, parallel_batches = self._batch_config(claim)
            campaign_id = str(claim.input_snapshot.get("campaign_id") or "")
            groups.setdefault(
                (
                    campaign_id,
                    metric_name,
                    batch_group_key,
                    batch_size,
                    parallel_batches,
                ),
                [],
            ).append(claim)

        # A single worker-wide semaphore bounds provider calls across all
        # compatibility groups.  Per-group semaphores multiply concurrency
        # when multiple groups are present, so the aggregate cap stays at two
        # calls regardless of grouping.
        global_semaphore = asyncio.Semaphore(2)

        async def run_chunk(
            campaign_id: str,
            metric_name: str,
            batch_group_key: str | None,
            chunk: list[ClaimedEvaluationWork],
        ) -> None:
            async with global_semaphore:
                await self._execute_chunk(
                    campaign_id, metric_name, batch_group_key, chunk, invocation_id
                )

        tasks: list[asyncio.Task[None]] = []
        invocation_id = str(uuid4())
        for (
            campaign_id,
            metric_name,
            batch_group_key,
            batch_size,
            parallel_batches,
        ), group in groups.items():
            for offset in range(0, len(group), batch_size):
                tasks.append(
                    asyncio.create_task(
                        run_chunk(
                            campaign_id,
                            metric_name,
                            batch_group_key,
                            group[offset : offset + batch_size],
                        )
                    )
                )
        if tasks:
            await asyncio.gather(*tasks)

    async def _execute_legacy_campaigns(
        self,
        claims: list[ClaimedEvaluationWork],
        legacy_evaluate: Any,
    ) -> None:
        groups: dict[tuple[str, str, str], list[ClaimedEvaluationWork]] = {}
        for claim in claims:
            snapshot = claim.input_snapshot
            user_id = str(snapshot.get("user_id") or "")
            campaign_id = str(snapshot.get("campaign_id") or "")
            groups.setdefault((user_id, campaign_id, claim.job_id), []).append(claim)

        for (user_id, campaign_id, _job_id), group in groups.items():
            first = group[0].input_snapshot
            selected_id_values: list[str] = []
            for claim in group:
                snapshot = claim.input_snapshot
                value = (
                    snapshot.get("campaign_result_id")
                    or snapshot.get("result_id")
                    or snapshot.get("result", {}).get("id")
                )
                if value:
                    selected_id_values.append(str(value))
            selected_ids = list(dict.fromkeys(selected_id_values))
            try:
                await run_with_retry(
                    legacy_evaluate,
                    user_id=user_id,
                    campaign_id=campaign_id,
                    ragas_batch_size=first.get("ragas_batch_size"),
                    ragas_parallel_batches=first.get("ragas_parallel_batches"),
                    ragas_rpm_limit=first.get("ragas_rpm_limit"),
                    selected_result_ids=selected_ids or None,
                )
                for claim in group:
                    await self._store.complete_ragas_attempt(
                        claim, RagasAttemptOutput(scores=[])
                    )
            except asyncio.CancelledError:
                await self._cancel_claims(group)
                await self._derive_campaign_state(group)
                raise
            except Exception as exc:  # noqa: BLE001
                decision = classify_evaluation_error(exc)
                await asyncio.gather(
                    *(self._fail_claim(claim, decision) for claim in group)
                )
            await self._derive_campaign_state(group)

    async def _execute_chunk(
        self,
        campaign_id: str,
        metric_name: str,
        batch_group_key: str | None,
        claims: list[ClaimedEvaluationWork],
        invocation_id: str,
    ) -> None:
        rows = [self._row_for_claim(claim) for claim in claims]
        scope = None
        try:
            if self._accounting_store is not None:
                scope = await start_ragas_batch_scope(
                    store=self._accounting_store,
                    sink=self._accounting_sink,
                    campaign_id=campaign_id,
                    metric_name=metric_name,
                    scope_key=self._ragas_scope_key(
                        campaign_id, metric_name, batch_group_key, claims, invocation_id
                    ),
                    targets=[
                        AccountingScopeTarget(
                            campaign_result_id=self._result_id(claim),
                            job_id=claim.job_id,
                            work_item_id=claim.work_item_id,
                            attempt_id=claim.attempt_id,
                            metric_name=metric_name,
                        )
                        for claim in claims
                    ],
                )
            if scope is None:
                values = await run_with_retry(
                    self._evaluator.evaluate_metric_batch,
                    metric_name,
                    rows,
                    self._evaluator_llm,
                    self._evaluator_embeddings,
                )
            else:
                with (
                    llm_accounting_scope(scope.context),
                    llm_accounting_phase("ragas_scoring"),
                ):
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
            normalized = [
                self._score_value(
                    value,
                    allow_none=bool(
                        getattr(self._evaluator, "allows_missing_metric_values", False)
                    ),
                )
                for value in values
            ]
        except asyncio.CancelledError:
            if scope is not None:
                await self._accounting_store.finalize_scope(scope.scope_id, "cancelled")
            await self._cancel_claims(claims)
            await self._derive_campaign_state(claims)
            raise
        except Exception as exc:  # noqa: BLE001
            if scope is not None:
                await self._accounting_store.finalize_scope(scope.scope_id, "failed")
            decision = classify_evaluation_error(exc)
            await asyncio.gather(
                *(self._fail_claim(claim, decision) for claim in claims)
            )
            await self._derive_campaign_state(claims)
            return

        # Promote each value separately.  This is intentionally outside one
        # transaction: a later promotion failure must not roll back checkpoints.
        try:
            for claim, value in zip(claims, normalized, strict=True):
                if value is None:
                    promoted_score_count = await self._store.complete_ragas_attempt(
                        claim, RagasAttemptOutput(scores=[])
                    )
                else:
                    score = {
                        "campaign_result_id": self._result_id(claim),
                        "metric_name": metric_name,
                        "metric_value": value,
                        "evaluation_signature": claim.input_snapshot.get(
                            "evaluation_signature"
                        ),
                        "details": {
                            "evaluator_model": getattr(
                                self._evaluator, "evaluator_model", None
                            ),
                            "question_id": self._row_for_claim(claim).question_id,
                            "invalid_metric": False,
                            "batch_group_key": batch_group_key,
                        },
                    }
                    promoted_score_count = await self._store.complete_ragas_attempt(
                        claim, RagasAttemptOutput(scores=[score])
                    )
                if scope is not None and self._did_promote_scores(
                    promoted_score_count, score_count=0 if value is None else 1
                ):
                    await self._accounting_store.mark_targets_official(
                        scope.scope_id, {claim.attempt_id: self._result_id(claim)}
                    )
            if scope is not None:
                await self._accounting_store.finalize_scope(scope.scope_id, "completed")
        except Exception as exc:  # noqa: BLE001
            if scope is not None:
                await self._accounting_store.finalize_scope(scope.scope_id, "failed")
            decision = classify_evaluation_error(exc)
            for claim in claims:
                await self._fail_claim(claim, decision)
        except asyncio.CancelledError:
            if scope is not None:
                await self._accounting_store.finalize_scope(scope.scope_id, "cancelled")
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
            )
            for claim in claims
            if claim.input_snapshot.get("user_id")
            and claim.input_snapshot.get("campaign_id")
        }
        for user_id, campaign_id in campaigns:
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
    def _did_promote_scores(result: object, *, score_count: int) -> bool:
        """Interpret new promotion counts while tolerating legacy store fakes."""
        if result is None:
            return score_count > 0
        return isinstance(result, int) and not isinstance(result, bool) and result > 0

    @staticmethod
    def _ragas_scope_key(
        campaign_id: str,
        metric_name: str,
        batch_group_key: str | None,
        claims: list[ClaimedEvaluationWork],
        invocation_id: str,
    ) -> str:
        attempt_ids = ",".join(sorted(claim.attempt_id for claim in claims))
        return "|".join(
            (
                campaign_id,
                metric_name,
                batch_group_key or "",
                attempt_ids,
                invocation_id,
            )
        )

    @staticmethod
    def _score_value(value: Any, *, allow_none: bool = False) -> float | None:
        if value is None:
            if allow_none:
                return None
            raise ValueError("RAGAS returned a missing metric value")
        if isinstance(value, bool):
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
        return str(
            snapshot.get("campaign_result_id")
            or snapshot.get("result_id")
            or snapshot.get("result", {}).get("id")
        )

    @classmethod
    def _row_for_claim(cls, claim: ClaimedEvaluationWork) -> Any:
        snapshot = claim.input_snapshot
        raw = (
            snapshot.get("result")
            or snapshot.get("result_snapshot")
            or snapshot.get("row")
        )
        if isinstance(raw, Mapping):
            payload = dict(raw)
        else:
            payload = {}
        payload.setdefault("id", cls._result_id(claim))
        payload.setdefault(
            "question_id", str(snapshot.get("question_id") or payload["id"])
        )
        return SimpleNamespace(**payload)
