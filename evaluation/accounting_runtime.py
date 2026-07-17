"""Runtime bridge from scoped provider callbacks to durable accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from core.llm_usage_context import LlmAccountingContext, RawLlmUsageEvent
from evaluation.accounting_schemas import (
    AccountingScopeStart,
    AccountingScopeTarget,
    UsageEventCreate,
)
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.token_cost import load_price_snapshot, price_normalized_usage
from evaluation.token_normalizers import normalize_provider_usage


class EvaluationAccountingSink:
    """Normalize and persist callback observations for one worker lifecycle."""

    def __init__(
        self,
        *,
        store: EvaluationAccountingStore,
        price_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self._store = store
        self._price_snapshot = (
            dict(price_snapshot)
            if price_snapshot is not None
            else load_price_snapshot()
        )

    async def record(self, raw: RawLlmUsageEvent) -> None:
        """Persist one normalized callback event using its callback ID as the key."""
        usage = normalize_provider_usage(raw.provider, raw.raw_usage)
        pricing = price_normalized_usage(raw.model_name, usage, self._price_snapshot)
        await self._store.record_event(
            UsageEventCreate(
                usage_event_id=raw.usage_event_id,
                scope_id=raw.scope_id,
                campaign_id=raw.campaign_id,
                scope_type=raw.scope_type,
                scope_key=raw.scope_key,
                run_id=raw.run_id,
                provider_run_id=raw.provider_run_id,
                phase=raw.phase,
                purpose=raw.purpose,
                metric_name=raw.metric_name,
                provider=raw.provider,
                model_name=raw.model_name,
                input_tokens=usage.input_tokens,
                output_text_tokens=usage.output_text_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                other_tokens=usage.other_tokens,
                reported_total_tokens=usage.reported_total_tokens,
                raw_usage=raw.raw_usage,
                usage_status=usage.usage_status,
                reconciliation_status=usage.reconciliation_status,
                estimated_cost_usd=pricing["estimated_cost_usd"],
                estimated_cost_twd=pricing["estimated_cost_twd"],
                pricing_status=pricing["pricing_status"],
                price_snapshot_id=pricing["price_snapshot_id"],
                latency_ms=raw.latency_ms,
                status="failed" if raw.status == "failed" else "success",
                error=raw.error,
                created_at=raw.created_at,
            )
        )


@dataclass(frozen=True)
class ExecutionAccountingSession:
    """One durable accounting scope and its task-local callback context."""

    scope_id: str
    context: LlmAccountingContext


async def start_execution_scope(
    *,
    store: EvaluationAccountingStore,
    campaign_id: str,
    run_id: str,
    job_id: str,
    work_item_id: str,
    attempt_id: str,
    sink: EvaluationAccountingSink | None = None,
    price_snapshot: dict[str, Any] | None = None,
) -> ExecutionAccountingSession:
    """Start the sole accounting scope owned by one durable execution attempt."""
    scope_id = str(uuid4())
    accounting_sink = sink or EvaluationAccountingSink(
        store=store, price_snapshot=price_snapshot
    )
    await store.start_scope(
        AccountingScopeStart(
            scope_id=scope_id,
            campaign_id=campaign_id,
            scope_type="execution_run",
            scope_key=run_id,
            run_id=run_id,
            targets=[
                AccountingScopeTarget(
                    job_id=job_id,
                    work_item_id=work_item_id,
                    attempt_id=attempt_id,
                )
            ],
        )
    )
    return ExecutionAccountingSession(
        scope_id=scope_id,
        context=LlmAccountingContext(
            scope_id=scope_id,
            campaign_id=campaign_id,
            scope_type="execution_run",
            scope_key=run_id,
            run_id=run_id,
            metric_name=None,
            sink=accounting_sink,
        ),
    )


async def start_ragas_batch_scope(
    *,
    store: EvaluationAccountingStore,
    campaign_id: str,
    metric_name: str,
    scope_key: str,
    targets: list[AccountingScopeTarget],
    sink: EvaluationAccountingSink | None = None,
    price_snapshot: dict[str, Any] | None = None,
) -> ExecutionAccountingSession:
    """Start one shared, campaign-level RAGAS batch accounting scope."""
    scope_id = str(uuid4())
    accounting_sink = sink or EvaluationAccountingSink(
        store=store, price_snapshot=price_snapshot
    )
    await store.start_scope(
        AccountingScopeStart(
            scope_id=scope_id,
            campaign_id=campaign_id,
            scope_type="ragas_batch",
            scope_key=scope_key,
            metric_name=metric_name,
            targets=targets,
        )
    )
    return ExecutionAccountingSession(
        scope_id=scope_id,
        context=LlmAccountingContext(
            scope_id=scope_id,
            campaign_id=campaign_id,
            scope_type="ragas_batch",
            scope_key=scope_key,
            run_id=None,
            metric_name=metric_name,
            sink=accounting_sink,
        ),
    )
