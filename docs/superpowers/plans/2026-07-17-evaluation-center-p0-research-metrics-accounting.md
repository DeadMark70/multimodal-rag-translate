# Evaluation Center P0 Research Metrics and Accounting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Evaluation Center proxy metrics and synthetic performance charts with official RAGAS quality, complete per-call inference accounting, separate RAGAS overhead, measured per-mode percentiles, and strict comparability states.

**Architecture:** A task-local LangChain callback records every evaluation LLM call into version-2 accounting scopes and usage events. Durable execution and RAGAS workers own scope lifecycle and attempt linkage; a new typed research-summary service joins official RAGAS scores, official attempts, latency, tokens, and pricing. The React overview consumes only that strict contract and renders missing or incompatible data as explicit states.

**Tech Stack:** Python 3.13, FastAPI, Pydantic 2, aiosqlite/SQLite WAL, LangChain Core 1.2.7, langchain-google-genai 4.1.1, pytest/pytest-asyncio, React 18, TypeScript 5.9, Chakra UI 2, Vitest 4.

## Global Constraints

- Missing, failed, partial, legacy, or unpriced data must never become synthetic zero.
- Official correctness, faithfulness, and relevance values come only from compatible RAGAS scores.
- Inference benchmark cost uses only official successful execution attempts.
- Operational execution cost includes retries and failed attempts.
- RAGAS evaluator usage is campaign-level overhead and is never divided across target runs.
- Historical campaigns are not backfilled; absence of a version-2 official scope means `incomplete_legacy`.
- Token categories are non-overlapping and must reconcile to provider-reported total before status is `complete`.
- Normal chat has no accounting scope and must create no evaluation accounting rows.
- Main-dashboard mappers may format API values but may not derive quality, percentile, token, or cost metrics.
- Existing evaluation metrics, overview, export, and run-detail APIs remain compatible.
- Backend working directory is `D:\flutterserver\pdftopng`; frontend working directory is `D:\flutterserver\Multimodal_RAG_System`.

## File Structure

### Backend files to create

- `core/llm_usage_context.py`: neutral task-local accounting and phase context plus sink protocol.
- `core/llm_usage_callback.py`: LangChain callback and raw provider-event extraction.
- `evaluation/accounting_schemas.py`: accounting persistence models and strict research-summary response models.
- `evaluation/accounting_store.py`: scope, target, usage-event persistence and aggregate reads.
- `evaluation/accounting_runtime.py`: evaluation sink, provider normalization, pricing, and worker scope lifecycle.
- `evaluation/token_normalizers.py`: provider-specific non-overlapping token normalization.
- `evaluation/research_analytics.py`: RAGAS, latency, accounting, pricing, and comparability aggregation.
- `tests/test_evaluation_accounting_schema.py`
- `tests/test_evaluation_accounting_store.py`
- `tests/test_evaluation_token_normalizers.py`
- `tests/test_llm_usage_callback.py`
- `tests/test_evaluation_accounting_runtime.py`
- `tests/test_evaluation_phase_attribution.py`
- `tests/test_evaluation_research_analytics.py`
- `tests/test_evaluation_research_api.py`
- `tests/test_evaluation_research_end_to_end.py`

### Backend files to modify

- `evaluation/db.py`: additive tables, migrations, and indexes.
- `evaluation/execution_worker.py`: execution-scope lifecycle and ledger-derived official token totals.
- `evaluation/ragas_worker.py`: batch-scope lifecycle and target linkage.
- `evaluation/token_cost.py`: audited price-snapshot loading and normalized pricing input.
- `evaluation/router.py`: authenticated research-summary route.
- `core/llm_factory.py`: install the callback on real cached chat models.
- `core/providers.py`: make fake-provider calls emit through the same active accounting context.
- `data_base/query_transformer.py`: query-expansion and rewrite phase tags.
- `data_base/RAG_QA_service.py`: answer-generation, rewrite, synthesis, and visual phase tags.
- `evaluation/agentic_evaluation_service.py`: agent-planning and synthesis phase tags.
- `data_base/research_execution_core.py`: agent-planning phase tag.
- `agents/planner.py`, `agents/evaluator.py`, `agents/synthesizer.py`: agent control/synthesis phase tags.
- `graph_rag/local_search.py`, `graph_rag/global_search.py`, `graph_rag/generic_mode.py`: graph-reasoning phase tags.
- `multimodal_rag/image_summarizer.py`: visual-verification phase tags.
- `tests/test_evaluation_execution_worker.py`, `tests/test_evaluation_ragas_worker.py`, `tests/test_evaluation_token_cost.py`, `tests/test_evaluation_api.py`: focused regressions.
- `docs/BACKEND.md`: research-summary and accounting semantics.

### Frontend files to modify

- `src/types/evaluation.ts`: strict research-summary contract.
- `src/types/index.ts`: type exports.
- `src/services/evaluationApi.ts`: `getCampaignResearchSummary`.
- `src/services/evaluationApi.test.ts`: endpoint contract.
- `src/pages/EvaluationCenter.tsx`: fetch and map only research-summary for overview claims.
- `src/pages/EvaluationCenter.ui.test.tsx`: strict source and state behavior.
- `src/components/evaluation/CampaignOverviewTab.tsx`: status badges, explicit N/A, separated cost panels, excluded-mode reasons.
- `src/components/evaluation/CampaignOverviewTab.test.tsx`: semantic rendering.
- `src/components/evaluation/ModeComparisonChart.tsx`: nullable official metrics and sample status.
- `src/components/evaluation/CostQualityScatter.tsx`: comparable rows only.
- `src/components/evaluation/LatencyWaterfall.tsx`: mean/p50/p95/sample/method.
- `src/components/evaluation/TokenBreakdownChart.tsx`: input/output-text/reasoning/other and phase attribution.
- `docs/design-docs/evaluation-center.md`: mounted strict-overview behavior.
- `docs/product-specs/evaluation-results-and-traces.md`: completeness and cost definitions.

---

### Task 1: Add Accounting Persistence Models and SQLite Schema

**Files:**
- Create: `evaluation/accounting_schemas.py`
- Modify: `evaluation/db.py`
- Create: `tests/test_evaluation_accounting_schema.py`

**Interfaces:**
- Produces: `AccountingScopeStart`, `AccountingScope`, `AccountingScopeTarget`, `UsageEventCreate`, `UsageEvent`, `ScopeStatus`, `ScopeType`.
- Consumes: existing `evaluation.db.init_db()` and `connect_db()`.

- [ ] **Step 1: Write failing schema and migration tests**

```python
# tests/test_evaluation_accounting_schema.py
import pytest
from evaluation import db as evaluation_db
from evaluation.accounting_schemas import AccountingScopeStart


def test_execution_scope_requires_one_target() -> None:
    scope = AccountingScopeStart(
        scope_id="scope-1",
        campaign_id="campaign-1",
        scope_type="execution_run",
        scope_key="run-1",
        run_id="run-1",
        metric_name=None,
        targets=[
            {
                "campaign_result_id": None,
                "job_id": "job-1",
                "work_item_id": "work-1",
                "attempt_id": "attempt-1",
                "metric_name": None,
            }
        ],
    )
    assert scope.accounting_schema_version == "2"
    assert len(scope.targets) == 1


@pytest.mark.asyncio
async def test_init_db_creates_accounting_tables(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")
    await evaluation_db.force_init_db()
    async with evaluation_db.connect_db() as connection:
        cursor = await connection.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name IN (
                   'evaluation_accounting_scopes',
                   'evaluation_accounting_scope_targets',
                   'evaluation_usage_events'
               )"""
        )
        names = {row["name"] for row in await cursor.fetchall()}
    assert names == {
        "evaluation_accounting_scopes",
        "evaluation_accounting_scope_targets",
        "evaluation_usage_events",
    }
```

- [ ] **Step 2: Run tests and verify the missing module/tables fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_schema.py -q`

Expected: FAIL during import of `evaluation.accounting_schemas` or because accounting tables do not exist.

- [ ] **Step 3: Add exact persistence models**

```python
# evaluation/accounting_schemas.py
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

ScopeType = Literal["execution_run", "ragas_batch"]
ScopeStatus = Literal["running", "completed", "failed", "interrupted", "cancelled"]
UsageStatus = Literal["measured", "missing", "failed"]
ReconciliationStatus = Literal["balanced", "partial", "unavailable"]
PricingStatus = Literal["priced", "unknown_model", "missing_price", "unavailable_usage"]


class AccountingScopeTarget(BaseModel):
    campaign_result_id: str | None = None
    job_id: str
    work_item_id: str
    attempt_id: str
    metric_name: str | None = None
    is_official: bool = False


class AccountingScopeStart(BaseModel):
    scope_id: str
    campaign_id: str
    scope_type: ScopeType
    scope_key: str
    run_id: str | None = None
    metric_name: str | None = None
    accounting_schema_version: str = "2"
    targets: list[AccountingScopeTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scope_shape(self) -> "AccountingScopeStart":
        if self.scope_type == "execution_run" and (not self.run_id or len(self.targets) != 1):
            raise ValueError("execution_run requires run_id and exactly one target")
        if self.scope_type == "ragas_batch" and not self.metric_name:
            raise ValueError("ragas_batch requires metric_name")
        return self


class AccountingScope(BaseModel):
    scope_id: str
    campaign_id: str
    scope_type: ScopeType
    scope_key: str
    run_id: str | None = None
    metric_name: str | None = None
    accounting_schema_version: str
    status: ScopeStatus
    observed_call_count: int = 0
    measured_call_count: int = 0
    missing_usage_call_count: int = 0
    unclassified_phase_call_count: int = 0
    started_at: datetime
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    targets: list[AccountingScopeTarget] = Field(default_factory=list)


class UsageEventCreate(BaseModel):
    usage_event_id: str
    scope_id: str
    campaign_id: str
    scope_type: ScopeType
    scope_key: str
    run_id: str | None = None
    provider_run_id: str | None = None
    phase: str
    purpose: str
    metric_name: str | None = None
    provider: str | None = None
    model_name: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_text_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    other_tokens: int = Field(default=0, ge=0)
    reported_total_tokens: int | None = Field(default=None, ge=0)
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    usage_status: UsageStatus
    reconciliation_status: ReconciliationStatus
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    estimated_cost_twd: float | None = Field(default=None, ge=0)
    pricing_status: PricingStatus
    price_snapshot_id: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    status: Literal["success", "failed"] = "success"
    error: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class UsageEvent(UsageEventCreate):
    pass
```

- [ ] **Step 4: Add tables and indexes to `_INIT_SQL`**

```sql
CREATE TABLE IF NOT EXISTS evaluation_accounting_scopes (
    scope_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    run_id TEXT,
    metric_name TEXT,
    accounting_schema_version TEXT NOT NULL,
    status TEXT NOT NULL,
    observed_call_count INTEGER NOT NULL DEFAULT 0,
    measured_call_count INTEGER NOT NULL DEFAULT 0,
    missing_usage_call_count INTEGER NOT NULL DEFAULT 0,
    unclassified_phase_call_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_accounting_scope_targets (
    scope_id TEXT NOT NULL,
    campaign_result_id TEXT,
    job_id TEXT NOT NULL,
    work_item_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    metric_name TEXT,
    is_official INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY(scope_id, attempt_id),
    FOREIGN KEY(scope_id) REFERENCES evaluation_accounting_scopes(scope_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_usage_events (
    usage_event_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    run_id TEXT,
    provider_run_id TEXT,
    phase TEXT NOT NULL,
    purpose TEXT NOT NULL,
    metric_name TEXT,
    provider TEXT,
    model_name TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_text_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    other_tokens INTEGER NOT NULL DEFAULT 0,
    reported_total_tokens INTEGER,
    raw_usage_json TEXT NOT NULL DEFAULT '{}',
    usage_status TEXT NOT NULL,
    reconciliation_status TEXT NOT NULL,
    estimated_cost_usd REAL,
    estimated_cost_twd REAL,
    pricing_status TEXT NOT NULL,
    price_snapshot_id TEXT,
    latency_ms REAL,
    status TEXT NOT NULL,
    error_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(scope_id) REFERENCES evaluation_accounting_scopes(scope_id) ON DELETE CASCADE,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);
```

Add indexes in `_apply_migrations()` for `(campaign_id, scope_type, status)`, `(scope_id, created_at)`, `(run_id, phase)`, `(attempt_id, is_official)`, and `(campaign_id, metric_name)`.

- [ ] **Step 5: Run migration tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_schema.py tests/test_evaluation_observability_schema.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add evaluation/accounting_schemas.py evaluation/db.py tests/test_evaluation_accounting_schema.py
git commit -m "feat(evaluation): add research accounting schema"
```

---

### Task 2: Implement the Durable Accounting Store

**Files:**
- Create: `evaluation/accounting_store.py`
- Create: `tests/test_evaluation_accounting_store.py`

**Interfaces:**
- Consumes: Task 1 persistence models and `evaluation.db.connect_db()`.
- Produces: `EvaluationAccountingStore.start_scope`, `record_event`, `finalize_scope`, `mark_targets_official`, `interrupt_running_scopes`, `get_scope`, `list_campaign_scopes`, `list_campaign_events`, and `summarize_scope_tokens`.

- [ ] **Step 1: Write failing round-trip and atomic-counter tests**

```python
from datetime import datetime, timezone
from uuid import uuid4
import pytest
import pytest_asyncio
from evaluation import db as evaluation_db
from evaluation.accounting_schemas import AccountingScopeStart, UsageEventCreate
from evaluation.accounting_store import EvaluationAccountingStore


async def seed_campaign(campaign_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """INSERT INTO campaigns (
                   id, user_id, name, status, phase, config_json,
                   completed_units, total_units, evaluation_completed_units,
                   evaluation_total_units, cancel_requested, created_at, updated_at
               ) VALUES (?, 'user-1', 'Accounting test', 'running', 'execution', '{}',
                         0, 1, 0, 0, 0, ?, ?)""",
            (campaign_id, now, now),
        )
        await connection.commit()


@pytest_asyncio.fixture
async def accounting_store(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")
    await seed_campaign("campaign-1")
    return EvaluationAccountingStore()


def _execution_scope() -> AccountingScopeStart:
    return AccountingScopeStart(
        scope_id="scope-1", campaign_id="campaign-1", scope_type="execution_run",
        scope_key="run-1", run_id="run-1",
        targets=[{"job_id": "job-1", "work_item_id": "work-1", "attempt_id": "attempt-1"}],
    )


def _ragas_scope(target_count: int) -> AccountingScopeStart:
    return AccountingScopeStart(
        scope_id="ragas-scope", campaign_id="campaign-1", scope_type="ragas_batch",
        scope_key="faithfulness:batch-1", metric_name="faithfulness",
        targets=[{
            "campaign_result_id": f"result-{index}", "job_id": "job-1",
            "work_item_id": f"work-{index}", "attempt_id": f"attempt-{index}",
            "metric_name": "faithfulness",
        } for index in range(target_count)],
    )


def _usage_event(*, phase: str) -> UsageEventCreate:
    return UsageEventCreate(
        usage_event_id=str(uuid4()), scope_id="scope-1", campaign_id="campaign-1",
        scope_type="execution_run", scope_key="run-1", run_id="run-1",
        phase=phase, purpose="rag_qa", provider="google", model_name="fake-model",
        input_tokens=10, output_text_tokens=5, reasoning_tokens=0, other_tokens=0,
        reported_total_tokens=15, raw_usage={"total_tokens": 15},
        usage_status="measured", reconciliation_status="balanced",
        estimated_cost_usd=0.01, pricing_status="priced", status="success",
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_record_event_updates_scope_counters_atomically(accounting_store):
    await accounting_store.start_scope(_execution_scope())
    await accounting_store.record_event(_usage_event(phase="answer_generation"))
    scope = await accounting_store.get_scope("scope-1")
    assert scope.observed_call_count == 1
    assert scope.measured_call_count == 1
    assert scope.missing_usage_call_count == 0


@pytest.mark.asyncio
async def test_ragas_scope_keeps_multiple_targets_without_cost_allocation(accounting_store):
    scope = _ragas_scope(target_count=3)
    await accounting_store.start_scope(scope)
    stored = await accounting_store.get_scope(scope.scope_id)
    assert [target.attempt_id for target in stored.targets] == ["attempt-0", "attempt-1", "attempt-2"]
```

- [ ] **Step 2: Run tests and verify missing store failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_store.py -q`

Expected: FAIL importing `EvaluationAccountingStore`.

- [ ] **Step 3: Implement scope and event transactions**

Use one short transaction for scope plus targets and one transaction per event. `record_event()` must `INSERT OR IGNORE` the event by `usage_event_id`; update counters only when the insert changes one row, so callback retries remain idempotent.

```python
class EvaluationAccountingStore:
    async def start_scope(self, request: AccountingScopeStart) -> AccountingScope:
        now = datetime.now(timezone.utc).isoformat()
        await init_db()
        async with connect_db() as connection:
            await connection.execute(
                """INSERT INTO evaluation_accounting_scopes (
                       scope_id, campaign_id, scope_type, scope_key, run_id, metric_name,
                       accounting_schema_version, status, started_at, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)""",
                (request.scope_id, request.campaign_id, request.scope_type, request.scope_key,
                 request.run_id, request.metric_name, request.accounting_schema_version,
                 now, now, now),
            )
            for target in request.targets:
                await connection.execute(
                    """INSERT INTO evaluation_accounting_scope_targets (
                           scope_id, campaign_result_id, job_id, work_item_id, attempt_id,
                           metric_name, is_official, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
                    (request.scope_id, target.campaign_result_id, target.job_id,
                     target.work_item_id, target.attempt_id, target.metric_name, now),
                )
            await connection.commit()
        return await self.get_scope(request.scope_id)

    async def record_event(self, event: UsageEventCreate) -> None:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(USAGE_EVENT_INSERT_SQL, usage_event_values(event))
            if cursor.rowcount == 1:
                await connection.execute(
                    """UPDATE evaluation_accounting_scopes SET
                           observed_call_count = observed_call_count + 1,
                           measured_call_count = measured_call_count + ?,
                           missing_usage_call_count = missing_usage_call_count + ?,
                           unclassified_phase_call_count = unclassified_phase_call_count + ?,
                           updated_at = ?
                       WHERE scope_id = ?""",
                    (int(event.usage_status == "measured"),
                     int(event.usage_status == "missing"),
                     int(event.phase == "unclassified"),
                     event.created_at.isoformat(), event.scope_id),
                )
            await connection.commit()
```

`finalize_scope()` accepts only terminal statuses. `mark_targets_official()` updates selected target rows and campaign result IDs in one transaction. `interrupt_running_scopes()` marks all running rows interrupted during startup recovery. `summarize_scope_tokens()` returns non-overlapping sums and event completeness.

- [ ] **Step 4: Run store and concurrent-write tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_store.py tests/test_evaluation_observability_repository.py -q`

Expected: PASS with duplicate event insertion leaving counters unchanged.

- [ ] **Step 5: Commit**

```powershell
git add evaluation/accounting_store.py tests/test_evaluation_accounting_store.py
git commit -m "feat(evaluation): persist accounting scopes and usage"
```

---

### Task 3: Normalize Provider Tokens and Load Audited Prices

**Files:**
- Create: `evaluation/token_normalizers.py`
- Modify: `evaluation/token_cost.py`
- Create: `tests/test_evaluation_token_normalizers.py`
- Modify: `tests/test_evaluation_token_cost.py`

**Interfaces:**
- Produces: `NormalizedTokenUsage`, `normalize_provider_usage(provider, raw_usage)`, `load_price_snapshot(path=None)`, and `price_normalized_usage(model_name, usage, snapshot)`.
- Consumes: raw callback usage dictionaries.

- [ ] **Step 1: Write failing Google, OpenAI-style, missing, and unbalanced fixtures**

```python
def test_google_usage_is_non_overlapping_and_balanced():
    usage = normalize_provider_usage(
        "google",
        {"prompt_token_count": 10, "candidates_token_count": 4,
         "thoughts_token_count": 3, "total_token_count": 17},
    )
    assert usage.model_dump() == {
        "input_tokens": 10, "output_text_tokens": 4, "reasoning_tokens": 3,
        "other_tokens": 0, "reported_total_tokens": 17,
        "usage_status": "measured", "reconciliation_status": "balanced",
    }


def test_completion_reasoning_subset_is_not_double_counted():
    usage = normalize_provider_usage(
        "openai",
        {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19,
         "output_token_details": {"reasoning": 4}},
    )
    assert usage.output_text_tokens == 5
    assert usage.reasoning_tokens == 4


def test_missing_usage_is_not_zero():
    usage = normalize_provider_usage("google", {})
    assert usage.reported_total_tokens is None
    assert usage.usage_status == "missing"
```

- [ ] **Step 2: Run and verify failures**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_token_normalizers.py -q`

Expected: FAIL importing `evaluation.token_normalizers`.

- [ ] **Step 3: Implement strict normalization**

```python
class NormalizedTokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_text_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    other_tokens: int = Field(default=0, ge=0)
    reported_total_tokens: int | None = Field(default=None, ge=0)
    usage_status: Literal["measured", "missing"]
    reconciliation_status: Literal["balanced", "partial", "unavailable"]


def normalize_provider_usage(provider: str | None, raw_usage: object) -> NormalizedTokenUsage:
    payload = extract_usage_dict(raw_usage)
    if not payload:
        return NormalizedTokenUsage(usage_status="missing", reconciliation_status="unavailable")
    input_tokens = first_int(payload, "input_tokens", "prompt_tokens", "prompt_token_count") or 0
    completion = first_int(payload, "output_tokens", "completion_tokens", "candidates_token_count") or 0
    details = payload.get("output_token_details") if isinstance(payload.get("output_token_details"), dict) else {}
    reasoning = first_int(payload, "reasoning_tokens", "thoughts_token_count")
    if reasoning is None:
        reasoning = first_int(details, "reasoning") or 0
    provider_key = (provider or "").lower()
    output_text = max(completion - reasoning, 0) if provider_key == "openai" else completion
    total = first_int(payload, "total_tokens", "total_token_count")
    if total is None:
        return NormalizedTokenUsage(
            input_tokens=input_tokens, output_text_tokens=output_text,
            reasoning_tokens=reasoning, reported_total_tokens=None,
            usage_status="missing", reconciliation_status="partial",
        )
    known = input_tokens + output_text + reasoning
    if known > total:
        return NormalizedTokenUsage(
            input_tokens=input_tokens, output_text_tokens=output_text,
            reasoning_tokens=reasoning, reported_total_tokens=total,
            usage_status="measured", reconciliation_status="partial",
        )
    return NormalizedTokenUsage(
        input_tokens=input_tokens, output_text_tokens=output_text,
        reasoning_tokens=reasoning, other_tokens=total - known,
        reported_total_tokens=total, usage_status="measured",
        reconciliation_status="balanced",
    )
```

- [ ] **Step 4: Add audited snapshot loading without hard-coded prices**

`load_price_snapshot()` reads `EVALUATION_PRICE_SNAPSHOT_PATH` or its explicit path. It validates `snapshot_id`, `currency="USD"`, and per-model non-negative input/output/reasoning rates. No path returns the existing unknown snapshot. Invalid configured JSON raises `ValueError` during evaluation startup rather than silently pricing at zero.

- [ ] **Step 5: Run normalizer and pricing tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_token_normalizers.py tests/test_evaluation_token_cost.py -q`

Expected: PASS; unknown models return `pricing_status="unknown_model"` and `estimated_cost_usd=None`.

- [ ] **Step 6: Commit**

```powershell
git add evaluation/token_normalizers.py evaluation/token_cost.py tests/test_evaluation_token_normalizers.py tests/test_evaluation_token_cost.py
git commit -m "feat(evaluation): normalize and price measured usage"
```

---

### Task 4: Add Task-Local Context and LangChain Callback

**Files:**
- Create: `core/llm_usage_context.py`
- Create: `core/llm_usage_callback.py`
- Modify: `core/llm_factory.py`
- Modify: `core/providers.py`
- Create: `tests/test_llm_usage_callback.py`

**Interfaces:**
- Produces: `LlmAccountingContext`, `RawLlmUsageEvent`, `LlmUsageSink`, `llm_accounting_scope`, `llm_accounting_phase`, `current_llm_accounting_context`, `emit_direct_usage`, and `EvaluationUsageCallback`.
- Consumes: a sink supplied later by `evaluation.accounting_runtime`.

- [ ] **Step 1: Write failing isolation, no-scope, async, error, and structured-output callback tests**

```python
from uuid import uuid4
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult


class MemorySink:
    def __init__(self) -> None:
        self.events = []

    async def record(self, event) -> None:
        self.events.append(event)


def _context(sink: MemorySink, scope_id: str = "scope-1") -> LlmAccountingContext:
    return LlmAccountingContext(
        scope_id=scope_id,
        campaign_id="campaign-1",
        scope_type="execution_run",
        scope_key=scope_id,
        run_id=scope_id,
        metric_name=None,
        sink=sink,
    )


async def simulate_callback(callback: EvaluationUsageCallback, *, usage: dict[str, int]) -> None:
    provider_run_id = uuid4()
    await callback.on_chat_model_start({}, [["prompt"]], run_id=provider_run_id)
    response = LLMResult(generations=[[
        ChatGeneration(message=AIMessage(content="ok", usage_metadata=usage))
    ]])
    await callback.on_llm_end(response, run_id=provider_run_id)


async def run_scoped_call(scope_id: str, sink: MemorySink) -> None:
    callback = EvaluationUsageCallback(purpose="rag_qa", provider="google")
    with llm_accounting_scope(_context(sink, scope_id)):
        await simulate_callback(
            callback,
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )


@pytest.mark.asyncio
async def test_callback_records_only_inside_active_scope():
    sink = MemorySink()
    callback = EvaluationUsageCallback(purpose="rag_qa", provider="google")
    await simulate_callback(callback, usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})
    assert sink.events == []
    with llm_accounting_scope(_context(sink)), llm_accounting_phase("answer_generation"):
        await simulate_callback(callback, usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})
    assert sink.events[0].phase == "answer_generation"


@pytest.mark.asyncio
async def test_concurrent_contexts_do_not_mix():
    left, right = MemorySink(), MemorySink()
    await asyncio.gather(run_scoped_call("left", left), run_scoped_call("right", right))
    assert {event.scope_id for event in left.events} == {"left"}
    assert {event.scope_id for event in right.events} == {"right"}
```

- [ ] **Step 2: Run and verify missing modules**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm_usage_callback.py -q`

Expected: FAIL importing the new context or callback.

- [ ] **Step 3: Implement neutral context and sink protocol**

```python
# core/llm_usage_context.py
@dataclass(frozen=True)
class RawLlmUsageEvent:
    usage_event_id: str
    scope_id: str
    campaign_id: str
    scope_type: str
    scope_key: str
    run_id: str | None
    provider_run_id: str | None
    phase: str
    purpose: str
    metric_name: str | None
    provider: str | None
    model_name: str | None
    raw_usage: dict[str, Any]
    latency_ms: float | None
    status: str
    error: dict[str, Any]
    created_at: datetime


class LlmUsageSink(Protocol):
    async def record(self, event: RawLlmUsageEvent) -> None:
        raise NotImplementedError


@dataclass
class LlmAccountingContext:
    scope_id: str
    campaign_id: str
    scope_type: str
    scope_key: str
    run_id: str | None
    metric_name: str | None
    sink: LlmUsageSink
    persistence_error_count: int = 0


_ACCOUNTING_CONTEXT: ContextVar[LlmAccountingContext | None] = ContextVar("llm_accounting_context", default=None)
_ACCOUNTING_PHASE: ContextVar[str] = ContextVar("llm_accounting_phase", default="unclassified")
```

Implement both context managers with `Token` reset in `finally` so exceptions and cancellation cannot leak state.

- [ ] **Step 4: Implement callback lifecycle**

`EvaluationUsageCallback` subclasses `AsyncCallbackHandler`. `on_chat_model_start` and `on_llm_start` snapshot the active context, phase, and monotonic start time by LangChain run UUID. `on_llm_end` extracts usage from the first generated `AIMessage.usage_metadata` and `llm_output`; `on_llm_error` emits a failed raw event. Every terminal callback removes its start entry in `finally`. Sink exceptions increment `persistence_error_count`, log a sanitized warning, and never change the provider result.

Add one integration test using a callback-equipped fake chat model and `model.with_structured_output(TestAnswer, include_raw=True)`. Assert the returned parsed object is unchanged and exactly one raw-message usage event reaches `MemorySink`. Add one streaming fixture that supplies usage on the terminal chunk and assert exactly one combined event. If the installed fake model cannot implement structured output, define a minimal `BaseChatModel` test subclass whose `_generate()` returns the same `AIMessage` fixture; do not call an external provider.

- [ ] **Step 5: Attach callback without changing the model interface**

In `_get_llm_cached()`, pass `callbacks=[EvaluationUsageCallback(purpose=purpose, provider="google")]` to `ChatGoogleGenerativeAI`. Update `_FakeLLM.ainvoke()` to call the shared direct-emission helper when a context is active, using its fake response usage metadata. Do not wrap the model object returned by `get_llm()`.

- [ ] **Step 6: Run callback, provider-registry, and model-override tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm_usage_callback.py tests/test_provider_registry.py tests/test_llm_factory_override.py -q`

Expected: PASS; the no-scope assertion remains empty.

- [ ] **Step 7: Commit**

```powershell
git add core/llm_usage_context.py core/llm_usage_callback.py core/llm_factory.py core/providers.py tests/test_llm_usage_callback.py
git commit -m "feat(evaluation): capture scoped LLM usage"
```

---

### Task 5: Wire Execution Attempt Accounting

**Files:**
- Create: `evaluation/accounting_runtime.py`
- Modify: `evaluation/execution_worker.py`
- Create: `tests/test_evaluation_accounting_runtime.py`
- Modify: `tests/test_evaluation_execution_worker.py`

**Interfaces:**
- Consumes: Tasks 1–4 models, store, normalizer, pricing, and context.
- Produces: `EvaluationAccountingSink`, `ExecutionAccountingSession`, and ledger-derived official `CampaignResult.token_usage/total_tokens`.

- [ ] **Step 1: Write failing execution success, failure, cancellation, and persistence-error tests**

```python
import pytest
from core.llm_usage_callback import emit_direct_usage
from core.llm_usage_context import llm_accounting_phase
from evaluation.accounting_store import EvaluationAccountingStore

TEST_PRICE_SNAPSHOT = {
    "snapshot_id": "test-v1",
    "currency": "USD",
    "usd_to_twd": None,
    "models": {
        "test-model": {"input_per_1m_usd": 1.0, "output_per_1m_usd": 2.0}
    },
}


@pytest.mark.asyncio
async def test_execution_worker_promotes_ledger_total_not_payload_total(store):
    async def runner(**runtime_inputs):
        for phase, usage in [
            ("query_expansion", {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}),
            ("answer_generation", {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}),
        ]:
            with llm_accounting_phase(phase):
                await emit_direct_usage(
                    purpose="rag_qa", provider="google", model_name="test-model",
                    raw_usage=usage, status="success", error={},
                )
        return _successful_payload(
            question_id=runtime_inputs["test_case"].id,
            mode=runtime_inputs["mode"],
            answer="42",
        )

    accounting_store = EvaluationAccountingStore()
    worker = DatasetExecutionWorker(
        store=store,
        runner=runner,
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )
    claim = await _claim_seeded_execution(store)
    await worker.execute(claim)
    results = await evaluation_db.CampaignResultRepository().list_for_campaign(
        user_id="user-a", campaign_id="cmp-1"
    )
    assert results[0].total_tokens == 37
    assert results[0].token_usage["accounting_schema_version"] == "2"


@pytest.mark.asyncio
async def test_failed_attempt_scope_is_not_official(store):
    accounting_store = EvaluationAccountingStore()
    worker = DatasetExecutionWorker(
        store=store,
        runner=AsyncMock(side_effect=RuntimeError("provider failed")),
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )
    claim = await _claim_seeded_execution(store)
    await worker.execute(claim)
    scopes = await accounting_store.list_campaign_scopes("cmp-1")
    scope = scopes[0]
    assert scope.status == "failed"
    assert all(not target.is_official for target in scope.targets)
```

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_runtime.py tests/test_evaluation_execution_worker.py -q`

Expected: FAIL because the worker does not start accounting scopes.

- [ ] **Step 3: Implement the persistence sink**

`EvaluationAccountingSink.record(raw)` normalizes raw usage, loads the configured price snapshot once per worker lifecycle, builds `UsageEventCreate`, and calls `EvaluationAccountingStore.record_event()`. It preserves `raw_usage`, maps unavailable usage to unknown cost, and uses one `usage_event_id` from the callback for idempotency.

- [ ] **Step 4: Generate `run_id` before provider execution and own one scope for the full attempt**

Inject `accounting_store` and `price_snapshot` into `DatasetExecutionWorker`. Before calling the runner:

```python
run_id = str(uuid4())
scope = await accounting_runtime.start_execution_scope(
    campaign_id=campaign_id,
    run_id=run_id,
    job_id=claim.job_id,
    work_item_id=claim.work_item_id,
    attempt_id=claim.attempt_id,
)
try:
    with llm_accounting_scope(scope.context):
        payload = await self._runner(
            test_case=unit.test_case,
            user_id=user_id,
            mode=unit.mode,
            model_config=model_config,
            run_number=unit.repeat_number,
            ablation_flags=unit.ablation_flags,
            budget=unit.budget,
        )
    token_summary = await self._accounting_store.summarize_scope_tokens(scope.scope_id)
    payload.token_usage = token_summary.as_legacy_usage(accounting_schema_version="2")
    execution = ExecutedCampaignUnit(
        unit=unit,
        payload=payload,
        run_id=run_id,
        request_id=str(uuid4()),
        started_at=started_at,
        completed_at=completed_at,
        total_latency_ms=total_latency_ms,
        model_config=model_config,
    )
    result = self._successful_result(campaign_id=campaign_id, execution=execution)
    promoted = await self._store.complete_execution_attempt(
        claim, ExecutionAttemptOutput(result=result)
    )
    await self._accounting_store.mark_targets_official(
        scope.scope_id, {claim.attempt_id: promoted.id}
    )
    await self._accounting_store.finalize_scope(scope.scope_id, "completed")
except asyncio.CancelledError:
    await self._accounting_store.finalize_scope(scope.scope_id, "cancelled")
    raise
except Exception:
    await self._accounting_store.finalize_scope(scope.scope_id, "failed")
    raise
```

If callback persistence errors occurred or any measured event is unbalanced, the scope still reflects the attempt terminal state, while its derived token-accounting status is partial.

- [ ] **Step 5: Recover interrupted accounting scopes with durable attempts**

Inject `accounting_store: EvaluationAccountingStore | None = None` into `EvaluationJobWorker`. Call `self._accounting_store.interrupt_running_scopes()` immediately before each `recover_interrupted_attempts()` invocation in `start()`, `run_until_idle()`, and `stop()`, covered by an extension to `tests/test_evaluation_job_worker.py`.

- [ ] **Step 6: Run execution and recovery tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_runtime.py tests/test_evaluation_execution_worker.py tests/test_evaluation_job_worker.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add evaluation/accounting_runtime.py evaluation/execution_worker.py evaluation/job_worker.py tests/test_evaluation_accounting_runtime.py tests/test_evaluation_execution_worker.py tests/test_evaluation_job_worker.py
git commit -m "feat(evaluation): account durable execution attempts"
```

---

### Task 6: Tag Every Evaluation-Reachable LLM Phase

**Files:**
- Modify all phase-site files listed in File Structure.
- Create: `tests/test_evaluation_phase_attribution.py`
- Modify focused tests for query transformer, RAG QA, Graph, Agentic, and visual verification.

**Interfaces:**
- Consumes: `llm_accounting_phase(name)` from Task 4.
- Produces: controlled phases `query_expansion`, `retrieval_rewrite`, `graph_reasoning`, `agent_planning`, `answer_generation`, `visual_verification`, `agent_synthesis`, and fallback `unclassified`.

- [ ] **Step 1: Write a failing phase matrix test**

```python
from pathlib import Path
import pytest

PHASE_CASES = [
    ("data_base/query_transformer.py", "query_expansion"),
    ("data_base/query_transformer.py", "retrieval_rewrite"),
    ("graph_rag/local_search.py", "graph_reasoning"),
    ("graph_rag/global_search.py", "graph_reasoning"),
    ("graph_rag/generic_mode.py", "graph_reasoning"),
    ("evaluation/agentic_evaluation_service.py", "agent_planning"),
    ("data_base/RAG_QA_service.py", "answer_generation"),
    ("multimodal_rag/image_summarizer.py", "visual_verification"),
    ("agents/synthesizer.py", "agent_synthesis"),
]


@pytest.mark.parametrize(("relative_path", "expected_phase"), PHASE_CASES)
def test_evaluation_call_sites_declare_controlled_phase(relative_path, expected_phase):
    source = Path(relative_path).read_text(encoding="utf-8")
    assert f'llm_accounting_phase("{expected_phase}")' in source
```

- [ ] **Step 2: Run and verify current calls are unclassified**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_phase_attribution.py -q`

Expected: FAIL with one or more observed phases equal to `unclassified`.

- [ ] **Step 3: Wrap each provider invocation at its semantic boundary**

Use this exact pattern and do not change prompts or model settings:

```python
with llm_accounting_phase("query_expansion"):
    response = await llm.ainvoke([message])
```

Map query decomposition/expansion to `query_expansion`, CRAG rewrite to `retrieval_rewrite`, graph search/extraction to `graph_reasoning`, planning/evaluator control to `agent_planning`, final response generation to `answer_generation`, visual caption/verification used during evaluation to `visual_verification`, and final multi-subtask composition to `agent_synthesis`.

- [ ] **Step 4: Run focused behavioral tests to prove phase tagging did not alter outputs**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_query_transformer.py tests/test_agentic_evaluation_service.py tests/test_graph_local_search_vector.py tests/test_research_execution_core_generic.py tests/test_rag_modes_agentic.py tests/test_evaluation_phase_attribution.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add data_base/query_transformer.py data_base/RAG_QA_service.py data_base/research_execution_core.py evaluation/agentic_evaluation_service.py agents/planner.py agents/evaluator.py agents/synthesizer.py graph_rag/local_search.py graph_rag/global_search.py graph_rag/generic_mode.py multimodal_rag/image_summarizer.py tests/test_evaluation_phase_attribution.py tests/test_query_transformer.py tests/test_agentic_evaluation_service.py
git commit -m "feat(evaluation): attribute LLM usage by phase"
```

---

### Task 7: Account RAGAS Batches Without Per-Run Allocation

**Files:**
- Modify: `evaluation/ragas_worker.py`
- Modify: `tests/test_evaluation_ragas_worker.py`
- Modify: `evaluation/accounting_runtime.py`

**Interfaces:**
- Consumes: batch scopes and multiple `AccountingScopeTarget` rows.
- Produces: one shared RAGAS accounting scope per metric provider chunk and campaign-level evaluation overhead.

- [ ] **Step 1: Write failing batch-target, retry, and cancellation tests**

```python
@pytest.mark.asyncio
async def test_ragas_batch_records_one_shared_scope_for_all_claims(accounting_store):
    evaluator = AccountingFakeEvaluator(results=[[0.8, 0.7, 0.9]])
    worker = RagasBatchWorker(
        store=FakeStore(),
        evaluator=evaluator,
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )
    await worker.execute([_claim(i) for i in range(3)])
    scopes = await accounting_store.list_campaign_scopes("campaign-1")
    ragas_scopes = [scope for scope in scopes if scope.scope_type == "ragas_batch"]
    assert len(ragas_scopes) == 1
    assert len(ragas_scopes[0].targets) == 3
    assert ragas_scopes[0].metric_name == "faithfulness"


@pytest.mark.asyncio
async def test_ragas_retry_cost_stays_operational_overhead(accounting_store):
    evaluator = AccountingFakeEvaluator(results=[TimeoutError(), [0.8]])
    worker = RagasBatchWorker(
        store=FakeStore(),
        evaluator=evaluator,
        accounting_store=accounting_store,
        price_snapshot=TEST_PRICE_SNAPSHOT,
    )
    await worker.execute([_claim(0)])
    events = await accounting_store.list_campaign_events("campaign-1")
    assert len(events) == 2
    assert sum(event.estimated_cost_usd or 0 for event in events) == pytest.approx(0.02)
```

Extend the existing `FakeEvaluator` into `AccountingFakeEvaluator`: before returning or raising each queued result, invoke the fake evaluator LLM under the active callback with usage `{input_tokens: 10, output_tokens: 5, total_tokens: 15}`. Define `TEST_PRICE_SNAPSHOT` with one evaluator-model input/output rate that prices each call at exactly `$0.01`. Add a local async `accounting_store` fixture that points `EVALUATION_DB_PATH` to `tmp_path / "evaluation.db"`, inserts campaign `campaign-1` owned by `user-1`, inserts three completed campaign results `result-0` through `result-2`, and returns `EvaluationAccountingStore()`. Use the existing `_claim()` helper after adding `user_id="user-1"` and `campaign_id="campaign-1"` to its input snapshot.

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_ragas_worker.py -q`

Expected: FAIL because no RAGAS accounting scope exists.

- [ ] **Step 3: Open one scope inside `_execute_chunk()`**

Build a stable scope key from campaign ID, metric name, batch-group key, sorted attempt IDs, and worker invocation UUID. Add one target per claim. Activate phase `ragas_scoring` around the existing `run_with_retry` call whose callable is `self._evaluator.evaluate_metric_batch` and whose positional arguments are `metric_name`, `rows`, `self._evaluator_llm`, and `self._evaluator_embeddings`. Finalize completed only after score promotion; failed/cancelled scopes retain every actual callback event and no target becomes official unless its score promotion succeeds.

- [ ] **Step 4: Verify no per-result cost field is written**

Add an assertion that RAGAS events have `run_id is None`, targets list all campaign result IDs, and research accounting exposes only campaign-level overhead.

- [ ] **Step 5: Run RAGAS and job-store tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_ragas_worker.py tests/test_evaluation_job_store.py tests/test_campaign_engine.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add evaluation/ragas_worker.py evaluation/accounting_runtime.py tests/test_evaluation_ragas_worker.py
git commit -m "feat(evaluation): account RAGAS batch overhead"
```

---

### Task 8: Build Strict Research Analytics and API

**Files:**
- Modify: `evaluation/accounting_schemas.py`
- Create: `evaluation/research_analytics.py`
- Modify: `evaluation/router.py`
- Create: `tests/test_evaluation_research_analytics.py`
- Create: `tests/test_evaluation_research_api.py`

**Interfaces:**
- Consumes: official `campaign_results.source_attempt_id`, `ragas_scores`, accounting scopes/targets/events, and campaign ownership.
- Produces: `CampaignResearchSummaryResponse` and `GET /api/evaluation/campaigns/{campaign_id}/research-summary`.

- [ ] **Step 1: Define strict response models and failing serialization tests**

Add typed models for `MetricObservation`, `LatencySummary`, `TokenBreakdown`, `CostSummary`, `ModeResearchSummary`, `EvaluationOverheadSummary`, `ResearchWarning`, and `CampaignResearchSummaryResponse`. Values that can be unavailable are nullable and always carry status/sample metadata.

```python
QualityStatus = Literal["complete", "evaluating", "partial", "failed", "not_requested"]
TokenAccountingStatus = Literal["complete", "partial", "incomplete_legacy"]
ResearchPricingStatus = Literal["complete", "partial", "unknown"]
PhaseAttributionStatus = Literal["complete", "partial", "not_available"]


class MetricObservation(BaseModel):
    value: float | None = None
    status: QualityStatus
    valid_samples: int = 0
    missing_samples: int = 0
    failed_samples: int = 0
    evaluator_model: str | None = None
    metric_version: str | None = None


class LatencySummary(BaseModel):
    mean_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    sample_count: int = 0
    method: Literal["nearest_rank"] = "nearest_rank"
    low_sample_size: bool = False


class TokenBreakdown(BaseModel):
    input_tokens: int | None = None
    output_text_tokens: int | None = None
    reasoning_tokens: int | None = None
    other_tokens: int | None = None
    total_tokens: int | None = None
    by_phase: dict[str, int] = Field(default_factory=dict)
    accounting_status: TokenAccountingStatus
    phase_attribution_status: PhaseAttributionStatus


class CostSummary(BaseModel):
    benchmark_usd: float | None = None
    operational_usd: float | None = None
    pricing_status: ResearchPricingStatus
    priced_call_count: int = 0
    unpriced_call_count: int = 0


class ModeResearchSummary(BaseModel):
    mode: str
    sample_count: int
    comparable: bool
    not_comparable_reasons: list[str] = Field(default_factory=list)
    quality: dict[str, MetricObservation] = Field(default_factory=dict)
    latency: LatencySummary
    tokens: TokenBreakdown
    execution_cost: CostSummary


class EvaluationOverheadSummary(BaseModel):
    tokens: TokenBreakdown
    cost_usd: float | None = None
    pricing_status: ResearchPricingStatus
    evaluator_models: list[str] = Field(default_factory=list)
    metric_names: list[str] = Field(default_factory=list)
    batch_count: int = 0
    retry_count: int = 0


class ResearchWarning(BaseModel):
    code: str
    message: str
    mode: str | None = None


class CampaignResearchSummaryResponse(BaseModel):
    campaign_id: str
    research_schema_version: Literal["2"] = "2"
    completed_run_count: int
    total_run_count: int
    failed_run_count: int
    quality_status: QualityStatus
    token_accounting_status: TokenAccountingStatus
    pricing_status: ResearchPricingStatus
    phase_attribution_status: PhaseAttributionStatus
    sample_count: int
    quality: dict[str, MetricObservation] = Field(default_factory=dict)
    latency: LatencySummary
    tokens: TokenBreakdown
    execution_cost: CostSummary
    modes: list[ModeResearchSummary]
    evaluation_overhead: EvaluationOverheadSummary
    warnings: list[ResearchWarning] = Field(default_factory=list)
```

- [ ] **Step 2: Write failing nearest-rank, legacy, missing-RAGAS, unknown-price, retry, and compatibility tests**

```python
def test_nearest_rank_percentiles_are_observed_values():
    assert nearest_rank([100, 200, 300, 400, 500], 0.50) == 300
    assert nearest_rank([100, 200, 300, 400, 500], 0.95) == 500


@pytest.mark.asyncio
async def test_legacy_campaign_is_visible_but_not_comparable(research_service):
    summary = await research_service.get_summary(user_id="user-1", campaign_id="legacy")
    assert summary.token_accounting_status == "incomplete_legacy"
    assert summary.modes[0].comparable is False
    assert "legacy_accounting" in summary.modes[0].not_comparable_reasons


@pytest.mark.asyncio
async def test_missing_faithfulness_stays_null(research_service):
    summary = await research_service.get_summary(user_id="user-1", campaign_id="partial")
    faithfulness = summary.modes[0].quality["faithfulness"]
    assert faithfulness.value is None
    assert faithfulness.status == "failed"
```

The `research_service` fixture uses a temporary evaluation DB and real repositories. Seed campaign `legacy` with one completed result and no accounting scopes. Seed campaign `partial` with one completed official result, one completed version-2 execution scope linked to that result's `source_attempt_id`, priced balanced usage, successful answer-correctness and answer-relevancy scores, and a terminal failed faithfulness work item with no score. Seed a third campaign with two modes, retries, unknown pricing, and mixed evaluation signatures for the remaining comparability tests. Keep all fixture model names, signatures, attempts, and expected statuses literal in the test file.

- [ ] **Step 3: Implement deterministic aggregation**

`ResearchAnalyticsService.get_summary()` must:

1. Authenticate campaign ownership through `CampaignRepository.get()`.
2. Load official completed results and their `source_attempt_id` values.
3. Load RAGAS scores and group only compatible evaluator model/version/signature rows.
4. Load accounting scopes, targets, and events once per campaign.
5. Select official completed execution scopes for benchmark tokens/cost.
6. Include all execution scopes in operational cost.
7. Include all RAGAS batch scopes only in evaluation overhead.
8. Compute nearest-rank mean/p50/p95 per mode from official `total_latency_ms`.
9. Compute campaign-level quality, latency, tokens, and execution costs directly from included official runs; never average per-mode aggregates.
10. Derive quality, accounting, pricing, and phase statuses without numerical fallback.
11. Set `comparable` only when the approved five comparability rules pass.

Use this exact official metric set:

```python
PRIMARY_QUALITY_METRICS = ("answer_correctness", "faithfulness", "answer_relevancy")
OPTIONAL_CONTEXT_METRICS = ("context_precision", "context_recall")
```

The primary metrics always appear in each mode response, even when their value is null. Optional context metrics appear only when requested by the campaign's RAGAS work. Unsupported-claim and evidence-coverage derived fields are never read by this service.

- [ ] **Step 4: Add the authenticated route**

```python
@router.get(
    "/campaigns/{campaign_id}/research-summary",
    response_model=CampaignResearchSummaryResponse,
)
async def get_campaign_research_summary(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: ResearchAnalyticsService = Depends(get_research_analytics_service),
) -> CampaignResearchSummaryResponse:
    return await analytics.get_summary(user_id=user_id, campaign_id=campaign_id)
```

- [ ] **Step 5: Run analytics/API tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py tests/test_evaluation_analytics_api.py tests/test_evaluation_api.py -q`

Expected: PASS with absent quality/cost values serialized as JSON `null`, never `0`.

- [ ] **Step 6: Commit**

```powershell
git add evaluation/accounting_schemas.py evaluation/research_analytics.py evaluation/router.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py
git commit -m "feat(evaluation): expose strict research summary"
```

---

### Task 9: Add Frontend Research Contract and Replace Overview Data Source

**Files:**
- Modify frontend: `src/types/evaluation.ts`, `src/types/index.ts`, `src/services/evaluationApi.ts`, `src/services/evaluationApi.test.ts`, `src/pages/EvaluationCenter.tsx`, `src/pages/EvaluationCenter.ui.test.tsx`.

**Interfaces:**
- Consumes: Task 8 JSON contract.
- Produces: `CampaignResearchSummaryResponse`, `getCampaignResearchSummary`, and `mapOverviewData(researchSummary)` with no derived research values.

- [ ] **Step 1: Write failing API and page-source tests**

```typescript
it('fetches the strict research summary', async () => {
  mockedApi.get.mockResolvedValueOnce({ data: researchSummaryFixture });
  expect(await getCampaignResearchSummary('cmp-1')).toEqual(researchSummaryFixture);
  expect(mockedApi.get).toHaveBeenCalledWith('/api/evaluation/campaigns/cmp-1/research-summary');
});

it('does not request mode-comparison to build Campaign Overview', async () => {
  renderEvaluationCenter();
  await screen.findByText(/CampaignOverviewTab/);
  expect(mockGetCampaignResearchSummary).toHaveBeenCalledWith('cmp-1');
  expect(mockGetModeComparison).not.toHaveBeenCalled();
});
```

Define `researchSummaryFixture` as a complete schema-version-2 response containing one comparable Agentic mode, all three primary quality metrics, measured latency, balanced tokens, complete pricing, and zero RAGAS overhead. Add `getCampaignResearchSummary: mockGetCampaignResearchSummary` to the existing service mock and remove the overview mock's dependency on `getModeComparison` data.

- [ ] **Step 2: Run and verify missing function/type failures**

Run: `npm test -- --run src/services/evaluationApi.test.ts src/pages/EvaluationCenter.ui.test.tsx`

Expected: FAIL because `getCampaignResearchSummary` and its types do not exist.

- [ ] **Step 3: Add exact TypeScript contract**

Mirror every backend enum and nullable field. Do not use `Record<string, unknown>` for mode research summaries. Define `ResearchMetricObservation`, `ResearchLatencySummary`, `ResearchTokenBreakdown`, `ResearchCostSummary`, `ModeResearchSummary`, `EvaluationOverheadSummary`, `ResearchWarning`, and `CampaignResearchSummaryResponse`.

```typescript
export type ResearchQualityStatus = 'complete' | 'evaluating' | 'partial' | 'failed' | 'not_requested';
export type TokenAccountingStatus = 'complete' | 'partial' | 'incomplete_legacy';
export type ResearchPricingStatus = 'complete' | 'partial' | 'unknown';
export type PhaseAttributionStatus = 'complete' | 'partial' | 'not_available';

export interface ResearchMetricObservation {
  value: number | null;
  status: ResearchQualityStatus;
  valid_samples: number;
  missing_samples: number;
  failed_samples: number;
  evaluator_model: string | null;
  metric_version: string | null;
}

export interface ResearchLatencySummary {
  mean_ms: number | null;
  p50_ms: number | null;
  p95_ms: number | null;
  sample_count: number;
  method: 'nearest_rank';
  low_sample_size: boolean;
}

export interface ResearchTokenBreakdown {
  input_tokens: number | null;
  output_text_tokens: number | null;
  reasoning_tokens: number | null;
  other_tokens: number | null;
  total_tokens: number | null;
  by_phase: Record<string, number>;
  accounting_status: TokenAccountingStatus;
  phase_attribution_status: PhaseAttributionStatus;
}

export interface ResearchCostSummary {
  benchmark_usd: number | null;
  operational_usd: number | null;
  pricing_status: ResearchPricingStatus;
  priced_call_count: number;
  unpriced_call_count: number;
}

export interface ModeResearchSummary {
  mode: string;
  sample_count: number;
  comparable: boolean;
  not_comparable_reasons: string[];
  quality: Record<string, ResearchMetricObservation>;
  latency: ResearchLatencySummary;
  tokens: ResearchTokenBreakdown;
  execution_cost: ResearchCostSummary;
}

export interface EvaluationOverheadSummary {
  tokens: ResearchTokenBreakdown;
  cost_usd: number | null;
  pricing_status: ResearchPricingStatus;
  evaluator_models: string[];
  metric_names: string[];
  batch_count: number;
  retry_count: number;
}

export interface ResearchWarning { code: string; message: string; mode: string | null; }

export interface CampaignResearchSummaryResponse {
  campaign_id: string;
  research_schema_version: '2';
  completed_run_count: number;
  total_run_count: number;
  failed_run_count: number;
  quality_status: ResearchQualityStatus;
  token_accounting_status: TokenAccountingStatus;
  pricing_status: ResearchPricingStatus;
  phase_attribution_status: PhaseAttributionStatus;
  sample_count: number;
  quality: Record<string, ResearchMetricObservation>;
  latency: ResearchLatencySummary;
  tokens: ResearchTokenBreakdown;
  execution_cost: ResearchCostSummary;
  modes: ModeResearchSummary[];
  evaluation_overhead: EvaluationOverheadSummary;
  warnings: ResearchWarning[];
}
```

- [ ] **Step 4: Add the API function**

```typescript
export async function getCampaignResearchSummary(
  campaignId: string,
): Promise<CampaignResearchSummaryResponse> {
  const response = await api.get<CampaignResearchSummaryResponse>(
    `/api/evaluation/campaigns/${campaignId}/research-summary`,
  );
  return response.data;
}
```

- [ ] **Step 5: Switch overview loading and mapping**

Add `researchSummary` to `DashboardApiData`. On campaign selection, clear prior campaign data and fetch the research summary. Tab 0 loads errors only. Replace proxy calculations with direct nullable field mapping; do not import or call `getModeComparison()` for Campaign Overview.

Top cards map `completed_run_count`, `total_run_count`, top-level `quality`, `latency`, `tokens`, and `execution_cost` directly. They must not average the `modes` array.

- [ ] **Step 6: Run API/page tests and TypeScript build**

Run: `npm test -- --run src/services/evaluationApi.test.ts src/pages/EvaluationCenter.ui.test.tsx`

Expected: PASS.

Run: `npm run build`

Expected: PASS.

- [ ] **Step 7: Commit in the frontend repository**

```powershell
git add src/types/evaluation.ts src/types/index.ts src/services/evaluationApi.ts src/services/evaluationApi.test.ts src/pages/EvaluationCenter.tsx src/pages/EvaluationCenter.ui.test.tsx
git commit -m "feat(evaluation): consume strict research summary"
```

---

### Task 10: Render Strict Quality, Latency, Token, and Cost States

**Files:**
- Modify frontend overview and chart components listed in File Structure.
- Modify: `src/components/evaluation/CampaignOverviewTab.test.tsx`.
- Create: `src/components/evaluation/ModeComparisonChart.test.tsx`.
- Create: `src/components/evaluation/CostQualityScatter.test.tsx`.
- Create: `src/components/evaluation/LatencyWaterfall.test.tsx`.
- Create: `src/components/evaluation/TokenBreakdownChart.test.tsx`.

**Interfaces:**
- Consumes: the typed view model produced by Task 9.
- Produces: explicit research-status header, nullable metric rows, comparable-only cost table, measured latency table, non-overlapping token table, separate evaluation overhead, and excluded-mode reasons.

- [ ] **Step 1: Write failing semantic rendering tests**

```typescript
it('renders missing RAGAS and unknown price without synthetic zero', () => {
  renderOverview(partialFixture);
  expect(screen.getByText('N/A')).toBeInTheDocument();
  expect(screen.getByText('Unknown')).toBeInTheDocument();
  expect(screen.queryByText('$0.00')).not.toBeInTheDocument();
});

it('shows measured percentiles and low sample warning', () => {
  renderOverview(completeFixture);
  expect(screen.getByText('3,900 ms')).toBeInTheDocument();
  expect(screen.getByText('7,100 ms')).toBeInTheDocument();
  expect(screen.getByText(/Low sample size/)).toBeInTheDocument();
});

it('excludes non-comparable modes from cost quality rows', () => {
  renderOverview(mixedFixture);
  expect(screen.getByTestId('cost-quality-agentic')).toBeInTheDocument();
  expect(screen.queryByTestId('cost-quality-graph')).not.toBeInTheDocument();
  expect(screen.getByText(/graph: unknown pricing/)).toBeInTheDocument();
});
```

Define `renderOverview(data)` in the test file as `render(<ChakraProvider theme={theme}><CampaignOverviewTab data={data} /></ChakraProvider>)`. `partialFixture` has null faithfulness, `pricingStatus='unknown'`, and null costs. `completeFixture` has p50 `3900`, p95 `7100`, sample count `4`, and `lowSampleSize=true`. `mixedFixture` has comparable Agentic and non-comparable Graph with reason `unknown pricing`. Use the exact Task 9 TypeScript interfaces so fixtures cannot omit status fields.

- [ ] **Step 2: Run and verify current proxy/zero presentation fails**

Run: `npm test -- --run src/components/evaluation/CampaignOverviewTab.test.tsx`

Expected: FAIL because components require numeric values and render `$0.00`.

- [ ] **Step 3: Update overview component contracts**

Quality fields become `number | null` plus status/helper metadata. Formatters return `N/A` for null. Status badges use the four independent campaign statuses. Legacy accounting displays a banner. Cost cards use `Unknown` for null and show benchmark, operational, and RAGAS overhead separately.

- [ ] **Step 4: Update chart rows without fallbacks**

- `ModeComparisonChart`: nullable metric cells with valid/missing/failed samples.
- `CostQualityScatter`: receives only comparable points and exposes stable `data-testid` values.
- `LatencyWaterfall`: mode, mean, p50, p95, sample count, method, and low-sample warning.
- `TokenBreakdownChart`: input, output text, reasoning, other, total, phase status, and visible unclassified tokens.

The components must not coerce null with `?? 0`, `numberValue`, or default numeric arguments.

- [ ] **Step 5: Run all Evaluation Center frontend tests**

Run: `npm test -- --run src/pages/EvaluationCenter.ui.test.tsx src/components/evaluation/CampaignOverviewTab.test.tsx src/components/evaluation/ModeComparisonChart.test.tsx src/components/evaluation/CostQualityScatter.test.tsx src/components/evaluation/LatencyWaterfall.test.tsx src/components/evaluation/TokenBreakdownChart.test.tsx`

Expected: PASS in all six listed test files.

- [ ] **Step 6: Run lint and build**

Run: `npm run lint:ci`

Expected: PASS with zero warnings.

Run: `npm run build`

Expected: PASS.

- [ ] **Step 7: Commit in the frontend repository**

```powershell
git add src/components/evaluation src/pages/EvaluationCenter.tsx src/pages/EvaluationCenter.ui.test.tsx
git commit -m "feat(evaluation): render strict research metrics"
```

---

### Task 11: End-to-End Accounting Verification and Documentation

**Files:**
- Create: `tests/test_evaluation_research_end_to_end.py`
- Modify backend docs: `docs/BACKEND.md`
- Modify frontend docs: `docs/design-docs/evaluation-center.md`, `docs/product-specs/evaluation-results-and-traces.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: deterministic cross-mode proof, updated contracts, and final verification evidence.

- [ ] **Step 1: Write the deterministic fake-provider campaign test**

Create four durable execution claims with a controlled runner that emits these scoped calls:

```python
EXPECTED_PHASES = {
    "naive": ["answer_generation"],
    "advanced": ["query_expansion", "answer_generation"],
    "graph": ["query_expansion", "graph_reasoning", "answer_generation"],
    "agentic": ["agent_planning", "answer_generation", "agent_synthesis"],
}
```

Every fake call reports 10 input, 5 output, and 15 total tokens. Use a test price snapshot with deterministic rates. Add one batched faithfulness call and one batched answer-correctness call. Assert exact official token totals, operational totals, per-phase totals, RAGAS overhead, status fields, and comparability through the HTTP API.

- [ ] **Step 2: Run the end-to-end test and fix only integration defects**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_end_to_end.py -q`

Expected: PASS with the four modes and two evaluator batches represented separately.

- [ ] **Step 3: Update documentation with exact semantics**

Document:

- version-2 accounting scope and legacy behavior;
- inference benchmark versus operational versus RAGAS overhead cost;
- official RAGAS-only quality labels;
- nearest-rank percentile method;
- nullable/partial status behavior;
- `EVALUATION_PRICE_SNAPSHOT_PATH` configuration;
- `/api/evaluation/campaigns/{campaign_id}/research-summary` response purpose.

- [ ] **Step 4: Run focused backend verification**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_accounting_schema.py tests/test_evaluation_accounting_store.py tests/test_evaluation_token_normalizers.py tests/test_llm_usage_callback.py tests/test_evaluation_accounting_runtime.py tests/test_evaluation_phase_attribution.py tests/test_evaluation_ragas_worker.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py tests/test_evaluation_research_end_to_end.py -q`

Expected: PASS with zero failures.

- [ ] **Step 5: Run full backend verification**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: PASS with zero failures. Record warning count separately; do not claim warnings are errors.

- [ ] **Step 6: Run full frontend verification**

Run: `npm test -- --run`

Expected: PASS with zero failed test files.

Run: `npm run lint:ci`

Expected: PASS with zero warnings.

Run: `npm run build`

Expected: PASS.

- [ ] **Step 7: Verify forbidden frontend fallback patterns are absent from the overview path**

Run from the frontend repository:

```powershell
rg -n "1 - unsupported|unsupported_claim_ratio_mean|p50Ms: overview\.avg|p95Ms: overview\.avg|completionTokens: 0|reasoningTokens: 0|total_cost_usd \?\? 0" src/pages/EvaluationCenter.tsx src/components/evaluation
```

Expected: no matches.

- [ ] **Step 8: Commit documentation and end-to-end proof**

Backend repository:

```powershell
git add tests/test_evaluation_research_end_to_end.py docs/BACKEND.md
git commit -m "test(evaluation): verify research accounting end to end"
```

Frontend repository:

```powershell
git add docs/design-docs/evaluation-center.md docs/product-specs/evaluation-results-and-traces.md
git commit -m "docs: document strict evaluation metrics"
```

## Final Review Checklist

- [ ] Every official metric displayed in Campaign Overview traces to `ragas_scores`.
- [ ] Every official execution result traces to one version-2 completed execution scope.
- [ ] Every observed provider call has one idempotent usage event.
- [ ] RAGAS batch events have no fabricated run cost.
- [ ] Unknown usage and price remain null and status-bearing.
- [ ] Benchmark and operational costs differ correctly under retry.
- [ ] Legacy campaigns remain inspectable but not comparable.
- [ ] Backend owns percentile and comparability calculations.
- [ ] Frontend contains no research fallback arithmetic.
- [ ] Normal chat produces no accounting rows.
- [ ] Both repositories are clean after all verification and intentional commits.
