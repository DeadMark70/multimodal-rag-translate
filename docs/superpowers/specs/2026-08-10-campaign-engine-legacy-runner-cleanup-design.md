# CampaignEngine Legacy Runner Cleanup Design

## Context

The evaluation runtime completed its durable-worker migration, but
`evaluation/campaign_engine.py` still contains the former process-local campaign
runner and the execution helpers that predate that migration.

The production request path no longer calls `CampaignEngine._run_campaign()`.
Campaign creation writes durable job items, and `EvaluationJobWorker` dispatches
those claims to `DatasetExecutionWorker`. Direct calls to `_run_campaign()` now
exist only in tests.

The resulting ownership is inverted:

- `CampaignEngine` is intended to be the enqueue/query/cancel/rerun facade;
- `DatasetExecutionWorker` is the production execution owner; but
- `evaluation/execution_worker.py` imports execution dataclasses and private
  projection/observability helpers from `evaluation/campaign_engine.py`.

This keeps `campaign_engine.py` at roughly 2,600 physical lines and encourages
tests to exercise a dead process-local path rather than the durable runtime.

The same cleanup also removes `tests/test_agents_static_analysis.py`. Its five
tests check symbol presence, selected signatures, the textual presence of
`try:`/`except`, and constructor annotations. Imports, Ruff, and maintained
behavior tests provide stronger coverage of those contracts.

## Goals

1. Make `CampaignEngine` the durable campaign facade described by the existing
   evaluation runtime design.
2. Remove the process-local campaign execution path that has no production
   caller.
3. Move shared execution contracts and projections to their actual owner so the
   durable worker does not import private helpers from the facade.
4. Preserve unique execution and observability behavior by testing the durable
   worker path.
5. Delete the obsolete agent static-analysis test without replacing heuristic
   source-text assertions.
6. Keep the change mechanical, backward compatible, and net-deleting.

## Non-goals

- No database migration or schema change.
- No API request, response, route, or public signature change.
- No frontend `CampaignRunner` change.
- No rewrite of the durable job state machine, retry policy, RAGAS worker, or
  provider execution.
- No split or simplification of `CampaignEngine.create_rerun()`.
- No new service layer, dependency-injection framework, or runtime class
  hierarchy.
- No removal of pre-ledger campaign recovery or other legacy data-reading
  compatibility.
- No cleanup of `tests/test_evaluator_audit.py` or unrelated source-boundary
  tests.
- No additional observability-module decomposition in this batch.

## Considered approaches

### Delete only the legacy runner

This is the smallest diff, but it leaves `DatasetExecutionWorker` importing
private helpers from the facade and leaves `campaign_engine.py` as a large
execution-helper container. It removes dead methods without fixing ownership,
so it is rejected.

### Extract one execution-support module and delete the legacy runner

Create `evaluation/campaign_execution.py` for execution contracts, result
projections, and campaign-specific observability persistence. Both
`CampaignEngine` and `DatasetExecutionWorker` import from that owner while the
old process-local runner is removed. This is the selected approach.

The new module may remain substantial because the move is intentionally
mechanical. Splitting campaign-specific observability into further modules can
be considered only after this boundary is established and measured.

### Fully decompose campaign execution

Separate unit factories, result projectors, observability adapters, recovery,
and facade services. Although individually narrow, those extra seams would add
constructor wiring and test doubles without changing current behavior. This is
rejected as unnecessary complexity.

## Module ownership

### `evaluation/campaign_engine.py`

Owns the durable campaign facade:

- campaign create/list/get/cancel operations;
- rerun and manual-evaluate requests;
- durable job and work-item construction;
- worker startup/notification;
- campaign status reconciliation;
- startup recovery; and
- the pre-ledger compatibility bridge.

It retains `_resolve_test_cases()`, `_build_units()`, and `_work_item_spec()`
because they construct durable work from campaign configuration. It retains
`_prepare_legacy_recovery()` because legacy database rows are still supported.

### `evaluation/campaign_execution.py`

Owns shared execution contracts and deterministic projections currently housed
above `CampaignEngine`:

- `CampaignRunner`;
- `CampaignUnit` and `ExecutedCampaignUnit`;
- duration, token, snapshot, hash, and derived-metric helpers;
- safe failure diagnostics;
- agent-trace enrichment;
- root-span and LLM-usage recording;
- campaign-result research-observability projection; and
- the bounded rerank/expected-source diagnostic helpers used by that
  projection.

The first extraction preserves function signatures and bodies except for import
adjustments. It does not redesign the large observability projector.

### `evaluation/execution_worker.py`

Remains the only production owner of claimed dataset execution. It imports
shared contracts and projections from `campaign_execution.py`, not from
`campaign_engine.py`.

Its claim, accounting, provider invocation, promotion, error classification,
observability, and downstream RAGAS behavior remain unchanged.

## Removed process-local path

After durable-worker behavior coverage exists, remove the process-local methods
and their exclusive helper:

- `_run_campaign()`;
- `_run_evaluation_only()`;
- `_run_ragas_evaluation()`;
- `_evaluate_campaign_results()`;
- `_execute_unit()`;
- `_persist_unit_result()`; and
- `_cancel_and_drain_tasks()`.

No compatibility wrapper remains for tests. Tests must use the same durable
claim-to-worker path used in production.

## Runtime data flow

The production flow remains:

```text
evaluation/router.py
  -> CampaignEngine creates durable job/items
  -> EvaluationJobWorker claims an item
  -> DatasetExecutionWorker executes the immutable snapshot
  -> campaign_execution projects result and observability rows
  -> repositories persist/promote official state
```

Startup recovery remains:

```text
CampaignEngine.recover_inflight_campaigns()
  -> _prepare_legacy_recovery() for campaigns without durable items
  -> ensure_campaign_task()
  -> durable worker resumes unresolved work
```

## Test migration

### Facade tests

`tests/test_campaign_engine.py` retains tests for:

- API integration and streaming;
- campaign creation and durable work expansion;
- query/status reconciliation;
- cancellation;
- rerun and manual evaluation requests;
- startup recovery and pre-ledger migration;
- work-item construction; and
- repository-level compatibility behavior owned by the facade.

It no longer calls `_run_campaign()` or tests execution through a private
process-local method.

### Durable execution tests

Tests already covered by `tests/test_evaluation_execution_worker.py` are removed
from `test_campaign_engine.py` rather than duplicated. Existing durable coverage
includes failure profiles, classified/redacted errors, oversized answers,
cancellation, measured provider usage, ablation identity, and v9 trace
materialization.

Unique observability cases move to
`tests/test_evaluation_execution_observability.py`. Each test creates a durable
claim and invokes `DatasetExecutionWorker.execute()`. Preserved cases include:

- result snapshots and minimal root spans;
- best-effort LLM observer failure and partial status;
- visual/final provider attempts;
- actual route persistence;
- retrieval, context, and evidence rows;
- rerank diagnostics;
- expected-source resolution and unresolved states; and
- claims and derived claim metrics.

The migration may reuse focused claim/repository fixtures, but must not introduce
a general-purpose test harness or retain the old runner as a fixture.

### Obsolete static test

Delete `tests/test_agents_static_analysis.py` in full. Do not recreate its
`hasattr`, signature, `try:`/`except`, or constructor-annotation assertions.
Maintained evaluator and agent behavior tests remain authoritative.

## Error and compatibility behavior

The cleanup must preserve all of the following:

- `asyncio.CancelledError` closes the accounting scope as cancelled and is not
  converted into a successful result.
- Oversized answers fail without scheduling RAGAS work.
- Provider, timeout, rate-limit, and configuration failures continue through
  `evaluation.error_policy`.
- Raw exception text and credential-like values are not persisted in result or
  diagnostic payloads.
- Observability write failure remains best-effort: the answer can succeed while
  observability is marked partial.
- A failed attempt does not replace an existing official result.
- Successful promotion remains idempotent and source-attempt aware.
- Pre-ledger recovery continues to backfill missing durable work and safely mark
  unrecoverable campaigns failed.
- Existing campaign public API signatures and response schemas do not change.

## Complexity baseline

The current production baseline contains:

- `evaluation/campaign_engine.py::_record_unit_research_observability` at 19;
- `evaluation/campaign_engine.py::create_rerun` at 17.

The observability finding moves to
`evaluation/campaign_execution.py::_record_unit_research_observability` without
increasing its score. `CampaignEngine.create_rerun` remains unchanged. The total
finding count and score must not increase.

## Implementation sequence

1. Delete `tests/test_agents_static_analysis.py` and verify maintained agent and
   evaluator tests.
2. Mechanically extract `evaluation/campaign_execution.py`, update imports, move
   the complexity-baseline key, and verify behavior before deleting anything.
3. Move unique observability tests to the durable worker path and remove tests
   already duplicated by durable-worker coverage.
4. Delete the process-local runner methods and clean imports.
5. Run focused evaluation tests, the full backend suite, and all maintenance
   gates.

These steps should be separate commits so a mechanical ownership move, test
migration, and dead-code deletion can each be reviewed independently.

## Acceptance criteria

- `rg "_run_campaign"` finds no production or test definition/call.
- `evaluation/execution_worker.py` does not import private helpers from
  `evaluation/campaign_engine.py`.
- `CampaignEngine` retains its existing public API and pre-ledger recovery.
- All preserved observability behaviors are exercised through durable claims.
- `tests/test_agents_static_analysis.py` is absent.
- `tests/test_evaluator_audit.py` and unrelated source-boundary tests are
  unchanged.
- No database, API schema, frontend, Docker, or dependency changes are present.
- Focused evaluation tests and the complete backend test suite pass within the
  committed warning budget.
- Ruff correctness checks, the C901 ratchet, OpenAPI drift check, Markdown-link
  check, and `git diff --check` pass.
- The implementation is net-deleting and does not add a DI framework or new
  runtime service class.
