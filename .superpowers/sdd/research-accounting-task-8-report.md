# Task 8: strict research analytics and authenticated API

## Delivered

- Added the version-2 typed research-summary contract in `evaluation/accounting_schemas.py`.
- Added `ResearchAnalyticsService`, using campaign/result/RAGAS/accounting repositories directly rather than legacy analytics or `RagasEvaluator`.
- Added authenticated `GET /api/evaluation/campaigns/{campaign_id}/research-summary`.
- Kept missing token, quality, and price totals nullable; RAGAS batch usage is reported only as evaluation overhead.
- Added regression coverage for nearest-rank percentile behaviour and HTTP null serialization.

## Verification

`python -m pytest tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py tests/test_evaluation_analytics_api.py tests/test_evaluation_api.py -q`

Result: `17 passed` (third-party Pydantic deprecation warnings only).

`python -m ruff check evaluation/accounting_schemas.py evaluation/research_analytics.py evaluation/router.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py`

Result: all checks passed.

## Remaining risk

The focused fixtures cover contract/percentile/API behaviour. Broader literal durable-DB fixtures for retry, mixed evaluator signatures, and failed RAGAS work-item state should be expanded before frontend rollout.

## Follow-up hardening

Added durable-repository regression fixtures for a legacy completed run without a v2 scope and a partially evaluated v2 run. The latter proves primary metrics remain present and an absent faithfulness result remains `null` with `failed` status after evaluation activity has begun.

Re-ran the strict and existing evaluation API suites: `19 passed`.

## Completion edge coverage (2026-07-18)

- Added a durable mixed-mode campaign fixture that proves official successful
  execution scopes exclusively provide benchmark tokens/cost, while every
  execution attempt remains in operational pricing. RAGAS batch usage remains
  evaluation overhead and never contributes to execution tokens or cost.
- Quality aggregation now selects a deterministic compatible group by evaluator
  model, metric version, and evaluation signature. Incompatible rows are not
  averaged and make the mode non-comparable with an explicit warning.
- Added unequal mode sample-count coverage proving campaign aggregates use raw
  included runs, including nearest-rank percentiles and requested-only optional
  context metrics.
- Added an HTTP ownership fixture: a campaign owned by another user returns 404;
  nullable response values remain JSON `null`.

Verification: `23 passed` across the strict and existing evaluation/API suites;
Ruff check and format check passed. Third-party Pydantic deprecation warnings
remain pre-existing environment noise.

## Review correction (2026-07-18)

- Per-mode operational execution cost now maps every execution scope through its
  durable run or target identity before aggregating, so a failed or unpriced
  retry in one mode cannot change another mode's cost or pricing status.
- Optional context metrics are requested from durable `ragas_batch.metric_name`
  records even when the batch is running or terminally failed without a score.
- RAGAS scores now require an exact result-to-`source_attempt_id` match; stale
  or cross-linked scores are excluded. Missing metric versions remain `null` and
  are not inferred from an evaluation signature.

Verification: `25 passed` across the strict and existing evaluation/API suites;
Ruff check, format check, and `git diff --check` passed. Existing third-party
Pydantic deprecation warnings remain.
