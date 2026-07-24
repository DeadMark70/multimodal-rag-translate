# Task 2 report: campaign-scoped release snapshots

## Scope

Replaced the release-report N+1 path with campaign-scoped result, accounting, and observability snapshots. The fast path for anchors without a benchmark remains before campaign discovery and all expensive reads.

## RED evidence

Added `test_release_metrics_builds_campaign_runs_from_one_snapshot_per_repository`, parameterized for 1 and 160 result fakes. Before production changes it failed twice with:

`AttributeError: '_Results' object has no attribute 'list_for_campaign'`

This demonstrated that the previous report path still requested the full result list rather than the release projection. The test requires one call per selected campaign to each of: release result list, scores, score work metadata, accounting snapshot, and observability snapshot. It also preserves the two-result quality projection `[('naive-1', 0.4), ('v9-1', 0.7)]`.

## Implementation

- Added frozen `CampaignReleaseResult` and `list_for_campaign_release`, using the existing campaign/user/order index shape and excluding answer, contexts, source IDs, and RAGAS text fields.
- Added `CampaignAccountingSnapshot` and `load_campaign_snapshot`, grouping scopes by run and events by scope from one connection.
- Added `CampaignReleaseObservabilitySnapshot` and `load_campaign_release_snapshot`, grouping only release-gate materializations, evidence packets, slot resolutions, claims, context packs, and graph events by run.
- `ReleaseMetricsService.get_report` now gathers independent campaign reads and calls synchronous `_build_release_runs` for in-memory derivation. The result loop no longer calls `run_detail`, `_run_tokens`, `list_graph_events_for_run`, or opens per-run connections.
- V9 reconstruction selects the result's exact `source_attempt_id`, preserving prior attempt-level lookup semantics even when multiple materializations exist for a run.

## GREEN evidence

`uv run --python 3.13 --with-requirements requirements.txt python -m pytest -q tests/test_evaluation_release_metrics.py tests/test_evaluation_accounting_store.py tests/test_evaluation_observability_repository.py`

Result: `29 passed` (third-party Pydantic deprecation warnings only).

`uv run --python 3.13 --with-requirements requirements.txt ruff check evaluation/release_metrics.py evaluation/db.py evaluation/accounting_store.py evaluation/observability_storage.py tests/test_evaluation_release_metrics.py`

Result: `All checks passed!`

`python -m compileall` over the changed backend modules completed successfully.

## Self-review

- The projection uses `campaign_id` plus `user_id` and the existing `idx_campaign_results_campaign_user_order` ordering.
- The release builder derives accounting from the supplied snapshot and preserves the prior schema-v2 filtering and `_tokens` calculation.
- V9 parsing retains the previous fail-closed validation behavior; invalid/partial materializations resolve to N/A as before.
- No frontend, source data, plans/specs, or original checkout files were modified. The pre-existing Task 1 report change remains unstaged and excluded from the Task 2 commit.
