# Task 4 report: bounded research-result projections

## RED

Added `test_research_aggregates_use_bounded_result_projection_for_large_payloads`.
The test persists a completed result with 2 MiB `answer`, `contexts_json`, and
`ground_truth` values, snapshots the legacy public responses, then injects a
repository spy which rejects `list_for_campaign`.

The pre-change run failed as intended:

```text
AssertionError: research aggregate loaded full campaign results
evaluation/research_analytics.py:96 in get_summary
```

## GREEN

Added `CampaignResearchResult` and
`CampaignResultRepository.list_for_campaign_research`, selecting only the
metadata consumed by summary, question-comparison, and agent-behavior
aggregation. The projection excludes `answer`, `contexts_json`, and
`ground_truth`; it extracts and parses only `required_modalities` from the
question snapshot plus `derived_metrics_json`.

All three aggregate methods now prefer the bounded repository method, retaining
`list_for_campaign` only when an injected legacy test double lacks the new
method. The regression test verifies response parity, three bounded-loader
calls, zero full-loader calls, and selected-column SQL without the heavy fields.

Verification:

```text
uv run --python 3.13 --with-requirements requirements.txt ... pytest -q tests/test_evaluation_research_analytics.py
24 passed in 2.79s

uv run --python 3.13 --with-requirements requirements.txt ruff check evaluation/db.py evaluation/research_analytics.py tests/test_evaluation_research_analytics.py
All checks passed!

git diff --check
exit 0
```

The test runner stubs the unrelated heavyweight
`evaluation.agentic_evaluation_service` import with its durable
`LEGACY_SHARED_PROFILE` constant because the declared requirements do not
install Torch outside the project Docker environment. This does not affect the
repositories or analytics behavior under test.
