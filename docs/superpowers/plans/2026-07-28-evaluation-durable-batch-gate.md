# Durable Evaluation Batch Gate Implementation Plan

## Scope

Fix only durable dataset-work admission. Do not change Agentic v9 retrieval,
reranking, deadlines, RAGAS execution, or the frontend.

## Task 1: Lock the contract with tests

Add store-level tests for:

1. `batch_size=1` admitting exactly one dataset item.
2. `batch_size=2` admitting exactly two dataset items.
3. independent campaigns each receiving capacity under a global claim.
4. a terminal attempt releasing capacity for the next item.
5. RAGAS claims remaining independent of the dataset gate.

Run the focused tests and confirm they fail against the current claim query.

## Task 2: Implement atomic admission

Update `EvaluationJobStore.claim_ready_items`:

1. Join candidates to their durable job and campaign.
2. derive a safe per-job batch size from `config_snapshot_json`;
3. calculate currently running dataset items for the campaign;
4. rank ready dataset items within each campaign;
5. retain only candidates that fit remaining campaign capacity;
6. apply global ordering and worker limit after admission filtering.

Keep the existing `BEGIN IMMEDIATE`, same-work-item exclusion, attempt
numbering, and append-only attempt creation unchanged.

## Task 3: Verify and commit

Run:

- the new focused admission tests;
- existing durable job-store tests;
- worker/campaign-engine tests relevant to claims;
- Ruff on modified Python files.

Review the diff for unrelated files, then commit the implementation separately
from this documentation commit.

