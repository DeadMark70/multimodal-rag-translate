# Durable Evaluation Batch Gate

## Problem

Production campaigns are executed by the durable job worker. The worker has a
global execution concurrency of four, but its claim operation does not enforce
the campaign's configured `batch_size`. A campaign configured with
`batch_size=1` can therefore run several dataset units concurrently, causing
GPU contention and avoidable timeouts.

The legacy in-process runner already slices work by `batch_size`, but that path
is not authoritative for production durable jobs.

## Decision

Enforce `batch_size` when durable dataset work is claimed:

- Keep the worker-wide concurrency limit of four. It remains the global safety
  ceiling and permits independent campaigns to make progress concurrently.
- Apply a per-campaign admission limit to `dataset_execution` work only.
- Read the limit from each job's immutable `config_snapshot_json`.
- Default missing or invalid values to one and clamp values to the supported
  range of one through four.
- Count all currently running dataset items for the same campaign, including
  items belonging to another job for that campaign.
- Admit only the remaining capacity for each campaign.
- Leave RAGAS work unchanged.

## Concurrency Safety

Claims already run inside `BEGIN IMMEDIATE`. SQLite serializes these write
transactions, so a second worker observes the first worker's newly running
items before calculating campaign capacity. The admission calculation and the
status transition therefore remain atomic without introducing a second lock.

## Fairness

Admission filtering must happen before the worker-wide `LIMIT`. Otherwise a
full campaign at the front of the queue could hide claimable work from another
campaign. Candidate rows are ranked within their campaign and filtered by
remaining capacity before the global ordering and limit are applied.

## Acceptance Criteria

- A campaign with `batch_size=1` never has more than one running dataset item.
- A campaign with `batch_size=2` may claim two, but not three, dataset items.
- Two campaigns with `batch_size=1` may each run one item concurrently.
- Completing, failing, or cancelling a running item releases its campaign slot.
- RAGAS claim behavior and same-work-item exclusion remain unchanged.

