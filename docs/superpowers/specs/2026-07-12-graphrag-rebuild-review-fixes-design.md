# GraphRAG Rebuild Final Review Fixes — Design

## Goal

Close the final-review gaps in resumable GraphRAG full rebuilds: job-control durability
across graph snapshot promotion, immutable rebuild inputs, maintenance-operation exclusion,
transient provider retry coverage, and accurate retry-limit UI text.

## Stable rebuild control data

`GraphRebuildJobStore` will store manifests, runner locks, staging graphs, and frozen inputs
under a stable per-user graph root, outside `versions/v*`. Snapshot promotion must never move
or hide this directory. Reads will support the pre-release location long enough to migrate a
found legacy current job into the stable root atomically.

## Frozen source inputs

Starting a job reads each eligible document's OCR markdown before the job is scheduled. The
job stores one UTF-8 markdown copy per document under its own `sources/` directory and records
its SHA-256 in the frozen document snapshot. Extraction and resume always read these copies;
the original upload/OCR files are only used while creating the job. Failure to freeze any
source fails job creation before a runner or maintenance lock is acquired.

## Shared maintenance lease

A small, atomic per-user maintenance lock lives beside the stable job root. It has a random
owner token, activity label, process identifier, and timestamp. Every graph-mutating operation
(full rebuild, legacy rebuild, optimize, document retry, document purge, and node-vector sync)
must acquire it before scheduling or changing graph state, then release it in `finally`.
`active_job_state` remains a display/status sidecar, not the concurrency primitive. A stale
owner process is reconciled so a restart cannot leave users permanently blocked.

## Retry and public contract

Retry classification covers timeout, transport failures, HTTP 408, 429, and HTTP 5xx, including
provider exceptions that expose `status_code`. The public full-rebuild status adds `max_attempts`
from the durable manifest; the progress UI displays that value rather than a fixed literal.

## Test plan

- Promote a live graph snapshot after job creation, then prove status/resume still finds the same job.
- Mutate the original OCR markdown after job creation, then prove extraction uses the frozen copy.
- Exercise competing maintenance starts and prove only one obtains the durable lock; verify release after completion.
- Cover HTTP 408 and 5xx retry classification and the existing 429 path.
- Verify the UI renders the backend-supplied retry limit.

## Compatibility and safety

No live graph is changed by this migration. Existing legacy rebuild jobs are discovered and moved
only when their stable-root counterpart does not already exist. No job is scheduled until all
source inputs are frozen successfully. All lock writes use atomic create/replace semantics.
