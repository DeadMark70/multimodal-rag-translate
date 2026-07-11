# Resumable GraphRAG Full Rebuild Design

## Summary

GraphRAG full rebuilds may process dozens of source documents and make many provider calls. The current implementation safely builds into a temporary `GraphStore` and replaces the live graph only after every document succeeds, but it always deletes the temporary directory when the task exits. A process crash, deployment, or unexpected exception therefore discards all completed work and forces the next rebuild to start from the first document.

This design replaces the disposable rebuild workspace with a persistent, user-scoped rebuild job. It checkpoints after each document, retries transient document failures up to three attempts, survives backend restarts, exposes pollable progress, and publishes the new graph only after every source document succeeds. Recovery is deliberately manual: after a backend restart, the job becomes `interrupted` and waits for the user to resume it.

## Goals

- Preserve completed document extraction work across document failures, process crashes, deployments, and backend restarts.
- Use one source document as the checkpoint unit. An interrupted in-progress document may be repeated; completed documents must not be repeated.
- Retry transient document failures automatically, with at most three total attempts per document.
- Continue processing remaining documents after one document exhausts its retries.
- Keep the existing live graph available until the replacement graph is complete and verified.
- Allow users to inspect overall and per-document progress from Graph Workspace.
- Allow manual resume after interruption and targeted retry of failed or partial documents.
- Preserve the current per-user graph-maintenance mutual exclusion across full rebuild, retry, purge, optimize, and node-vector sync operations.

## Non-goals

- Chunk-level checkpointing within a document.
- Automatically resuming model calls during application startup.
- Parallel document extraction.
- General-purpose distributed job scheduling.
- Redesigning GraphRAG as independently mergeable per-document graph fragments.
- Adding cancellation or arbitrary job-history management in the first release.

## Existing Behavior and Constraints

`rebuild_full_graph_task` currently creates a temporary graph directory, processes all OCR-complete documents sequentially, and replaces the live graph only when no document is `failed` or `partial`. This publication rule remains correct.

The recovery gap is caused by unconditional temporary-directory deletion in the task's `finally` block. Existing `GraphDocumentStatus` sidecars show document outcomes but do not identify a rebuild job, freeze its source set, persist its current phase, or point to a recoverable staging graph. `active_job_state` is a useful UI and compatibility field, but it is not sufficient as a durable job record or a cross-worker lease.

The implementation must continue using workspace-local directories because Windows temporary-directory permissions have been unreliable in this project. It must reuse `GraphStore` atomic sidecar writes and the existing live-graph replacement boundary.

## Architecture

### Persistent job layout

Each user may have one nonterminal full rebuild job. Its files live below the user's graph index directory:

```text
uploads/<user>/rag_index/rebuild_jobs/<job_id>/
├── manifest.json
├── runner.lock
└── staging_graph/
    ├── graph.pkl
    ├── graph.metadata.json
    ├── graph.documents.json
    ├── graph.extractions.json
    ├── graph.provenance.json
    └── existing GraphStore sidecars
```

`manifest.json` is the control-plane record. `staging_graph/` is a normal `GraphStore` with a non-live `storage_dir`. The live graph is never mutated by document extraction during a full rebuild.

### GraphRebuildJobStore

A focused job-store component owns:

- creation of a user-scoped job and immutable source snapshot;
- atomic manifest reads and writes;
- discovery of the current nonterminal job;
- validation that a job path belongs to the authenticated user;
- progress aggregation and public response projection;
- runner lease acquisition, heartbeat, release, and stale-lease detection;
- retention and explicit replacement of an unrecoverable job.

The job store must not perform GraphRAG extraction or graph optimization.

### GraphRebuildCoordinator

A coordinator owns the state machine and calls existing GraphRAG capabilities:

- select the next document from the frozen source snapshot;
- update document and job states;
- load OCR artifacts;
- call `run_graph_extraction(..., store=staging_store, autosync=False)`;
- apply retry classification and backoff;
- checkpoint successful document work;
- run final optimization, community regeneration, and required sidecar synchronization;
- verify staging completeness and atomically replace live graph files.

The FastAPI router remains thin: authenticate, validate the requested transition, acquire or reject the runner lease, schedule the coordinator, and return the job projection.

### Existing GraphStore

`GraphStore` remains authoritative for graph content and existing sidecars. It must not become responsible for job orchestration. The implementation reuses its atomic persistence and live replacement logic, adding only narrowly scoped helpers if needed for checkpoint validation.

## Manifest Model

The manifest is versioned so future migrations can be explicit.

```json
{
  "schema_version": 1,
  "job_id": "uuid",
  "user_id": "user-uuid",
  "state": "running",
  "phase": "extracting",
  "created_at": "RFC3339 timestamp",
  "started_at": "RFC3339 timestamp",
  "updated_at": "RFC3339 timestamp",
  "completed_at": null,
  "published_at": null,
  "current_doc_id": "document-uuid",
  "source_snapshot_hash": "sha256",
  "max_attempts": 3,
  "documents": [],
  "last_error": null,
  "lease": {
    "owner_token": "random token",
    "acquired_at": "RFC3339 timestamp",
    "heartbeat_at": "RFC3339 timestamp"
  }
}
```

Each document snapshot contains `doc_id`, display file name, OCR artifact location or stable lookup data, state, attempt count, timestamps, sanitized last error, and extraction outcome counts. The source list and its hash are immutable after job creation. Newly uploaded documents belong to a later rebuild; deleted or missing artifacts become explicit failures rather than silently changing the denominator.

Manifest paths returned to the frontend must never expose server filesystem paths.

## State Machines

### Job states

- `pending`: job exists but a runner has not begun extraction.
- `running`: a runner owns a valid lease and is actively processing or finalizing.
- `interrupted`: a previously running job has no valid live runner and can be resumed manually.
- `completed_with_failures`: all eligible documents were attempted, but one or more are `failed` or `partial`; the old graph remains live.
- `failed`: the job cannot safely continue because its manifest, staging graph, or checkpoint invariants are invalid.
- `completed`: staging validation and live publication succeeded.

Allowed primary transitions are:

```text
pending -> running -> completed
                   -> completed_with_failures -> running
                   -> interrupted -> running
                   -> failed
```

`completed` is terminal. `failed` requires an explicit new-rebuild action rather than blind resume.

### Job phases

- `preparing`
- `extracting`
- `retry_wait`
- `optimizing`
- `building_communities`
- `syncing_sidecars`
- `validating`
- `publishing`
- `done`

The phase gives the UI useful detail and enables finalize-only recovery. If extraction is complete and a later phase fails, resume continues finalization rather than repeating documents.

### Document states

- `pending`
- `running`
- `retry_wait`
- `indexed`
- `empty`
- `partial`
- `failed`

`indexed` and `empty` are successful checkpoint states. `partial` and `failed` block publication. A document found in `running` after interruption is returned to `pending` and its prior staging contribution is removed before the whole document is attempted again.

## Execution and Checkpoint Flow

1. Snapshot all OCR-complete source documents and create a new persistent job directory.
2. Create and clear a staging `GraphStore` within that job.
3. Acquire the runner lease and set the job to `running` / `extracting`.
4. For the next pending document, persist `running`, increment its attempt count, and set `current_doc_id`.
5. Load OCR artifacts and run extraction into the staging store with autosync disabled.
6. On a successful `indexed` or `empty` result:
   1. persist the staging graph and all relevant sidecars;
   2. reload or validate that the staging graph document set and latest extraction manifest agree with the result;
   3. atomically persist the document's successful state in the job manifest.
7. On a retryable failure, persist the sanitized error and `retry_wait`, wait with backoff, then retry if fewer than three attempts have occurred.
8. On a permanent failure or exhausted retry budget, persist `failed` or `partial` and continue to the next document.
9. After all documents have been attempted:
   - if any are `failed` or `partial`, set `completed_with_failures` and retain staging;
   - otherwise proceed through optimization, community regeneration, sidecar synchronization, validation, and publication.
10. Clear the staging store's maintenance marker, atomically replace the live graph files, then set the job to `completed` / `done`.

The checkpoint order intentionally commits graph data before marking a document successful. A crash may cause one document to be repeated, but it must never cause an unpersisted graph contribution to be treated as complete.

## Retry Policy

Each document receives at most three total attempts per execution cycle. The default delays before attempts two and three are 5 seconds and 20 seconds plus bounded jitter.

Retryable failures include provider rate limits, timeouts, and recognized transport failures. Permanent local failures such as missing OCR artifacts, invalid persisted inputs, or deterministic schema incompatibility do not consume unnecessary provider calls. The existing transport-error feature-detection rule must be preserved when classifying optional third-party exception symbols.

After `completed_with_failures`, the user can request retry of failed work. This transition resets only `failed` and `partial` documents to `pending`, resets their per-cycle attempt counters, and keeps `indexed` and `empty` checkpoints unchanged. Cumulative attempt and error history may be retained separately for diagnostics.

## Recovery Semantics

Application startup must not automatically invoke providers. A durable job whose manifest says `running` but whose lease is stale is projected as `interrupted`. The transition may be persisted during status reconciliation or the first authorized status read.

Manual resume performs these checks before acquiring a new runner lease:

- manifest schema and user ownership are valid;
- staging graph and required sidecars can be loaded;
- every `indexed` or `empty` checkpoint has a consistent graph/extraction record;
- the frozen source snapshot has not been silently modified;
- any document left `running` is reset for whole-document replay;
- phase and document states identify a deterministic next action.

If extraction is complete, resume continues the first unfinished finalize phase. A publication failure retains staging and may be retried without new extraction calls.

If validation cannot establish a safe checkpoint, the job becomes `failed` with a sanitized recovery error. The system must not publish questionable staging data.

## Concurrency and Lease

`active_job_state` remains available for compatibility and UI status, but correctness depends on a persistent exclusive runner lease.

Lease acquisition uses an atomic filesystem operation within the job directory and records a random owner token in the manifest. The runner refreshes its heartbeat while processing and before/after long provider calls. Only the matching owner may update active execution state or release the lease. A lease becomes reclaimable only after a configured stale interval and explicit manual resume.

The start and resume endpoints are idempotent. Concurrent requests can result in only one successful lease owner; other requests return the current job without launching another runner. Full rebuild continues to exclude retry, purge, optimize, and node-vector sync for the same user.

The implementation must account for multiple Uvicorn workers and must not rely solely on an in-memory lock.

## Publication Safety and Retention

The old graph remains the query target throughout rebuild and all failure states. Publication requires:

- no pending, running, retrying, failed, or partial documents;
- staging graph and sidecars load successfully;
- source snapshot coverage matches successful document checkpoints;
- optimization and community generation complete;
- required node-vector state is either synchronized or explicitly marked dirty according to the existing contract;
- the replacement helper completes without corrupting the live graph.

Successful job artifacts are retained for a configurable diagnostic period and may later be cleaned by a bounded maintenance policy. Failed, interrupted, and `completed_with_failures` jobs must not be automatically deleted. The first implementation may leave cleanup manual if a safe retention worker is outside scope.

## API Design

### Start

`POST /graph/rebuild-full`

- Starts a new job only when no recoverable nonterminal job exists.
- Returns the existing job projection when a rebuild already exists instead of creating a duplicate.
- Returns `job_id`, initial state, phase, and source document count.

### Status

`GET /graph/rebuild-full/status`

- Returns the current nonterminal job or most recent full rebuild.
- Reconciles a stale running lease into `interrupted` without starting work.
- Includes aggregate counts, percent, current document, timestamps, sanitized errors, per-document rows, and allowed actions.

The principal response fields are:

```json
{
  "job_id": "uuid",
  "state": "running",
  "phase": "extracting",
  "total": 50,
  "processed": 23,
  "succeeded": 21,
  "empty": 0,
  "failed": 1,
  "partial": 1,
  "pending": 27,
  "progress_percent": 46,
  "current_document": {
    "doc_id": "uuid",
    "file_name": "paper.pdf",
    "attempt": 2,
    "max_attempts": 3
  },
  "can_resume": false,
  "can_retry_failed": false,
  "live_graph_unchanged": true
}
```

`processed` means documents in `indexed`, `empty`, `failed`, or `partial`. `succeeded` means `indexed + empty`. `progress_percent` is `processed / total`, rounded consistently; a zero-document job is rejected at creation rather than producing an ambiguous percentage.

### Resume

`POST /graph/rebuild-full/resume`

- Resumes an `interrupted` job.
- For `completed_with_failures`, resets only `failed` and `partial` documents for another retry cycle.
- For an interrupted finalize phase, resumes finalization without repeating successful documents.
- Is idempotent under duplicate requests.

### Replace unrecoverable job

Starting over after a `failed` job requires an explicit request flag or separate confirmation endpoint. It must never occur as a side effect of a normal resume call. The exact endpoint shape may follow existing frontend confirmation patterns during implementation planning.

## Frontend Design

Graph Workspace gains a full-rebuild progress panel that is restored exclusively from the status API.

The panel displays:

- total progress bar and `processed / total` percentage;
- current phase in user-facing language;
- current document name and attempt number;
- indexed/empty, failed, partial, and pending counts;
- an expandable per-document status list with sanitized last errors;
- a persistent notice that queries still use the old graph until publication;
- `Continue rebuild` for `interrupted`;
- `Retry failed documents` for `completed_with_failures`;
- diagnostic guidance and explicit restart confirmation for `failed`.

React Query polling is preferred to SSE because the job must survive browser refresh, disconnection, deployment, and server restart. Poll approximately every two seconds only while the state is `pending` or `running`; use a slower refresh or user-triggered refetch for stable states. Duplicate button clicks must be disabled while the mutation is pending, while backend idempotency remains the correctness boundary.

## Error Handling and Security

- Persist and return sanitized error summaries, not provider payloads, credentials, stack traces, or absolute filesystem paths.
- Log job ID, user ID, phase, document ID, and attempt at each transition without logging document content.
- Treat manifest and persisted state as untrusted input and validate them through typed schemas.
- Resolve every job and staging path beneath the authenticated user's expected rebuild root before reading, writing, replacing, or deleting files.
- Do not add environment-flag auth bypasses to graph endpoints.
- Use atomic writes for manifests and sidecars; corrupted manifest recovery must fail closed rather than guessing progress.

## Testing Strategy

### Backend unit and integration tests

- A crash while processing document 24 of 50 retains checkpoints for documents 1 through 23.
- Status reconciliation after simulated restart yields `interrupted` and performs no provider call.
- Manual resume starts at document 24 and never re-extracts the first 23 documents.
- A document that fails twice with retryable errors and succeeds on attempt three completes normally.
- A permanent error is not retried three times.
- A document that exhausts retries does not stop later documents.
- Any `failed` or `partial` document produces `completed_with_failures` and preserves the old graph.
- Retrying only failed documents can lead to finalization and atomic publication.
- Failure during optimize, community generation, sidecar sync, validation, or publication does not repeat extracted documents.
- A document left `running` is safely removed/replayed as a whole document during resume.
- Missing staging data, corrupted manifests, ownership mismatches, and inconsistent successful checkpoints prevent publication.
- Concurrent start or resume requests yield one lease owner and one coordinator invocation.
- Stale leases require manual resume; application startup does not launch extraction.
- Status counts, percentage, current document, attempts, phases, errors, and allowed actions are correct.
- Existing graph retry, purge, optimize, node-vector sync, and live-replacement tests continue to pass.

### Frontend tests

- The progress panel renders aggregate and current-document state.
- Polling runs only for active states and survives component remount/refetch.
- Interrupted and failure states expose the correct actions.
- Mutation pending state prevents duplicate UI submissions.
- Per-document errors and old-graph notice render correctly.
- Completed state stops active polling and reports publication.

### Acceptance verification

- Run focused backend GraphRAG rebuild tests, then the full backend pytest suite.
- Run frontend `lint:ci`, `tsc --noEmit`, focused/full Vitest, and production build.
- Verify the Graph Workspace flow against a multi-document local test set, including a forced interruption and manual resume.
- Update `docs/BACKEND.md`, `docs/generated/api-surface.md`, `docs/FRONTEND.md`, and `docs/generated/ui-surface.md` in the implementation change because API and UI surfaces change.

## Success Criteria

- No successfully checkpointed document is reprocessed after a process restart.
- A single document failure cannot discard completed work or stop remaining documents.
- The system performs no automatic provider work after restart.
- Users can observe accurate progress and manually continue or retry failed work.
- The existing live graph remains queryable until the replacement graph is complete.
- A replacement graph is published only when every snapshotted source document is successful and final validation passes.
