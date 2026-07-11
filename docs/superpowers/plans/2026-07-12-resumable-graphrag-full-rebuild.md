# Resumable GraphRAG Full Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a durable, manually resumable GraphRAG full rebuild with document-level checkpoints, bounded retries, atomic publication, and observable frontend progress.

**Architecture:** Persist one user-scoped rebuild manifest and staging `GraphStore` beneath `rag_index/rebuild_jobs/<job_id>`. A dedicated job store owns typed atomic state and cross-worker leases; a coordinator owns extraction, retry, recovery, validation, and publication; thin FastAPI routes expose start/status/resume; React Query polls durable status and Graph Workspace renders progress and recovery actions.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, asyncio, NetworkX/GraphStore sidecars, pytest/pytest-asyncio, React 18, TypeScript, TanStack Query, Chakra UI, Vitest/Testing Library.

## Global Constraints

- Use one source document as the checkpoint unit; never add chunk-level resume in this work.
- Each execution cycle allows at most three total attempts per failed document, with delays of 5 seconds and 20 seconds plus bounded jitter.
- Do not automatically invoke provider/model work during application startup or stale-job reconciliation.
- Keep the live graph unchanged until all snapshotted documents are `indexed` or `empty` and final validation passes.
- A `failed` or `partial` document blocks publication; retrying a failed job cycle resets only those documents.
- Newly uploaded documents do not alter an existing job's immutable source snapshot.
- Preserve same-user mutual exclusion across full rebuild, retry, purge, optimize, and node-vector sync.
- Do not rely solely on process-local locks; multiple Uvicorn workers must not launch duplicate runners.
- Use workspace-local persistent directories and atomic JSON writes; do not rely on `tempfile.mkdtemp(...)`.
- Never return absolute paths, credentials, raw provider payloads, or stack traces through status APIs.
- Update backend/frontend top-level guides and generated API/UI inventories in the implementation change set.

---

## File Structure

### Backend

- Create `graph_rag/rebuild_jobs.py`: typed durable manifest repository, progress projection, atomic lease, stale reconciliation, path containment, and error sanitization.
- Create `graph_rag/rebuild_coordinator.py`: document checkpoint loop, retry classification/backoff, recovery validation, finalization, and publication.
- Modify `graph_rag/schemas.py`: rebuild job/document/phase/state API and persistence models.
- Modify `graph_rag/maintenance.py`: retire the disposable full-rebuild implementation and delegate the compatibility task entrypoint to the coordinator; retain existing retry/purge/optimize helpers.
- Modify `graph_rag/router.py`: start/status/resume endpoints and mutual-exclusion integration.
- Create `tests/test_graph_rebuild_job_store.py`: manifest, path, lease, stale-state, and aggregation tests.
- Create `tests/test_graph_rebuild_coordinator.py`: checkpoint, retry, resume, failure, and publication tests.
- Modify `tests/test_graph_router_rebuild_full.py`: endpoint contracts, idempotency, auth-scoped status, and background scheduling tests.

### Frontend

- Modify `Multimodal_RAG_System/src/types/graph.ts`: typed rebuild job/document status contracts.
- Modify `Multimodal_RAG_System/src/services/graphApi.ts`: start/status/resume calls.
- Modify `Multimodal_RAG_System/src/services/graphApi.test.ts`: transport contract tests.
- Modify `Multimodal_RAG_System/src/hooks/useGraphData.ts`: polling status query and start/resume mutations.
- Modify `Multimodal_RAG_System/src/hooks/useGraphData.test.tsx`: polling and invalidation tests.
- Create `Multimodal_RAG_System/src/components/graph/GraphRebuildProgress.tsx`: focused progress/status/actions component.
- Create `Multimodal_RAG_System/src/components/graph/GraphRebuildProgress.test.tsx`: rendering/action tests.
- Modify `Multimodal_RAG_System/src/pages/GraphDemo.tsx`: compose progress panel and replace fire-and-forget rebuild handling.
- Modify `Multimodal_RAG_System/src/pages/GraphDemo.test.tsx`: page integration regression tests.

### Documentation

- Modify `docs/BACKEND.md` and `docs/generated/api-surface.md`.
- Modify `Multimodal_RAG_System/docs/FRONTEND.md` and `Multimodal_RAG_System/docs/generated/ui-surface.md`.

---

### Task 1: Define Typed Rebuild State and Atomic Manifest Storage

**Files:**
- Modify: `graph_rag/schemas.py`
- Create: `graph_rag/rebuild_jobs.py`
- Create: `tests/test_graph_rebuild_job_store.py`

**Interfaces:**
- Consumes: `GraphStore.storage_dir`, Pydantic v2 `model_validate`/`model_dump`, existing atomic sidecar conventions.
- Produces: `GraphRebuildManifest`, `GraphRebuildDocument`, `GraphRebuildStatusResponse`, and `GraphRebuildJobStore(user_id: str)` with `create_job`, `load_current`, `save`, `to_status`, and `mark_interrupted_if_stale`.

- [ ] **Step 1: Write failing schema and manifest round-trip tests**

```python
def test_create_job_freezes_sources_and_round_trips(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = store.create_job([
        {"doc_id": "doc-1", "file_name": "a.pdf", "original_path": "uploads/user-1/doc-1/a.pdf"},
        {"doc_id": "doc-2", "file_name": "b.pdf", "original_path": None},
    ])

    restored = store.load_current()
    assert restored is not None
    assert restored.job_id == manifest.job_id
    assert [item.doc_id for item in restored.documents] == ["doc-1", "doc-2"]
    assert restored.source_snapshot_hash == manifest.source_snapshot_hash
    assert restored.state == "pending"


def test_status_aggregates_terminal_document_counts(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = store.create_job(SOURCES)
    manifest.documents[0].state = "indexed"
    manifest.documents[1].state = "failed"
    store.save(manifest)

    status = store.to_status(manifest)
    assert status.processed == 2
    assert status.succeeded == 1
    assert status.failed == 1
    assert status.progress_percent == 100
    assert status.live_graph_unchanged is True
```

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_job_store.py -v`

Expected: collection fails because `graph_rag.rebuild_jobs` and rebuild schemas do not exist.

- [ ] **Step 3: Add exact state models to `graph_rag/schemas.py`**

```python
GraphRebuildJobState = Literal[
    "pending", "running", "interrupted", "completed_with_failures", "failed", "completed"
]
GraphRebuildPhase = Literal[
    "preparing", "extracting", "retry_wait", "optimizing",
    "building_communities", "syncing_sidecars", "validating", "publishing", "done",
]
GraphRebuildDocumentState = Literal[
    "pending", "running", "retry_wait", "indexed", "empty", "partial", "failed"
]


class GraphRebuildDocument(BaseModel):
    doc_id: str
    file_name: str | None = None
    original_path: str | None = Field(default=None, exclude=True)
    state: GraphRebuildDocumentState = "pending"
    attempt: int = Field(default=0, ge=0)
    cumulative_attempts: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    chunks_succeeded: int = Field(default=0, ge=0)
    chunks_failed: int = Field(default=0, ge=0)
    last_error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class GraphRebuildLease(BaseModel):
    owner_token: str
    acquired_at: datetime
    heartbeat_at: datetime


class GraphRebuildManifest(BaseModel):
    schema_version: Literal[1] = 1
    job_id: str
    user_id: str
    state: GraphRebuildJobState = "pending"
    phase: GraphRebuildPhase = "preparing"
    created_at: datetime
    started_at: datetime | None = None
    updated_at: datetime
    completed_at: datetime | None = None
    published_at: datetime | None = None
    current_doc_id: str | None = None
    source_snapshot_hash: str
    max_attempts: int = 3
    documents: list[GraphRebuildDocument]
    last_error: str | None = None
    lease: GraphRebuildLease | None = None
```

- [ ] **Step 4: Implement minimal atomic repository and status projection**

```python
class GraphRebuildJobStore:
    def __init__(self, user_id: str, rebuild_root: Path | None = None) -> None:
        self.user_id = user_id
        live_store = GraphStore(user_id)
        self.root = (rebuild_root or live_store.storage_dir / "rebuild_jobs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_job(self, sources: list[dict[str, str | None]]) -> GraphRebuildManifest:
        if not sources:
            raise ValueError("A full rebuild requires at least one source document")
        normalized = sorted(sources, key=lambda item: str(item["doc_id"]))
        snapshot = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        now = datetime.now(timezone.utc)
        manifest = GraphRebuildManifest(
            job_id=str(uuid4()), user_id=self.user_id, created_at=now, updated_at=now,
            source_snapshot_hash=hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
            documents=[GraphRebuildDocument(**source) for source in normalized],
        )
        self.save(manifest)
        return manifest

    def save(self, manifest: GraphRebuildManifest) -> None:
        manifest.updated_at = datetime.now(timezone.utc)
        self._assert_owned_path(self._manifest_path(manifest.job_id))
        self._atomic_write_json(self._manifest_path(manifest.job_id), manifest.model_dump(mode="json"))
        self._atomic_write_json(self.root / "current.json", {"job_id": manifest.job_id})
```

Implement `load_current()` with typed validation, `_assert_owned_path()` using `Path.resolve().is_relative_to(self.root)`, and `to_status()` with `processed = indexed + empty + failed + partial`, `succeeded = indexed + empty`, and integer percentage.

- [ ] **Step 5: Run focused tests**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_job_store.py -v`

Expected: all manifest/schema/aggregation tests pass.

- [ ] **Step 6: Commit the typed persistence boundary**

```bash
git add graph_rag/schemas.py graph_rag/rebuild_jobs.py tests/test_graph_rebuild_job_store.py
git commit -m "feat: persist GraphRAG rebuild jobs"
```

---

### Task 2: Add Cross-Worker Lease and Manual Stale Reconciliation

**Files:**
- Modify: `graph_rag/rebuild_jobs.py`
- Modify: `tests/test_graph_rebuild_job_store.py`

**Interfaces:**
- Consumes: `GraphRebuildJobStore.save(manifest)` from Task 1.
- Produces: `acquire_lease(job_id) -> str | None`, `heartbeat(job_id, owner_token)`, `release_lease(job_id, owner_token)`, and `reconcile_status(manifest) -> GraphRebuildManifest`.

- [ ] **Step 1: Write failing atomic lease and stale reconciliation tests**

```python
def test_only_one_store_can_acquire_runner_lease(tmp_path: Path) -> None:
    first = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    manifest = first.create_job(SOURCES)
    second = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)

    token = first.acquire_lease(manifest.job_id)
    assert token is not None
    assert second.acquire_lease(manifest.job_id) is None


def test_stale_running_job_becomes_interrupted_without_starting_work(tmp_path: Path) -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path, lease_ttl=timedelta(seconds=30))
    manifest = store.create_job(SOURCES)
    manifest.state = "running"
    manifest.lease = GraphRebuildLease(
        owner_token="dead", acquired_at=OLD_TIME, heartbeat_at=OLD_TIME
    )
    store.save(manifest)

    reconciled = store.reconcile_status(store.load_current())
    assert reconciled.state == "interrupted"
    assert reconciled.lease is None
```

- [ ] **Step 2: Run tests and verify lease behavior is absent**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_job_store.py -k "lease or stale" -v`

Expected: FAIL because lease methods are undefined.

- [ ] **Step 3: Implement owner-token lease with atomic lock creation**

```python
def acquire_lease(self, job_id: str) -> str | None:
    lock_path = self._job_dir(job_id) / "runner.lock"
    token = secrets.token_urlsafe(32)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"owner_token": token}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    manifest = self.load(job_id)
    now = self._now()
    manifest.lease = GraphRebuildLease(owner_token=token, acquired_at=now, heartbeat_at=now)
    self.save(manifest)
    return token
```

`heartbeat` and `release_lease` must compare the manifest and lock-file owner token. Stale reconciliation removes only a stale lock, clears the lease, writes `interrupted`, and never calls a coordinator or provider.

- [ ] **Step 4: Run focused and full job-store tests**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_job_store.py -v`

Expected: all tests pass, including duplicate acquisition and wrong-owner release rejection.

- [ ] **Step 5: Commit lease safety**

```bash
git add graph_rag/rebuild_jobs.py tests/test_graph_rebuild_job_store.py
git commit -m "feat: lease GraphRAG rebuild runners"
```

---

### Task 3: Implement Document-Level Checkpoints and Bounded Retry

**Files:**
- Create: `graph_rag/rebuild_coordinator.py`
- Create: `tests/test_graph_rebuild_coordinator.py`
- Modify: `graph_rag/maintenance.py`

**Interfaces:**
- Consumes: `GraphRebuildJobStore`, `GraphRebuildManifest`, `list_graph_source_documents`, `run_graph_extraction`, `load_ocr_artifacts`, `GraphStore`.
- Produces: `GraphRebuildCoordinator.run(user_id: str, job_id: str, owner_token: str) -> None` and compatibility `rebuild_full_graph_task(user_id, job_id, owner_token)`.

- [ ] **Step 1: Write failing checkpoint and retry tests**

```python
@pytest.mark.asyncio
async def test_checkpointed_documents_are_not_reprocessed_on_resume(job_fixture) -> None:
    manifest = job_fixture.manifest
    manifest.documents[0].state = "indexed"
    job_fixture.job_store.save(manifest)
    extract = AsyncMock(return_value=indexed_result("doc-2"))
    coordinator = GraphRebuildCoordinator(job_fixture.job_store, run_extraction=extract)

    await coordinator.run("user-1", manifest.job_id, job_fixture.owner_token)

    assert [call.kwargs["doc_id"] for call in extract.await_args_list] == ["doc-2"]


@pytest.mark.asyncio
async def test_retryable_failure_succeeds_on_third_attempt(job_fixture) -> None:
    extract = AsyncMock(side_effect=[TimeoutError(), TimeoutError(), indexed_result("doc-1")])
    sleep = AsyncMock()
    coordinator = GraphRebuildCoordinator(job_fixture.job_store, run_extraction=extract, sleep=sleep)

    await coordinator.run("user-1", job_fixture.job_id, job_fixture.owner_token)

    assert extract.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [5, 20]
```

- [ ] **Step 2: Verify tests fail before coordinator exists**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_coordinator.py -k "checkpointed or third_attempt" -v`

Expected: collection fails for missing `GraphRebuildCoordinator`.

- [ ] **Step 3: Implement the sequential coordinator loop**

```python
class GraphRebuildCoordinator:
    RETRY_DELAYS = (5.0, 20.0)

    async def run(self, user_id: str, job_id: str, owner_token: str) -> None:
        manifest = self.jobs.load(job_id)
        staging = GraphStore(user_id, storage_dir=self.jobs.staging_dir(job_id))
        manifest.state = "running"
        manifest.phase = "extracting"
        self.jobs.save(manifest)
        for document in manifest.documents:
            if document.state in {"indexed", "empty"}:
                continue
            await self._process_document(manifest, document, staging, owner_token)
        await self._finish_extraction(manifest, staging, owner_token)
```

In `_process_document`, persist `running` before the call; on success save the staging store, validate `doc_id` plus latest extraction manifest, then persist `indexed`/`empty`. Before replaying a previously `running` document, call `staging.remove_document(doc_id)` and remove its prior status.

- [ ] **Step 4: Implement explicit retry classification**

```python
def is_retryable_rebuild_error(exc: Exception) -> bool:
    optional_httpcore_error = getattr(httpcore, "HTTPError", None)
    retry_types = (TimeoutError, httpx.TransportError)
    if isinstance(optional_httpcore_error, type):
        retry_types = (*retry_types, optional_httpcore_error)
    status_code = getattr(exc, "status_code", None)
    return isinstance(exc, retry_types) or status_code == 429
```

Permanent OCR artifact and validation failures go directly to `failed`. Provider errors use `RETRY_DELAYS` plus injected bounded jitter in production; tests inject zero jitter.

- [ ] **Step 5: Replace disposable task body with coordinator delegation**

```python
async def rebuild_full_graph_task(user_id: str, job_id: str, owner_token: str) -> None:
    coordinator = GraphRebuildCoordinator(GraphRebuildJobStore(user_id))
    await coordinator.run(user_id, job_id, owner_token)
```

Do not delete the job directory in `finally`. Release only the owned lease and refresh compatibility `active_job_state` after the durable state is saved.

- [ ] **Step 6: Run coordinator and legacy maintenance tests**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_coordinator.py tests/test_graph_router_rebuild_full.py -v`

Expected: checkpoint/retry tests pass; legacy tests may require signature updates completed in Task 5, but unrelated retry/purge tests remain green.

- [ ] **Step 7: Commit resumable extraction**

```bash
git add graph_rag/rebuild_coordinator.py graph_rag/maintenance.py tests/test_graph_rebuild_coordinator.py
git commit -m "feat: checkpoint GraphRAG full rebuilds"
```

---

### Task 4: Add Failure Completion, Finalize-Only Recovery, and Atomic Publication

**Files:**
- Modify: `graph_rag/rebuild_coordinator.py`
- Modify: `graph_rag/rebuild_jobs.py`
- Modify: `tests/test_graph_rebuild_coordinator.py`

**Interfaces:**
- Consumes: coordinator loop from Task 3 and `_replace_live_graph_files`/`optimize_existing_graph` from existing maintenance code.
- Produces: `_finish_extraction`, `_validate_for_publication`, and resume-safe phase transitions.

- [ ] **Step 1: Write failing failure/publication recovery tests**

```python
@pytest.mark.asyncio
async def test_failed_document_keeps_old_graph_and_retains_staging(job_fixture) -> None:
    coordinator = coordinator_with_results(job_fixture, [failed_result("doc-1"), indexed_result("doc-2")])
    await coordinator.run("user-1", job_fixture.job_id, job_fixture.owner_token)

    manifest = job_fixture.job_store.load(job_fixture.job_id)
    assert manifest.state == "completed_with_failures"
    assert job_fixture.replace_live_graph_files.call_count == 0
    assert job_fixture.job_store.staging_dir(job_fixture.job_id).exists()


@pytest.mark.asyncio
async def test_resume_after_optimize_failure_does_not_extract_again(job_fixture) -> None:
    first = coordinator_with_finalize_failure(job_fixture, phase="optimizing")
    await first.run("user-1", job_fixture.job_id, job_fixture.owner_token)
    second = coordinator_for_resume(job_fixture)
    await second.run("user-1", job_fixture.job_id, job_fixture.new_owner_token)

    assert second.run_extraction.await_count == 0
    second.replace_live_graph_files.assert_called_once()
```

- [ ] **Step 2: Run the new tests and verify missing finalization behavior**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_coordinator.py -k "old_graph or optimize_failure" -v`

Expected: FAIL because the coordinator does not yet persist/recover finalize phases.

- [ ] **Step 3: Implement phase-persisted finalization**

```python
async def _finalize(self, manifest: GraphRebuildManifest, staging: GraphStore) -> None:
    manifest.phase = "optimizing"
    self.jobs.save(manifest)
    await optimize_existing_graph(staging, regenerate_communities=True)
    manifest.phase = "syncing_sidecars"
    self.jobs.save(manifest)
    await self._sync_required_sidecars(staging)
    manifest.phase = "validating"
    self.jobs.save(manifest)
    self._validate_for_publication(manifest, staging)
    manifest.phase = "publishing"
    self.jobs.save(manifest)
    self.replace_live_graph_files(staging, GraphStore(manifest.user_id))
    manifest.state = "completed"
    manifest.phase = "done"
    manifest.published_at = self.now()
    manifest.completed_at = self.now()
    self.jobs.save(manifest)
```

Resume branches on persisted `phase`: if all documents are successful, bypass extraction and rerun the idempotent first incomplete finalize operation. Validation checks successful document coverage and loadability before publication.

- [ ] **Step 4: Implement retry-failed reset semantics**

```python
def reset_failed_documents(manifest: GraphRebuildManifest) -> GraphRebuildManifest:
    for document in manifest.documents:
        if document.state in {"failed", "partial"}:
            document.state = "pending"
            document.attempt = 0
            document.last_error = None
    manifest.state = "pending"
    manifest.phase = "extracting"
    manifest.last_error = None
    return manifest
```

- [ ] **Step 5: Run all coordinator tests**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_coordinator.py -v`

Expected: all checkpoint, retry, failure, recovery, corruption, and publication tests pass.

- [ ] **Step 6: Commit finalization safety**

```bash
git add graph_rag/rebuild_coordinator.py graph_rag/rebuild_jobs.py tests/test_graph_rebuild_coordinator.py
git commit -m "feat: safely finalize resumable graph rebuilds"
```

---

### Task 5: Expose Idempotent Start, Status, and Resume APIs

**Files:**
- Modify: `graph_rag/router.py`
- Modify: `tests/test_graph_router_rebuild_full.py`

**Interfaces:**
- Consumes: job store/coordinator task signatures from Tasks 1-4.
- Produces: `POST /graph/rebuild-full`, `GET /graph/rebuild-full/status`, and `POST /graph/rebuild-full/resume`, all returning `GraphRebuildStatusResponse`.

- [ ] **Step 1: Write failing endpoint contract tests**

```python
def test_rebuild_full_start_returns_durable_job_and_schedules_owner_token() -> None:
    response = client.post("/graph/rebuild-full")
    assert response.status_code == 200
    assert response.json()["state"] in {"pending", "running"}
    assert response.json()["total"] == 2
    mock_task.assert_awaited_once_with(TEST_USER_ID, "job-1", "owner-token")


def test_status_reconciles_stale_job_without_scheduling_task() -> None:
    response = client.get("/graph/rebuild-full/status")
    assert response.status_code == 200
    assert response.json()["state"] == "interrupted"
    mock_task.assert_not_called()


def test_duplicate_resume_schedules_only_one_runner() -> None:
    first = client.post("/graph/rebuild-full/resume")
    second = client.post("/graph/rebuild-full/resume")
    assert first.status_code == 200
    assert second.status_code == 200
    assert mock_task.await_count == 1
```

- [ ] **Step 2: Verify old endpoint response/signature fails the new tests**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/test_graph_router_rebuild_full.py -k "durable or stale or duplicate_resume" -v`

Expected: FAIL because status/resume routes and durable response are absent.

- [ ] **Step 3: Implement thin route helpers and endpoints**

```python
async def _schedule_rebuild(
    background_tasks: BackgroundTasks,
    store: GraphRebuildJobStore,
    manifest: GraphRebuildManifest,
) -> GraphRebuildStatusResponse:
    owner_token = store.acquire_lease(manifest.job_id)
    if owner_token is not None:
        manifest.state = "running"
        store.save(manifest)
        background_tasks.add_task(_rebuild_full_graph_task, store.user_id, manifest.job_id, owner_token)
    return store.to_status(store.load(manifest.job_id))
```

Start returns an existing recoverable job instead of creating a second one. Status calls only `reconcile_status`. Resume accepts only `interrupted` or `completed_with_failures`; the latter calls `reset_failed_documents` before acquiring a lease. Retain explicit auth dependency and existing graph-maintenance exclusion.

- [ ] **Step 4: Run router and boundary tests**

Run: `\.\.venv\Scripts\python.exe -m pytest tests/test_graph_router_rebuild_full.py tests/test_router_boundaries.py -v`

Expected: all pass; router modules still do not import other router modules.

- [ ] **Step 5: Commit the API surface**

```bash
git add graph_rag/router.py tests/test_graph_router_rebuild_full.py
git commit -m "feat: expose resumable graph rebuild API"
```

---

### Task 6: Add Frontend Contracts, API Calls, and Polling Hooks

**Files:**
- Modify: `../Multimodal_RAG_System/src/types/graph.ts`
- Modify: `../Multimodal_RAG_System/src/services/graphApi.ts`
- Modify: `../Multimodal_RAG_System/src/services/graphApi.test.ts`
- Modify: `../Multimodal_RAG_System/src/hooks/useGraphData.ts`
- Modify: `../Multimodal_RAG_System/src/hooks/useGraphData.test.tsx`

**Interfaces:**
- Consumes: backend `GraphRebuildStatusResponse` from Task 5.
- Produces: `getFullGraphRebuildStatus`, `resumeFullGraphRebuild`, `useFullGraphRebuildStatus`, and `useResumeFullGraphRebuild`.

- [ ] **Step 1: Write failing transport and polling tests**

```typescript
it('gets durable full rebuild status', async () => {
  mockedApi.get.mockResolvedValueOnce({ data: rebuildStatus });
  await expect(getFullGraphRebuildStatus()).resolves.toEqual(rebuildStatus);
  expect(mockedApi.get).toHaveBeenCalledWith('/graph/rebuild-full/status');
});

it('polls every two seconds only while running', () => {
  const { result } = renderHook(() => useFullGraphRebuildStatus(), { wrapper });
  const refetchInterval = capturedQueryOptions.refetchInterval;
  expect(refetchInterval({ state: { data: runningStatus } })).toBe(2000);
  expect(refetchInterval({ state: { data: interruptedStatus } })).toBe(false);
});
```

- [ ] **Step 2: Run focused frontend tests and verify missing exports**

Run from `../Multimodal_RAG_System`: `npm test -- --run src/services/graphApi.test.ts src/hooks/useGraphData.test.tsx`

Expected: FAIL because new API functions and hooks are undefined.

- [ ] **Step 3: Add exact TypeScript contracts**

```typescript
export type GraphRebuildJobState =
  | 'pending' | 'running' | 'interrupted'
  | 'completed_with_failures' | 'failed' | 'completed';

export interface GraphRebuildDocumentStatus {
  doc_id: string;
  file_name: string | null;
  state: 'pending' | 'running' | 'retry_wait' | 'indexed' | 'empty' | 'partial' | 'failed';
  attempt: number;
  cumulative_attempts: number;
  last_error: string | null;
}

export interface GraphRebuildStatus {
  job_id: string;
  state: GraphRebuildJobState;
  phase: string;
  total: number;
  processed: number;
  succeeded: number;
  empty: number;
  failed: number;
  partial: number;
  pending: number;
  progress_percent: number;
  current_document: GraphRebuildDocumentStatus | null;
  documents: GraphRebuildDocumentStatus[];
  can_resume: boolean;
  can_retry_failed: boolean;
  live_graph_unchanged: boolean;
  last_error: string | null;
}
```

- [ ] **Step 4: Implement service functions and query hooks**

```typescript
export async function getFullGraphRebuildStatus(): Promise<GraphRebuildStatus | null> {
  const response = await api.get<GraphRebuildStatus | null>('/graph/rebuild-full/status');
  return response.data;
}

export async function resumeFullGraphRebuild(): Promise<GraphRebuildStatus> {
  const response = await api.post<GraphRebuildStatus>('/graph/rebuild-full/resume');
  return response.data;
}

export function useFullGraphRebuildStatus() {
  return useQuery({
    queryKey: ['graph', 'rebuild-full', 'status'],
    queryFn: getFullGraphRebuildStatus,
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return state === 'pending' || state === 'running' ? 2000 : false;
    },
  });
}
```

Start and resume mutations invalidate both `['graph']` and the rebuild status key. Use `import type` for all newly added TanStack Query/types imports.

- [ ] **Step 5: Run service/hook tests and strict lint**

Run from `../Multimodal_RAG_System`: `npm test -- --run src/services/graphApi.test.ts src/hooks/useGraphData.test.tsx`

Expected: PASS.

Run: `npm run lint:ci`

Expected: PASS with zero warnings.

- [ ] **Step 6: Commit frontend data boundaries**

```bash
git add src/types/graph.ts src/services/graphApi.ts src/services/graphApi.test.ts src/hooks/useGraphData.ts src/hooks/useGraphData.test.tsx
git commit -m "feat: query GraphRAG rebuild progress"
```

---

### Task 7: Render Rebuild Progress and Recovery Actions in Graph Workspace

**Files:**
- Create: `../Multimodal_RAG_System/src/components/graph/GraphRebuildProgress.tsx`
- Create: `../Multimodal_RAG_System/src/components/graph/GraphRebuildProgress.test.tsx`
- Modify: `../Multimodal_RAG_System/src/pages/GraphDemo.tsx`
- Modify: `../Multimodal_RAG_System/src/pages/GraphDemo.test.tsx`

**Interfaces:**
- Consumes: `GraphRebuildStatus` and hooks from Task 6.
- Produces: `GraphRebuildProgress({ status, isActionPending, onResume })`.

- [ ] **Step 1: Write failing component behavior tests**

```typescript
it('shows durable progress and the old-graph notice', () => {
  render(<GraphRebuildProgress status={runningStatus} isActionPending={false} onResume={vi.fn()} />);
  expect(screen.getByText('23 / 50（46%）')).toBeInTheDocument();
  expect(screen.getByText(/目前查詢仍使用舊圖譜/)).toBeInTheDocument();
  expect(screen.getByText(/paper.pdf/)).toBeInTheDocument();
  expect(screen.getByText(/第 2 次，共 3 次/)).toBeInTheDocument();
});

it('offers resume for interrupted jobs', async () => {
  const onResume = vi.fn();
  render(<GraphRebuildProgress status={interruptedStatus} isActionPending={false} onResume={onResume} />);
  await userEvent.click(screen.getByRole('button', { name: '繼續重建' }));
  expect(onResume).toHaveBeenCalledOnce();
});
```

- [ ] **Step 2: Run component tests and verify missing component failure**

Run from `../Multimodal_RAG_System`: `npm test -- --run src/components/graph/GraphRebuildProgress.test.tsx`

Expected: FAIL because the component does not exist.

- [ ] **Step 3: Implement the focused Chakra component**

```tsx
export function GraphRebuildProgress({ status, isActionPending, onResume }: Props) {
  const actionLabel = status.state === 'completed_with_failures'
    ? '重試失敗文件'
    : status.state === 'interrupted' ? '繼續重建' : null;
  return (
    <Card data-testid="graph-rebuild-progress">
      <CardBody>
        <HStack justify="space-between">
          <Heading size="sm">完整重建進度</Heading>
          <Text>{status.processed} / {status.total}（{status.progress_percent}%）</Text>
        </HStack>
        <Progress value={status.progress_percent} mt={3} />
        {status.current_document && (
          <Text mt={2}>{status.current_document.file_name ?? status.current_document.doc_id}：第 {status.current_document.attempt} 次，共 3 次</Text>
        )}
        {status.live_graph_unchanged && <Alert status="info" mt={3}><AlertIcon />目前查詢仍使用舊圖譜。</Alert>}
        {actionLabel && <Button mt={3} onClick={onResume} isLoading={isActionPending}>{actionLabel}</Button>}
      </CardBody>
    </Card>
  );
}
```

Add phase labels, success/failure/partial/pending statistics, expandable document rows, sanitized error display, and failed-state restart guidance without rendering any server path.

- [ ] **Step 4: Integrate the panel into `GraphDemo`**

```tsx
const rebuildStatusQuery = useFullGraphRebuildStatus();
const resumeRebuildMutation = useResumeFullGraphRebuild();

{rebuildStatusQuery.data && (
  <GraphRebuildProgress
    status={rebuildStatusQuery.data}
    isActionPending={resumeRebuildMutation.isPending}
    onResume={() => resumeRebuildMutation.mutate()}
  />
)}
```

The existing full rebuild button starts a durable job and then relies on the status query. Disable conflicting graph maintenance buttons while durable state is `pending` or `running`, not merely while the start request is pending.

- [ ] **Step 5: Run component and page tests**

Run from `../Multimodal_RAG_System`: `npm test -- --run src/components/graph/GraphRebuildProgress.test.tsx src/pages/GraphDemo.test.tsx`

Expected: PASS, including remount/status restoration and action-button tests.

- [ ] **Step 6: Run frontend lint and type checking immediately**

Run: `npm run lint:ci`

Expected: PASS with zero warnings.

Run: `npx tsc --noEmit`

Expected: PASS.

- [ ] **Step 7: Commit the observable UI**

```bash
git add src/components/graph/GraphRebuildProgress.tsx src/components/graph/GraphRebuildProgress.test.tsx src/pages/GraphDemo.tsx src/pages/GraphDemo.test.tsx
git commit -m "feat: show GraphRAG rebuild progress"
```

---

### Task 8: Update System-of-Record Docs and Run Acceptance Verification

**Files:**
- Modify: `docs/BACKEND.md`
- Modify: `docs/generated/api-surface.md`
- Modify: `../Multimodal_RAG_System/docs/FRONTEND.md`
- Modify: `../Multimodal_RAG_System/docs/generated/ui-surface.md`
- Modify: `agent.md` only if implementation reveals a new project-wide prevention rule.

**Interfaces:**
- Consumes: completed backend/frontend behavior from Tasks 1-7.
- Produces: current API/UI documentation and verified release evidence.

- [ ] **Step 1: Update backend runtime and API documentation**

Document the persistent job directory, manual restart recovery, document checkpoint, three-attempt retry policy, old-graph publication rule, and exact start/status/resume endpoints. In `docs/generated/api-surface.md`, add:

```markdown
- `POST /graph/rebuild-full` — create or return the current durable full rebuild job
- `GET /graph/rebuild-full/status` — poll aggregate and per-document rebuild progress
- `POST /graph/rebuild-full/resume` — manually resume interruption or retry failed/partial documents
```

- [ ] **Step 2: Update frontend runtime and generated UI documentation**

Document Graph Workspace's progress panel, two-second active polling, old-graph notice, current document/attempt display, expandable failures, and manual continue/retry actions.

- [ ] **Step 3: Run focused backend verification**

Run from `pdftopng`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_job_store.py tests/test_graph_rebuild_coordinator.py tests/test_graph_router_rebuild_full.py -v
```

Expected: all focused rebuild tests pass.

- [ ] **Step 4: Run full backend acceptance**

Run: `.\.venv\Scripts\python.exe -m pytest`

Expected: full backend suite passes with no failures.

- [ ] **Step 5: Run full frontend acceptance**

Run from `../Multimodal_RAG_System`:

```powershell
npm run lint:ci
npx tsc --noEmit
npx vitest run
npm run build
```

Expected: all four commands pass; the production build emits without TypeScript or Rollup errors.

- [ ] **Step 6: Perform the forced-interruption smoke test**

Using a local multi-document test user, start a rebuild, force the runner to stop after at least one completed document, restart the backend, and verify:

```text
status == interrupted
provider calls after startup == 0
processed checkpoint count is unchanged
manual resume skips completed documents
live graph remains unchanged until publication
```

- [ ] **Step 7: Review the final diff for secrets, paths, and unrelated changes**

Run:

```powershell
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors, no config/secrets/output artifacts, and only planned backend/frontend/docs files changed.

- [ ] **Step 8: Commit backend documentation in the backend repository**

```bash
git add docs/BACKEND.md docs/generated/api-surface.md
git commit -m "docs: describe resumable graph rebuilds"
```

- [ ] **Step 9: Commit frontend documentation in the frontend repository**

Run from `../Multimodal_RAG_System`:

```bash
git add docs/FRONTEND.md docs/generated/ui-surface.md
git commit -m "docs: describe graph rebuild progress"
```

---

## Implementation Completion Checklist

- [ ] Durable manifest and staging graph survive task/process termination.
- [ ] Successful documents are skipped on resume; an interrupted current document is replayed whole.
- [ ] Retryable failures use the bounded three-attempt policy; permanent failures fail fast.
- [ ] Failed/partial documents do not stop later documents and always block publication.
- [ ] Startup/status reconciliation performs no provider work.
- [ ] Duplicate start/resume requests cannot create duplicate runners across workers.
- [ ] Finalize-only resume does not repeat model extraction.
- [ ] Live graph remains queryable and unchanged until validated atomic publication.
- [ ] Status API and Graph Workspace show accurate aggregate/per-document progress.
- [ ] Backend and frontend system-of-record docs match the shipped surface.
- [ ] Focused and full backend/frontend verification commands pass.
