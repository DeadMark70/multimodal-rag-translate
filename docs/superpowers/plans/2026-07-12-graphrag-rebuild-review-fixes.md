# GraphRAG Rebuild Final Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make resumable GraphRAG full rebuild control data, inputs, concurrency, retries, and UI progress safe after snapshot promotion, errors, and restarts.

**Architecture:** Keep rebuild control-plane state outside immutable graph snapshots and preserve frozen OCR markdown in each job directory. Add one stable, owner-token maintenance lock shared by every graph-mutating operation; retain `active_job_state` only as UI metadata. Extend the durable status contract for retry limits and use a shared HTTP-status retry classifier in both extraction service and coordinator.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, pytest, React 18, TypeScript, TanStack Query, Vitest.

## Global Constraints

- Work directly on the existing `main` / `master` branches; do not create a worktree.
- Every task must start with a failing focused regression test and end with its own commit.
- Rebuild job manifests, locks, frozen inputs, and staging data must never live inside `versions/v*`.
- No graph-mutating operation may run without the shared maintenance lease.
- A full rebuild must publish only after all frozen source documents are successfully checkpointed and validated.

---

### Task 1: Stable rebuild-control root and immutable source inputs

**Files:**
- Modify: `graph_rag/store.py`
- Modify: `graph_rag/rebuild_jobs.py`
- Modify: `graph_rag/rebuild_coordinator.py`
- Modify: `graph_rag/router.py`
- Modify: `graph_rag/schemas.py`
- Test: `tests/test_graph_rebuild_job_store.py`
- Test: `tests/test_graph_rebuild_coordinator.py`
- Test: `tests/test_graph_router_rebuild_full.py`

**Interfaces:**
- Consumes: `GraphStore._root_storage_dir`, `load_ocr_artifacts`, `GraphRebuildDocument`.
- Produces: `GraphRebuildJobStore.stable_root`, `create_or_load_active(sources, source_markdown)`, `load_source_markdown(job_id, doc_id)`, and `GraphRebuildDocument.source_markdown_sha256`.

- [ ] **Step 1: Write failing tests for snapshot-stable control data and source freezing**

```python
def test_current_job_survives_live_snapshot_promotion(tmp_path: Path) -> None:
    jobs = GraphRebuildJobStore("user-1", rebuild_root=tmp_path / "graph")
    manifest = jobs.create_job(SOURCES, source_markdown={"doc-1": "frozen", "doc-2": "frozen"})
    GraphStore("user-1", storage_dir=tmp_path / "graph").save_snapshot()
    assert jobs.load_current().job_id == manifest.job_id

def test_coordinator_reads_frozen_source_not_mutated_original(tmp_path: Path) -> None:
    jobs, coordinator, job_id, owner = _coordinator_with_frozen_source(tmp_path, "frozen")
    await coordinator.run("user-1", job_id, owner)
    assert coordinator.run_extraction.await_args.kwargs["markdown_text"] == "frozen"
```

- [ ] **Step 2: Run focused tests to verify they fail**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_graph_rebuild_job_store.py tests/test_graph_rebuild_coordinator.py tests/test_graph_router_rebuild_full.py`

Expected: FAIL because job root follows `storage_dir` and coordinator reloads OCR artifacts.

- [ ] **Step 3: Implement stable control data and frozen source input APIs**

```python
class GraphRebuildJobStore:
    def __init__(self, user_id: str, rebuild_root: Path | None = None, ...):
        root = rebuild_root or GraphStore(user_id).root_storage_dir / "rebuild_jobs"
        self.root = root.resolve()

    def create_job(self, sources, *, source_markdown: dict[str, str]) -> GraphRebuildManifest:
        # write sources/<doc_id>.md atomically, record sha256 on each document, then save manifest

    def load_source_markdown(self, job_id: str, doc_id: str) -> str:
        # verify SHA-256 against manifest before returning UTF-8 markdown
```

Expose a read-only `GraphStore.root_storage_dir` property. In the full-rebuild start route, load every artifact before creating the job, pass markdown by `doc_id`, and return a validation error without scheduling when freezing fails. Replace coordinator artifact loading with `jobs.load_source_markdown()`.

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_graph_rebuild_job_store.py tests/test_graph_rebuild_coordinator.py tests/test_graph_router_rebuild_full.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graph_rag/store.py graph_rag/rebuild_jobs.py graph_rag/rebuild_coordinator.py graph_rag/router.py graph_rag/schemas.py tests/test_graph_rebuild_job_store.py tests/test_graph_rebuild_coordinator.py tests/test_graph_router_rebuild_full.py
git commit -m "fix: stabilize graph rebuild job inputs"
```

### Task 2: Shared durable graph-maintenance lease

**Files:**
- Create: `graph_rag/maintenance_lock.py`
- Modify: `graph_rag/router.py`
- Modify: `graph_rag/maintenance.py`
- Test: `tests/test_graph_maintenance_lock.py`
- Test: `tests/test_graph_router_rebuild_full.py`
- Test: `tests/test_graph_router_copy.py`

**Interfaces:**
- Consumes: `GraphStore.root_storage_dir`, `GraphStore.set_active_job_state`.
- Produces: `GraphMaintenanceLock.acquire(activity) -> str | None`, `release(owner_token) -> bool`, `reconcile() -> str | None`.

- [ ] **Step 1: Write failing lock and router tests**

```python
def test_only_one_maintenance_operation_can_acquire_stable_lock(tmp_path: Path) -> None:
    first = GraphMaintenanceLock(tmp_path)
    second = GraphMaintenanceLock(tmp_path)
    token = first.acquire("rebuild_full")
    assert token is not None
    assert second.acquire("optimize") is None

def test_legacy_rebuild_sets_maintenance_lock_before_scheduling(client) -> None:
    response = client.post("/graph/rebuild", json={"force": True})
    assert response.json()["status"] == "started"
    assert maintenance_lock_activity() == "rebuild"
```

- [ ] **Step 2: Run focused tests to verify they fail**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_graph_maintenance_lock.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py`

Expected: FAIL because no shared stable lock exists and legacy rebuild/optimize do not own one.

- [ ] **Step 3: Implement the shared lease and wire every mutating operation**

```python
class GraphMaintenanceLock:
    def acquire(self, activity: str) -> str | None: ...
    def release(self, owner_token: str) -> bool: ...
    def reconcile(self) -> str | None: ...
```

Use atomic exclusive creation for `maintenance.lock` below the stable graph root. Start routes acquire before setting `active_job_state` or adding a task; pass the owner token into background tasks. Each background task releases in `finally`. Wrap synchronous optimize in acquire/release with `try/finally`. On failed scheduling, release the token and restore UI sidecar state.

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_graph_maintenance_lock.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graph_rag/maintenance_lock.py graph_rag/router.py graph_rag/maintenance.py tests/test_graph_maintenance_lock.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py
git commit -m "fix: serialize graph maintenance operations"
```

### Task 3: Complete transient retry classification

**Files:**
- Create: `graph_rag/retry.py`
- Modify: `graph_rag/service.py`
- Modify: `graph_rag/rebuild_coordinator.py`
- Test: `tests/test_graph_rag_extractor.py`
- Test: `tests/test_graph_rebuild_coordinator.py`

**Interfaces:**
- Produces: `is_retryable_graph_error(exc: Exception) -> bool` for timeout, transport, 408, 429, and 500–599 status codes.

- [ ] **Step 1: Write failing HTTP status tests**

```python
class ProviderError(Exception):
    status_code = 503

def test_provider_5xx_is_retryable() -> None:
    assert is_retryable_graph_error(ProviderError()) is True

def test_provider_400_is_not_retryable() -> None:
    assert is_retryable_graph_error(type("BadRequest", (Exception,), {"status_code": 400})()) is False
```

- [ ] **Step 2: Run focused tests to verify they fail**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_graph_rag_extractor.py tests/test_graph_rebuild_coordinator.py`

Expected: FAIL because current classifiers only accept 429 and transport/timeout errors.

- [ ] **Step 3: Implement and consume the shared classifier**

```python
def is_retryable_graph_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return (
        isinstance(exc, (TimeoutError, httpx.TransportError))
        or status_code in {408, 429}
        or isinstance(status_code, int) and 500 <= status_code <= 599
    )
```

Use this function when producing `GraphExtractionRunResult.retryable` and when handling coordinator-raised exceptions.

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_graph_rag_extractor.py tests/test_graph_rebuild_coordinator.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graph_rag/retry.py graph_rag/service.py graph_rag/rebuild_coordinator.py tests/test_graph_rag_extractor.py tests/test_graph_rebuild_coordinator.py
git commit -m "fix: retry transient graph extraction failures"
```

### Task 4: Durable retry-limit status and progress UI

**Files:**
- Modify: `graph_rag/schemas.py`
- Modify: `graph_rag/rebuild_jobs.py`
- Modify: `src/types/graph.ts`
- Modify: `src/components/graph/GraphRebuildProgress.tsx`
- Test: `tests/test_graph_rebuild_job_store.py`
- Test: `src/components/graph/GraphRebuildProgress.test.tsx`

**Interfaces:**
- Produces: `GraphRebuildStatusResponse.max_attempts: int` and `GraphRebuildStatus.max_attempts: number`.

- [ ] **Step 1: Write failing backend and frontend contract tests**

```python
def test_status_exposes_manifest_retry_limit(tmp_path: Path) -> None:
    manifest = store.create_job(SOURCES, source_markdown=MARKDOWN)
    manifest.max_attempts = 5
    assert store.to_status(manifest).max_attempts == 5
```

```tsx
expect(screen.getByText(/第 2 次，共 5 次/)).toBeInTheDocument();
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_graph_rebuild_job_store.py`

Run: `npx vitest run src/components/graph/GraphRebuildProgress.test.tsx`

Expected: FAIL because the API omits `max_attempts` and the UI displays a literal `3`.

- [ ] **Step 3: Add the field through the backend and frontend contracts**

```python
class GraphRebuildStatusResponse(BaseModel):
    max_attempts: int = Field(ge=1)

return GraphRebuildStatusResponse(..., max_attempts=manifest.max_attempts)
```

```tsx
第 {status.current_document.attempt} 次，共 {status.max_attempts} 次
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_graph_rebuild_job_store.py`

Run: `npx vitest run src/components/graph/GraphRebuildProgress.test.tsx`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graph_rag/schemas.py graph_rag/rebuild_jobs.py tests/test_graph_rebuild_job_store.py
git commit -m "feat: expose graph rebuild retry limit"
git -C ../Multimodal_RAG_System add src/types/graph.ts src/components/graph/GraphRebuildProgress.tsx src/components/graph/GraphRebuildProgress.test.tsx
git -C ../Multimodal_RAG_System commit -m "fix: display graph rebuild retry limit"
```

### Task 5: End-to-end verification and documentation

**Files:**
- Modify: `docs/BACKEND.md`
- Test: all changed backend and frontend test suites

**Interfaces:**
- Consumes: completed Tasks 1–4.
- Produces: documented stable rebuild behavior and verified clean worktrees.

- [ ] **Step 1: Document stable job storage, frozen inputs, and maintenance exclusion**

Add a concise GraphRAG maintenance section stating that full rebuild inputs are frozen at start, all graph maintenance operations are serialized, and interrupted jobs resume from the frozen job directory.

- [ ] **Step 2: Run backend full suite**

Run: `.venv\\Scripts\\python.exe -m pytest -q`

Expected: PASS.

- [ ] **Step 3: Run frontend full validation**

Run: `npm run lint:ci`

Run: `npx vitest run`

Run: `npm run build`

Expected: all commands exit 0.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/BACKEND.md
git commit -m "docs: clarify durable graph rebuild safeguards"
```
