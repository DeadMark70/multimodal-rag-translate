# Graph Rebuild Publication Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make completed GraphRAG rebuilds explicitly report and display that the new graph is now live.

**Architecture:** The durable backend manifest remains the source of truth. `GraphRebuildJobStore.to_status()` derives whether the live graph remains unchanged from `manifest.state`; the frontend maps that boolean and state to one mutually exclusive publication notice.

**Tech Stack:** Python 3.11, FastAPI/Pydantic, pytest; React, TypeScript, Chakra UI, Vitest.

## Global Constraints

- Do not change graph rebuild, staging, publishing, or retry behavior.
- `live_graph_unchanged` is `false` only after a manifest is `completed`.
- A completed UI state must state that the new graph is live and must not display the old-graph warning.
- Preserve the running-state old-graph warning.
- Commit each task separately on the existing main/master branches.

---

### Task 1: Correct the backend publication-status projection

**Files:**
- Modify: `graph_rag/rebuild_jobs.py:153-189`
- Modify: `tests/test_graph_rebuild_job_store.py`

**Interfaces:**
- Consumes: `GraphRebuildManifest.state` and `GraphRebuildStatusResponse.live_graph_unchanged`.
- Produces: `GraphRebuildJobStore.to_status(manifest)` with an explicit publication-state boolean.

- [ ] **Step 1: Write the failing test**

Add a test that projects both a running and completed manifest:

```python
def test_status_reports_live_graph_changed_only_after_publication() -> None:
    store = GraphRebuildJobStore("user-1", rebuild_root=tmp_path)
    running = _manifest(state="running")
    completed = running.model_copy(update={"state": "completed"})

    assert store.to_status(running).live_graph_unchanged is True
    assert store.to_status(completed).live_graph_unchanged is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_job_store.py -k publication -q
```

Expected: failure because both projections currently inherit the schema default `live_graph_unchanged=True`.

- [ ] **Step 3: Write the minimal implementation**

Pass the derived value when constructing `GraphRebuildStatusResponse`:

```python
live_graph_unchanged=manifest.state != "completed",
```

- [ ] **Step 4: Run the focused backend tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graph_rebuild_job_store.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add graph_rag/rebuild_jobs.py tests/test_graph_rebuild_job_store.py
git commit -m "fix: report completed graph rebuild publication"
```

### Task 2: Render an unambiguous GraphRAG publication notice

**Files:**
- Modify: `../Multimodal_RAG_System/src/components/graph/GraphRebuildProgress.tsx`
- Modify: `../Multimodal_RAG_System/src/components/graph/GraphRebuildProgress.test.tsx`
- Modify: `../Multimodal_RAG_System/docs/FRONTEND.md`

**Interfaces:**
- Consumes: `GraphRebuildStatus.state` and corrected `live_graph_unchanged` from Task 1.
- Produces: one publication alert: old graph for unchanged states, new graph for a completed and published job.

- [ ] **Step 1: Write the failing test**

Add a completed status and assert it has only the success copy:

```tsx
it('confirms the new graph is live after a completed rebuild', () => {
  renderProgress({
    ...runningStatus,
    state: 'completed',
    phase: 'done',
    live_graph_unchanged: false,
  });

  expect(screen.getByText(/新圖譜已切換/)).toBeInTheDocument();
  expect(screen.queryByText(/目前查詢仍使用舊圖譜/)).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run from `D:\flutterserver\Multimodal_RAG_System`:

```powershell
npx vitest run src/components/graph/GraphRebuildProgress.test.tsx
```

Expected: failure because the component has no completed-and-published success alert.

- [ ] **Step 3: Write the minimal implementation**

Keep the existing old-graph alert behind `status.live_graph_unchanged`. Add an exclusive success alert for `status.state === 'completed' && !status.live_graph_unchanged`:

```tsx
{status.state === 'completed' && !status.live_graph_unchanged && (
  <Alert status="success" mt={3} borderRadius="md">
    <AlertIcon />新圖譜已切換，查詢目前使用新圖譜。
  </Alert>
)}
```

- [ ] **Step 4: Document the completed-state meaning and run checks**

Append to the `/graph-demo` GraphRAG workspace description in `docs/FRONTEND.md`: completed full rebuild progress confirms that the staged graph has been published and current queries use it; the old-graph notice appears only before publication.

Run:

```powershell
npx vitest run src/components/graph/GraphRebuildProgress.test.tsx
npm run lint:ci
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/components/graph/GraphRebuildProgress.tsx src/components/graph/GraphRebuildProgress.test.tsx docs/FRONTEND.md
git commit -m "fix: clarify graph rebuild publication status"
```

## Plan Self-Review

- Spec coverage: Task 1 corrects the durable API signal; Task 2 renders the two exclusive user-facing states and documents them.
- No placeholders: test names, assertions, implementation conditions, paths, commands, and commits are explicit.
- Type consistency: both tasks use the existing `GraphRebuildStatus.live_graph_unchanged` boolean and `completed` state.
