# Gemini Graph Extraction Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use explicit Gemini 3.1 Flash-Lite GraphRAG policies and persist an auditable extraction manifest with graph snapshots.

**Architecture:** The LLM factory owns the policy; a typed profile travels from retry API through extractor; GraphStore persists successful-run metadata atomically. The UI chooses normal or confirmed high-precision retry.

**Tech Stack:** Python, FastAPI, Pydantic, LangChain Google GenAI, pytest, React, TypeScript, TanStack Query, Vitest.

## Global Constraints

- Provider remains Google Gemini; no OpenRouter work in this plan.
- Standard extraction is `gemini-3.1-flash-lite` plus `medium`; high-precision one-document retry is the same model plus `high`.
- Community summaries are the same model plus `low`, independent of retry profile.
- Full rebuild always passes `standard`; only one-document retry accepts `high_precision`.
- Explicit non-Gemini-3 models retain existing purpose-specific `thinking_budget` behavior.
- Persist `graph.extraction_runs.json` atomically with snapshots; failed retries never promote graph or manifests.
- Missing legacy metadata remains readable and is shown as unavailable.
- Test first; each task ends with one commit.

---

### Task 1: Explicit Gemini GraphRAG Policy

**Files:** `core/llm_factory.py`, create `tests/test_graph_extraction_policy.py`.

**Produces:** `ExtractionProfile = Literal["standard", "high_precision"]`; profile-aware `get_graph_rag_runtime_overrides` and `graph_rag_llm_runtime_override`.

- [ ] Write failing tests asserting: default extraction resolves model `gemini-3.1-flash-lite` and `thinking_level="medium"`; high-precision resolves `high`; community summary resolves `low`; explicit `gemini-2.5-flash-lite` returns `thinking_budget=2048` and no thinking level.
- [ ] Run: `$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; & 'D:\flutterserver\pdftopng\.venv\Scripts\python.exe' -m pytest tests/test_graph_extraction_policy.py -q -p no:cacheprovider`. Expected: FAIL before implementation.
- [ ] Set both GraphRAG purpose mappings to a `_GRAPH_RAG_MODEL = "gemini-3.1-flash-lite"`. Add level mapping `{graph_extraction: {standard: medium, high_precision: high}, community_summary: {standard: low}}`. For Gemini 3 return `include_thoughts=False` and the mapped level; otherwise return existing thinking budget. Ignore extraction profile for community summaries.
- [ ] Re-run the pytest command and `& 'D:\flutterserver\pdftopng\.venv\Scripts\python.exe' -m ruff check core/llm_factory.py tests/test_graph_extraction_policy.py`. Expected: exit 0.
- [ ] Commit with `git add core/llm_factory.py tests/test_graph_extraction_policy.py` then `git commit -m "feat: define Gemini graph extraction policy"`.

### Task 2: Versioned Extraction Manifest Sidecar

**Files:** `graph_rag/schemas.py`, `graph_rag/store.py`, `graph_rag/maintenance.py`, create `tests/test_graph_extraction_manifest.py`.

**Produces:** `GraphStore.record_extraction_manifest(manifest)`, `get_latest_extraction_manifest(doc_id)`, and durable `graph.extraction_runs.json`.

- [ ] Write failing tests that record a manifest for `doc-1`, save/promote a snapshot, reload GraphStore, and receive its `thinking_level="medium"`; assert a store without the sidecar returns `None` rather than failing.
- [ ] Run: `$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; & 'D:\flutterserver\pdftopng\.venv\Scripts\python.exe' -m pytest tests/test_graph_extraction_manifest.py -q -p no:cacheprovider`. Expected: FAIL before the API exists.
- [ ] Extend `GraphExtractionRunManifest` with `extractor_provider="google"`, optional low/medium/high thinking level, standard/high_precision profile, and snapshot version. Store a manifest list in GraphStore; skip individually malformed legacy records with warnings. Include the sidecar in sidecar save/load, hash/copy/restore, snapshot save, and `_copy_graph_sidecars`. Set new records to the promoted snapshot version before atomic promotion.
- [ ] Re-run the manifest test plus `tests/test_graph_snapshot_atomic_save.py tests/test_graph_legacy_store_compatibility.py`. Expected: all pass.
- [ ] Commit with `git add graph_rag/schemas.py graph_rag/store.py graph_rag/maintenance.py tests/test_graph_extraction_manifest.py` then `git commit -m "feat: persist graph extraction manifests"`.

### Task 3: Profile-aware Extraction and Retry API

**Files:** `graph_rag/extractor.py`, `graph_rag/service.py`, `graph_rag/maintenance.py`, `graph_rag/router.py`, `graph_rag/schemas.py`; modify `tests/test_graph_extraction_policy.py`, `tests/test_graph_router_rebuild_full.py`, `tests/test_graph_router_copy.py`.

**Consumes:** Tasks 1-2 profiles and manifest store.

**Produces:** `POST /graph/documents/{doc_id}/retry` accepts optional `{"extraction_profile":"standard"|"high_precision"}` and graph/document API metadata.

- [ ] Write failing tests: full rebuild calls `run_graph_extraction(..., extraction_profile="standard")`; retry body `high_precision` queues maintenance task with that exact third argument; a no-body retry queues `standard`; response rows expose latest manifest model/profile/thinking fields.
- [ ] Run: `$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; & 'D:\flutterserver\pdftopng\.venv\Scripts\python.exe' -m pytest tests/test_graph_extraction_policy.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py -q -p no:cacheprovider`. Expected: FAIL before API/profile plumbing exists.
- [ ] Add `GraphDocumentRetryRequest` defaulting to `standard`; accept it in router and queue it. Thread the profile through maintenance, `run_graph_extraction`, `extract_and_add_to_graph`, and every extractor `graph_rag_llm_runtime_override("graph_extraction", extraction_profile=...)`. Explicitly pass standard from full rebuild. Construct and record a manifest only for a successful extraction that is part of the temporary graph being promoted. Enrich status/document rows from the latest persisted manifest.
- [ ] Re-run: `$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; & 'D:\flutterserver\pdftopng\.venv\Scripts\python.exe' -m pytest tests/test_graph_extraction_policy.py tests/test_graph_extraction_manifest.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py tests/test_graph_rag_extractor.py tests/test_graphrag_integration.py -q -p no:cacheprovider`; then run ruff on all five changed `graph_rag` modules and the policy test. Expected: exit 0.
- [ ] Commit with `git add graph_rag/extractor.py graph_rag/service.py graph_rag/maintenance.py graph_rag/router.py graph_rag/schemas.py tests/test_graph_extraction_policy.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py` then `git commit -m "feat: add profile-aware graph extraction retry"`.

### Task 4: Graph Policy UI

**Files:** `src/types/graph.ts`, `src/services/graphApi.ts`, `src/hooks/useGraphData.ts`, `src/pages/GraphDemo.tsx`; modify `src/services/graphApi.test.ts`, `src/hooks/useGraphData.test.tsx`, `src/pages/GraphDemo.test.tsx`.

**Consumes:** Task 3 retry body and optional response metadata.

**Produces:** default standard retry, confirmed high-precision retry, legacy-safe metadata display.

- [ ] Write failing tests asserting `retryGraphDocument('doc-123')` posts `{ extraction_profile: 'standard' }`; selected high precision calls the mutation with `('doc-1', 'high_precision')` only after confirmation; absent metadata renders unavailable rather than error.
- [ ] Run: `npm run test -- src/services/graphApi.test.ts src/hooks/useGraphData.test.tsx src/pages/GraphDemo.test.tsx --run`. Expected: FAIL before type and control changes.
- [ ] Add `GraphExtractionProfile` union; API client optional second profile parameter; mutation forwarding; optional model/profile/thinking/prompt/timestamp fields on status types. Add standard and high-precision actions in existing document controls. Reuse the repository confirmation-dialog pattern for high precision and show that it replaces only that document, refreshes communities, and preserves live graph on failure. Render page default policies and unrecorded legacy metadata.
- [ ] Re-run the focused tests, `npm run lint:ci`, `npx tsc --noEmit`, and `npm run build`. Expected: exit 0.
- [ ] Commit with `git add src/types/graph.ts src/services/graphApi.ts src/hooks/useGraphData.ts src/pages/GraphDemo.tsx src/services/graphApi.test.ts src/hooks/useGraphData.test.tsx src/pages/GraphDemo.test.tsx` then `git commit -m "feat: expose graph extraction policy controls"`.

### Task 5: One Focused Review

**Scope:** Tasks 1-4 diff only.

- [ ] Request one combined review focused on task-local profile isolation, legacy sidecar reads, snapshot atomicity, and retry UI/API compatibility.
- [ ] Reproduce every reported issue with a focused test before changing code; reject unsupported findings.
- [ ] For validated corrections, stage only the source and test files changed by the reproduced finding, then commit with `git commit -m "fix: harden graph extraction policy"`.
