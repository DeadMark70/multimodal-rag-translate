# Agentic v9 Smoke Accounting Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore production comparison planning without weakening subject safety, and eliminate unbudgeted GraphRAG provider calls from the v9 locator path.

**Architecture:** Keep provider JSON parsing separate from trusted `ComparisonPlan` construction. The provider transport accepts harmless formatting variation, while deterministic promotion still requires independently named entities anchored in the question. Keep legacy GraphRAG generation unchanged, but make the v9 source locator consume stored community summaries without generating community answers.

**Tech Stack:** Python 3.11, asyncio, Pydantic v2, pytest.

## Global Constraints

- Do not change retrieval, reranker, final context packing, or synthesis behavior.
- Comparison planner failures remain fail-soft and preserve base retrieval.
- The v9 graph locator may use deterministic routing without a `graph_route` provider call.
- Every provider token counted by a v9 run must have a durable phase-linked attempt, otherwise accounting must be partial.

---

### Task 1: Tolerant comparison transport with strict promotion

**Files:**
- Modify: `data_base/agentic_v9/comparison_planner.py`
- Test: `tests/test_agentic_v9_comparison_planner.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**
- Consumes: `ComparisonPlanner.plan(question, authorized_source_names, timeout_seconds)`
- Produces: either a trusted `ComparisonPlan` or a typed fail-soft fallback.

- [ ] Add failing tests using production-like Gemini JSON with harmless extra fields, entity-role synonyms, and omitted redundant exact-span metadata.
- [ ] Verify the tests fail with `schema_violation`.
- [ ] Introduce a provider-facing transport model that ignores harmless extras and normalizes entity-like roles.
- [ ] Construct trusted `ComparisonSubject` values only after exact question anchoring, unique identity checks, source-reference rejection, and numeric-value rejection.
- [ ] Verify Q3 remains `not_comparison`/`invalid_subjects`, while Q4 and Q14 produce `planned`.
- [ ] Run the focused comparison planner and campaign runtime tests.
- [ ] Commit the planner repair.

### Task 2: Zero-generation v9 graph community hints

**Files:**
- Modify: `graph_rag/global_search.py`
- Modify: `data_base/RAG_QA_service.py`
- Test: `tests/test_graphrag_integration.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**
- Consumes: `global_search_hints(..., generate_answers=False)` from the v9 graph evidence bundle.
- Produces: ranked stored community-summary `GraphHint` values without provider calls.

- [ ] Add a failing test proving the v9 graph bundle does not call `query_community` or `synthesize_answers`.
- [ ] Verify the test fails because the current locator generates community answers.
- [ ] Add a default-preserving flag to `global_search_hints`; legacy callers keep generated answers, while v9 passes `generate_answers=False`.
- [ ] Verify deterministic graph fast paths need no `graph_route` attempt and produce no hidden `graph_reasoning` usage.
- [ ] Run focused GraphRAG and campaign runtime tests.
- [ ] Commit the graph locator repair.

### Task 3: Reconciliation fail-closed verification

**Files:**
- Modify only if existing reconciliation does not already fail closed: `evaluation/research_analytics.py`
- Test: `tests/test_evaluation_research_analytics.py`
- Test: `tests/test_agentic_v9_smoke_runner.py`

**Interfaces:**
- Consumes: runtime token totals and durable provider attempts.
- Produces: complete only when totals reconcile; otherwise partial with an explicit reason.

- [ ] Add or identify a test where runtime total exceeds durable attempts.
- [ ] Verify the result is partial; if it already passes, make no production change.
- [ ] Run smoke-verification and analytics tests.
- [ ] Commit only if a production change is required.

### Task 4: Verification and review

**Files:**
- Review all files changed by Tasks 1–3.

- [ ] Run focused pytest suites for comparison planning, GraphRAG integration, campaign runtime, accounting, and smoke verification.
- [ ] Run Ruff on all changed Python files.
- [ ] Inspect `git diff --check` and the final scoped diff.
- [ ] Dispatch one fresh read-only reviewer subagent against the base and head commits.
- [ ] Resolve every Critical or Important finding and rerun verification.

