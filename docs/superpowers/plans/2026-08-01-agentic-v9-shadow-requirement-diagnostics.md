# Agentic v9 Shadow Requirement Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record question-only atomic requirements, evidence-representation availability, and fail-soft visual eligibility in Agentic v9 traces without changing runtime behavior.

**Architecture:** A new pure analyzer deterministically decomposes only explicit question structure and projects bounded requirement diagnostics from retrieved documents. The campaign runtime stores the projection under `agentic_v9.requirement_shadow`; the projection is observational only and never mutates the query contract, retrieval tasks, capability execution, sufficiency, context packing, or final prompt.

**Tech Stack:** Python 3.11, Pydantic v2, LangChain `Document`, pytest.

## Global Constraints

- Do not add an LLM call.
- Do not use evaluation ground truth, golden atomic facts, expected evidence, or source filenames.
- Do not change routing, retrieval, reranking, graph/visual execution, repair, sufficiency, context packing, or synthesis.
- Treat Markdown tables as structured text, not visual assets.
- Treat Figure summaries as text representations of a visual asset; require visual inspection only when the information need exceeds the summary representation.
- Bound the trace to at most 8 requirements and 8 evidence references per requirement.

---

### Task 1: Pure shadow analyzer

**Files:**
- Create: `data_base/agentic_v9/requirement_shadow.py`
- Test: `tests/test_agentic_v9_requirement_shadow.py`

**Interfaces:**
- Consumes: `question: str` and `documents: Sequence[Document]`.
- Produces: `build_requirement_shadow(question, documents) -> RequirementShadowAnalysis`.

- [ ] Write failing tests proving that Markdown-table questions are `not_requested`, qualitative Figure questions with a summary are `optional`, exact graph-reading questions require an available visual asset, and explicit numbered subquestions become separate requirements.
- [ ] Run `pytest tests/test_agentic_v9_requirement_shadow.py -q` and confirm failures are caused by the missing analyzer.
- [ ] Implement bounded deterministic decomposition, representation detection, candidate-only coverage, and Pydantic serialization.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Runtime trace integration

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**
- Consumes: all documents returned by existing retrieval tasks.
- Produces: `agent_trace.agentic_v9.requirement_shadow` with `behavior_influence=false`.

- [ ] Write a failing runtime test asserting that the new trace projection exists while the original query contract, visual execution, retrieved documents, and final answer remain unchanged.
- [ ] Run the single runtime test and confirm it fails because `requirement_shadow` is absent.
- [ ] Accumulate retrieved documents for diagnostics and build the shadow projection only after execution succeeds.
- [ ] Re-run the runtime test and focused Agentic v9 tests.

### Task 3: Verification and commit

**Files:**
- Verify only the files above; do not stage unrelated workspace files.

- [ ] Run `pytest tests/test_agentic_v9_requirement_shadow.py tests/test_agentic_v9_campaign_runtime.py -q`.
- [ ] Run formatter/lint checks used by the backend for the touched files.
- [ ] Inspect `git diff --check` and `git status --short`.
- [ ] Commit only scoped files with message `feat(agentic-v9): add shadow requirement diagnostics`.
