# Question Comparison and v9 Phase Accounting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Question Analysis compare `agentic` against `naive` deterministically and record v9 provider usage in its actual phases.

**Architecture:** `ResearchAnalyticsService` will separate the descriptive best-mode calculation from the fixed comparison pair. The v9 invocation boundary will propagate phase context to provider usage callbacks; execution accounting already derives its summary from those records.

**Tech Stack:** Python 3.11, pytest, Pydantic, evaluation accounting store.

## Global Constraints

- `naive` is the comparison baseline and `agentic` is the comparison target for Question Analysis.
- Missing or unreconciled usage remains partial; never synthesize totals or phases.
- Do not weaken formal benchmark release gates.
- Stage and commit only files changed for each task.

---

### Task 1: Fixed Question Analysis comparison pair

**Files:**
- Modify: `evaluation/research_analytics.py`
- Modify: `tests/test_evaluation_question_comparison.py`

**Interfaces:**
- Consumes: `QuestionModeComparison` rows for `naive` and `agentic`.
- Produces: measured deltas when both complete rows exist, independent of `best_quality_mode`.

- [ ] Write a failing regression test where `naive` is best but agentic is complete, then assert deltas are present.
- [ ] Run `pytest tests/test_evaluation_question_comparison.py -q` and verify it fails as `comparison_mode_missing` before the production change.
- [ ] Select `naive` as baseline and `agentic` as target explicitly; retain `best_quality_mode` only for display.
- [ ] Run the focused test file and commit the production and test changes together.

### Task 2: v9 phase-specific durable usage accounting

**Files:**
- Modify: `data_base/agentic_v9/budgeted_llm.py`
- Modify: focused v9 accounting/runtime tests

**Interfaces:**
- Consumes: `phase` from `BudgetedLlmInvoker.invoke()` and the existing evaluation accounting context.
- Produces: one measured accounting record per actual provider call, attributed to its v9 phase.

- [ ] Write a failing test proving a final-answer invocation exposes `final_answer` as accounting phase.
- [ ] Run focused v9 tests and verify the pre-change failure shows `unclassified` phase.
- [ ] Add `llm_accounting_phase(phase)` at the provider boundary.
- [ ] Run focused and adjacent evaluation tests, then commit the implementation and tests together.
