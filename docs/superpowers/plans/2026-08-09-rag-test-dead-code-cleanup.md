# RAG Test And Dead-Code Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove obsolete RAG/Graph manual tooling, one dead evaluator helper, and brittle structural tests while preserving all runtime behavior and public APIs.

**Architecture:** Replace the process-global GraphRAG structural test with one scoped behavior test, then remove Arena-only artifacts and source-text assertions whose behavior is already covered. Keep each deletion in a small independently verifiable commit; do not introduce shared helpers or refactor the RAG pipeline.

**Tech Stack:** Python 3.13, pytest, pytest-asyncio, NetworkX, Ruff, Git, PowerShell

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-09-rag-test-dead-code-cleanup-design.md` exactly.
- Runtime behavior and public imports must remain unchanged.
- Do not modify `data_base/rag_pipeline.py`, `data_base/rag_graph_runtime.py`, `CampaignEngine`, evaluation schemas, public APIs, Docker, or dependency versions.
- Do not consolidate `tests/test_reranker_logic.py` in this batch.
- Do not delete `experiments/evaluation_pipeline.py`.
- Do not create a shared test utility or dependency-injection framework.
- Historical checklists, archived plans, and dead-code audit records keep their original paths.
- The complete backend test gate must remain at or below 56 warnings and must not require external API access.

---

## File Responsibility Map

- `tests/test_community_builder_budget.py`: owns executable community-builder behavior, including the missing-Leiden fallback.
- `tests/test_graphrag_static_analysis.py`: obsolete process-global structural test; delete it after behavior coverage is present.
- `tests/run_arena.py`: obsolete uncollected Arena command; delete it.
- `tests/verify_agentic_fix.py`: obsolete uncollected one-off visual verification command; delete it.
- `agents/evaluator.py`: retains supported evaluator behavior; remove only the Arena-only `compare_rag_vs_pure_llm()` helper.
- `agentlog/codebase_overview.md`: active overview; remove the obsolete Arena script entry.
- `tests/test_rag_retrieval_generation_split.py`: retains behavior and compatibility contracts; remove only source-text and unrelated assertions.
- `tests/test_rag_qa_prompts.py`: retains prompt registry behavior; remove only the negative source-text assertion.

---

### Task 1: Replace The Order-Dependent GraphRAG Structural Test

**Files:**
- Modify: `tests/test_community_builder_budget.py:1-105`
- Delete: `tests/test_graphrag_static_analysis.py`
- Test: `tests/test_community_builder_budget.py`
- Verify with: `tests/test_reranker_logic.py`

**Interfaces:**
- Consumes: `graph_rag.community_builder.detect_communities_leiden(store) -> list[Community]` and a store-like object exposing a NetworkX `graph`.
- Produces: one behavior-level fallback test that uses scoped import-state mutation and leaves later test collection unaffected.

- [ ] **Step 1: Capture the current order-dependent collection failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest --collect-only -q tests\test_graphrag_static_analysis.py tests\test_reranker_logic.py
```

Expected: FAIL during `tests/test_reranker_logic.py` collection with `ValueError: networkx.__spec__ is not set`.

- [ ] **Step 2: Add the scoped Leiden fallback behavior test**

Add these imports to `tests/test_community_builder_budget.py`:

```python
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import networkx as nx
import pytest

from graph_rag.community_builder import build_communities, detect_communities_leiden
```

Keep the existing schema imports. Add this test before the build-budget tests:

```python
@pytest.mark.asyncio
async def test_detect_communities_falls_back_to_connected_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = nx.Graph()
    graph.add_edge("paper-a", "paper-b")
    graph.add_node("paper-c")
    store = SimpleNamespace(graph=graph)

    monkeypatch.setitem(sys.modules, "igraph", None)

    communities = await detect_communities_leiden(store)

    assert {frozenset(community.node_ids) for community in communities} == {
        frozenset({"paper-a", "paper-b"}),
        frozenset({"paper-c"}),
    }
```

Do not assign a `MagicMock` to dependency names outside `monkeypatch`.

- [ ] **Step 3: Run the new behavior test**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_community_builder_budget.py::test_detect_communities_falls_back_to_connected_components -q
```

Expected: PASS using real NetworkX and the connected-components fallback.

- [ ] **Step 4: Delete the obsolete structural test file**

Delete `tests/test_graphrag_static_analysis.py` in full. Do not move its `hasattr`, signature, or `inspect.getsource()` assertions elsewhere.

- [ ] **Step 5: Verify collection and execution no longer depend on file order**

Run:

```powershell
.venv\Scripts\python.exe -m pytest --collect-only -q tests\test_community_builder_budget.py tests\test_reranker_logic.py
.venv\Scripts\python.exe -m pytest -q tests\test_community_builder_budget.py tests\test_reranker_logic.py
```

Expected: both commands PASS; there is no `networkx.__spec__` error.

- [ ] **Step 6: Run focused lint and commit**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check tests\test_community_builder_budget.py --select E9,F63,F7,F82,F401,F841
git diff --check
```

Expected: PASS.

Commit:

```powershell
git add -- tests/test_community_builder_budget.py tests/test_graphrag_static_analysis.py
git commit -m "test(graph): replace structural fallback checks"
```

---

### Task 2: Remove Obsolete Arena And One-Off Verification Tooling

**Files:**
- Delete: `tests/run_arena.py`
- Delete: `tests/verify_agentic_fix.py`
- Modify: `agents/evaluator.py:621-677`
- Modify: `agentlog/codebase_overview.md:219`
- Test: `tests/test_evaluator.py`
- Verify with: `tests/test_agentic_v9_smoke_runner.py`
- Verify with: `tests/test_visual_tool_trigger.py`

**Interfaces:**
- Consumes: the maintained `RAGEvaluator`, agentic smoke runner, and visual verification tests.
- Produces: no replacement runtime interface; removes only uncollected commands and their sole Arena-only helper.

- [ ] **Step 1: Confirm the helper has exactly one live Python caller**

Run:

```powershell
rg -n "compare_rag_vs_pure_llm" agents data_base evaluation scripts tests -g "*.py"
```

Expected: matches only `agents/evaluator.py` and `tests/run_arena.py`. If another Python caller appears, stop and reassess.

- [ ] **Step 2: Delete the two uncollected manual scripts**

Delete these files in full:

```text
tests/run_arena.py
tests/verify_agentic_fix.py
```

Do not move them to `scripts/` or `experiments/`.

- [ ] **Step 3: Remove the Arena-only evaluator helper**

Delete the complete `compare_rag_vs_pure_llm()` function, from:

```python
async def compare_rag_vs_pure_llm(
    question: str,
    rag_answer: str,
    pure_llm_answer: str,
    documents: List[Document],
) -> dict:
```

through the closing result dictionary. Keep `evaluate_rag_result()` unchanged. Remove an import only if Ruff reports that this deletion made it unused.

- [ ] **Step 4: Remove the obsolete active documentation entry**

Delete this line from `agentlog/codebase_overview.md`:

```markdown
- **Arena 腳本**: `tests/run_arena.py` RAG vs Pure LLM A/B 測試
```

Do not edit historical checklists, archived plans, audits, or the approved spec.

- [ ] **Step 5: Verify supported evaluator and replacement workflows**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_evaluator.py tests\test_agentic_v9_smoke_runner.py tests\test_visual_tool_trigger.py
```

Expected: PASS without external API access.

- [ ] **Step 6: Verify live references are gone**

Run:

```powershell
rg -n "compare_rag_vs_pure_llm" agents data_base evaluation scripts tests -g "*.py"
rg -n "tests/(run_arena|verify_agentic_fix)\.py" README.md docs agentlog/codebase_overview.md -g "*.md" -g "!docs/superpowers/**"
```

Expected: no live matches. Historical records may retain their original paths.

- [ ] **Step 7: Run focused lint and commit**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check agents\evaluator.py tests\test_evaluator.py tests\test_agentic_v9_smoke_runner.py tests\test_visual_tool_trigger.py --select E9,F63,F7,F82,F401,F841
.venv\Scripts\python.exe scripts\check_markdown_links.py
git diff --check
```

Expected: PASS.

Commit:

```powershell
git add -- agents/evaluator.py agentlog/codebase_overview.md tests/run_arena.py tests/verify_agentic_fix.py
git commit -m "refactor(evaluation): remove obsolete Arena tooling"
```

---

### Task 3: Remove Brittle RAG Source-Text Assertions

**Files:**
- Modify: `tests/test_rag_retrieval_generation_split.py:1-451`
- Modify: `tests/test_rag_qa_prompts.py:1-77`
- Verify with: `tests/test_visual_tool_trigger.py`

**Interfaces:**
- Consumes: existing facade delegation, generation behavior, visual verification, and prompt-registry tests.
- Produces: unchanged behavior-level coverage without assertions about source layout, removed constant names, or an old syntax-error spelling.

- [ ] **Step 1: Run the behavior tests that justify removing source assertions**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q `
  tests\test_rag_retrieval_generation_split.py::test_facade_delegates_through_pipeline_module_lookup `
  tests\test_rag_retrieval_generation_split.py::test_legacy_generation_keeps_visual_synthesis_inside_legacy_module `
  tests\test_visual_tool_trigger.py `
  tests\test_rag_qa_prompts.py
```

Expected: PASS.

- [ ] **Step 2: Remove source-text tests and the unrelated assertion**

From `tests/test_rag_retrieval_generation_split.py`, delete these complete tests:

```text
test_legacy_wrapper_delegates_generation_without_exposing_visual_synthesis
test_legacy_generation_avoids_python_311_incompatible_fstring_backslashes
```

Within `test_legacy_generation_keeps_visual_synthesis_inside_legacy_module`, delete only:

```python
    assert "data_base.visual_tools" not in Path("data_base/agentic_v9/model_paths.py").read_text(
        encoding="utf-8"
    )
```

Remove `from pathlib import Path` after confirming no `Path` use remains.

- [ ] **Step 3: Remove the obsolete prompt source assertion**

From `tests/test_rag_qa_prompts.py`, delete `test_rag_qa_service_no_long_prompt_constants()` and remove `from pathlib import Path`.

Keep all registry key, required-variable, and formatting tests unchanged.

- [ ] **Step 4: Run the focused RAG and prompt files**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_rag_retrieval_generation_split.py tests\test_rag_qa_prompts.py tests\test_visual_tool_trigger.py
```

Expected: PASS.

- [ ] **Step 5: Verify no source-reading import remains and commit**

Run:

```powershell
rg -n "read_text\(|from pathlib import Path" tests\test_rag_retrieval_generation_split.py tests\test_rag_qa_prompts.py
.venv\Scripts\python.exe -m ruff check tests\test_rag_retrieval_generation_split.py tests\test_rag_qa_prompts.py tests\test_visual_tool_trigger.py --select E9,F63,F7,F82,F401,F841
git diff --check
```

Expected: the search returns no matches and quality commands pass.

Commit:

```powershell
git add -- tests/test_rag_retrieval_generation_split.py tests/test_rag_qa_prompts.py
git commit -m "test(rag): remove brittle source assertions"
```

---

### Task 4: Run The Complete Verification Gate

**Files:**
- Verify: all files changed by Tasks 1-3

**Interfaces:**
- Consumes: the completed cleanup commits.
- Produces: evidence that cleanup changes no runtime behavior and leaves repository gates green.

- [ ] **Step 1: Run the focused cleanup regression suite**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q `
  tests\test_community_builder_budget.py `
  tests\test_reranker_logic.py `
  tests\test_evaluator.py `
  tests\test_agentic_v9_smoke_runner.py `
  tests\test_rag_retrieval_generation_split.py `
  tests\test_rag_qa_prompts.py `
  tests\test_visual_tool_trigger.py `
  tests\test_graphrag_integration.py
```

Expected: PASS.

- [ ] **Step 2: Run the complete backend test gate**

Run:

```powershell
.venv\Scripts\python.exe scripts\run_pytest_with_warning_budget.py --max-warnings 56 -- -q
```

Expected: all tests pass, warning count is 56 or lower, and no external API access is required.

- [ ] **Step 3: Run repository maintenance gates**

Run:

```powershell
.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
.venv\Scripts\python.exe scripts\check_markdown_links.py
.venv\Scripts\python.exe scripts\check_complexity_ratchet.py --check
.venv\Scripts\python.exe -m ruff check . --select E9,F63,F7,F82,F401,F841
git diff --check
```

Expected: OpenAPI artifacts, Markdown links, complexity, Ruff, and whitespace checks pass.

- [ ] **Step 4: Review final scope and deletion size**

Run:

```powershell
git status --short
git diff --stat 534836c..HEAD
git diff --name-status 534836c..HEAD
git diff --numstat 534836c..HEAD
```

Confirm:

- Only files listed in this plan changed.
- `data_base/rag_pipeline.py`, `data_base/rag_graph_runtime.py`, `evaluation/campaign_engine.py`, and `tests/test_reranker_logic.py` are unchanged.
- The implementation removes substantially more lines than it adds; a net deletion around 450 lines is expected.
- There are exactly three implementation commits from Tasks 1-3.

- [ ] **Step 5: Perform the final clean-worktree check**

Run:

```powershell
git status --porcelain=v1
git log -5 --oneline --decorate
```

Expected: clean worktree with three cleanup commits above `534836c`; do not push unless explicitly requested.
