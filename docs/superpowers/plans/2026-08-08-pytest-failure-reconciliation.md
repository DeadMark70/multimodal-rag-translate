# Pytest Failure Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile all 17 reproducible backend pytest failures with current production contracts and finish with a hermetic, cross-platform suite that has zero failures.

**Architecture:** Preserve current application behavior and repair the enforcement layer around it. Add one production-scope exclusion with direct regression coverage, then update stale contract assertions and patch seams in four independently testable groups; local secret validation becomes explicitly opt-in and golden hashes canonicalize checkout line endings.

**Tech Stack:** Python 3.13, pytest 9, unittest.mock, pathlib, Git, FastAPI TestClient

## Global Constraints

- Work directly on `main`, as explicitly requested by the user.
- Do not delete useful tests or revert intentional production contracts.
- Do not read, print, commit, move, or rewrite `.env` or `config.env` values.
- Default pytest must not inspect local secrets files; strict validation runs only with `VALIDATE_LOCAL_ENV=1`.
- Do not delete or modify registered worktrees.
- Do not broaden Gemini or router allowlists.
- Do not modify API, response schema, provider budget, routing, retrieval, or answer behavior.
- Preserve unrelated untracked documentation.

---

## File Structure

- Modify `core/production_scope.py`: treat `.worktrees` as non-production input.
- Modify `tests/test_production_scope.py`: prove normal production files remain included and worktree copies are excluded.
- Modify `tests/test_agentic_v9_execution_policy.py`: synchronize the exact timeout contract.
- Modify `tests/test_agentic_v9_golden_dataset.py`: hash canonical LF bytes on every platform.
- Modify `tests/test_dependency_env.py`: make real local-env checks opt-in while retaining value-free key validation.
- Modify `tests/test_agentic_v9_full_rollback.py`: synchronize the empty-visual-result contract.
- Modify `tests/test_campaign_engine.py`: assert safe failure redaction and structured diagnostics.
- Modify `tests/test_evaluation_analytics_api.py`: synchronize v9 defaults and five-call pre-route admission.
- Modify `tests/test_evaluation_question_comparison.py`: mirror the bounded result DTO.
- Modify `tests/test_evaluation_phase_attribution.py`: point the phase contract at the current generation owner.
- Modify `tests/test_graph_auto_gate.py`: test the current locator boundary and scope filter.
- Modify `tests/test_graph_context_packing.py`: patch current locator and generation owners.

### Task 1: Make Production Scans Ignore Registered Worktrees

**Files:**
- Modify: `tests/test_production_scope.py`
- Modify: `core/production_scope.py`

**Interfaces:**
- Consumes: `is_production_path(path: Path, *, project_root: Path) -> bool`
- Produces: a production scan that excludes any path with a `.worktrees` component while retaining ordinary production paths

- [ ] **Step 1: Add the failing production-scope boundary test**

Update the import and add this test to `tests/test_production_scope.py`:

```python
from core.production_scope import (
    PROJECT_ROOT,
    is_production_path,
    iter_production_python_files,
)


def test_registered_worktree_copies_are_not_production_paths(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    production_file = project_root / "core" / "app_factory.py"
    worktree_copy = (
        project_root
        / ".worktrees"
        / "feature"
        / "core"
        / "app_factory.py"
    )

    assert is_production_path(production_file, project_root=project_root)
    assert not is_production_path(worktree_copy, project_root=project_root)
```

- [ ] **Step 2: Verify the new test is RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_production_scope.py::test_registered_worktree_copies_are_not_production_paths -q
```

Expected: FAIL because `.worktrees/feature/core/app_factory.py` is currently classified as production.

- [ ] **Step 3: Add the minimal scanner exclusion**

Add one entry to `NON_PRODUCTION_ROOTS` in `core/production_scope.py`:

```python
NON_PRODUCTION_ROOTS = {
    ".venv",
    ".worktrees",
    "venv",
    # existing entries remain unchanged
}
```

- [ ] **Step 4: Verify the boundary and architecture guards are GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_production_scope.py tests/test_gemini_layering.py tests/test_router_boundaries.py -q
```

Expected: all tests pass; the existing Gemini and router allowlists remain unchanged.

- [ ] **Step 5: Commit the hermetic scanner fix**

```powershell
git add -- core/production_scope.py tests/test_production_scope.py
git commit -m "fix(test): exclude worktrees from production scans"
```

### Task 2: Make Repository Contract Tests Hermetic and Cross-Platform

**Files:**
- Modify: `tests/test_agentic_v9_execution_policy.py`
- Modify: `tests/test_agentic_v9_golden_dataset.py`
- Modify: `tests/test_dependency_env.py`

**Interfaces:**
- Consumes: `ExecutionPolicy.phase_timeouts_s`, frozen JSON artifacts, `VALIDATE_LOCAL_ENV`
- Produces: exact current policy coverage, canonical SHA-256 checks, and opt-in local-env validation

- [ ] **Step 1: Re-run the existing RED contract group**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_execution_policy.py::test_execution_policy_has_the_initial_runtime_bounds tests/test_agentic_v9_golden_dataset.py::test_agentic_v9_golden_paths_and_hashes_are_frozen tests/test_dependency_env.py::test_env_keys_match_example -q
```

Expected: three failures for the missing `comparison_plan` expectation, CRLF-sensitive hash, and local missing key.

- [ ] **Step 2: Synchronize the exact execution policy contract**

Add the intentional comparison-planner timeout to the expected dictionary:

```python
assert policy.phase_timeouts_s == {
    "route_plan": 32.0,
    "comparison_plan": 68.0,
    "retrieval_judge": 32.0,
    "evidence_extract": 64.0,
    "visual_extract": 16.0,
    "final_answer": 32.0,
}
```

- [ ] **Step 3: Canonicalize only CRLF before golden hashing**

Replace `_sha256` in `tests/test_agentic_v9_golden_dataset.py` with:

```python
def _sha256(path: Path) -> str:
    canonical_bytes = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical_bytes).hexdigest()
```

Do not change the expected hashes or any golden JSON file.

- [ ] **Step 4: Make local secret validation explicitly opt-in**

Add `import os`, lift the value-free key parser to module scope, and gate both real local-file tests before any path inspection:

```python
def _require_local_env_validation() -> None:
    if os.getenv("VALIDATE_LOCAL_ENV") != "1":
        pytest.skip("Set VALIDATE_LOCAL_ENV=1 to validate local env files")


def _parse_env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0].strip())
    return keys
```

The first statement in both `test_env_file_exists()` and
`test_env_keys_match_example()` must be:

```python
_require_local_env_validation()
```

Use `_parse_env_keys()` for the strict comparison and keep the assertion limited
to missing key names. Remove the debug path print so default pytest performs no
local secret-file inspection or disclosure.

- [ ] **Step 5: Add opt-in gate and value-free parser coverage**

Add these tests without referencing `.env` or `config.env`:

```python
def test_local_env_validation_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VALIDATE_LOCAL_ENV", raising=False)

    with pytest.raises(pytest.skip.Exception):
        _require_local_env_validation()


def test_local_env_validation_can_be_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATE_LOCAL_ENV", "1")

    _require_local_env_validation()


def test_env_key_comparison_reports_only_missing_names(tmp_path: Path) -> None:
    example = tmp_path / "example.env"
    actual = tmp_path / "actual.env"
    example.write_text("FIRST=example\nSECOND=example\n", encoding="utf-8")
    actual.write_text("FIRST=local-value\n", encoding="utf-8")

    missing = _parse_env_keys(example) - _parse_env_keys(actual)

    assert missing == {"SECOND"}
```

- [ ] **Step 6: Verify the repository contract group is GREEN by default**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_execution_policy.py tests/test_agentic_v9_golden_dataset.py tests/test_dependency_env.py -q
```

Expected: zero failures and the two real local-env checks skipped.

- [ ] **Step 7: Commit the hermetic repository contract repairs**

```powershell
git add -- tests/test_agentic_v9_execution_policy.py tests/test_agentic_v9_golden_dataset.py tests/test_dependency_env.py
git commit -m "fix(test): make repository contracts hermetic"
```

### Task 3: Synchronize Agentic v9 and Evaluation Analytics Contracts

**Files:**
- Modify: `tests/test_agentic_v9_full_rollback.py`
- Modify: `tests/test_campaign_engine.py`
- Modify: `tests/test_evaluation_analytics_api.py`
- Modify: `tests/test_evaluation_question_comparison.py`

**Interfaces:**
- Consumes: current empty-visual semantics, safe campaign failure projection, v9 campaign defaults, five-call pre-route feasibility, bounded research result DTO
- Produces: six integration tests aligned with current public behavior

- [ ] **Step 1: Re-run the six existing RED tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_full_rollback.py::test_missing_visual_capability_preserves_authorized_text_answer tests/test_campaign_engine.py::test_campaign_integration_keeps_running_when_one_mode_fails tests/test_evaluation_analytics_api.py::test_research_analytics_endpoints_return_owned_run_details tests/test_evaluation_analytics_api.py::test_v9_campaign_preflight_uses_golden_routes_and_reports_incompatible_setup tests/test_evaluation_analytics_api.py::test_v9_campaign_preflight_admits_visual_requirements_when_contract_reserves_three_calls tests/test_evaluation_question_comparison.py::test_question_comparison_is_measured_and_fail_closed -q
```

Expected: six failures matching the diagnosed contract drift.

- [ ] **Step 2: Preserve complete text answers for an empty visual result**

In `tests/test_agentic_v9_full_rollback.py`, change only the response-status
assertion:

```python
assert result.agent_trace["response_status"] == "complete"
```

Keep the answer, documents, source IDs, and
`visual_execution.state == "required_but_not_satisfied"` assertions unchanged.

- [ ] **Step 3: Assert safe public campaign failure diagnostics**

Replace the raw exception assertion in `tests/test_campaign_engine.py` with:

```python
assert failed_row["error_message"] == "Provider error details were redacted."
assert "graph retrieval blew up" not in failed_row["error_message"]
diagnostics = failed_row["derived_metrics"]["failure_diagnostics"]
assert diagnostics["error_code"] == "RuntimeError"
assert diagnostics["safe_error_message"] == "Provider error details were redacted."
```

- [ ] **Step 4: Synchronize new campaign analytics with the v9 default**

In `test_research_analytics_endpoints_return_owned_run_details`, replace the
pre-v9 comment and expectation with:

```python
# New campaigns default to the current v9 execution contract when the request
# does not explicitly select a legacy version.
assert run_list_item["condition_id"] is None
assert run_list_item["agentic_execution_version"] == "v9"
```

Keep `_seed_legacy_campaign_result()` and its legacy endpoint assertions intact.

- [ ] **Step 5: Use the fail-closed five-call setup budget in both preflight tests**

Change both `max_llm_calls: 4` values in
`test_v9_campaign_preflight_uses_golden_routes_and_reports_incompatible_setup`
to `5`. This makes the incompatible case isolate `thinking_reserve_unknown`.

Rename the visual test to
`test_v9_campaign_preflight_admits_visual_route_with_five_call_setup_reserve`
and change its request from:

```python
"runtime_token_budget": 10000, "max_llm_calls": 3,
```

to:

```python
"runtime_token_budget": 10000, "max_llm_calls": 5,
```

Keep the expected route, feasible status, and empty issues assertions.

- [ ] **Step 6: Mirror the bounded research-result DTO**

In `_result()` inside `tests/test_evaluation_question_comparison.py`, add the
direct field consumed by `ResearchAnalyticsService`:

```python
required_modalities=["text"],
question_snapshot={"required_modalities": ["text"]},
```

Keep `question_snapshot` to preserve compatibility coverage, while the direct
field proves the current bounded projection contract.

- [ ] **Step 7: Verify the six repaired contracts and adjacent controls**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_full_rollback.py tests/test_campaign_engine.py::test_campaign_integration_keeps_running_when_one_mode_fails tests/test_campaign_engine.py::test_run_campaign_persists_safe_failure_diagnostics tests/test_evaluation_analytics_api.py tests/test_evaluation_question_comparison.py -q
```

Expected: zero failures. The adjacent extractor-error, safe redaction, v9
default, feasibility, and bounded-projection controls must remain green.

- [ ] **Step 8: Commit the current-contract test repairs**

```powershell
git add -- tests/test_agentic_v9_full_rollback.py tests/test_campaign_engine.py tests/test_evaluation_analytics_api.py tests/test_evaluation_question_comparison.py
git commit -m "fix(test): align evaluation tests with current contracts"
```

### Task 4: Retarget Generation and Graph Tests After Module Extraction

**Files:**
- Modify: `tests/test_evaluation_phase_attribution.py`
- Modify: `tests/test_graph_auto_gate.py`
- Modify: `tests/test_graph_context_packing.py`

**Interfaces:**
- Consumes: `data_base.rag_generation`, `data_base.rag_graph_locator.locate_graph_sources`, current compatibility adapters in `RAG_QA_service`
- Produces: four tests that exercise the modules actually owning generation and Graph location behavior

- [ ] **Step 1: Re-run the four existing RED tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation_phase_attribution.py::test_evaluation_call_sites_declare_controlled_phase tests/test_graph_auto_gate.py::test_auto_gate_planning_records_skip_without_bundling_merging_or_prompting tests/test_graph_auto_gate.py::test_graph_located_chunks_outside_scoped_doc_ids_are_excluded tests/test_graph_context_packing.py::test_graph_to_chunk_flag_uses_source_chunks_and_falls_back_on_lookup_failure -q
```

Expected: four failures from the stale source-owner entry or patch targets.

- [ ] **Step 2: Point phase ownership at the extracted generation module**

Replace this `PHASE_CASES` entry:

```python
("data_base/RAG_QA_service.py", "answer_generation"),
```

with:

```python
("data_base/rag_generation.py", "answer_generation"),
```

- [ ] **Step 3: Test planning-only skip at the public locator boundary**

In `test_auto_gate_planning_records_skip_without_bundling_merging_or_prompting`:

```python
locator = AsyncMock()
```

Patch the current public boundary:

```python
patch("data_base.RAG_QA_service.locate_graph_sources", new=locator),
```

Remove the nonexistent `merge_vector_and_graph_docs` patch and its
`assert_not_called()`. Retarget generation-owned test doubles:

```python
patch("data_base.rag_generation.get_llm_usage_metrics", return_value={}),
patch(
    "data_base.rag_generation.fetch_document_filenames",
    new=AsyncMock(return_value={"doc-allowed": "allowed.pdf"}),
),
```

After execution assert:

```python
locator.assert_not_awaited()
graph_context.assert_not_awaited()
graph_bundle.assert_not_awaited()
```

Keep the observability assertion that records `route_decision.path == "skip"`.

- [ ] **Step 4: Prove Graph scope filtering happens before scoring**

In `test_graph_located_chunks_outside_scoped_doc_ids_are_excluded`, create a
scorer that reflects only its actual candidates:

```python
score = Mock(
    side_effect=lambda chunks, **_kwargs: [chunk.document for chunk in chunks]
)
```

Retarget the patches:

```python
patch("data_base.rag_generation.get_llm_usage_metrics", return_value={}),
patch(
    "data_base.rag_generation.fetch_document_filenames",
    new=AsyncMock(return_value={"doc-allowed": "allowed.pdf"}),
),
patch(
    "data_base.rag_graph_locator.expand_graph_evidence_to_chunks",
    return_value=[GraphLocatedChunk(out_of_scope, item)],
),
patch("data_base.rag_graph_locator.score_graph_located_chunks", new=score),
```

After the existing output assertions, add:

```python
score.assert_called_once()
assert score.call_args.args[0] == []
```

This prevents the test from passing merely because the merge ratio rounds a
graph-only candidate count down to zero.

- [ ] **Step 5: Retarget lookup and generation patches in context packing**

In `test_graph_to_chunk_flag_uses_source_chunks_and_falls_back_on_lookup_failure`,
replace the three stale patch owners with:

```python
patch("data_base.rag_generation.get_llm_usage_metrics", return_value={}),
patch(
    "data_base.rag_generation.fetch_document_filenames",
    new=AsyncMock(return_value={"doc-1": "doc.pdf"}),
),
patch("data_base.rag_graph_locator.VectorStoreChunkLookup", return_value=lookup),
```

Keep the identity assertion proving `chunk_lookup is lookup`, vector fallback,
legacy-path non-use, and `fallback=no_packed_graph_chunks` observability.

- [ ] **Step 6: Verify the refactor-aware group and current locator controls**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation_phase_attribution.py tests/test_graph_auto_gate.py tests/test_graph_context_packing.py tests/test_rag_graph_locator.py -q
```

Expected: zero failures and no live database metadata warning from the repaired
Graph context test.

- [ ] **Step 7: Commit the refactor-aware test repairs**

```powershell
git add -- tests/test_evaluation_phase_attribution.py tests/test_graph_auto_gate.py tests/test_graph_context_packing.py
git commit -m "fix(test): retarget graph tests after module extraction"
```

### Task 5: Verify the Complete Reconciliation

**Files:**
- Verify: every file changed in Tasks 1–4
- Verify unchanged: `.env`, `config.env`, Gemini/router allowlists, registered worktrees

**Interfaces:**
- Consumes: four independently green repair commits
- Produces: a zero-failure full backend suite and a scoped main-branch handoff

- [ ] **Step 1: Inspect implementation scope**

Run:

```powershell
git diff ce72cd8..HEAD --stat
git diff ce72cd8..HEAD -- core/production_scope.py tests/test_production_scope.py tests/test_agentic_v9_execution_policy.py tests/test_agentic_v9_golden_dataset.py tests/test_dependency_env.py tests/test_agentic_v9_full_rollback.py tests/test_campaign_engine.py tests/test_evaluation_analytics_api.py tests/test_evaluation_question_comparison.py tests/test_evaluation_phase_attribution.py tests/test_graph_auto_gate.py tests/test_graph_context_packing.py
git status --short --branch
```

Expected: only the approved production-scope helper and test repairs are in the
implementation commits; unrelated untracked docs remain untouched.

- [ ] **Step 2: Run all formerly failing tests together**

Run the four Task 1 architecture files, the three Task 2 files, the Task 3
files, and the Task 4 files in one pytest command:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_production_scope.py tests/test_gemini_layering.py tests/test_router_boundaries.py tests/test_agentic_v9_execution_policy.py tests/test_agentic_v9_golden_dataset.py tests/test_dependency_env.py tests/test_agentic_v9_full_rollback.py tests/test_campaign_engine.py::test_campaign_integration_keeps_running_when_one_mode_fails tests/test_campaign_engine.py::test_run_campaign_persists_safe_failure_diagnostics tests/test_evaluation_analytics_api.py tests/test_evaluation_question_comparison.py tests/test_evaluation_phase_attribution.py tests/test_graph_auto_gate.py tests/test_graph_context_packing.py tests/test_rag_graph_locator.py -q
```

Expected: zero failures; local real-env tests are skipped by default.

- [ ] **Step 3: Run the complete backend suite**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expected: zero failures. Record the exact passed/skipped/warning counts.

- [ ] **Step 4: Run final patch checks**

```powershell
git diff --check ce72cd8..HEAD
git status --short --branch
git log -5 --oneline
```

Expected: no whitespace errors, no uncommitted candidate changes, and the four
repair commits on `main`.

- [ ] **Step 5: Perform the completion review**

Confirm all 17 original failures are absent, no test was deleted, default pytest
did not inspect local secrets, worktree copies are excluded without changing
allowlists, and every intentional current production contract remains intact.
