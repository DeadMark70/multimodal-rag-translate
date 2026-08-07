# Backend Maintainability Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete backend CI, warning/complexity ratchets, deterministic OpenAPI and documentation drift checks, and accurate execution-plan lifecycle documentation.

**Architecture:** Small Python CLIs expose pure parsers/comparators plus thin subprocess and filesystem adapters. CI runs the complete fake-provider suite and read-only drift checks. Generated Markdown is confined to marker blocks so human prose remains intact.

**Tech Stack:** Python 3.11/3.13, pytest, Ruff, FastAPI OpenAPI, GitHub Actions, Markdown.

## Global Constraints

- Work only in `D:\flutterserver\pdftopng` on `main`; preserve unrelated tracked and untracked files.
- Before test edits, read `superpowers:test-driven-development/writing-good-tests.md`; use strict RED/GREEN/REFACTOR.
- Set `TEST_MODE=true`, `USE_FAKE_PROVIDERS=true`, and `CI_BLOCK_EXTERNAL_NETWORK=true`; never contact real providers.
- Warning ceiling is exactly 56. Lower/equal passes; higher fails.
- Complexity identity is normalized production path plus function name. Existing equal/lower scores pass; a new score above 10 or any increase fails.
- Ruff correctness rules are exactly `E9,F63,F7,F82`; do not reformat the repository.
- OpenAPI hashes use recursively key-sorted compact JSON bytes, so whitespace alone cannot cause drift.
- `--check` is read-only; `--write` changes only declared artifacts; preserve prose outside generated markers.
- Commit only task-listed files and inspect staged paths before every commit.

---

### Task 1: Add the pytest warning-budget runner

**Files:**

- Create: `scripts/run_pytest_with_warning_budget.py`
- Create: `tests/test_warning_budget.py`

**Interfaces:** `parse_warning_count(output: str) -> int`, `warning_budget_exit_code(pytest_exit_code: int, warning_count: int, max_warnings: int) -> int`, and `main(argv: Sequence[str] | None = None) -> int`.

CLI: `python scripts/run_pytest_with_warning_budget.py --max-warnings 56 -- -q`. It runs `sys.executable -m pytest`, streams combined output, treats a missing warning summary as zero, preserves pytest failures, and returns 1 only when pytest passed but exceeded the budget.

- [ ] Write tests for zero, singular/plural summaries, unrelated numbers, lower/equal/higher ceilings, and pytest exit-code preservation.
- [ ] RED: run `python -m pytest tests/test_warning_budget.py -q`; expect import failure.
- [ ] Implement `subprocess.Popen` streaming and print `warning budget: COUNT/56 (pass|fail)`.
- [ ] GREEN: run `python -m pytest tests/test_warning_budget.py -q`; expect pass.
- [ ] Real check: set the three safe-test variables and run the CLI; expect full pytest success and at most 56 warnings.
- [ ] Commit: stage only the two files and use `ci: add pytest warning budget ratchet`.

---

### Task 2: Add Ruff correctness and C901 ratchets

**Files:**

- Create: `scripts/check_complexity_ratchet.py`
- Create: `tests/test_complexity_ratchet.py`
- Create: `quality/ruff-complexity-baseline.json`
- Create: `ruff.toml`

**Interfaces:** `ComplexityFinding(path, function, score)`, `parse_ruff_findings(payload, repo_root) -> dict[str, int]`, `compare_complexity(baseline, current, threshold=10) -> list[str]`, and `main(argv=None) -> int`. Keys are `relative/posix/path.py::function_name`.

Production roots are `core`, `data_base`, `evaluation`, `graph_rag`, `pdfserviceMD`, `multimodal_rag`, `conversations`, `stats`, and `image_service`. The tool invokes Ruff C901 JSON; Ruff exit 1 is valid findings output, while malformed output and execution errors fail closed.

- [ ] Test normalization, Ruff JSON parsing, unchanged/decreased/removed entries, increases, and new scores at 10 versus above 10.
- [ ] RED: `python -m pytest tests/test_complexity_ratchet.py -q`; expect import failure.
- [ ] Implement `--write-baseline` and `--check`; store schema version 1, threshold 10, and sorted score mapping without timestamps/absolute paths.
- [ ] Generate the baseline with `python scripts/check_complexity_ratchet.py --write-baseline`; expect the current 71 production findings.
- [ ] Create `ruff.toml` with only exclusions needed for generated, venv, worktree, experiment, archive, and test paths.
- [ ] GREEN: run the focused test, correctness Ruff command with `--select E9,F63,F7,F82`, and `--check`; all must pass.
- [ ] Commit the four listed files as `ci: ratchet backend ruff complexity`.

---

### Task 3: Generate and check OpenAPI artifacts

**Files:**

- Create: `scripts/sync_openapi_artifacts.py`
- Create: `tests/test_openapi_artifacts.py`
- Create: `contracts/openapi-contract.json`
- Modify: `openapi.json`
- Modify: `docs/generated/api-surface.md`

**Interfaces:** `canonical_openapi_bytes(schema)`, `openapi_sha256(schema)`, `render_route_inventory(schema)`, `replace_marker_block(document, generated)`, `build_outputs(schema)`, and `main(argv=None)`. Markers are `<!-- BEGIN GENERATED OPENAPI ROUTES -->` and `<!-- END GENERATED OPENAPI ROUTES -->`.

The tool sets safe-test defaults before importing the runtime FastAPI `app` and calling `app.openapi()`. The manifest has exact keys `schema_version: 1`, `sha256`, and `snapshot: openapi.json`. The route table is sorted by path/method with Method, Path, and Operation ID columns.

- [ ] Test recursive canonicalization, formatting-insensitive hashes, route ordering, missing/duplicate marker failures, human-prose preservation, write mode, and non-mutating drift failure.
- [ ] RED: `python -m pytest tests/test_openapi_artifacts.py -q`; expect import failure.
- [ ] Implement `--write` and `--check`; check mode prints every stale file and writes nothing.
- [ ] Insert exactly one marker block into the existing human-maintained API-surface document.
- [ ] GREEN: run `--write`, then `--check`, then focused pytest with safe variables; all pass.
- [ ] Commit the five listed artifacts as `docs: add deterministic openapi drift check`.

---

### Task 4: Validate internal Markdown links

**Files:**

- Create: `scripts/check_markdown_links.py`
- Create: `tests/test_markdown_links.py`

**Interfaces:** `iter_markdown_files(root)`, `extract_local_links(markdown)`, `resolve_local_link(source, target, repo_root)`, `find_broken_links(root)`, and `main(argv=None)`.

Obtain the Markdown input set from `git ls-files -- '*.md'` so untracked user drafts are outside the gate. Skip fenced code, images, HTTP(S), mailto, bare anchors, and generated/vendor/build directories. Decode escaped paths, remove anchor suffixes for existence checks, reject repository escapes, sort diagnostics, and never access the network.

- [ ] Test relative/root links, anchors, escaped spaces, fences, external URLs, missing files, and repository escapes in temporary trees.
- [ ] RED: `python -m pytest tests/test_markdown_links.py -q`; expect import failure.
- [ ] Implement actionable `source.md: broken-target` diagnostics.
- [ ] GREEN: run focused pytest and `python scripts/check_markdown_links.py`; repair only genuinely broken or newly stale links.
- [ ] Stage the two files and only directly repaired docs after inspecting `git diff --cached --name-only`; commit `docs: validate backend markdown links`.

---

### Task 5: Correct execution-plan lifecycle

**Files:**

- Move: `docs/exec-plans/active/2026-07-evaluation-chat-loading-performance.md` to `docs/exec-plans/completed/2026-07-evaluation-chat-loading-performance.md`
- Move: `docs/exec-plans/active/google-genai-stage2-langchain-paths.md` to `docs/exec-plans/references/google-genai-stage2-langchain-paths.md`
- Modify: `docs/exec-plans/active/index.md`
- Modify: `docs/exec-plans/completed/index.md`
- Modify: `docs/exec-plans/references/index.md`
- Modify: `tests/test_markdown_links.py`

- [ ] Add a lifecycle test asserting the performance plan is completed, the Google GenAI memo is a reference, and `genai-langchain-layering-plan.md` remains active.
- [ ] RED: run the focused link test; expect lifecycle assertions to fail against current locations/indexes.
- [ ] Move both documents and update all indexes with one accurate entry each and no duplicates.
- [ ] GREEN: run focused pytest and the repository link checker; expect pass.
- [ ] Stage only the named moves/indexes/test and commit `docs: organize backend execution plans`.

---

### Task 6: Replace partial CI with complete backend gates

**Files:**

- Modify: `.github/workflows/no-external-api-test.yml`
- Modify: `README.md`
- Modify: `tests/test_warning_budget.py`

Required jobs: `deployment-compile` on Python 3.11, and `quality-and-tests` on Python 3.13. The second job installs Ruff, sets all safe-test variables, then runs correctness Ruff, complexity check, full pytest through the warning wrapper, OpenAPI check, and Markdown-link check. Remove the old four-test subset as the primary gate. Use `contents: read`, dependency caching, and explicit timeouts.

- [ ] Add workflow-contract assertions for both Python versions, safe variables, full warning-wrapper command, Ruff commands, OpenAPI check, and link check.
- [ ] RED: run `python -m pytest tests/test_warning_budget.py -q`; expect workflow assertions to fail.
- [ ] Implement both jobs and document canonical local commands plus ratchet semantics in README.
- [ ] GREEN: run all four infrastructure test modules; expect pass.
- [ ] Stage only the three named files and commit `ci: run complete backend quality gates`.

---

### Task 7: Final backend verification and contract handoff

**Files:** Modify generated OpenAPI artifacts only if write mode reports drift.

- [ ] Run OpenAPI `--write` once, then `--check`; commit only changed generated artifacts as `docs: refresh backend api contract`.
- [ ] Run correctness Ruff, complexity `--check`, Markdown links, and the full warning-budget pytest command with safe variables.
- [ ] Run `git diff --check`, inspect `git status --short`, and scan tracked files for common OpenAI/Google/AWS key patterns; expect no new secret-like value.
- [ ] Run `git rev-parse HEAD` and read `contracts/openapi-contract.json`.
- [ ] Pass those final exact values—not an earlier provisional commit—to the frontend plan's `--write-pin` step.
