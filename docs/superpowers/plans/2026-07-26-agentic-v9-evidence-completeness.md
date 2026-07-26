# Agentic v9 Evidence Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Agentic v9 completeness atomic and trustworthy, constrain partial answers at content level, repair missing evidence precisely, establish a working visual-evidence path, and export complete per-phase research observability without leaking benchmark answers into runtime.

**Architecture:** Evolve the existing typed v9 pipeline rather than creating a second orchestration path. `QueryContract` v2 carries an answer-free atomic slot plan and an actual route decision. Retrieval, repair, sufficiency, visual resolution, and final synthesis all operate on those stable slot IDs. `BudgetedLlmInvoker` remains the only provider boundary and gains an optional observer that records each provider attempt. Existing `GraphAssetLink` sidecars become the canonical visual manifest, with a resolver/authorized loader bridging them into `AssetLocator`. Legacy v1 runs remain readable but explicitly N/A for atomic completeness.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, SQLite evaluation repository, LangChain provider adapters, pytest/pytest-asyncio, React 18, TypeScript 5.9, Chakra UI, TanStack Query, Vitest, Testing Library.

**Source of truth:** `docs/superpowers/specs/2026-07-26-agentic-v9-evidence-completeness-design.md` at commit `3404794`.

## Global Constraints

- Never pass `key_points`, `ground_truth`, `ground_truth_short`, expected values, gold atomic facts, or expected-evidence answer content into contract planning, retrieval planning, repair, or final generation.
- Evaluation Setup is authoritative for model, thinking, token/call limits, deadlines, and prompt-capture policy.
- Every provider call must pass through `BudgetedLlmInvoker`; no direct `ainvoke()` may be added in v9 orchestration.
- A degraded contract can return useful evidence, but cannot return `response_status="complete"`.
- Required graph/visual capability outcomes must update affected slot resolutions before final synthesis.
- Missing telemetry remains missing/partial. Never replace unknown token, score, route, asset, or prompt data with zero.
- Observability write failure must not erase an otherwise usable answer; it must mark observability/release comparability partial.
- Preserve v8, Naive, Advanced, and Graph mode behavior.
- Preserve v1 contract deserialization. Historical generic slots display `slot_semantics="legacy_generic"` and `atomic_completeness=null`.
- Add type hints to all modified signatures, use structured logging, and keep router modules free of orchestration logic.
- Each task starts with a failing focused test, implements the minimum behavior, reruns the focused tests, and commits only that task's files.
- Do not add the existing untracked `.pytest-tmp/`, `data/`, or unrelated plan/spec files to any commit.

## File and Interface Map

### Backend contract and runtime

- Modify `data_base/agentic_v9/schemas.py`
  - Add v2 route-decision, slot-plan, answer-shape, visual-policy, supported-finding, and unresolved-requirement models.
  - Keep v1-compatible defaults.
- Add `data_base/agentic_v9/contract_planner.py`
  - Own deterministic question decomposition, one ambiguity call, safe fallback, and v2 contract assembly.
- Modify `data_base/agentic_v9/route_planner.py`
  - Retain a compatibility wrapper; delegate v2 planning to `QuestionContractPlanner`.
- Add `prompts/agentic_v9_contract_planner.json`
  - Strict answer-free route-and-slot JSON prompt.
- Modify `data_base/agentic_v9/budget_feasibility.py`
  - Reserve worst-case planning and downstream calls during preflight.
- Modify `evaluation/agentic_v9_admission.py`
  - Return deterministic preflight analysis without calling a model.
- Modify `evaluation/agentic_v9_campaign_runtime.py`
  - Build the runtime planner after budget controller creation, persist v2 contract/route, wire slot-aware visual outcomes, structured final synthesis, and observer.
- Modify `data_base/agentic_v9/execution_core.py`
  - Carry final sufficiency and pre-final capability-adjusted resolutions into generation.

### Retrieval, sufficiency, and final answer

- Modify `data_base/agentic_v9/retrieval_tasks.py`
  - Compile grouped tasks while retaining individual target slot IDs and source/locator constraints.
- Modify `data_base/agentic_v9/repair.py`
  - Group only `not_found` slots by authorized source and locator; enforce 2 tasks × 2 rounds.
- Modify `data_base/agentic_v9/sufficiency_gate.py`
  - Keep slot resolution authoritative and add explicit degraded-plan/capability downgrade reasons.
- Modify `data_base/agentic_v9/final_answer.py`
  - Parse structured supported findings/unresolved requirements, validate slot/evidence boundaries, and render deterministic fallback.
- Modify `data_base/agentic_v9/citation_renderer.py`
  - Render supported and unresolved sections without presenting unsupported text as a claim.
- Add `prompts/agentic_v9_final_answer.json`
  - Load the structured final-synthesis prompt through `core/prompt_loader.py`.

### Visual asset bridge

- Modify `graph_rag/schemas.py`
  - Extend `GraphAssetLink` with storage reference, content hash, dimensions, printed page, and formula identifier using optional defaults.
- Modify `graph_rag/assets.py`
  - Populate manifest metadata from `VisualElement.image_path` and extracted Markdown assets.
- Modify `graph_rag/store.py`
  - Add bounded manifest lookup by authorized document IDs and locator fields.
- Modify `pdfserviceMD/indexing_tasks.py`
  - Persist the enriched manifest after visual processing.
- Add `data_base/agentic_v9/visual_asset_resolver.py`
  - Resolve manifest entries, authorize paths, load selected images, and return diagnostics.
- Modify `data_base/agentic_v9/asset_locator.py`
  - Accept manifest-backed candidates without requiring image base64 in retrieved chunks.
- Modify `data_base/agentic_v9/visual_evidence_extractor.py`
  - Emit covered slot IDs and terminal diagnostic reasons.
- Add `scripts/backfill_visual_asset_manifest.py`
  - Idempotently rebuild manifest metadata for existing canonical upload directories.

### Evaluation observability and export

- Modify `data_base/agentic_v9/budgeted_llm.py`
  - Add `LlmCallObserver`, attempt numbering, reservation linkage, prompt/response hashing, terminal status, and capture policy.
- Modify `evaluation/trace_schemas.py`
  - Add explicit phase, reservation ID, provider attempt, capture status, and token-component fields.
- Modify `evaluation/db.py`
  - Add backward-compatible SQLite columns/indexes through the existing idempotent schema initializer.
- Modify `evaluation/observability.py`
  - Record attempt-level LLM rows and expose non-fatal observer write failures.
- Modify `evaluation/observability_storage.py`
  - Persist/read the new fields.
- Modify `evaluation/campaign_schemas.py`
  - Persist execution-time prompt-capture policy and export availability summary.
- Modify `evaluation/campaign_engine.py`
  - Inject run/campaign-aware observer and persist the actual v9 route decision.
- Modify `evaluation/analytics.py`
  - Report capture availability and exact phase-token reconciliation.
- Modify `evaluation/router.py`
  - Return typed availability warnings; do not synthesize uncaptured prompts.

### Frontend

- Modify `Multimodal_RAG_System/src/types/evaluation.ts`
  - Add v2 contract, atomic slot, actual route, repair, visual diagnostics, capture availability, and legacy semantics.
- Modify `Multimodal_RAG_System/src/services/evaluationApi.ts`
  - Send capture policy and parse export summary/warnings.
- Modify `Multimodal_RAG_System/src/pages/EvaluationCenter.mappers.ts`
  - Map v2 fields N/A-safely and preserve v1 legacy state.
- Modify `Multimodal_RAG_System/src/components/evaluation/AgentBehaviorTab.tsx`
  - Display contract version/status, actual route rationale, atomic slots, repair rounds, and capability outcomes.
- Modify `Multimodal_RAG_System/src/components/evaluation/ClaimEvidenceTab.tsx`
  - Display claim → slot → evidence mappings and unresolved requirements.
- Modify `Multimodal_RAG_System/src/components/evaluation/RouterLabTab.tsx`
  - Separate actual routing from retrospective analysis.
- Modify `Multimodal_RAG_System/src/components/evaluation/AblationDashboardTab.tsx`
  - Display execution capture policy, availability counts, warnings, and actual export counts.
- Modify `Multimodal_RAG_System/src/components/evaluation/CampaignRunner.tsx`
  - Add prompt-capture settings with a clear full-prompt privacy warning.

---

## Wave 1 — Content-level Fail-closed Final

### Task 1: Define structured final-output contracts

**Files:**
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `data_base/agentic_v9/final_answer.py`
- Test: `tests/test_agentic_v9_final_answer.py`
- Test: `tests/test_agentic_v9_schemas.py`

- [ ] Add failing schema tests for `SupportedFinding(slot_id, statement, evidence_ids)`, `UnresolvedRequirement(slot_id, reason)`, and a strict `FinalAnswerDraft` containing only those two arrays.
- [ ] Add a failing compatibility test showing existing `FinalClaim`/`FinalAnswerResult` payloads still deserialize.
- [ ] Run:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_schemas.py tests/test_agentic_v9_final_answer.py -q
  ```

  Expected: failures for missing structured models/parser.

- [ ] Implement the strict models with `extra="forbid"` and stable defaults.
- [ ] Convert validated supported findings into persisted `FinalClaim` rows while retaining `slot_id`.
- [ ] Rerun the focused tests and confirm they pass.
- [ ] Commit:

  ```powershell
  git add data_base/agentic_v9/schemas.py data_base/agentic_v9/final_answer.py tests/test_agentic_v9_schemas.py tests/test_agentic_v9_final_answer.py
  git commit -m "feat(agentic-v9): define structured final findings"
  ```

### Task 2: Enforce claim → slot → evidence boundaries

**Files:**
- Modify: `data_base/agentic_v9/final_answer.py`
- Modify: `data_base/agentic_v9/citation_renderer.py`
- Add: `prompts/agentic_v9_final_answer.json`
- Test: `tests/test_agentic_v9_final_answer.py`

- [ ] Add failing tests that reject:
  - a finding for a `not_found`, `conflicted`, or `explicitly_unavailable` slot;
  - an unknown slot ID;
  - evidence not assigned to that slot;
  - omission of a required unresolved slot.
- [ ] Add a failing Q14 regression fixture where the provider emits `SegmentAnyBone（推測）`; assert the unsupported finding is absent from rendered output and the unresolved source requirement is present.
- [ ] Run the focused final-answer test and verify the new cases fail.
- [ ] Load the final prompt through `core/prompt_loader.py`. Instruct it to return supported findings and unresolved requirements only.
- [ ] Validate every finding against `SlotResolution` and `EvidencePacket.slot_ids`; build required unresolved rows deterministically from non-supported required slots.
- [ ] Render two explicit sections: supported conclusions and unresolved/unverifiable requirements.
- [ ] Ensure qualifiers such as “推測” never turn an unsupported finding into an accepted claim.
- [ ] Rerun `tests/test_agentic_v9_final_answer.py`.
- [ ] Commit:

  ```powershell
  git add data_base/agentic_v9/final_answer.py data_base/agentic_v9/citation_renderer.py prompts/agentic_v9_final_answer.json tests/test_agentic_v9_final_answer.py
  git commit -m "fix(agentic-v9): enforce slot-bound final synthesis"
  ```

### Task 3: Move capability downgrade before final generation

**Files:**
- Modify: `data_base/agentic_v9/execution_core.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Test: `tests/test_agentic_v9_execution_core.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`

- [ ] Add failing tests proving:
  - required graph failure changes affected slot resolutions before `generate_final`;
  - required visual failure changes affected slots to `explicitly_unavailable`;
  - degraded or incomplete resolution cannot be upgraded to `complete`;
  - invalid provider output uses deterministic supported/unresolved rendering.
- [ ] Run:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py -q
  ```

- [ ] Replace the runtime's ad-hoc final call and hardcoded `response_status="complete"` with `generate_final_answer()`.
- [ ] Pass the final sufficiency report and capability-adjusted slot resolutions into final generation.
- [ ] Remove the post-generation global-only downgrade as the primary safety mechanism; retain a defensive invariant check.
- [ ] Rerun the focused tests.
- [ ] Commit:

  ```powershell
  git add data_base/agentic_v9/execution_core.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py
  git commit -m "fix(agentic-v9): fail closed before final generation"
  ```

**Wave 1 gate:**

- [ ] Run:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py -q
  ```

- [ ] Confirm the Q14 regression contains no unsupported SegmentAnyBone conclusion.

---

## Wave 2 — Atomic Question Contract Planner

### Task 4: Introduce QueryContract v2 and legacy projection

**Files:**
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `evaluation/trace_schemas.py`
- Modify: `evaluation/research_analytics.py`
- Modify: `evaluation/release_metrics.py`
- Test: `tests/test_agentic_v9_schemas.py`
- Test: `tests/test_evaluation_research_analytics.py`
- Test: `tests/test_evaluation_release_metrics.py`

- [ ] Add failing tests for:
  - `RouteDecision` provenance fields;
  - `RequiredSlot` source hints, authorized IDs, answer type, dependencies, and visual policy;
  - `QueryContract(contract_version="2", slot_plan_status=...)`;
  - v1 projection as `legacy_generic` with `atomic_completeness=None`.
- [ ] Run the three focused test modules and verify failure.
- [ ] Implement additive optional fields/defaults so persisted v1 contracts remain readable.
- [ ] Make release metrics fail closed for degraded v2 plans and show N/A for v1 atomic completeness.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add data_base/agentic_v9/schemas.py evaluation/trace_schemas.py evaluation/research_analytics.py evaluation/release_metrics.py tests/test_agentic_v9_schemas.py tests/test_evaluation_research_analytics.py tests/test_evaluation_release_metrics.py
  git commit -m "feat(agentic-v9): version atomic query contracts"
  ```

### Task 5: Build deterministic answer-free decomposition

**Files:**
- Add: `data_base/agentic_v9/contract_planner.py`
- Modify: `data_base/agentic_v9/route_planner.py`
- Modify: `data_base/agentic_v9/__init__.py`
- Test: `tests/test_agentic_v9_contract_planner.py`
- Modify: `tests/test_agentic_v9_route_planner.py`

- [ ] Add parameterized failing tests for Q5, Q7, Q11, Q14, and Q16 question text.
- [ ] Assert Q16 yields seven ordered slots and none of their descriptions contains expected numeric answers.
- [ ] Assert numbered/bulleted clauses, parallel values, document names, and Figure/Table/Appendix/Formula/Equation/Theorem locators split correctly.
- [ ] Add a guard test that injects a `question_snapshot` containing gold fields and proves the planner API accepts only question, authorized source names/IDs, and setup policy.
- [ ] Run:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_route_planner.py -q
  ```

- [ ] Implement deterministic decomposition with stable `S1`–`S8` IDs, expected answer types, route candidates, matched rules, confidence, and source/locator hints.
- [ ] Keep `RoutePlanner` as a compatibility delegate, not a second planning implementation.
- [ ] Rerun tests and inspect Q16 descriptions for answer leakage.
- [ ] Commit:

  ```powershell
  git add data_base/agentic_v9/contract_planner.py data_base/agentic_v9/route_planner.py data_base/agentic_v9/__init__.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_route_planner.py
  git commit -m "feat(agentic-v9): decompose questions into atomic slots"
  ```

### Task 6: Add one budgeted ambiguity contract-planning call

**Files:**
- Add: `prompts/agentic_v9_contract_planner.json`
- Modify: `data_base/agentic_v9/contract_planner.py`
- Modify: `data_base/agentic_v9/budget_feasibility.py`
- Modify: `evaluation/agentic_v9_admission.py`
- Test: `tests/test_agentic_v9_contract_planner.py`
- Test: `tests/test_agentic_v9_budget_feasibility.py`
- Test: `tests/test_agentic_v9_provider_boundary.py`

- [ ] Add failing tests that:
  - deterministic confident questions use zero planner calls;
  - ambiguous questions use exactly one `contract_planning` call;
  - invalid JSON, timeout, budget rejection, or unauthorized source expansion yields `decision_source="safe_fallback"` and `slot_plan_status="degraded"`;
  - preflight never calls a provider and reserves worst-case planning/downstream calls;
  - provider limits come from Evaluation Setup.
- [ ] Run the focused tests and verify failure.
- [ ] Implement strict JSON parsing for route plus slots, source-scope intersection, answer/value leakage checks, and the deterministic fallback.
- [ ] Update post-contract feasibility to evaluate the exact runtime contract and whether the planner call was used.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add prompts/agentic_v9_contract_planner.json data_base/agentic_v9/contract_planner.py data_base/agentic_v9/budget_feasibility.py evaluation/agentic_v9_admission.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_provider_boundary.py
  git commit -m "feat(agentic-v9): budget ambiguous contract planning"
  ```

### Task 7: Wire runtime planning and actual route provenance

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `evaluation/campaign_engine.py`
- Modify: `evaluation/observability.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`
- Test: `tests/test_campaign_engine.py`
- Test: `tests/test_evaluation_observability_repository.py`

- [ ] Add failing integration tests showing runtime invokes the planner at most once, persists the exact v2 contract, and records one `analysis_type="actual"` routing decision with decision source, candidates, matched rules, reason, confidence, and fallback reason.
- [ ] Add a test proving retrospective routing remains a separate row and cannot overwrite actual routing.
- [ ] Run focused tests.
- [ ] Construct the budget controller before optional ambiguity planning and inject the budgeted invoker.
- [ ] Persist the exact runtime contract in attempt materialization and the actual route in routing decisions.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_engine.py evaluation/observability.py tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_engine.py tests/test_evaluation_observability_repository.py
  git commit -m "feat(evaluation): persist actual v9 route decisions"
  ```

**Wave 2 gate:**

- [ ] Run all contract, route, admission, feasibility, provider-boundary, runtime, and campaign-engine tests.
- [ ] Confirm Q16 produces seven answer-free slots and an actual route record.

---

## Wave 3 — Slot-bound Retrieval and Corrective Repair

### Task 8: Compile source- and locator-aware retrieval tasks

**Files:**
- Modify: `data_base/agentic_v9/retrieval_tasks.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Test: `tests/test_agentic_v9_retrieval_tasks.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`

- [ ] Add failing tests for grouping compatible slots while preserving each `target_slot_id`.
- [ ] Add failing tests proving a document outside a slot's authorized IDs cannot satisfy the slot.
- [ ] Add Q16 tests for separate ODES formula, U-KAN Table 3, and Theorem 1 task groups.
- [ ] Run focused tests.
- [ ] Compile query text from answer-free slot descriptions, source hints, entities, and locator hints.
- [ ] Enforce per-slot source intersection during evidence packet creation.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add data_base/agentic_v9/retrieval_tasks.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py
  git commit -m "feat(agentic-v9): bind retrieval to atomic slots"
  ```

### Task 9: Group and persist missing-slot repair

**Files:**
- Modify: `data_base/agentic_v9/repair.py`
- Modify: `data_base/agentic_v9/execution_core.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `evaluation/trace_schemas.py`
- Test: `tests/test_agentic_v9_repair.py`
- Test: `tests/test_agentic_v9_execution_core.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`

- [ ] Add failing tests that repair only `not_found` required slots.
- [ ] Assert grouping key is authorized source group + locator identifier/type + compatible terms.
- [ ] Assert a maximum of two tasks per round and two rounds total.
- [ ] Assert sufficiency is recomputed after every round and stop reasons are persisted.
- [ ] Assert supported, conflicted, or explicitly unavailable slots are never repaired and authorized scope never expands.
- [ ] Run focused tests.
- [ ] Implement grouped repair and trace payloads containing round, slot IDs, constraints, query, resulting evidence IDs, and stop reason.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add data_base/agentic_v9/repair.py data_base/agentic_v9/execution_core.py evaluation/agentic_v9_campaign_runtime.py evaluation/trace_schemas.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py
  git commit -m "feat(agentic-v9): repair unresolved atomic slots"
  ```

**Wave 3 gate:**

- [ ] Execute a deterministic Q16 fixture where ODES formula and Theorem 1 are initially missing.
- [ ] Confirm repair queries contain the correct source/locator constraints and no expected answers.

---

## Wave 4 — Visual Asset Manifest and Positive Control

### Task 10: Enrich the existing GraphAssetLink manifest

**Files:**
- Modify: `graph_rag/schemas.py`
- Modify: `graph_rag/assets.py`
- Modify: `graph_rag/store.py`
- Modify: `pdfserviceMD/indexing_tasks.py`
- Test: `tests/test_graph_asset_links.py`
- Test: `tests/test_pdfservice_background_processing.py`

- [ ] Add failing round-trip tests for optional `storage_reference`, `sha256`, `width`, `height`, `printed_page_label`, and `formula_id`.
- [ ] Add a failing lookup test by authorized doc ID plus page/figure/table/formula locator.
- [ ] Add a failing ingestion test that stores `VisualElement.image_path` as a canonical upload-root-relative reference rather than base64.
- [ ] Run focused tests.
- [ ] Extend `GraphAssetLink` additively and retain old sidecar compatibility.
- [ ] Populate hashes/dimensions when the file exists; never fail text indexing solely because optional visual metadata cannot be read.
- [ ] Add bounded locator lookup to `GraphStore`.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add graph_rag/schemas.py graph_rag/assets.py graph_rag/store.py pdfserviceMD/indexing_tasks.py tests/test_graph_asset_links.py tests/test_pdfservice_background_processing.py
  git commit -m "feat(visual): persist resolvable asset manifests"
  ```

### Task 11: Resolve and load authorized visual assets

**Files:**
- Add: `data_base/agentic_v9/visual_asset_resolver.py`
- Modify: `data_base/agentic_v9/asset_locator.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `core/uploads.py`
- Test: `tests/test_agentic_v9_visual_asset_resolver.py`
- Modify: `tests/test_agentic_v9_visual_evidence_extractor.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

- [ ] Add failing tests for the pipeline:
  `doc ID + locator → manifest candidates → authorization → load → AssetLocator`.
- [ ] Cover each terminal diagnostic: `asset_manifest_empty`, `source_not_authorized`, `locator_not_matched`, `asset_load_failed`, `asset_exceeds_cap`.
- [ ] Add traversal and cross-document authorization tests.
- [ ] Run focused tests.
- [ ] Resolve only authorized `GraphAssetLink` rows, validate the canonical path with centralized upload helpers, and load only selected assets.
- [ ] Remove runtime dependence on `page_image_base64` inside retrieved chunk metadata.
- [ ] Return counts for manifest, authorization, locator matches, loads, selection, drops, evidence packets, and covered slots.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add data_base/agentic_v9/visual_asset_resolver.py data_base/agentic_v9/asset_locator.py evaluation/agentic_v9_campaign_runtime.py core/uploads.py tests/test_agentic_v9_visual_asset_resolver.py tests/test_agentic_v9_visual_evidence_extractor.py tests/test_agentic_v9_campaign_runtime.py
  git commit -m "feat(agentic-v9): resolve authorized visual assets"
  ```

### Task 12: Apply visual policy and establish a positive control

**Files:**
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `data_base/agentic_v9/visual_evidence_extractor.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Add: `scripts/backfill_visual_asset_manifest.py`
- Add: `tests/fixtures/agentic_v9_visual_positive_control/manifest.json`
- Add: `tests/fixtures/agentic_v9_visual_positive_control/page.png`
- Test: `tests/test_agentic_v9_visual_evidence_extractor.py`
- Test: `tests/test_agentic_v9_visual_manifest_backfill.py`

- [ ] Add failing tests for `never`, `preferred`, and `required`.
- [ ] Assert preferred visual runs only after matching text slots remain unresolved.
- [ ] Assert required failure marks only affected slots explicitly unavailable.
- [ ] Add a fixed positive-control asset and assert it yields a provenance-bound visual evidence packet with covered slot IDs.
- [ ] Add idempotent backfill tests and an explicit `visual_assets_unavailable` result.
- [ ] Run focused tests.
- [ ] Implement policy behavior, extractor diagnostics, and bounded backfill.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add data_base/agentic_v9/schemas.py data_base/agentic_v9/visual_evidence_extractor.py evaluation/agentic_v9_campaign_runtime.py scripts/backfill_visual_asset_manifest.py tests/fixtures/agentic_v9_visual_positive_control tests/test_agentic_v9_visual_evidence_extractor.py tests/test_agentic_v9_visual_manifest_backfill.py
  git commit -m "feat(agentic-v9): validate visual evidence end to end"
  ```

**Wave 4 gate:**

- [ ] Run graph asset, ingestion, resolver, extractor, runtime, and backfill tests.
- [ ] Run the positive control twice; confirm stable asset ID, one selected asset, at least one packet, and no scope escape.

---

## Wave 5 — Per-phase LLM Observability and Export

### Task 13: Extend attempt-level LLM storage

**Files:**
- Modify: `evaluation/trace_schemas.py`
- Modify: `evaluation/db.py`
- Modify: `evaluation/observability_storage.py`
- Test: `tests/test_evaluation_observability_schema.py`
- Test: `tests/test_evaluation_observability_repository.py`

- [ ] Add failing schema/repository round-trip tests for:
  `phase`, `reservation_id`, `provider_attempt`, `prompt_capture_status`,
  `full_prompt_capture_status`, `reasoning_tokens`, and `other_tokens`.
- [ ] Add migration tests that open a legacy SQLite schema and verify additive columns/defaults.
- [ ] Assert retry rows are append-only and uniquely addressable.
- [ ] Run focused tests.
- [ ] Add idempotent SQLite columns/indexes and update insert/list projections.
- [ ] Preserve old rows as unknown/unclassified rather than inventing phase.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add evaluation/trace_schemas.py evaluation/db.py evaluation/observability_storage.py tests/test_evaluation_observability_schema.py tests/test_evaluation_observability_repository.py
  git commit -m "feat(evaluation): store phase-linked provider attempts"
  ```

### Task 14: Observe every BudgetedLlmInvoker attempt

**Files:**
- Modify: `data_base/agentic_v9/budgeted_llm.py`
- Modify: `evaluation/observability.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `evaluation/campaign_engine.py`
- Test: `tests/test_agentic_v9_budgeted_llm.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`
- Test: `tests/test_campaign_engine.py`

- [ ] Add failing tests for success, timeout, cancellation, provider failure, retry, budget rejection, and observability-write failure.
- [ ] Assert every admitted provider attempt has phase, purpose, reservation ID, attempt number, provider/model, prompt/response hash, latency, status, safe error, and normalized usage.
- [ ] Assert retries create separate rows with attempts 1 and 2.
- [ ] Assert observer failure preserves the answer but marks observability partial.
- [ ] Run focused tests.
- [ ] Add an optional `LlmCallObserver` protocol and terminal callbacks around the existing reserve/invoke/reconcile sequence.
- [ ] Inject a run/campaign-aware observer from the campaign engine/runtime.
- [ ] Use these canonical phases: `contract_planning`, `evidence_extract`, `retrieval_judge`, `visual_extract`, `final_answer`.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add data_base/agentic_v9/budgeted_llm.py evaluation/observability.py evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_engine.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_engine.py
  git commit -m "feat(agentic-v9): observe every provider attempt"
  ```

### Task 15: Make prompt capture execution-time authoritative

**Files:**
- Modify: `evaluation/campaign_schemas.py`
- Modify: `evaluation/analytics.py`
- Modify: `evaluation/router.py`
- Modify: `data_base/agentic_v9/budgeted_llm.py`
- Test: `tests/test_evaluation_export_redaction.py`
- Test: `tests/test_evaluation_api.py`
- Test: `tests/test_agentic_v9_budgeted_llm.py`

- [ ] Add failing tests for capture policy defaults:
  hash `true`, preview `true`, full prompt `false`.
- [ ] Assert previews are sanitized/bounded and secrets are removed.
- [ ] Assert an export request cannot reveal a prompt that was not captured at execution.
- [ ] Assert exports distinguish `captured`, `redacted`, `not_captured_at_execution`, and `capture_failed`.
- [ ] Assert export summary contains run count, LLM-call count, per-phase counts, and hash/preview/full availability counts.
- [ ] Run focused tests.
- [ ] Persist capture policy in campaign manifest/setup snapshot.
- [ ] Hash the canonical serialized messages for every call; capture sanitized preview/full text according to the frozen campaign policy.
- [ ] Update export to reveal only captured fields and return explicit availability warnings.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add evaluation/campaign_schemas.py evaluation/analytics.py evaluation/router.py data_base/agentic_v9/budgeted_llm.py tests/test_evaluation_export_redaction.py tests/test_evaluation_api.py tests/test_agentic_v9_budgeted_llm.py
  git commit -m "feat(evaluation): enforce execution-time prompt capture"
  ```

### Task 16: Reconcile official tokens by phase

**Files:**
- Modify: `evaluation/analytics.py`
- Modify: `evaluation/release_metrics.py`
- Modify: `evaluation/research_analytics.py`
- Test: `tests/test_evaluation_analytics_context.py`
- Test: `tests/test_evaluation_release_metrics.py`
- Test: `tests/test_evaluation_research_analytics.py`

- [ ] Add failing tests proving phase attribution is complete only when summed official provider tokens equal official runtime tokens.
- [ ] Cover missing usage, mismatched totals, retries, reasoning/other tokens, and observability failure.
- [ ] Assert unknown totals remain N/A/partial and are never coerced to zero.
- [ ] Run focused tests.
- [ ] Aggregate by `llm_call_id`, phase, reservation, and provider attempt; report explicit mismatch reasons.
- [ ] Block official release comparability when required observability/accounting is partial.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add evaluation/analytics.py evaluation/release_metrics.py evaluation/research_analytics.py tests/test_evaluation_analytics_context.py tests/test_evaluation_release_metrics.py tests/test_evaluation_research_analytics.py
  git commit -m "fix(evaluation): reconcile official tokens by phase"
  ```

**Wave 5 gate:**

- [ ] Run all observability, export, analytics, release-metric, budgeted-invoker, runtime, and campaign-engine tests.
- [ ] Export a fixture campaign with full capture disabled and verify the export says not captured rather than returning null without explanation.

---

## Wave 6 — Evaluation Center Contracts and UI

### Task 17: Extend frontend contracts and N/A-safe mappers

**Files:**
- Modify: `Multimodal_RAG_System/src/types/evaluation.ts`
- Modify: `Multimodal_RAG_System/src/services/evaluationApi.ts`
- Modify: `Multimodal_RAG_System/src/pages/EvaluationCenter.mappers.ts`
- Test: `Multimodal_RAG_System/src/types/evaluation.contract.test.ts`
- Test: `Multimodal_RAG_System/src/services/evaluationApi.test.ts`
- Test: `Multimodal_RAG_System/src/pages/EvaluationCenter.mappers.test.ts`

- [ ] Add failing tests for v2 contract/slots/route/repair/visual/capture fields.
- [ ] Add a v1 fixture asserting `legacy_generic` and N/A atomic completeness.
- [ ] Assert missing values never become zero, empty success, or complete.
- [ ] Run:

  ```powershell
  npm test -- --run src/types/evaluation.contract.test.ts src/services/evaluationApi.test.ts src/pages/EvaluationCenter.mappers.test.ts
  ```

- [ ] Implement additive types, API requests/responses, and mapper projections.
- [ ] Rerun focused tests.
- [ ] Commit from the frontend repository:

  ```powershell
  git add src/types/evaluation.ts src/services/evaluationApi.ts src/pages/EvaluationCenter.mappers.ts src/types/evaluation.contract.test.ts src/services/evaluationApi.test.ts src/pages/EvaluationCenter.mappers.test.ts
  git commit -m "feat(evaluation-ui): map v9 evidence completeness"
  ```

### Task 18: Render route, slots, repair, and claim alignment

**Files:**
- Modify: `Multimodal_RAG_System/src/components/evaluation/AgentBehaviorTab.tsx`
- Modify: `Multimodal_RAG_System/src/components/evaluation/ClaimEvidenceTab.tsx`
- Modify: `Multimodal_RAG_System/src/components/evaluation/RouterLabTab.tsx`
- Test: `Multimodal_RAG_System/src/components/evaluation/AgentBehaviorTab.test.tsx`
- Test: `Multimodal_RAG_System/src/components/evaluation/ClaimEvidenceTab.test.tsx`
- Test: `Multimodal_RAG_System/src/components/evaluation/RouterLabTab.test.tsx`

- [ ] Add failing UI tests for:
  - contract version/slot-plan status;
  - actual route rationale separate from retrospective;
  - atomic slot state, source, locator, and answer type;
  - repair rounds and stop reasons;
  - visual/graph outcome by affected slot;
  - claim → slot → evidence and unresolved requirements;
  - legacy N/A.
- [ ] Run the three focused Vitest modules.
- [ ] Implement compact tables/detail panels without exposing raw full prompts.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add src/components/evaluation/AgentBehaviorTab.tsx src/components/evaluation/ClaimEvidenceTab.tsx src/components/evaluation/RouterLabTab.tsx src/components/evaluation/AgentBehaviorTab.test.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/RouterLabTab.test.tsx
  git commit -m "feat(evaluation-ui): show atomic v9 behavior"
  ```

### Task 19: Add capture setup and truthful export availability

**Files:**
- Modify: `Multimodal_RAG_System/src/components/evaluation/CampaignRunner.tsx`
- Modify: `Multimodal_RAG_System/src/components/evaluation/AblationDashboardTab.tsx`
- Test: `Multimodal_RAG_System/src/components/evaluation/CampaignRunner.test.tsx`
- Test: `Multimodal_RAG_System/src/components/evaluation/AblationDashboardTab.test.tsx`
- Test: `Multimodal_RAG_System/src/pages/EvaluationCenter.integration.test.tsx`

- [ ] Add failing tests for execution-time capture toggles and privacy warning.
- [ ] Add failing tests showing recorded phase counts and availability counts.
- [ ] Assert requesting full prompts after a non-capture campaign displays `full_prompts_not_captured_at_execution`.
- [ ] Assert the completed export displays actual exported run/LLM-call counts.
- [ ] Run focused tests.
- [ ] Implement setup controls and export availability/warning display.
- [ ] Rerun focused tests.
- [ ] Commit:

  ```powershell
  git add src/components/evaluation/CampaignRunner.tsx src/components/evaluation/AblationDashboardTab.tsx src/components/evaluation/CampaignRunner.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/pages/EvaluationCenter.integration.test.tsx
  git commit -m "feat(evaluation-ui): expose prompt capture availability"
  ```

**Wave 6 gate:**

- [ ] Run:

  ```powershell
  npm test -- --run
  npm run lint:ci
  npm run build
  ```

- [ ] Confirm v1 campaigns remain usable and display N/A rather than fabricated atomic metrics.

---

## Wave 7 — Integration, Smoke, and Release Verification

### Task 20: Add end-to-end contract fixtures

**Files:**
- Add: `tests/fixtures/agentic_v9_contract_cases.json`
- Add: `tests/test_agentic_v9_evidence_completeness_integration.py`
- Modify: `tests/test_evaluation_v9_attempt_persistence.py`
- Modify: `Multimodal_RAG_System/src/pages/EvaluationCenter.integration.test.tsx`

- [ ] Add fixed, answer-free question fixtures for Q5, Q7, Q11, Q14, and Q16.
- [ ] Test planner → retrieval task → repair → sufficiency → final → persistence → analytics projection.
- [ ] Assert:
  - Q16 has seven slots;
  - no planner/retrieval prompt contains fixture gold values;
  - Q14 unsupported inference is absent;
  - repair is source/locator-specific;
  - actual route and phase calls are present;
  - legacy data remains N/A-safe.
- [ ] Run backend and frontend integration tests.
- [ ] Commit backend and frontend fixture/test changes in their respective repositories with message:

  ```text
  test(agentic-v9): cover evidence completeness flow
  ```

### Task 21: Run production-focused verification

- [ ] Backend focused suite:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_schemas.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_route_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_v9_visual_asset_resolver.py tests/test_agentic_v9_visual_evidence_extractor.py tests/test_agentic_v9_visual_manifest_backfill.py tests/test_graph_asset_links.py tests/test_pdfservice_background_processing.py tests/test_evaluation_observability_schema.py tests/test_evaluation_observability_repository.py tests/test_evaluation_export_redaction.py tests/test_evaluation_analytics_context.py tests/test_evaluation_research_analytics.py tests/test_evaluation_release_metrics.py tests/test_evaluation_v9_attempt_persistence.py tests/test_agentic_v9_evidence_completeness_integration.py -q
  ```

- [ ] Backend lint for changed production/test paths:

  ```powershell
  .\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9 evaluation graph_rag pdfserviceMD/indexing_tasks.py core/uploads.py scripts/backfill_visual_asset_manifest.py tests
  .\.venv\Scripts\python.exe -m ruff format --check data_base/agentic_v9 evaluation graph_rag pdfserviceMD/indexing_tasks.py core/uploads.py scripts/backfill_visual_asset_manifest.py tests
  ```

- [ ] Backend full suite:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest
  ```

  If the known frozen golden SHA mismatch remains, record its exact test and hashes separately; do not describe the full suite as passing.

- [ ] Frontend full verification from `D:\flutterserver\Multimodal_RAG_System`:

  ```powershell
  npm test -- --run
  npm run lint:ci
  npm run build
  ```

- [ ] Review `git status --short` in both repositories and ensure no user-owned data/cache files are staged.

### Task 22: Run bounded smoke and evaluate release gates

- [ ] Execute Agentic v9 smoke for Q5, Q7, Q11, Q14, and Q16 with one repeat under a named Evaluation Setup preset.
- [ ] Execute paired Naive/v9 smoke for the same five questions only where comparison semantics are required.
- [ ] Verify each v9 run has:
  - contract v2 and non-empty actual route rationale;
  - atomic slots and final slot resolutions;
  - repair traces when slots remain missing;
  - phase-linked provider-attempt rows;
  - exact or explicitly partial token reconciliation;
  - capture availability matching setup;
  - no unsupported finding emitted as supported.
- [ ] Run the 16-question, single-repeat Agentic v9 evaluation only after the five-question smoke passes.
- [ ] Do not run the multi-repeat formal benchmark until all release gates pass.
- [ ] Record a release-verification manifest containing commit IDs for both repositories, Evaluation Setup snapshot/hash, dataset identity, campaign IDs, release-gate results, and residual failures.
- [ ] Commit only the verification manifest/documentation:

  ```text
  docs(agentic-v9): record evidence completeness verification
  ```

## Final Acceptance Checklist

- [ ] Q16 produces seven stable, answer-free atomic slots.
- [ ] Runtime planning inputs contain no benchmark key points, gold facts, or reference answers.
- [ ] Q14 cannot emit unsupported SegmentAnyBone claims as supported conclusions.
- [ ] Missing slots produce source- and locator-specific repair tasks within 2 × 2 bounds.
- [ ] At least one visual positive-control run selects an authorized asset and emits slot-bound visual evidence.
- [ ] Every new v9 run persists one actual route decision independently of retrospective routing.
- [ ] Every admitted provider attempt has a phase-linked, reservation-linked observability row.
- [ ] Export distinguishes captured, redacted, not captured, and capture failed.
- [ ] Official phase tokens reconcile exactly or remain explicitly partial.
- [ ] Legacy v1 runs display `legacy_generic` and N/A atomic completeness.
- [ ] Official release metrics fail closed for degraded contract, partial accounting, or missing required observability.
- [ ] Backend focused tests, frontend tests, frontend lint, and frontend build pass.
- [ ] Full backend test result is recorded honestly, including any pre-existing golden SHA mismatch.
