# Agentic v9 Wave 7 Release Gates Implementation Plan

> **For agentic workers:** Execute in the existing isolated backend and frontend worktrees. Use TDD, one scoped commit per deliverable, and a single combined review after all Wave 7 work.

**Goal:** Produce fail-closed v9 release metrics, scientifically valid benchmark identities/statistics, a 9-run smoke artifact, and the corresponding Evaluation Center release UI/tests; do not launch the formal 384-run matrix or promote v9.

**Architecture:** The backend owns all release calculations and comparability decisions. It emits a typed release report bound to an immutable benchmark manifest. The frontend renders that report verbatim, preserving unknown/partial state rather than deriving metric values. Smoke is an operational artifact produced only after the backend and frontend contracts are built; it is a gate for a later human-authorized formal benchmark, not a release decision.

**Repositories:**
- Backend: `D:\flutterserver\pdftopng\.worktrees\agentic-v9-backend` (`feature/agentic-v9-release-gates`)
- Frontend: `D:\flutterserver\_worktrees\Multimodal_RAG_System-agentic-v9-frontend` (`feature/agentic-v9-release-ui`)

## Global constraints

- Formal benchmark, promotion/default switch, and release verdict are out of scope.
- Token-only: never introduce price/cost fallbacks.
- API data is authoritative. Missing/partial/instrumentation gaps are `N/A` or an explicit failed gate, never zero.
- Formal and shadow v9 are distinct `(mode, condition_id, execution_profile)` arms.
- Comparability requires equal golden, corpus/index, prompt, model/thinking, phase, evaluator, and code snapshots.
- Official P95 and token ratio use only successful, accounting-complete official rows; a smoke/formal gate still fails if any official row fails or times out.
- Frontend must not fabricate delta, confidence interval, release gate, or Graph traversal state.

---

### Task 17A: Backend release-metrics API and fail-closed gates

**Files:**
- Create: `evaluation/release_metrics.py`
- Modify: `evaluation/campaign_schemas.py`, `evaluation/research_analytics.py`, `evaluation/router.py`, `openapi.json`
- Test: `tests/test_evaluation_release_metrics.py`, `tests/test_evaluation_api.py`

**Contract:** `GET /api/evaluation/campaigns/{campaign_id}/release-metrics` returns a typed report containing an immutable manifest summary, per-arm response-status counts, required-slot coverage, important unsupported-claim rate, provenance failure rate, pack efficiency, Graph locator success/fallback, final-generation count, latency, token ratio, paired deltas/CI, comparability state, and gate reasons. Every metric has a value-or-unavailable status plus a concrete reason.

- [ ] Write fixtures for complete comparable v9/v8/naive arms, partial accounting, missing golden/evidence mapping, and incompatible snapshots.
- [ ] Verify each invalid fixture produces a blocking gate and no numeric substitute.
- [ ] Implement typed release metric schemas and deterministic calculation helpers.
- [ ] Implement the API route through the existing research-analytics/repository ownership path.
- [ ] Regenerate `openapi.json` from the application, then assert the generated route/schema matches the pinned runtime contract.
- [ ] Run the focused backend metrics/API tests and commit `feat(evaluation): derive v9 evidence release metrics`.

### Task 17C: Benchmark manifest, arm identity, and clustered statistics

**Files:**
- Create: `evaluation/benchmark_release.py`
- Modify: `evaluation/release_metrics.py`, `evaluation/campaign_engine.py`, `evaluation/campaign_schemas.py`
- Test: `tests/test_evaluation_benchmark_release.py`, `tests/test_evaluation_release_metrics.py`

**Contract:** A benchmark manifest records condition identity, ordered paired blocks, snapshot fingerprints, evaluator blinding metadata, and run eligibility. Pairing is by `(question_id, repeat_number)`; eight repeats are aggregated before the 16-question deterministic bootstrap. Its seed and resampling method are returned in the report.

- [ ] Write failing tests for v9/shadow separation, duplicate/missing paired arms, incompatible snapshots, aggregate-before-bootstrap behavior, deterministic seed repeatability, and ratio-of-sums token accounting.
- [ ] Implement canonical arm identity and manifest validation; reject unpaired, shadow-mixed, snapshot-incompatible, failed, timed-out, or partial-accounting official data.
- [ ] Implement deterministic arm-order randomization and label-blinding metadata without changing RAGAS score semantics.
- [ ] Implement question-cluster paired bootstrap and successful-accounting-complete P95.
- [ ] Add the non-blocking `v8 + A-type phase policy` ablation arm to manifest/report metadata only; it must never affect official promotion comparators.
- [ ] Run focused benchmark/release tests and commit `feat(evaluation): enforce clustered v9 benchmark statistics`.

### Task Smoke: Produce and validate the 9-run smoke artifact

**Files:**
- Create or modify only the smallest existing campaign/benchmark runner surface required to invoke the backend with an explicit smoke manifest.
- Test: deterministic local smoke-manifest/gate tests under `tests/test_evaluation_benchmark_release.py`.

**Operational contract:** use Q9/Q15/Q16 × `naive`, `agentic-v8`, `agentic-v9` × repeat 1, with immutable snapshots captured in the manifest. It is successful only when all 9 official runs finish, required RAGAS metrics are complete, accounting/phase attribution is complete, and the smoke gates have no blocking reason.

- [ ] Add a dry-run manifest test that proves exactly nine official arms and excludes shadow/ablation arms.
- [ ] Discover the configured evaluation service and credentials without logging secrets.
- [ ] Launch the smoke only when the service and all required snapshots are available; save campaign/artifact IDs and its gate report.
- [ ] If the environment is unavailable or any gate fails, preserve the artifact and report `BLOCKED`/`FAILED`; do not retry with altered snapshots and do not start the formal matrix.

### Task F4: Frontend release-metric comparison semantics

**Files:**
- Modify: `src/types/evaluation.ts`, `src/services/evaluationApi.ts`, `src/components/evaluation/CampaignOverviewTab.tsx`, `src/pages/EvaluationCenter.tsx`
- Create or modify: focused release report presentation component under `src/components/evaluation/`
- Test: `src/services/evaluationApi.test.ts`, `src/components/evaluation/CampaignOverviewTab.test.tsx`, `src/pages/EvaluationCenter.integration.test.tsx`

- [ ] Write API/type fixtures for complete smoke, non-comparable, partial-accounting, and formal-manifest states.
- [ ] Render response status, evidence/provenance/pack/Graph/final-generation metrics exactly as backend supplied.
- [ ] Exclude partial accounting from token comparison and show the backend reason.
- [ ] Label one-repeat reports as smoke; only label formal when the backend manifest proves 16×3×8 comparable snapshots.
- [ ] Render grouped arms by mode + condition + profile/version; never merge v9 shadow with official v9.
- [ ] Render paired delta/CI/category/per-question results and the clustered-by-question / ratio-of-sums explanation without client computation.
- [ ] Run focused UI tests and commit `feat(evaluation): show v9 release metrics`.

### Task F5: Frontend integration and release-verification tests

**Files:**
- Modify: the focused Evaluation Center/release component test files created in F4, `src/pages/EvaluationCenter.integration.test.tsx`

- [ ] Add v8 historical, complete v9, qualified partial, insufficient, partial accounting, no Graph traversal, Graph fallback, cancelled, and missing instrumentation fixtures.
- [ ] Assert unknown numeric fields never render `0`, `$0.000`, or `0.0%`; assert measured Graph zero remains distinct from unavailable.
- [ ] Assert selector changes update all selected detail panels; cover v8/v9/shadow controls, incompatibility/preflight, request races, escaped malicious evidence, pagination, and `agentic_v9=null`.
- [ ] Verify independent shadow campaign notices/status only. Do not reintroduce nonexistent persisted `shadow_progress` linkage.
- [ ] Run lint, typecheck, focused unit/integration tests, and production build with captured exit status; commit `test(evaluation): verify agentic v9 observability ui`.

### Final wave verification and review

- [ ] Regenerate backend OpenAPI and refresh the frontend contract pin before F4/F5 tests.
- [ ] Run backend focused suite, frontend focused suite/lint/build, and diff checks in both repositories.
- [ ] One fresh, read-only Terra High reviewer checks the entire backend/frontend Wave 7 diff and smoke artifact. Critical/Important findings require a single fix pass and reviewer closure.
- [ ] Do not perform formal 16×3×8 execution, release promotion, or default-version change.
