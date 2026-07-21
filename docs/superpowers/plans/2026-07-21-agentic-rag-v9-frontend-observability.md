# Agentic RAG v9 Frontend Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` task-by-task. Every task uses TDD, focused verification, review, and a scoped commit.
> **Status:** Ready after backend contract and campaign-version APIs
> **Repository:** `https://github.com/DeadMark70/Multimodal_RAG_System_Web.git`
> **Baseline branch/commit:** `master` / `1ab15449af756886039614fab6b6cc64781d1d23`
> **Backend prerequisites:** F0 requires Backend 13B; F0.5–F3 require Backend 14; F4 requires Backend 17A; F5 formal verification requires smoke artifacts from Backend 17B

This plan is intentionally separate because the UI is a different repository. Refresh and approve the baseline before implementation; never combine frontend and backend changes in one commit.

## UI rules

- API values are authoritative. Missing/not-instrumented values display `N/A`, never zero.
- Evaluation remains token-only; no monetary fallback is added.
- Historical v8 campaigns remain readable and show v9 sections as unavailable.
- The UI distinguishes complete, qualified partial, insufficient, failed, timed out, and cancelled.
- Graph mode alone is not proof of traversal; show locator/traversal evidence or an explicit not-instrumented state.

## File map

Primary integration files:

- `src/types/evaluation.ts`
- `src/services/evaluationApi.ts`
- `src/pages/EvaluationCenter.tsx`
- `src/pages/EvaluationCenter.mappers.ts`
- `src/pages/EvaluationCenter.mappers.test.ts`
- `src/components/evaluation/CampaignRunner.tsx`
- `src/components/evaluation/CampaignRunner.test.tsx`
- `src/components/evaluation/RunContextSelector.tsx`
- `src/components/evaluation/RunTraceTab.tsx`
- `src/components/evaluation/RunTraceTree.tsx`
- `src/components/evaluation/RetrievalEvidenceTab.tsx`
- `src/components/evaluation/ClaimEvidenceTab.tsx`
- `src/components/evaluation/CampaignOverviewTab.tsx`
- `src/components/evaluation/QuestionAnalysisTab.tsx`

Create focused presentation components rather than enlarging the tabs: `V9ContractPanel.tsx`, `V9SlotResolutionTable.tsx`, `V9EvidencePacketTable.tsx`, `V9ContextPackPanel.tsx`, `V9BudgetPanel.tsx`, `V9DecisionPanel.tsx`, and `V9ReleaseGatePanel.tsx`, with colocated tests. `AgentTraceViewer.tsx` remains legacy or reuses presentation components; it is not the primary `/evaluation` integration surface.

## Task F0 — Pin generated API client contract

- [ ] Record backend commit/OpenAPI hash and frontend baseline in a manifest/test fixture.
- [ ] Define typed `V9QueryContract`, required slot, source/scope, packet, slot resolution, sufficiency, reservation/budget snapshot, context pack, repair, conflict, final claim, metrics, and `V9ExecutionObservability`.
- [ ] Nest versioned data as `RunDetailResponse.agentic_v9?: V9ExecutionObservability | null`; do not flatten v9 fields or leave them as `Record<string, unknown>`.
- [ ] Add contract tests proving optional v9 fields do not break v8 payloads.
- [ ] Commit: `chore(evaluation): pin agentic v9 api contract`

## Task F0.5 — Add Agentic execution control-plane

**Modify:** `src/types/evaluation.ts`, `src/components/evaluation/CampaignRunner.tsx`, `CampaignRunner.test.tsx`, API tests.

- [ ] Add `AgenticExecutionVersion = 'v8'|'v9'`, optional `agentic_execution_version`, and `agentic_v9_shadow` to campaign input.
- [ ] Controls appear only when Agentic mode is selected; default is v8 and v9 is labelled Evidence-First.
- [ ] Shadow is an explicit research-only checkbox with runtime-token warning; reject authoritative v9 + v9 shadow.
- [ ] Persisted campaign history displays saved version/condition and never guesses from current environment.
- [ ] Render backend `configuration_incompatible` reasons before/after campaign creation; never silently change thinking or budgets.
- [ ] Commit: `feat(evaluation): control agentic execution version`

## Task F1 — Add selected-run v9 evidence detail mapping

**Likely files:** `src/pages/EvaluationCenter.tsx`, evaluation API client/types, mapper tests.

- [ ] Load details by selected run ID; never reuse the first campaign run.
- [ ] Map QueryContract, SlotResolution, EvidencePacket, packed/dropped evidence, and final claim evidence IDs.
- [ ] Preserve null/N/A semantics for scores, tokens, Graph, and unsupported-claim metrics.
- [ ] Commit: `feat(evaluation): map agentic v9 run evidence`

## Task F1.5 — Make run identity condition/profile/version aware

**Modify:** `RunContextSelector.tsx`, `EvaluationCenter.tsx`, mappers/types/tests.

- [ ] Extend run option with `conditionId`, `executionProfile`, `agenticExecutionVersion`, and `responseStatus`.
- [ ] Labels distinguish `Agentic v8`, `Agentic v9`, and `Agentic v9 shadow` for the same question/repeat.
- [ ] Selected-run keys use run ID and never collapse rows by mode/question/repeat alone.
- [ ] Changing run clears expanded/raw/citation state and retains existing request-ID race protection.
- [ ] Commit: `fix(evaluation): preserve agentic condition identity`

## Task F2 — Render Evidence-First execution trace

- [ ] Show contract/route, authorized document scope, required slots, retrieval/repair rounds, final prose batch, sufficiency, conflict, context pack, final answer, and verification.
- [ ] Extend existing `RunTraceTree` lifecycle grouping rather than building a second grouping system; retain raw payload disclosure.
- [ ] Show provider attempt count, reserved/reconciled tokens, timeout/cancel status, and final-generation count.
- [ ] Do not show cost fields.
- [ ] Commit: `feat(evaluation): render agentic v9 execution trace`

## Task F3 — Render evidence, slots, and Graph observability

- [ ] Evidence table shows packet/source/scope/locator/support type/slot/round/packed/used/cited state.
- [ ] Slot table distinguishes supported, conflicted, explicitly unavailable, and not found.
- [ ] Context pack shows input/packed/dropped counts and per-slot/per-source tokens.
- [ ] Graph panel shows never/locator fallback/required locator, actual traversal, resolved source evidence, fallback reason, or not instrumented.
- [ ] Graph node/edge/path values are nullable. Distinguish actual zero, N/A, Not requested, Not triggered, and Fallback/Failed; replace mapper `numberValue(...,0)` with nullable mapping.
- [ ] Visual panel shows asset locator and evidence extraction, never a subtask answer.
- [ ] Render evidence as escaped plain text with `whiteSpace="pre-wrap"`; never inject HTML/Markdown.
- [ ] Limit excerpts by default, expand only authorized content, paginate/window long lists, collapse raw JSON, and support packet/claim ID copy plus citation-to-packet navigation.
- [ ] Commit: `feat(evaluation): expose evidence and graph provenance`

## Task F4 — Update overview and comparison semantics

- [ ] Show response-status counts, required-slot coverage, important unsupported claims, provenance failures, pack efficiency, Graph locator success, and final-generation counts.
- [ ] Exclude partial accounting from token comparisons with a clear reason.
- [ ] Label one-repeat campaigns as smoke; label formal gates only when the benchmark manifest proves 16×3×8 comparable snapshots.
- [ ] Show paired delta/CI/category/per-question regression outputs from backend without client-side fabrication.
- [ ] Group comparison arms by mode + condition ID + execution profile/version; formal v9 and shadow v9 are never merged.
- [ ] Explain that confidence intervals are clustered by question and official token ratio is ratio of summed official runtime tokens.
- [ ] Commit: `feat(evaluation): show v9 release metrics`

## Task F5 — Integration tests and release verification

- [ ] Test v8 historical payload, complete v9, qualified partial, insufficient, partial accounting, no Graph traversal, Graph fallback, cancelled, and missing instrumentation fixtures.
- [ ] Assert unknown numeric values never render `0`, `$0.000`, or `0.0%`.
- [ ] Test actual Graph zero separately from uninstrumented Graph values and every policy state label.
- [ ] Assert run selector changes all detail panels consistently.
- [ ] Test v8/v9/shadow campaign controls, condition-aware labels, configuration incompatibility, request races, escaped malicious evidence, pagination, and historical `agentic_v9=null`.
- [ ] Run lint, typecheck, unit/integration tests, and production build with an explicit timeout/log capture.
- [ ] Commit: `test(evaluation): verify agentic v9 observability ui`

## Frontend done criteria

The selected run is always identifiable; no metric is fabricated; evidence-to-claim and Graph-to-source provenance are inspectable; token-only semantics hold; historical campaigns render safely; lint/typecheck/tests/build pass against the pinned backend contract.
