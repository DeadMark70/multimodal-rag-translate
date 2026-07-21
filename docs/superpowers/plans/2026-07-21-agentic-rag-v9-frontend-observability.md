# Agentic RAG v9 Frontend Observability Implementation Plan

> **Status:** Ready after backend Task 13B
> **Repository:** `https://github.com/DeadMark70/Multimodal_RAG_System_Web.git`
> **Baseline branch/commit:** `master` / `1ab15449af756886039614fab6b6cc64781d1d23`
> **Backend prerequisite:** reviewed `openapi.json` from Agentic v9 backend Task 13B

This plan is intentionally separate because the UI is a different repository. Refresh and approve the baseline before implementation; never combine frontend and backend changes in one commit.

## UI rules

- API values are authoritative. Missing/not-instrumented values display `N/A`, never zero.
- Evaluation remains token-only; no monetary fallback is added.
- Historical v8 campaigns remain readable and show v9 sections as unavailable.
- The UI distinguishes complete, qualified partial, insufficient, failed, timed out, and cancelled.
- Graph mode alone is not proof of traversal; show locator/traversal evidence or an explicit not-instrumented state.

## Task F0 — Pin generated API client contract

- [ ] Record backend commit/OpenAPI hash and frontend baseline in a manifest/test fixture.
- [ ] Regenerate or update TypeScript types for QueryContract, slots, packets, context pack, budgets, repairs, conflicts, Graph outcome, and execution metrics.
- [ ] Add contract tests proving optional v9 fields do not break v8 payloads.
- [ ] Commit: `chore(evaluation): pin agentic v9 api contract`

## Task F1 — Add selected-run v9 evidence detail mapping

**Likely files:** `src/pages/EvaluationCenter.tsx`, evaluation API client/types, mapper tests.

- [ ] Load details by selected run ID; never reuse the first campaign run.
- [ ] Map QueryContract, SlotResolution, EvidencePacket, packed/dropped evidence, and final claim evidence IDs.
- [ ] Preserve null/N/A semantics for scores, tokens, Graph, and unsupported-claim metrics.
- [ ] Commit: `feat(evaluation): map agentic v9 run evidence`

## Task F2 — Render Evidence-First execution trace

- [ ] Show contract/route, authorized document scope, required slots, retrieval/repair rounds, final prose batch, sufficiency, conflict, context pack, final answer, and verification.
- [ ] Collapse lifecycle running/success rows under one logical stage while retaining raw payload access.
- [ ] Show provider attempt count, reserved/reconciled tokens, timeout/cancel status, and final-generation count.
- [ ] Do not show cost fields.
- [ ] Commit: `feat(evaluation): render agentic v9 execution trace`

## Task F3 — Render evidence, slots, and Graph observability

- [ ] Evidence table shows packet/source/scope/locator/support type/slot/round/packed/used/cited state.
- [ ] Slot table distinguishes supported, conflicted, explicitly unavailable, and not found.
- [ ] Context pack shows input/packed/dropped counts and per-slot/per-source tokens.
- [ ] Graph panel shows never/locator fallback/required locator, actual traversal, resolved source evidence, fallback reason, or not instrumented.
- [ ] Visual panel shows asset locator and evidence extraction, never a subtask answer.
- [ ] Commit: `feat(evaluation): expose evidence and graph provenance`

## Task F4 — Update overview and comparison semantics

- [ ] Show response-status counts, required-slot coverage, important unsupported claims, provenance failures, pack efficiency, Graph locator success, and final-generation counts.
- [ ] Exclude partial accounting from token comparisons with a clear reason.
- [ ] Label one-repeat campaigns as smoke; label formal gates only when the benchmark manifest proves 16×3×8 comparable snapshots.
- [ ] Show paired delta/CI/category/per-question regression outputs from backend without client-side fabrication.
- [ ] Commit: `feat(evaluation): show v9 release metrics`

## Task F5 — Integration tests and release verification

- [ ] Test v8 historical payload, complete v9, qualified partial, insufficient, partial accounting, no Graph traversal, Graph fallback, cancelled, and missing instrumentation fixtures.
- [ ] Assert unknown numeric values never render `0`, `$0.000`, or `0.0%`.
- [ ] Assert run selector changes all detail panels consistently.
- [ ] Run lint, typecheck, unit/integration tests, and production build with an explicit timeout/log capture.
- [ ] Commit: `test(evaluation): verify agentic v9 observability ui`

## Frontend done criteria

The selected run is always identifiable; no metric is fabricated; evidence-to-claim and Graph-to-source provenance are inspectable; token-only semantics hold; historical campaigns render safely; lint/typecheck/tests/build pass against the pinned backend contract.
