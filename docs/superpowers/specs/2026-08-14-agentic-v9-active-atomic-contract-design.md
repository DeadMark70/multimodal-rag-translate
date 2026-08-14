# Agentic RAG v9 Active Atomic Contract Design

**Status:** Approved

**Date:** 2026-08-14

## Objective

Make atomic requirement decomposition the active and mandatory Query Contract path for every Agentic RAG v9 evaluation run. The change must improve multi-part question coverage without changing the deterministic route, introducing a second planning architecture, or coupling this checkpoint to evidence qualification and structured final synthesis.

This is the first independently deployable checkpoint in the broader Agentic v9 recovery sequence. It answers one question only: what direct evidence does this question require the system to retrieve?

## Motivation

The current runtime can route and retrieve successfully while still representing a multi-part question with one or two broad slots. This makes downstream retrieval, sufficiency, repair, and observability too coarse:

- multiple direct evidence needs may collapse into one vague slot;
- comparison conclusions may be confused with source facts;
- sufficiency can report complete without demonstrating coverage of each direct requirement;
- repair cannot target the exact missing fact;
- later final synthesis cannot reliably distinguish evidence collection from reasoning obligations.

The repository already contains deterministic requirement decomposition, a v2 contract model, a contract planner, and comparison-aware retrieval. The problem is ownership and wiring, not the absence of another architecture.

## Scope

### In scope

- Agentic v9 evaluation Query Contract planning.
- Active atomic evidence requirements for every Agentic v9 run.
- Typed synthesis obligations and response constraints.
- Deterministic decomposition with at most one low-confidence planning call.
- Consolidation of atomic and comparison semantic planning into that one call.
- Retrieval-task targeting, sufficiency, repair, packing, and observability changes required to consume the atomic contract.
- Focused tests and documentation for this checkpoint.

### Out of scope

- New route classifiers or learned routing.
- Changes to the existing deterministic route decision.
- Span or quote extraction.
- Semantic evidence qualification or entailment verification.
- Claim verification.
- Structured final synthesis or changes to the final-answer prompt.
- Native RAG, v8, general chat, retriever, reranker, embedding, and model changes.
- A new automatic Agentic/Naive token-ratio gate.

## Fixed Decisions

- Atomic Contract is active for all Agentic v9 runs. There is no feature flag, shadow path, or per-campaign opt-in.
- All runtime Query Contracts are version 2. There is no v1 runtime fallback.
- Rollback is operationally simple: revert the atomic checkpoint commit.
- The existing deterministic route remains authoritative.
- The atomic overlay cannot change route, intent, source scope, graph policy, visual policy, route budgets, or evidence-extraction policy.
- Only direct evidence requirements become `RequiredSlot` records.
- Comparative conclusions, selections, causal conclusions, aggregations, and qualifications are synthesis obligations, not evidence slots.
- Response-format and conditional instructions are response constraints, not evidence slots.
- Only evidence slots participate in retrieval, sufficiency, repair, and required-first packing.
- A deterministic plan is preferred. Low-confidence or structurally ambiguous cases may use at most one LLM call.
- Atomic and comparison semantic planning share that one call. The independent comparison-planner provider call is removed from the active runtime.
- Comparison-aware retrieval behavior remains. Comparison subjects and dimensions become metadata on the atomic contract.
- Planner failure produces a degraded v2 contract with one safe evidence slot; it does not restore v1 or fail the run.
- This checkpoint deliberately retains task-target-inherited packet binding and the existing final prompt. Observability must state that semantic qualification is not enabled.

## Architecture

The active planning path is:

```text
authorized source admission
  -> existing deterministic RoutePlanner
  -> immutable base Query Contract fields
  -> deterministic AtomicRequirementDecomposer
  -> optional one-call semantic planner when confidence is insufficient
  -> AtomicContractOverlay
  -> one active QueryContract v2
```

`AtomicContractOverlay` is a narrow composition boundary. It replaces or adds only:

- atomic evidence requirements;
- synthesis obligations;
- response constraints;
- comparison subjects and dimensions;
- atomic planning provenance.

It copies without reinterpretation:

- route;
- intent;
- authorized source scope;
- graph and visual policy;
- route budgets and deadlines;
- evidence-extraction policy;
- other base contract controls.

The overlay approach is preferred over replacing `RoutePlanner` or adding a second orchestration service. It makes the new responsibility explicit while preserving the already-working routing and budget owners.

## Contract Data Model

### Evidence requirements

Direct evidence requirements continue to use `RequiredSlot`. Each slot represents a fact that should be retrievable from an authorized source, for example:

- the reported value of a metric for Model A;
- the reported value of the same metric for Model B;
- the source-defined evaluation condition;
- an explicit limitation stated by the source.

All Agentic v9 contracts must have between one and eight evidence slots, identified by stable code-assigned IDs `S1` through `S8`.

The contract remains experimental with respect to semantic atomic completeness. `slot_semantics` remains `heuristic_experimental`, and `atomic_completeness` remains unavailable until a later semantic qualification checkpoint can measure it honestly.

### Synthesis obligations

Add a typed `SynthesisObligation` with:

- `obligation_id`, assigned by code as `O1`, `O2`, and so on;
- `kind`: `comparison`, `selection`, `causal`, `aggregation`, or `qualification`;
- `description`;
- `depends_on_slot_ids`.

An obligation describes reasoning the final answer must eventually perform over retrieved facts. It never creates a retrieval task and never directly satisfies or blocks the sufficiency gate in this checkpoint.

Example:

```text
S1: retrieve Model A's reported score
S2: retrieve Model B's reported score
O1: compare S1 and S2 using the requested criterion
```

The conclusion “Model A is better” is not an evidence slot unless an authorized source explicitly states that conclusion as a fact requested by the question.

### Response constraints

Add a typed response constraint with:

- stable code-assigned ID `C1`, `C2`, and so on;
- kind: `conditional_scope`, `output_format`, `prohibition`, or `allowed_labels`;
- description.

Examples include “answer only if the source reports the metric,” “return the top two items,” and “do not infer missing values.” Constraints are retained for later synthesis but do not create evidence tasks or participate in sufficiency.

### Planning provenance

Route provenance and slot-plan provenance remain separate. The contract records:

- `slot_plan_status`: `complete` or `degraded`;
- `slot_plan_source`: `deterministic`, `llm_planner`, or `safe_fallback`;
- `slot_plan_confidence`: `high`, `medium`, or `low`;
- `slot_plan_fallback_reason`;
- `truncated_requirement_count`.

Route provenance continues to describe how the route was chosen. Slot provenance describes only how evidence requirements were derived.

## Deterministic Decomposition

The existing decomposition strategies remain the first path:

- numbered requirements;
- coordinated clauses;
- entity-distributive comparison requirements;
- fallback parsing;
- response-constraint separation.

A deterministic result is accepted without a provider call only when all of the following hold:

- it contains one to eight direct evidence requirements;
- it is not low confidence;
- it is not truncated;
- dependencies reference valid evidence requirements;
- evidence requirements are distinguishable from synthesis obligations;
- source and locator intent can be mapped safely within the authorized scope;
- no suspected compound question has collapsed into one vague slot;
- response constraints have not been misclassified as evidence needs.

High- and acceptable medium-confidence deterministic results add no provider tokens.

## Single Semantic Planning Call

The system may invoke one planning LLM when deterministic decomposition is insufficient. Triggers include:

- low-confidence or fallback decomposition;
- multiple question clauses collapsed into one requirement;
- unclear comparison subjects or dimensions;
- unclear evidence-versus-synthesis classification;
- unclear dependencies;
- unclear authorized-source mapping;
- more than eight candidate requirements requiring semantic merge;
- complex, unpunctuated Chinese that cannot be split reliably.

The single response contains:

- `evidence_requirements`;
- `synthesis_obligations`;
- `response_constraints`;
- optional `comparison.subjects` and `comparison.dimensions`.

The provider cannot choose or alter the route. It cannot emit document IDs, source IDs, answer values, golden answers, or external sources. It may name only sources already present in the authorized source scope; code performs the final identity mapping.

The response is strict, answer-free structured output with extra fields forbidden. Code assigns all IDs, verifies dependency references, enforces the eight-slot limit, and rejects invented numerical answers or unauthorized sources.

There is no retry. Any invalid response discards the entire semantic plan instead of partially mixing it with the deterministic plan.

## Comparison Planning Consolidation

Comparison retrieval remains a specialization, but comparison semantic planning no longer owns a separate provider call.

- Deterministic decomposition may identify comparison subjects and dimensions directly.
- When semantic planning is required, the one Atomic Contract call returns those fields together with the evidence requirements.
- Existing subject balancing, subject-specific retrieval queries, and comparison evidence limits remain in use.
- The former independent Comparison Planner call count must be zero in the active runtime.

This avoids paying once to decompose the question and again to rediscover the same comparison structure.

## Active Runtime Flow

### Retrieval task compilation

The final contract contains the immutable base route fields plus atomic slots, comparison metadata, synthesis obligations, and response constraints.

Retrieval tasks are compiled only from evidence slots. Compatible slots are grouped using existing route-aware rules such as subject, authorized source, locator, visual requirement, and dependency. Synthesis obligations never produce queries.

For a comparison question, a typical plan is:

```text
Task A -> target S1 and S2 for subject A
Task B -> target S3 and S4 for subject B
O1 -> compare the results of S1..S4; no retrieval task
```

The atomic contract does not authorize unbounded query fan-out. It remains constrained by the existing route, retrieval-round, repair, deadline, and runtime-token budgets.

### Packet binding

This checkpoint does not add span or quote extraction. Retrieved chunks continue to inherit the target slot IDs of the task that retrieved them.

The runtime must report:

- `slot_binding_method = task_target_inherited`;
- `semantic_qualification = not_enabled`.

This distinction is important: the checkpoint measures whether retrieval covered the right atomic requirements, not whether each chunk semantically entails its assigned slot.

### Sufficiency

The sufficiency gate uses the active atomic evidence slots. Synthesis obligations and response constraints are excluded.

In this checkpoint, `complete` means every direct evidence requirement has a packet accepted under the current validation boundary. It does not mean that every derived conclusion has been verified.

### Repair

Repair targets only missing evidence slots. Compatible gaps may be grouped under existing bounded repair behavior. Repair cannot target a synthesis obligation and cannot use answer or golden-answer content.

### Context packing and final generation

Required evidence slots retain required-first packing priority and the existing soft maximum behavior. The final prompt remains the currently deployed `Question + Evidence` form.

Slots, resolutions, obligations, constraints, structured claims, and semantic qualification are intentionally not added to final synthesis in this checkpoint. Those changes must be evaluated separately after atomic retrieval coverage is measured.

## Failure Handling

Atomic planning failure is fail-soft at the planning boundary and fail-closed with respect to invented structure.

### Degraded v2 fallback

The following conditions produce a degraded v2 contract:

- deterministic decomposition raises or returns an unusable result;
- the semantic planner times out;
- the semantic planner returns malformed or non-conforming output;
- the planner introduces answer content or unauthorized sources;
- the planning call cannot be admitted without consuming the final-answer reserve;
- the planning deadline is unavailable.

The degraded contract:

- preserves the deterministic route and all immutable base fields;
- creates one safe `S1` requiring an answer to the complete original question;
- preserves only response constraints that can be identified safely;
- records `slot_plan_status = degraded`;
- records `slot_plan_source = safe_fallback`;
- records a bounded, explicit fallback reason.

It does not return to v1 and does not cause `configuration_incompatible` merely because atomic planning failed.

### Hard failures

Existing hard failures remain hard failures when the system cannot safely establish an authorized source scope or a valid deterministic route. Atomic overlay logic does not conceal source-authorization or route-contract errors.

## Cost and Budget Boundaries

This checkpoint adds no evidence-qualification, claim-verification, or final-synthesis calls.

- Accepted deterministic decomposition: zero new provider calls.
- Low-confidence planning: at most one provider call per run.
- Independent comparison-planner provider calls: zero.
- Planner retries: zero.
- Per-slot and per-chunk planning calls: zero.

Planning uses the existing budget and deadline authorities. The final-answer reserve has priority. If the planner cannot be admitted safely, execution uses the degraded v2 contract.

The existing Agentic/Naive runtime-token ratio target of at most 3.0 remains a campaign-level release measure. This checkpoint does not duplicate or automatically enforce that policy.

## Observability

Each Agentic v9 run must expose enough data to distinguish decomposition behavior from later retrieval and evidence-quality behavior:

- `contract_version = 2`;
- `slot_plan_status`;
- `slot_plan_source`;
- `slot_plan_confidence`;
- `slot_plan_fallback_reason`;
- `evidence_requirement_count`;
- `synthesis_obligation_count`;
- `response_constraint_count`;
- `truncated_requirement_count`;
- `atomic_planner_call_count`;
- `comparison_planner_call_count = 0`;
- `slot_binding_method = task_target_inherited`;
- `semantic_qualification = not_enabled`.

The persisted contract must include the typed evidence slots, obligations, constraints, dependencies, comparison metadata, and unchanged route fields needed for run-level inspection.

No field may imply semantic validation, atomic completeness, or verified conclusions before the corresponding future stage is implemented.

## Testing Strategy

Implementation follows test-driven development.

### Schema and overlay tests

- v2 accepts fixed evidence requirements, obligations, constraints, and provenance.
- invalid IDs, dependencies, counts, or enum values are rejected.
- overlay preserves route, budgets, authorized sources, graph policy, visual policy, and evidence-extraction policy exactly.
- all active runtime contracts are v2.

### Deterministic decomposition tests

- numbered English and Chinese requirements;
- coordinated clauses;
- entity-distributive comparisons;
- locators and source constraints;
- conditional and formatting constraints;
- causal and comparative synthesis obligations;
- maximum eight slots and explicit truncation behavior;
- compound-question detection that prevents one vague accepted slot.

The imported Q1-Q32 question set should be used as a fixed regression corpus where licensing and repository policy permit. Tests should assert structural invariants rather than golden answers.

### Semantic planner tests

- accepted deterministic plans make zero provider calls;
- low-confidence plans make exactly one call;
- comparison metadata returns in the same call;
- provider output cannot alter the route;
- unknown sources, answer leakage, malformed JSON, invalid dependencies, extra fields, timeout, and budget rejection all produce degraded v2;
- no retry occurs;
- no independent comparison-planner call occurs.

### Runtime integration tests

- retrieval tasks target existing atomic evidence slots only;
- obligations and constraints generate no retrieval tasks;
- comparison task grouping remains bounded and subject-aware;
- sufficiency considers atomic evidence slots only;
- repair targets only missing evidence slots;
- context packing preserves required-first behavior;
- final prompt remains unchanged;
- planner budget rejection preserves final-answer reserve;
- materialized observability reports actual call counts and non-semantic binding accurately.

### Regression tests

- Native RAG and v8 behavior are unchanged;
- current retrieval and repair budget caps remain authoritative;
- source authorization remains fail-closed;
- existing campaign persistence and analytics can read the new contract fields;
- no `golden_expected_route` or built-in Q1-Q16 dependency is introduced.

## Deployment Checkpoint

This design is deployed and tested separately from semantic evidence qualification and structured final synthesis.

### Hard acceptance

- every successful Agentic v9 run persists contract version 2;
- no atomic-planning failure creates a v1 fallback;
- no run becomes `configuration_incompatible` solely because atomic planning failed;
- deterministic route and route budgets match the pre-checkpoint behavior;
- atomic planning uses at most one provider call per run;
- comparison planning uses no additional provider call;
- every retrieval task references valid evidence slots;
- synthesis obligations do not enter sufficiency or repair;
- final-answer reserve remains protected;
- no answer, golden answer, or unauthorized source leaks into the contract;
- focused and existing Agentic v9 regression suites pass.

### Campaign analysis

Run the same evaluation corpus before and after the checkpoint and inspect:

- evidence-slot counts and descriptions;
- synthesis-obligation classification;
- comparison-subject coverage;
- degraded fallback rate and reasons;
- retrieval-query count and distribution;
- required-slot coverage;
- correctness, faithfulness, relevancy, latency, and runtime tokens.

The expected first-order effect is improved multi-part retrieval coverage and potentially better correctness and relevancy. A large faithfulness improvement is not promised because semantic qualification and structured final synthesis remain intentionally absent.

If slot coverage improves but faithfulness does not, the next checkpoint should target span/quote evidence qualification. If atomic decomposition itself degrades retrieval or quality, this checkpoint can be reverted as one coherent commit without retaining partial runtime branches.

## Compatibility and Migration

Historical v1 and shadow-v2 observations remain readable. No database rewrite or retroactive contract migration is required.

New Agentic v9 executions no longer create v1 contracts. Analytics must continue to identify historical contract versions honestly and must not reinterpret historical generic slots as atomic.

Existing comparison observability may remain readable for historical runs, while new runs record the consolidated atomic-planner provenance and zero independent comparison-planner calls.

## Authority and Follow-up Boundary

This document supersedes the shadow-only activation posture of `2026-08-01-agentic-v9-shadow-requirements-v2-design.md` for new Agentic v9 evaluation runs. It does not supersede the broader evidence-first architecture.

The next checkpoint, if this one is retained after campaign testing, is span/quote extraction and semantic evidence qualification. Final prompt and structured claim changes remain a later independent checkpoint so their cost and quality effects can be measured separately.
