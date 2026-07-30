# Agentic v9 Wave B Planner Validation Design

**Date:** 2026-07-31
**Status:** Approved design
**Scope:** Comparison-planner eligibility, subject validation, and graph-planner provider accounting

## 1. Goal

Improve Agentic v9 comparison planning without changing retrieval, reranking, final context packing, or synthesis behavior.

Wave B must:

1. prevent claims, capabilities, conditions, and metrics from being promoted into comparison subjects;
2. fail soft to the existing query contract when fewer than two valid comparison subjects remain;
3. route the generic graph planner's ambiguity-only provider call through the existing budgeted and observable v9 provider boundary.

## 2. Evidence and Root Causes

The 2026-07-31 evaluation export showed two independent planner-boundary defects.

### 2.1 Invalid comparison subjects

For Q3, the comparison planner produced:

- `MedSAM-2`
- `single-prompt segmentation`
- `initial bounding box / prompt quality`

Only `MedSAM-2` is an independent subject. The other values are a capability and a condition used to judge two claims about the same model. The current schema verifies shape and uniqueness but does not verify that subjects are explicit, independent entities grounded in the question.

### 2.2 Unattributed graph-planner usage

For Q14:

- runtime accounting reported 2,283 tokens;
- persisted `comparison_plan` and `final_answer` calls accounted for 2,066 tokens;
- the remaining 217 tokens were 195 input and 22 output tokens.

The difference came from `GenericGraphRouter` creating its own provider through `get_llm("graph_extraction")` when its deterministic fast path could not choose a route. That call participated in the global accounting scope but bypassed `BudgetedLlmInvoker` and the v9 LLM-call observer.

## 3. Non-goals and Frozen Behavior

Wave B must not change:

- hybrid retrieval candidate count or query execution;
- reranker model, candidate count, `top_k`, fail-soft policy, or scores;
- per-subject document limits;
- corrective retrieval behavior;
- final context packing or diversity policy;
- final synthesis prompt, response parsing, or answer-generation count;
- source authorization policy;
- graph retrieval algorithms or graph route semantics.

The comparison planner remains answer-free and may make at most one provider call.

## 4. Design

### 4.1 Comparison candidate admission

The existing lightweight comparison-marker check remains a recall-oriented prefilter. It may request a planner call for an ambiguous question, but it is not sufficient to activate comparison specialization.

Comparison specialization becomes active only after the returned plan passes subject validation.

This preserves recall for new question phrasings while preventing an imprecise lexical gate from becoming a behavioral hard gate.

### 4.2 Subject validation

After schema validation and invented-number rejection, validate every proposed subject against the original question.

A valid subject must:

1. have its display name or one of its aliases explicitly anchored in the question after Unicode-aware normalization;
2. represent an independent entity, model, method, dataset, document, or other comparison target;
3. not be merely a requested dimension, metric, capability, condition, prompt type, result, or claim wording;
4. remain distinct from every other accepted subject after normalized identity comparison.

Validation must not call another model.

The planner prompt will also clarify that:

- two claims about one model are not two subjects;
- capabilities, conditions, metrics, and comparison dimensions belong in `dimensions`;
- subject names and aliases must be copied from the question.

Prompt guidance improves provider behavior, while deterministic validation remains authoritative.

### 4.3 Validation outcome

If two to four valid subjects remain, construct the existing `ComparisonPlan` and apply the existing comparison overlay.

If fewer than two valid subjects remain:

- return a fail-soft planner outcome;
- record a dedicated safe fallback reason such as `invalid_subjects`;
- retain the pre-comparison `QueryContract`;
- continue the run without comparison specialization.

Do not partially activate a one-subject comparison plan.

Examples:

- Q3: fallback to the original contract because it contains one entity and two claims/conditions.
- Q4: retain `nnMamba` and `EfficientMedNeXt-L`.
- Q7: retain the four explicitly named segmentation models.
- Q14: retain `SAM`, `SegmentAnyBone`, and `SegVol`.

### 4.4 Graph planner provider boundary

Extend the graph locator boundary so its ambiguity-only graph router receives an injected `LlmInvoker`.

For Agentic v9:

- construct the invoker from the run's existing `RunBudgetController`;
- use the Evaluation Setup provider/model configuration;
- attach the existing LLM-call observer;
- invoke the graph route phase as `graph_route` with a graph-planning purpose;
- preserve the graph router's existing deterministic fast path and fallback decision.

The graph router may retain its legacy self-created provider only for non-v9 compatibility callers that do not inject an invoker.

This change affects provider admission and telemetry, not graph retrieval semantics.

### 4.5 Accounting and phase attribution

After the graph planner call is routed through `BudgetedLlmInvoker`:

- its provider reservation is included in the run budget;
- its measured input/output tokens are persisted as an LLM call;
- phase attribution includes the graph-planning phase;
- runtime totals must reconcile with persisted LLM calls;
- failures remain safe and sanitized.

No fabricated zero-token usage is allowed.

## 5. Execution Flow

```text
Existing route contract
  -> comparison marker prefilter
  -> at most one comparison planner call
  -> schema and numeric validation
  -> subject grounding and semantic-role validation
       -> >= 2 valid subjects: existing comparison overlay
       -> < 2 valid subjects: original contract, fail-soft diagnostic
  -> unchanged retrieval
  -> unchanged reranker
  -> graph route when required
       -> deterministic fast path, or
       -> injected budgeted graph planner call
  -> unchanged evidence processing
  -> unchanged context packing
  -> unchanged synthesis
```

## 6. Diagnostics

Persist enough safe metadata to distinguish:

- planner not requested;
- planner returned `is_comparison=false`;
- planner response/schema failure;
- planner returned invalid subjects;
- planner planned a valid comparison;
- graph route selected deterministically;
- graph route selected by an observed provider call;
- graph route provider fallback.

Do not persist hidden chain-of-thought or raw secrets.

## 7. Tests

### 7.1 Comparison planner unit tests

Add tests that prove:

- Q3-like one-entity claim arbitration falls back;
- a capability cannot become a subject;
- a condition or metric cannot become a subject;
- an unanchored model name cannot be introduced by the planner;
- aliases explicitly present in the question are accepted;
- Q4-like two-model comparison remains planned;
- Q14-like three-model lineage judgment remains planned;
- invalid subjects do not raise or fail the run.

### 7.2 Graph provider-boundary tests

Add tests that prove:

- the v9 graph router receives the injected budgeted invoker;
- deterministic graph routing does not consume a provider call;
- ambiguity routing produces one observed `graph_route` LLM call;
- timeout/provider failure retains the existing safe graph fallback;
- no Agentic v9 graph-planner call bypasses the run budget.

### 7.3 Runtime and accounting regression tests

Add a focused Q14-shaped runtime test that proves:

- graph planner usage is persisted under its actual phase;
- runtime total tokens equal the sum of persisted measured LLM calls;
- phase attribution is complete;
- retrieval/reranker/synthesis configuration remains unchanged.

## 8. Acceptance Criteria

Wave B is complete when:

- Q3 no longer produces capability/condition subjects;
- Q4 and Q14 still activate comparison specialization with their real subjects;
- subject-validation failure is fail-soft;
- Q14's 217-token attribution gap is eliminated in an equivalent smoke run;
- all v9 provider calls pass through the run budget boundary;
- no retrieval, reranker, context-packing, or synthesis behavior changes;
- focused tests and lint pass.
