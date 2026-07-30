# Agentic v9 Comparison-Aware Retrieval Design

## Status

Approved design for specification review. This document defines one bounded
Agentic v9 capability change. It does not authorize implementation until the
written spec and the later implementation plan are separately approved.

## Problem

Agentic v9 can retrieve many relevant candidates yet still send evidence from
only one comparison subject to final synthesis. Q4 is the concrete observed
case:

- the question compares `nnMamba` with the realized efficiency of other
  architectures;
- hybrid retrieval represented multiple documents before reranking;
- all four final contexts came from the nnMamba document;
- the run was treated as sufficiently supported and produced the wrong
  conclusion;
- the benchmark ground truth required the EfficientMedNeXt-L Params and FLOPs
  evidence that never reached final synthesis.

This is not primarily a global retrieval-recall or reranker-execution failure.
The missing capability is semantic decomposition of an explicit comparison
into its real subjects, subject-specific retrieval, balanced final selection,
and honest subject coverage.

The current deterministic entity extraction is not a safe substitute. For Q4,
ordinary technical dimensions such as `Params` and `FLOPs` may be extracted
alongside model names, and the bounded comparison compiler can choose the
wrong first two values. The current generic slots also allow evidence from one
task to satisfy the entire comparison.

## Goals

1. Detect explicit comparison or judgment questions semantically.
2. Identify the actual comparison subjects and requested dimensions without
   using benchmark answers or expected source files.
3. Retrieve evidence independently for each subject.
4. Preserve at least minimal evidence coverage for every explicit subject in
   the final context pack.
5. Perform at most one targeted corrective retrieval when a subject remains
   uncovered.
6. Fail soft on planner/provider problems and retain the existing whole-question
   v9 path.
7. Make planner, retrieval, coverage, repair, and fallback decisions auditable.
8. Keep the first experiment isolated: do not change the existing final
   synthesis prompt or add per-slot LLM calls.

## Non-goals

- Do not use evaluation `expected_sources`, `ground_truth`, expected evidence,
  or atomic benchmark facts at runtime.
- Do not add exact textual locator gates, quote binding, or filename matching.
- Do not turn source diversity into a hard rule for non-comparison questions.
- Do not add an iterative autonomous retrieval loop.
- Do not retry the semantic planner.
- Do not change Native RAG.
- Do not redesign graph or visual capability handling.
- Do not tune the final synthesis prompt in this wave.
- Do not claim quality improvement without a smoke test and paired evaluation.

## Selected Architecture

The selected approach is a semantic comparison planner followed by
subject-specific retrieval and a balanced merge:

```text
question
→ comparison-intent check
→ semantic Comparison Planner
→ subject-specific retrieval tasks
→ each task: Hybrid 8 → rerank 8 → subject top 2
→ balanced merge
→ subject coverage check
→ at most one deterministic corrective retrieval for a missing subject
→ final context pack of 4–6 chunks
→ existing final synthesis
```

This is preferred over global reranking with only document quotas because
global retrieval may never formulate a query for the missing subject. It is
also preferred over a fully iterative agent because that would add unstable
LLM calls, latency, and token cost before the bounded design is validated.

## Components and Responsibilities

### 0. Integration Point and Authority

Comparison specialization is an overlay between the admitted base contract and
retrieval-task compilation:

1. existing admission resolves the authorized source scope and base v9
   contract;
2. the comparison-intent boundary decides whether specialization is requested;
3. post-contract feasibility includes one additional comparison-planning call
   before any provider request is made;
4. the budget controller is created before invoking the planner;
5. a valid comparison plan supplies subject slots and retrieval tasks without
   changing the authorized scope;
6. a planner fallback discards only the specialization overlay and compiles the
   existing base contract.

The semantic planner is not a new source-authority or answer-authority. It
cannot add authorized documents, alter the user's setup snapshot, or replace
the base route recorded by admission.

### 1. Comparison Intent Boundary

Only a question that plausibly asks for a comparison, selection, relative
judgment, contradiction decision, or bounded classification enters the
comparison-planning path. The intent boundary may use existing deterministic
signals as a cheap pre-filter, but the Comparison Planner is authoritative for
whether a suspected question contains explicit comparison subjects.

Non-comparison questions keep their current v9 contract, retrieval tasks,
context packing, and final synthesis unchanged.

The intent boundary must not classify a question as a comparison merely
because it contains technical metrics such as Params, FLOPs, Dice, latency, or
memory. Those terms are dimensions, not necessarily subjects.

### 2. Semantic Comparison Planner

For every suspected comparison or judgment question, call the Comparison
Planner exactly once. The planner receives:

- the user question;
- the runtime-authorized source scope identifiers or safe display names already
  available to v9;
- a compact schema and instruction that it must identify subjects and
  dimensions, not answer the question.

It must not receive:

- evaluation expected source filenames;
- ground truth;
- benchmark atomic facts;
- expected answer values;
- expected evidence locators.

The planner returns a strictly validated structure equivalent to:

```json
{
  "is_comparison": true,
  "subjects": [
    {
      "subject_id": "nnmamba",
      "display_name": "nnMamba",
      "aliases": ["nnMamba", "Mamba segmentation model"],
      "retrieval_query": "nnMamba parameters FLOPs computational efficiency"
    },
    {
      "subject_id": "efficientmednext_l",
      "display_name": "EfficientMedNeXt-L",
      "aliases": ["EfficientMedNeXt-L", "Efficient MedNeXt L"],
      "retrieval_query": "EfficientMedNeXt-L parameters FLOPs computational efficiency"
    }
  ],
  "dimensions": ["parameters", "FLOPs", "computational efficiency"],
  "qualification": "cross-paper relative comparison, not a same-configuration benchmark"
}
```

Validation rules:

- `is_comparison` is required.
- A comparison requires 2–4 unique subjects.
- `subject_id`, `display_name`, aliases, and retrieval query are bounded in
  length.
- Subject IDs are normalized and unique within the plan.
- Empty aliases and duplicate aliases are removed.
- The retrieval query must contain its subject name or alias.
- The output schema has no answer, winner, expected-source, or result fields,
  and unknown fields are rejected.
- Retrieval queries may preserve numeric or locator tokens already present in
  the question, but may not introduce new numeric answer values, filenames, or
  document IDs.
- Unknown fields are rejected.

If more than four subjects appear, the planner must group semantically
equivalent names or retain only the explicit core subjects required by the
question. The runtime never creates more than four subject tasks.

Planner execution policy:

- timeout: 64 seconds or the remaining overall deadline, whichever is lower;
- retry count: zero;
- one provider call at most;
- the call is reserved and recorded through the existing budgeted v9 LLM
  boundary;
- provider usage is attributed to a dedicated comparison-planning phase.

### 3. Planner Fail-Soft Behavior

Planner timeout, provider error, invalid response, schema violation, or a
validated `is_comparison=false` result must never clear contexts or fail the
run.

For planner failure:

1. record `comparison_planner_fallback`;
2. record one safe reason from `timeout`, `provider_error`,
   `invalid_response`, `schema_violation`, or `not_comparison`;
3. execute the current whole-question v9 retrieval, reranking, packing, and
   synthesis path;
4. do not fabricate subjects or subject coverage;
5. keep the overall Agentic run deadline at 128 seconds.

The fallback must preserve the current answer-producing behavior. It is an
observability-visible loss of comparison specialization, not a fatal error.

### 4. Subject-Specific Retrieval Tasks

For a valid comparison plan, compile one first-round retrieval task per
subject. Each task:

- uses the subject-specific planner query;
- remains constrained to the normal runtime-authorized source scope;
- targets only that subject's required coverage slot;
- uses the existing hybrid retrieval and reranker boundary;
- retrieves up to 8 hybrid candidates, reranks up to 8 candidates, and exposes
  up to 2 selected chunks to the balanced merge.

No task may target every comparison slot. This prevents evidence about one
subject from satisfying another subject.

The subject planner supplements, rather than replaces, existing source
authorization. Authorization continues to prevent evidence outside the user's
allowed corpus, but benchmark source metadata never narrows runtime retrieval.

### 5. Balanced Merge and Final Context Limits

The balanced merge is deterministic after reranking.

For two subjects:

- retain up to the best two chunks per subject;
- final context limit is four.

For three or four subjects:

- reserve at least the best available chunk for each covered subject;
- allocate remaining positions by rerank score;
- final context limit is six.

Across all cases:

- exact duplicate chunks are removed;
- a chunk keeps its originating `subject_id`;
- a subject cannot consume another subject's reserved minimum position;
- no fixed rerank-score threshold rejects a subject's only available evidence;
- source authorization remains mandatory;
- the existing safe reranker fallback remains valid if reranking times out or
  fails.

The merge is a coverage-aware allocation rule, not a claim that equal chunk
counts imply equal evidence quality.

### 6. Subject Coverage and Sufficiency

The comparison plan creates one required subject slot for every explicit
subject. A subject is covered only when at least one valid evidence packet
originating from that subject's retrieval task survives into the usable
evidence set.

Coverage rules:

- evidence binds only to the subject task that produced it;
- evidence from subject A cannot satisfy subject B;
- evidence does not require an exact quotation;
- evidence does not require an exact filename or text locator;
- no global score threshold is introduced;
- an optional qualification does not become a required subject slot.

The pre-repair coverage result contains:

- covered subject IDs;
- missing subject IDs;
- evidence IDs and selected chunk IDs per covered subject.

If all required subjects are covered, execution proceeds to existing final
synthesis. If any subject is missing, the runtime may execute one corrective
retrieval.

### 7. Bounded Corrective Retrieval

Corrective retrieval does not call the Comparison Planner again. It reuses:

- the missing subject's normalized name and aliases;
- the original requested dimensions;
- the original question;
- the same authorized source scope.

The corrective query is built deterministically and runs through the existing
hybrid retrieval and reranker path. Only one corrective round is allowed for
the entire comparison, even if multiple subjects are missing.

Corrective results enter the balanced merge without exceeding the same final
context limit:

- two-subject comparison: maximum four chunks;
- three- or four-subject comparison: maximum six chunks.

After repair, coverage is evaluated once more. There is no loop.

If a required subject remains missing:

- the run remains successful;
- status becomes `qualified_partial`;
- all supported evidence is retained;
- final synthesis is not allowed to represent the result as a complete
  comparison;
- the runtime records the missing subjects and repair outcome.

The first implementation wave must enforce this status through existing
sufficiency/final-result controls without rewriting the final prompt.

## Data Flow

```text
Authorized question and corpus
        │
        ▼
Comparison intent pre-filter
        │ suspected
        ▼
Comparison Planner ─────────────── failure ──► current whole-question path
        │ valid plan
        ▼
Subject task compiler
        │
        ├─► subject A: Hybrid 8 → rerank 8 → top 2
        ├─► subject B: Hybrid 8 → rerank 8 → top 2
        └─► optional subjects C/D
        │
        ▼
Balanced merge and coverage
        │ missing subject
        ▼
One deterministic corrective retrieval
        │
        ▼
Final coverage
        ├─► complete
        └─► qualified_partial
        │
        ▼
Existing final synthesis
```

## Observability

Every comparison-capable run must persist enough structured data to reconstruct
the decision without full prompts:

- `planner_status`;
- `planner_latency_ms`;
- `planner_fallback_reason`;
- `is_comparison`;
- normalized subjects, aliases, and dimensions;
- a safe query preview or query hash for each subject task;
- hybrid candidate count per subject;
- reranker candidate and selected counts per subject;
- reranker executed/fallback state;
- `coverage_before_repair`;
- missing subjects before repair;
- whether corrective retrieval executed;
- `coverage_after_repair`;
- missing subjects after repair;
- final chunk IDs, document IDs, and `subject_id`;
- final comparison status: `complete`, `qualified_partial`, or
  `comparison_planner_fallback`;
- planner token usage and phase attribution.

The existing candidate-stage diagnostics remain available. The new fields add
semantic subject identity and coverage; they do not replace document/chunk
diagnostics.

### Redacted Export

Redacted exports may include:

- planner status and safe fallback reason;
- normalized subject display names and aliases;
- dimensions;
- query hashes and bounded previews;
- document and chunk identifiers under the existing redaction policy;
- counts, coverage, repair, and final status.

They must not expose full prompts unless the existing explicit export control
allows them. They must never export benchmark ground truth or expected source
metadata as runtime planner input.

## Budget and Deadline Behavior

- Comparison Planner: one call, 64-second timeout or remaining overall
  deadline, whichever is lower, and no retry.
- Overall Agentic v9 execution deadline: 128 seconds.
- Corrective retrieval adds embedding, hybrid retrieval, and reranking work but
  no planner LLM call.
- The final synthesis call remains unchanged.
- The planner reservation must be reflected in post-contract feasibility and
  token accounting before execution starts.
- A planner fallback releases or settles its reservation according to the
  existing budget-controller rules.
- Timeout or provider failure must be safely classified and attributed rather
  than converted into an unclassified token/accounting gap.

## Error Handling

| Condition | Runtime behavior | Recorded status |
| --- | --- | --- |
| Planner timeout | Use current whole-question path | `comparison_planner_fallback:timeout` |
| Planner provider error | Use current whole-question path | `comparison_planner_fallback:provider_error` |
| Invalid or fenced non-schema JSON | Use current whole-question path | `comparison_planner_fallback:invalid_response` |
| Schema violation | Use current whole-question path | `comparison_planner_fallback:schema_violation` |
| Planner says not comparison | Use current whole-question path | `comparison_planner_fallback:not_comparison` |
| Reranker error/timeout | Preserve hybrid order through existing fail-soft path | existing reranker fallback plus subject ID |
| Subject missing before repair | Run one corrective retrieval | `repair_attempted` |
| Subject still missing after repair | Keep evidence, answer partially | `qualified_partial` |
| Unauthorized evidence | Reject under existing source-authorization rule | existing authorization failure |

## Expected Code Boundaries

Implementation should prefer small isolated units and existing interfaces:

- add a typed comparison-plan schema near the v9 schemas;
- add a comparison planner component with strict parsing and safe failure
  classification;
- integrate it at contract/task planning without changing Native RAG;
- extend retrieval-task compilation with subject task identity;
- add a deterministic balanced merge/coverage helper;
- adapt evidence projection so packets bind to the originating subject slot;
- extend one-shot repair construction for missing comparison subjects;
- extend v9 runtime diagnostics and redacted export projection;
- update post-contract budget feasibility for the optional planner call.

No unrelated refactor of the route planner, execution core, evaluation center,
graph pipeline, or visual pipeline belongs in this change.

## Test Strategy

### Unit Tests

1. Planner schema accepts 2–4 valid subjects and dimensions.
2. Planner schema rejects answer/winner/source fields, invented numeric query
   tokens, source filenames, duplicate subjects, overlong queries, and unknown
   fields while allowing numeric tokens copied from the question.
3. Timeout, provider error, invalid response, and schema violation map to the
   correct safe fallback reason.
4. The planner is called once at most and never retried.
5. Non-comparison questions keep the existing task plan.
6. Q4-like questions identify `nnMamba` and `EfficientMedNeXt-L`, not `Params`
   and `FLOPs`, as subjects.
7. Two-subject tasks each target only their own subject slot.
8. Balanced merge returns at most two chunks per subject and at most four
   chunks for two subjects.
9. Three- and four-subject merges reserve one available chunk per subject and
   never exceed six.
10. Evidence from one subject cannot satisfy another subject.
11. Corrective retrieval runs once at most and uses the missing subject plus
    original dimensions.
12. Missing post-repair coverage produces `qualified_partial`, not failure.
13. Planner fallback preserves the current whole-question path and does not
    fabricate coverage.
14. Token usage is attributed to the comparison-planning phase.

### Integration Tests

1. Q4 smoke executes three repeats:
   - planner identifies nnMamba and EfficientMedNeXt-L;
   - planner call count is at most one per run;
   - final contexts include both subject groups when evidence exists;
   - final context count is at most four;
   - contexts are not all nnMamba when EfficientMedNeXt-L evidence is
     retrievable;
   - coverage and repair telemetry is persisted.
2. A forced planner timeout still produces a usable answer through the old
   whole-question path.
3. A reranker failure keeps subject-specific hybrid candidates through the
   existing fallback.
4. An unavailable subject yields a successful `qualified_partial` result with
   the missing subject recorded.
5. Redacted export exposes structured planner/coverage data without benchmark
   leakage or unauthorized full prompts.

### Generalization Cases

- `SwinUNETR` versus `MedNeXt`;
- three-model latency comparison using `Model A`, `Model B`, and `Model C`;
- mixed Chinese/English model names;
- a general summary question with no explicit comparison subjects;
- a comparison with aliases or punctuation variants;
- planner timeout and invalid JSON.

## Experiment and Rollout

### Stage 1: Q4 Smoke

Run Q4 three times under the same model/setup used for the current baseline.
Proceed only if:

- no run fails because of planner specialization;
- both subjects are detected consistently;
- final context includes both subject groups when both are retrievable;
- the planner uses one call at most;
- fallback succeeds under an injected timeout;
- token and phase accounting remain complete.

### Stage 2: Fixed 16-Question Paired Evaluation

Compare the new v9 condition with the current v9 baseline under the same:

- model preset and thinking configuration;
- authorized corpus;
- reranker model and device;
- retrieval limits;
- repeat count;
- RAGAS evaluator metadata.

Review per-question correctness, faithfulness, relevancy, latency, total tokens,
planner tokens, fallback frequency, and subject coverage.

### Decision Rules

- Retain the change only if comparison-question correctness or evidence
  coverage improves without a meaningful non-comparison regression.
- If non-comparison questions change, treat that as a routing leak and roll
  back before prompt tuning.
- If both sides' evidence reaches final synthesis but the answer still mixes
  subjects, overgeneralizes, or declares an unsupported winner, keep retrieval
  findings separate and propose a later evidence-aware synthesis wave.
- Do not combine that synthesis change with this retrieval experiment.

## Acceptance Criteria

- Q4 comparison subjects are nnMamba and EfficientMedNeXt-L, not metric words.
- Every valid explicit comparison subject has an independent retrieval task and
  coverage slot.
- Hybrid retrieval and reranking remain `8 → 8`, with subject selection
  `top 2`.
- Two-subject final context is at most four; three- or four-subject context is
  at most six.
- One subject's evidence cannot mark another subject supported.
- At most one corrective retrieval occurs.
- Planner timeout/provider/parse failures do not fail the run or clear
  contexts.
- Planner timeout is 64 seconds; overall v9 deadline is 128 seconds.
- Non-comparison v9 behavior and Native RAG behavior remain unchanged.
- No benchmark source, answer, or expected evidence leaks into runtime.
- Comparison-planner usage, coverage, repair, fallback, and final subject
  mapping are present in durable diagnostics and redacted export.
- Existing final synthesis prompt remains unchanged in this wave.
- Scoped tests, Q4 smoke, and fixed paired evaluation are required before a
  quality claim.

## Rollback Boundary

The feature must be isolated behind the comparison-specialization boundary so
the semantic planner, subject tasks, balanced merge, and coverage-aware repair
can be disabled together. Disabling the feature restores the current
whole-question Agentic v9 path without changing Native RAG, source
authorization, reranker defaults, graph handling, visual handling, or final
synthesis.
