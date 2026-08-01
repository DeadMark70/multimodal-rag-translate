# Agentic v9 Shadow Requirements v2 Design

## Status

Approved design for a deterministic, behavior-neutral upgrade of the Agentic
v9 shadow requirement diagnostics.

## Problem

The first shadow implementation proves that requirement diagnostics can be
persisted and exported without changing Agentic v9 execution. Its parser is
not yet reliable enough for behavioral use:

- Parenthesized identifiers such as `tooth 1)` and decimals such as `0.4` can
  be mistaken for numbered subquestions.
- Questions containing several answer obligations can remain one coarse
  requirement.
- Conditional answer rules and output labels can be mistaken for evidence
  requirements, producing false `missing` results.
- A single `information_need` value cannot represent questions that reference
  both a Figure and a Markdown Table.
- Candidate evidence means only that a retrieved document may be relevant; it
  must never be reported as verified support.

## Goal

Build `shadow_requirements_v2`, a conservative deterministic parser that
separates atomic answer obligations from response constraints, records mixed
representation needs, and remains strictly observational.

## Non-goals

- No new LLM call.
- No QID-, paper-, model-, dataset-, answer-, or ground-truth-specific rule.
- No changes to routing, retrieval, reranking, graph execution, visual
  execution, sufficiency, context packing, synthesis, or response status.
- No corrective retrieval driven by v2 output.
- No claim that candidate evidence supports an atomic fact.

## Design Principles

1. Split only when syntax supplies positive evidence for a boundary.
2. Prefer one low-confidence coarse requirement over several false atomic
   requirements.
3. Keep answer obligations and response constraints in separate collections.
4. Preserve every requested representation instead of choosing one by
   precedence.
5. Fail soft: diagnostic failure cannot fail, downgrade, or alter a run.
6. Bound output to prevent telemetry growth.

## Architecture

The analyzer remains a pure post-retrieval projection:

```text
question + already-retrieved documents
  -> protected-span scanner
  -> top-level block parser
  -> obligation and constraint classifier
  -> conservative coordinated-clause splitter
  -> optional entity-distributive expansion
  -> mixed representation classifier
  -> candidate-only evidence mapper
  -> bounded shadow_requirements_v2 trace/export payload
```

The runtime invokes the analyzer after normal Agentic v9 execution data has
already been collected. Its result is written only to trace and redacted
export data.

## Parsing Pipeline

### 1. Normalize without destroying boundaries

Normalize repeated whitespace but retain punctuation required to distinguish
sentences, numbered blocks, clauses, quotations, and parenthetical spans.

### 2. Protect non-boundary numeric spans

Before numbered-block detection, mark spans that cannot start a subquestion:

- decimals and version-like values such as `0.4`, `2.5`, and `v3.1`;
- numbers inside parentheses or paired Chinese parentheses;
- Figure, Table, Equation, Theorem, Appendix, page, tooth, class, and model
  identifiers;
- percentages, ranges, coordinates, and measurement values.

Protected spans retain their original text in emitted requirements.

### 3. Detect top-level numbered blocks

A numbered marker is accepted only when all of the following hold:

- it begins the normalized question or follows a sentence/section boundary;
- at least two markers form a monotonic sequence beginning at `1`;
- the marker uses an accepted top-level form such as `1.`, `1、`, `（一）`,
  or `一、`;
- it does not overlap a protected span.

If these conditions are not met, numbered parsing is skipped rather than
guessed.

### 4. Separate response constraints

Clauses led by generic control language are emitted as constraints, not
requirements. Supported constraint kinds are:

- `conditional_scope`: for example, "若不能，必須按 claim scope 分開回答";
- `output_format`: requested ordering, labels, or response form;
- `prohibition`: statements such as "不要寫成通用排名";
- `allowed_labels`: category definitions such as `A.` and `B.` used to
  classify subjects.

A constraint is never assigned evidence coverage and cannot increase the
`missing_count`.

### 5. Extract answer obligations

The parser recognizes generic obligation cues in Chinese and English:

- identify, point out, determine, list;
- calculate, report a value, give a range;
- compare, classify, decide whether;
- explain, describe, reconstruct;
- define, provide an equation or theorem condition.

Coordinated clauses are split only when each side has its own obligation cue,
or when an explicit continuation cue such as "此外" introduces a new requested
answer. A bare conjunction does not create a boundary.

### 6. Apply entity-distributive expansion conservatively

An explicit entity list is expanded only when the question contains a
distributive cue such as "每個", "各自", "分別", or "另外三者". The expansion
creates at most one requirement per named entity for the shared predicate.
No implicit Cartesian product is allowed.

### 7. Enforce bounds and fallback

- A non-empty question produces at least one requirement.
- At most eight requirements and eight constraints are emitted.
- If valid obligations exceed eight, the first eight remain observable and
  `truncated=true` records the overflow.
- If decomposition is uncertain or fails, emit the whole question as one
  `fallback` requirement with `low` confidence.

## Data Contract

The new payload uses `schema_version: shadow_requirements_v2`.

Each requirement preserves the v1 diagnostic fields and adds:

```json
{
  "requirement_id": "R1",
  "text": "...",
  "answer_kind": "number",
  "information_need": "markdown_table",
  "information_needs": ["markdown_table", "plain_text"],
  "decomposition_method": "numbered",
  "decomposition_confidence": "high",
  "coverage_status": "candidate",
  "candidate_evidence_refs": ["doc-id:chunk-id"]
}
```

`information_need` remains as a primary compatibility projection;
`information_needs` is authoritative for v2 analysis.

Constraints use this shape:

```json
{
  "constraint_id": "C1",
  "kind": "conditional_scope",
  "text": "若不能，必須按 claim scope 分開回答"
}
```

The analysis-level payload adds:

```json
{
  "behavior_influence": false,
  "support_assessment": "candidate_only",
  "response_constraints": [],
  "truncated": false
}
```

The summary adds `constraint_count`, `low_confidence_count`, and
`truncated_requirement_count`. `supported_count` remains exactly zero.

## Representation and Visual Semantics

Representation classification is multi-valued:

- `markdown_table`: tabular values already represented as Markdown;
- `text_structured`: equations, definitions, theorem conditions, or structured
  prose;
- `plain_text`: normal prose evidence;
- `visual_pattern`: spatial, geometric, color, curve, heatmap, or other
  image-dependent information.

Mentioning a Figure does not by itself require visual execution:

- If the Figure is only provenance and the retrieved image summary contains
  candidate text for the requested fact, the shadow decision is `optional`.
- If the question asks for spatial/pattern information not represented in an
  eligible summary, the shadow decision is `required`.
- A Markdown Table is structured text and does not require visual execution.
- Mixed Figure/Table questions retain both information needs.

These decisions remain diagnostic and cannot trigger or suppress the existing
visual runtime.

## Candidate Evidence Mapping

Candidate mapping remains deliberately permissive and never produces
`supported`:

- representation compatibility is evaluated first;
- generic lexical anchors and named entities may identify candidate chunks;
- duplicate documents are removed by stable evidence identity;
- when canonical document identity is unavailable, use a run-local content
  hash fallback instead of the ambiguous `unknown:chunk-N` form;
- no ground truth, expected source, or evaluation-only metadata is consumed.

`candidate` means "worth inspecting", not "answers this requirement".

## Error Handling

The existing runtime fail-soft boundary remains authoritative. Any v2 parser,
classification, or serialization exception produces an unavailable diagnostic
payload with `behavior_influence=false`. The answer, documents, tokens,
latency, response status, and all Agentic v9 stages remain unchanged.

## Verification Strategy

### Parser regression cases

- `tooth 1)`, `tooth 32)`, and `0.4` do not form numbered boundaries.
- A real `1. / 2. / 3.` sequence forms exactly three top-level blocks.
- Q5-like coordinated obligations split into flow, branches, operation, and
  accumulation requirements without recognizing any paper-specific name.
- Q7-like distributive questions create bounded per-entity eligibility
  requirements plus a global selection requirement.
- Conditional scope instructions become constraints.
- A/B classification labels become `allowed_labels` constraints.
- A Figure plus Markdown Table question preserves both representation needs.
- A new bilingual question not present in the evaluation set exercises the
  same generic rules.

### Runtime and export regressions

- Successful v9 execution persists `shadow_requirements_v2`.
- Analyzer failure cannot fail or downgrade a run.
- Redacted export includes v2 requirements and constraints.
- No LLM call uses a shadow phase.
- Provider-call count and total token accounting are unchanged by the
  diagnostic analyzer.

## Acceptance Criteria

1. All 16 evaluation questions produce a v2 payload without parser failure.
2. Q16-like text yields six valid obligations and never splits protected
   numeric spans.
3. Q5-like text yields four obligations.
4. Q7-like text yields no more than five requirements and retains all four
   explicit entities.
5. Q11-like and Q13-like response rules do not create false missing evidence.
6. Q15-like text retains Figure and Markdown Table needs and identifies four
   answer obligations.
7. Non-empty ambiguous questions fall back to one low-confidence requirement.
8. `supported_count` is always zero.
9. The analyzer performs no network, provider, embedding, retrieval, reranker,
   graph, or visual call.
10. Existing v9 answer, status, documents, token usage, and latency semantics
    remain unchanged.

## Rollout

Deploy v2 as shadow-only telemetry. Run the fixed 16-question smoke plus at
least several structurally different new questions. Compare decomposition,
constraints, representation decisions, and candidate mappings manually. Do
not connect the results to corrective retrieval or sufficiency until a later
design is separately approved.
