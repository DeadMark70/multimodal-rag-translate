# Agentic v9 Evidence Completeness and Research Observability Design

**Date:** 2026-07-26  
**Status:** Approved design, pending implementation plan  
**Scope:** Agentic v9 runtime, evaluation observability/export, visual asset bridge, and Evaluation Center projections

## 1. Objective

Make Agentic v9 completeness claims trustworthy at the level of individual
question requirements, ensure partial answers are content-level fail-closed,
enable targeted missing-evidence repair, establish one real visual-evidence
success path, and make every runtime LLM phase reproducible in evaluation
exports.

The design addresses six confirmed gaps:

1. Route-generic slots are currently reported as if they were atomic question
   requirements.
2. Repair exists but receives generic slot descriptions and broad source scope.
3. `qualified_partial` is applied to status without constraining generated
   answer content.
4. Retrieved text chunks do not reliably bridge to persisted visual assets.
5. Ambiguous routing falls back without an actual, persisted route rationale.
6. Export redaction works only on recorded prompts, while Agentic v9 does not
   record per-phase prompts.

## 2. Non-goals

- Do not feed benchmark key points, reference answers, expected values, or gold
  atomic facts into runtime planning or retrieval.
- Do not change Naive or Agentic v8 behavior.
- Do not make visual evidence mandatory solely because a question contains the
  word “Figure” or “Table.”
- Do not reconstruct historical prompts that were never captured.
- Do not reinterpret legacy generic slots as atomic completeness.
- Do not allow observability failure to erase an otherwise usable answer.

## 3. Versioned Contract

Introduce `QueryContract` version `2`.

### 3.1 Route decision

```json
{
  "selected_route": "multi_document_exact",
  "decision_source": "deterministic",
  "matched_rules": [
    "numbered_subquestions",
    "multiple_named_sources"
  ],
  "candidate_routes": [
    "multi_document_exact",
    "exact_structured"
  ],
  "route_reason": "Three named sources and multiple exact locators were detected.",
  "planner_call_used": false,
  "fallback_reason": null,
  "confidence": 1.0
}
```

`decision_source` is one of:

- `deterministic`
- `llm_planner`
- `safe_fallback`

An evaluation routing row generated from this decision is an actual routing
decision and uses `analysis_type="actual"`. Retrospective policy analysis
remains a distinct record type.

### 3.2 Atomic required slot

```json
{
  "slot_id": "S1",
  "description": "Retrieve the tooth 1 to tooth 32 penalty value.",
  "required": true,
  "entity_ids": ["GEPAR3D", "tooth 1", "tooth 32"],
  "source_name_hints": ["GEPAR3D"],
  "authorized_source_doc_ids": ["resolved-doc-id"],
  "locator_hints": ["Appendix D", "Wasserstein matrix"],
  "expected_answer_type": "number",
  "depends_on_slot_ids": []
}
```

Slot descriptions state what must be found. They never contain the expected
answer unless that value already appears as a constraint in the user question.

Q16 decomposes into seven slots:

1. Tooth 1 to tooth 32 penalty value.
2. Reason the penalty is higher.
3. ODES regional impurity equation.
4. Meaning of `|A^c(x,y)|`.
5. U-KAN Dice at noise level 0.4.
6. Proposed method Dice at noise level 0.4.
7. Theorem 1 range for `m`.

### 3.3 Contract status and visual policy

`slot_plan_status` is one of:

- `complete`
- `degraded`

`visual_policy` is one of:

- `never`
- `preferred`
- `required`

A degraded slot plan can produce a useful answer but cannot produce
`response_status="complete"`.

### 3.4 Legacy compatibility

Version 1 contracts remain readable and are projected as:

```text
contract_version: 1
slot_semantics: legacy_generic
atomic_completeness: N/A
```

No migration invents atomic slots for historical runs.

## 4. Question-only Hybrid Contract Planner

Replace the route-only admission decision with a
`QuestionContractPlanner`.

### 4.1 Inputs

The planner may consume only:

- Original question.
- Authorized source names and canonical IDs.
- Evaluation Setup model and budget policy.

The planner must not consume:

- `key_points`
- `ground_truth`
- `ground_truth_short`
- gold `atomic_facts`
- expected evidence content containing correct answers

Gold facts remain available only after execution for scoring and attrition
analysis.

### 4.2 Deterministic stage

The deterministic stage extracts:

- Numbered or bulleted subquestions.
- Independent interrogative clauses.
- Parallel requested values.
- Named documents and technical entities.
- Figure, Table, Appendix, Formula, Equation, Theorem, page, and section
  locators.
- Expected answer shape such as number, equation, definition, comparison, or
  explanation.

It returns route candidates, matched rules, atomic slots, and a confidence
assessment.

### 4.3 Ambiguity stage

When deterministic decomposition cannot produce a confident contract, one
budgeted `contract_planning` LLM call returns both route and slots in a strict
JSON schema.

The call:

- Is limited to one provider attempt unless the global retry policy explicitly
  admits another provider attempt.
- Counts toward setup-authoritative LLM call and token budgets.
- Is recorded as phase `contract_planning`.
- Cannot emit expected answers or values.
- Cannot expand authorized source scope.

Invalid JSON, timeout, budget rejection, or schema violation results in a
deterministic safe fallback with:

```text
decision_source: safe_fallback
slot_plan_status: degraded
```

### 4.4 Slot bounds

- Slots are ordered and assigned stable IDs `S1` through `S8`.
- The maximum is eight required slots.
- Independent values remain independent slots.
- Items may be grouped only when they share the same source and locator and
  cannot be independently satisfied.

### 4.5 Preflight and runtime

Preflight does not invoke the planner model.

Preflight:

- Runs deterministic analysis.
- Detects whether one ambiguity call may be required.
- Reserves the worst-case legal planning and downstream provider-call budget.

Runtime:

- Invokes the ambiguity planner at most once.
- Persists the resulting contract.
- Runs post-contract feasibility against that exact contract.

This avoids duplicate preflight/runtime planning calls and nondeterministic
contract drift.

## 5. Slot-bound Evidence and Corrective Retrieval

Every retrieval task and evidence packet identifies its target slots.

### 5.1 Initial retrieval

The retrieval compiler may group slots with compatible source and locator
constraints, but the resulting evidence packets retain individual slot IDs.

Evidence from a document outside a slot’s authorized source IDs cannot satisfy
that slot.

### 5.2 Sufficiency

The existing slot statuses remain authoritative:

- `supported`
- `not_found`
- `conflicted`
- `explicitly_unavailable`

Required-slot completeness alone determines the response status:

- All required slots supported: `complete`
- At least one supported and at least one unresolved: `qualified_partial`
- No supported required slots: `insufficient`

### 5.3 Repair grouping

Corrective retrieval operates only on `not_found` required slots.

Missing slots are grouped by:

```text
authorized source group
+ locator type/identifier
+ compatible query terms
```

For Q16, representative groups are:

- ODES + formula: equation and variable-definition slots.
- Implicit U-KAN2.0 + Table 3: two Dice slots.
- Implicit U-KAN2.0 + Theorem 1: theorem-boundary slot.

Bounds:

- At most two retrieval tasks per repair round.
- At most two repair rounds.
- Recompute sufficiency after every round.
- Stop when no repairable slots remain.
- Stop when deadline, retrieval budget, or final-answer reserve is insufficient.
- Never expand authorized source scope.
- Never repair supported or explicitly unavailable slots.

Persist:

- Repair round.
- Grouped target slot IDs.
- Source and locator constraints.
- Generated query.
- Retrieved evidence IDs.
- Stop reason.

## 6. Content-level Fail-closed Final Synthesis

Final synthesis consumes:

- Query contract.
- Final sufficiency report.
- Slot resolutions.
- Packed evidence grouped by slot.
- Graph and visual capability outcomes attached to affected slots.

It returns structured JSON:

```json
{
  "supported_findings": [
    {
      "slot_id": "S1",
      "statement": "A supported statement.",
      "evidence_ids": ["evidence-1"]
    }
  ],
  "unresolved_requirements": [
    {
      "slot_id": "S7",
      "reason": "Theorem 1 was not found in the authorized evidence."
    }
  ]
}
```

Backend validation enforces:

- Every finding maps to a supported slot.
- Every evidence ID belongs to that slot.
- A missing, conflicted, or unavailable slot cannot appear in supported
  findings.
- The model cannot upgrade the sufficiency-derived response status.
- Required unresolved slots appear in `unresolved_requirements`.

Invalid JSON, unknown slots, evidence-boundary violations, or an unavailable
final provider call results in deterministic rendering of the supported and
unresolved slot sets.

The final user-facing text is rendered from the validated structure with
separate sections for supported conclusions and unverifiable requirements.
Words such as “推測” do not convert missing evidence into a supported claim.

Graph or visual capability failure updates the affected slot resolutions before
final synthesis. It is not applied only as a global status downgrade after the
answer has already been generated.

## 7. Visual Asset Pipeline

### 7.1 Persistent asset manifest

PDF ingestion persists a visual asset manifest independent of vector chunks:

```json
{
  "asset_id": "asset-id",
  "doc_id": "canonical-doc-id",
  "asset_type": "page",
  "pdf_page_index": 12,
  "printed_page_label": "10",
  "figure_id": "Figure 1(b)",
  "table_id": null,
  "formula_id": null,
  "bbox": [0.1, 0.2, 0.8, 0.7],
  "storage_reference": "authorized-storage-reference",
  "sha256": "content-hash",
  "width": 1600,
  "height": 2200
}
```

The database does not duplicate full page base64 into every vector chunk.

### 7.2 Asset resolution

Runtime uses:

```text
doc ID + page/figure/table/formula locator
→ AssetResolver
→ AssetLocator
→ authorized image loader
→ VisualEvidenceExtractor
```

Only selected assets are loaded from storage.

### 7.3 Policy behavior

- `never`: no visual stage.
- `preferred`: run visual only when corresponding slots remain unresolved
  after text retrieval.
- `required`: missing visual evidence marks corresponding slots explicitly
  unavailable and prevents a complete answer.

A textual Table or Figure reference does not by itself imply `required`.

### 7.4 Diagnostics

Persist:

- Manifest candidate count.
- Authorized candidate count.
- Locator-matched count.
- Successfully loaded count.
- Selected count.
- Dropped count and reasons.
- Evidence packet count.
- Covered slot IDs.

Specific terminal reasons include:

- `asset_manifest_empty`
- `source_not_authorized`
- `locator_not_matched`
- `asset_load_failed`
- `asset_exceeds_cap`
- `extractor_returned_no_evidence`

### 7.5 Backfill and positive control

Existing documents receive a bounded manifest backfill. Documents that cannot
be backfilled are marked `visual_assets_unavailable`.

At least one fixed positive-control test uses a known document, page, asset
manifest, locator, and expected slot-bound visual evidence packet.

## 8. Per-phase LLM Observability

`BudgetedLlmInvoker` receives an optional `LlmCallObserver`.

Each provider attempt persists:

- Run and campaign IDs.
- Phase and purpose.
- Provider attempt number.
- Budget reservation ID.
- Provider and model.
- Prompt hash and capture status.
- Optional safe preview.
- Optional full prompt.
- Response hash.
- Input, output, reasoning, other, and total tokens.
- Latency.
- Success, timeout, cancellation, or failure status.
- Safe error classification.

Required Agentic v9 phases:

- `contract_planning`
- `evidence_extract`
- `retrieval_judge`
- `visual_extract`
- `final_answer`

Retries create new rows and never overwrite prior attempts.

### 8.1 Capture policy

Evaluation Setup persists:

```json
{
  "capture_prompt_hashes": true,
  "capture_prompt_previews": true,
  "capture_full_prompts": false
}
```

Rules:

- Canonical prompt hashes are always captured.
- Previews are sanitized and length-bounded.
- Full prompts are captured only when enabled before execution.
- Credentials, authorization headers, cookies, and provider secrets are always
  removed.
- The capture policy is part of the campaign manifest.

### 8.2 Export

Export options reveal only data captured during execution.

If full prompts were requested at export but not captured, return:

```text
full_prompts_not_captured_at_execution
```

Export reports:

- Run count.
- LLM call count.
- Per-phase call counts.
- Prompt hash, preview, and full-prompt availability counts.
- Capture failures and reasons.

### 8.3 Token reconciliation

Every usage row links:

```text
run_id
→ llm_call_id
→ phase
→ reservation_id
→ provider_attempt
```

Phase attribution is complete only when:

```text
sum(per-phase official provider tokens)
= run official runtime tokens
```

Any mismatch remains `partial`; missing values are not replaced with zero.

## 9. Failure Semantics

- Degraded slot plan: final status is at most `qualified_partial`.
- Exhausted repair budget: preserve supported slots and expose unresolved ones.
- Preferred visual failure: complete is allowed only if text evidence supports
  all required slots.
- Required visual failure: affected slots become explicitly unavailable.
- Observability write failure: preserve the answer, mark observability partial,
  and block official release comparability.
- Full-prompt capture failure: retain safe metadata when possible and report
  `capture_failed`.

## 10. Evaluation Center

### 10.1 Agent Behavior

Display:

- Contract version and slot-plan status.
- Actual route decision and rationale.
- Atomic slot states.
- Initial and repair queries.
- Per-slot source and locator constraints.
- Graph and visual capability outcomes by slot.

### 10.2 Claim Evidence

Display:

- Claim to slot mapping.
- Slot to evidence mapping.
- Unresolved requirements.
- Final status downgrade reason.

### 10.3 Export Controls

Display:

- Execution-time capture policy.
- Per-phase recorded call counts.
- Hash, preview, and full-prompt availability.
- Actual counts in the completed export.
- Capture warnings.

Old version 1 runs remain N/A-safe.

## 11. Delivery Waves

### Wave 1: Content-level fail-closed final

- Structured final output.
- Sufficiency-derived status inheritance.
- Claim, slot, and evidence-boundary validation.
- Deterministic partial rendering.

### Wave 2: Atomic contract planner

- Contract version 2.
- Deterministic decomposition.
- Single ambiguity planner call.
- Route provenance.
- Preflight and budget integration.

### Wave 3: Corrective retrieval

- Per-slot source and locator scope.
- Grouped repair planning.
- Two-round bounded repair.

### Wave 4: Visual asset pipeline

- Asset manifest and resolver.
- Existing-document backfill.
- Visual policy.
- Positive-control path.

### Wave 5: Per-phase observability and export

- LLM call observer.
- Prompt capture policy.
- Phase token reconciliation.
- Export availability reporting.

### Wave 6: Evaluation Center

- Contract, slot, repair, route, visual, prompt, and compatibility UI.

### Wave 7: Verification

1. Unit and contract tests.
2. Q5, Q7, Q11, Q14, and Q16 Agentic v9 smoke.
3. Paired Naive/v9 smoke where comparison behavior is affected.
4. Sixteen-question single-repeat evaluation.
5. Multi-repeat formal benchmark only after all release gates pass.

## 12. Acceptance Criteria

- Q16 produces seven answer-free atomic slots.
- Runtime planner input contains no benchmark key points or gold values.
- Q14 does not output unsupported SegmentAnyBone inferences.
- Missing slots produce source- and locator-specific repair tasks.
- At least one visual positive-control run selects an asset and creates a
  provenance-bound visual evidence packet.
- Every new Agentic v9 run records an actual route decision.
- Every provider attempt has a phase-linked LLM call row.
- Prompt export accurately distinguishes not captured, captured, redacted, and
  capture failed.
- Per-phase token totals reconcile with official runtime tokens or remain
  explicitly partial.
- Version 1 runs never display fabricated atomic completeness.
- Official release metrics fail closed when contract, accounting, or required
  observability is incomplete.
