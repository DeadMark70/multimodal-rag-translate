# Agentic RAG v9 Grounded Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the working Wave 2 qualification pipeline while making qualification model-stable, separating source facts from derived synthesis, reconnecting the existing typed final renderer, and making `complete` depend on accepted claims and resolved synthesis obligations.

**Architecture:** Keep Query Contract v2, the deterministic route, authorized source scope, retrieval/reranking, repair controller, budget controller, `EvidenceExtractor`, `FinalAnswerRenderer`, and `ClaimVerifier`. Qualification returns identifiers only; backend-owned packets supply exact prose and provenance. Direct source facts remain `RequiredSlot`s, while comparison/calculation/rounding/cross-source conclusions become `SynthesisObligation`s. One compact structured final call produces hidden direct and synthesized findings, deterministic validation derives claims and used evidence, and one pure reducer owns terminal status.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, pytest, Ruff, LangChain structured output, React 18, TypeScript, Zod, Vitest, Chakra UI, generated OpenAPI artifacts.

## Global Constraints

- This plan implements the approved design in `docs/superpowers/specs/2026-08-16-agentic-v9-grounded-completion-design.md`.
- It preserves completed Wave 1 and Wave 2 work and supersedes only pending Wave 3 Tasks 11–14 in `docs/superpowers/plans/2026-08-15-agentic-v9-grounded-recovery.md`.
- Do not add a feature flag, compatibility execution branch, second evidence pipeline, or second final renderer.
- Do not change the deterministic route, retrieval engine, reranker, graph policy, repair limits, or user-selected provider/model.
- Keep provider cardinality bounded: at most one batched qualification call per admitted round, one final synthesis call, and one batched high-risk verifier call.
- Do not add per-slot or per-chunk LLM calls.
- Provider schemas guide generation; Pydantic and deterministic authorization remain the trust boundary.
- User-facing answers remain natural language. Provider JSON is internal only.
- Historical exports remain readable through existing optional/default conventions. New-profile fields must not silently turn missing data into zero.
- Use TDD for every behavior change: capture a focused RED failure before modifying production code, then run the stated GREEN gate.
- Each task produces exactly one scoped backend commit, except Task 4, which produces one backend commit and one frontend commit because the repositories are separate.
- Stop at each checkpoint. Do not begin the next checkpoint until the user has deployed and validated the stated real-system cases.

## Repository And Ownership Map

**Backend:** `D:\flutterserver\pdftopng`

- Qualification provider schema: `data_base/agentic_v9/provider_boundary.py`
- Qualification parsing and canonical packet copy: `data_base/agentic_v9/evidence_extractor.py`
- Atomic semantic decomposition: `data_base/agentic_v9/requirement_decomposition.py`
- Planned/deterministic contract assembly: `data_base/agentic_v9/contract_planner.py`
- Canonical v9 models: `data_base/agentic_v9/schemas.py`
- Compact final projection: new `data_base/agentic_v9/final_synthesis_context.py`
- Existing typed final path: `data_base/agentic_v9/final_answer.py`
- Claim validation/rendering: `data_base/agentic_v9/claim_verifier.py`, `data_base/agentic_v9/citation_renderer.py`
- Campaign adapter: `evaluation/agentic_v9_campaign_runtime.py`
- Export/API contract: `evaluation/campaign_schemas.py`, `evaluation/export_schemas.py`, `evaluation/export_service.py`
- Release/profile validation: `evaluation/retrieval_profiles.py`, `evaluation/smoke_verification.py`

**Frontend:** `D:\flutterserver\Multimodal_RAG_System`

- Canonical TypeScript contract: `src/types/evaluation.ts`
- Runtime export decoder: `src/services/evaluationExportSchema.ts`
- Interactive mapping: `src/pages/EvaluationCenter.mappers.ts`
- Final-claim/status UI: `src/components/evaluation/ClaimEvidenceTab.tsx`, `src/components/evaluation/AgenticV9Trace.tsx`

## Current Code Seams (Before Implementation)

- `data_base/agentic_v9/provider_boundary.py:69` owns the qualification response schema.
- `data_base/agentic_v9/evidence_extractor.py:501` owns row parsing and canonical packet validation.
- `data_base/agentic_v9/requirement_decomposition.py:230` owns deterministic decomposition; `_classify_block()` and synthesis extraction are in the lower half of that file.
- `data_base/agentic_v9/contract_planner.py:300` selects planned versus deterministic assembly; `_build_slots_from_decision()` begins near line 791 and deterministic builders near line 874.
- `data_base/agentic_v9/schemas.py:637` owns final draft/finding/claim/result models; `validate_active_atomic_contract()` begins near line 392.
- `data_base/agentic_v9/final_answer.py:42` owns the reusable renderer; direct claim construction begins near line 239 and current status reduction near line 350.
- `evaluation/agentic_v9_campaign_runtime.py:772` is the simplified free-text adapter that Task 3 removes.
- `evaluation/export_schemas.py:434` owns export-specific final claims; the full v9 export container begins near line 524.
- `src/types/evaluation.ts:945` and `src/types/evaluation.ts:1600` own interactive/export final-claim types.
- `src/services/evaluationExportSchema.ts:555` owns the strict runtime claim/metrics decoder.

---

# Wave 3A — Stable Evidence Semantics

## Task 1: Make evidence qualification identifier-only

**Repository:** `D:\flutterserver\pdftopng`

**Files:**

- Modify: `data_base/agentic_v9/provider_boundary.py`
- Modify: `data_base/agentic_v9/evidence_extractor.py`
- Modify: `prompts/agentic_rag_prompts.json`
- Modify: `tests/test_agentic_v9_provider_boundary.py`
- Modify: `tests/test_agentic_v9_evidence_extractor.py`
- Modify: `tests/test_agentic_rag_prompts.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

### Contract to implement

The provider may return only identifiers:

```json
{
  "packets": [
    {
      "source_evidence_id": "E1",
      "slot_ids": ["S1", "S2"]
    }
  ]
}
```

The backend must create the accepted packet from the canonical `EvidencePoolItem`; it must never trust provider-authored prose:

```python
accepted = _derived_packet(
    item.packet,
    evidence_id=f"curated:{actual_source_id}:{':'.join(deduped_slot_ids)}",
    slot_ids=deduped_slot_ids,
    statement=item.packet.statement,
    extractor_version="v9-id-qualification-1",
    prompt_version="2",
)
```

Keep `validate_prose_packet(accepted, source=item.packet, source_text=_source_text(item))` so exact source/provenance validation remains authoritative. Unknown aliases, empty/unknown slots, unauthorized source-slot pairs, unexpected row keys, and malformed rows are skipped individually. Valid sibling rows survive.

- [ ] **Step 1: Add focused RED provider-schema tests**

In `tests/test_agentic_v9_provider_boundary.py`, assert:

- response row required keys are exactly `source_evidence_id` and `slot_ids`;
- `statement` is absent from provider schema;
- the campaign factory still binds the shared qualification provider for purpose `evidence_extraction`;
- the user-selected synthesizer model remains the provider source.

- [ ] **Step 2: Add focused RED parser tests**

In `tests/test_agentic_v9_evidence_extractor.py`, add behavior tests proving:

1. accepted identifier-only row copies the exact canonical packet statement, source, locator, scope, numeric metadata, and premise IDs;
2. two returned slot IDs are de-duplicated in first-seen order;
3. one unknown alias does not discard a valid sibling row;
4. one unauthorized source-slot pair does not discard a valid sibling row;
5. a legacy/provider-injected `statement` key is rejected and cannot rewrite evidence;
6. new identifier-only acceptance never increments `statement_not_verbatim`;
7. provider failure remains batch-fail-closed and does not promote raw candidates.

- [ ] **Step 3: Update prompt-format RED coverage**

Update `tests/test_agentic_rag_prompts.py` so the `evidence_extract` prompt requires selection of source and slot IDs, explicitly forbids returning/copying/rephrasing source text, and still formats `question`, `unresolved_slots`, and `source_evidence`.

- [ ] **Step 4: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_rag_prompts.py tests/test_agentic_v9_campaign_runtime.py -k "identifier_only or canonical_statement or mixed_rows or evidence_qualification_schema" -q
```

Expected: failures show the current schema requires `statement` and current parser trusts provider text.

- [ ] **Step 5: Implement the minimal production change**

- Remove `statement` from `evidence_qualification_response_schema()`.
- Change `_parse_curated_packets()` to accept exact two-key rows.
- Resolve aliases and authorize slot bindings before packet creation.
- Copy `item.packet.statement` and all other canonical fields through `model_copy`/`_derived_packet`.
- Keep the existing row-tolerant loop and existing rejection counters.
- Update `evidence_extract` prompt version and wording; do not change any other prompt.

- [ ] **Step 6: Run GREEN and impacted regression**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_validator.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_rag_prompts.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/provider_boundary.py data_base/agentic_v9/evidence_extractor.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_rag_prompts.py
git diff --check
```

- [ ] **Step 7: Commit Task 1**

```powershell
git add data_base/agentic_v9/provider_boundary.py data_base/agentic_v9/evidence_extractor.py prompts/agentic_rag_prompts.json tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_rag_prompts.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): qualify evidence by identifier"
```

## Task 2: Separate direct evidence slots from synthesis obligations

**Repository:** `D:\flutterserver\pdftopng`

**Files:**

- Modify: `data_base/agentic_v9/requirement_decomposition.py`
- Modify: `data_base/agentic_v9/contract_planner.py`
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `prompts/agentic_v9_contract_planner.json`
- Modify: `tests/test_agentic_v9_requirement_decomposition.py`
- Modify: `tests/test_agentic_v9_contract_planner.py`
- Modify: `tests/test_agentic_v9_schemas.py`
- Reuse: `tests/fixtures/agentic_v9_atomic_questions_v1.json`

### Semantic rule

Use one bounded role classifier for both deterministic output validation and provider-planned output validation:

```python
RequirementRole = Literal["direct", "synthesis"]

def classify_requirement_role(text: str) -> RequirementRole:
    """Classify whether source text can directly satisfy the requirement."""

def validate_requirement_roles(
    *, required_slots: Sequence[RequiredSlot],
    synthesis_obligations: Sequence[SynthesisObligation],
) -> None:
    """Reject derived-only slots and obligations without valid direct dependencies."""
```

The classifier is deliberately narrow. It recognizes explicit derived operations already represented by `SynthesisObligationKind`: comparison/ranking/selection, arithmetic/ratio/percentage/recomputation, rounding interpretation/confirmation, causal conclusion, and whole-flow reconstruction. It does not infer a new route or retrieve anything.

For deterministic decomposition, `_classify_block()` must remove a derived clause from the direct requirement text and emit a `SynthesisObligationDraft` whose dependencies name the direct requirements produced from that same block. For provider-planned contracts, a derived-only `evidence_requirement` is a semantic rejection; use the already-computed deterministic decomposition as the degraded atomic contract rather than accepting it as a slot or collapsing immediately to the whole-question S1 fallback.

### Required Q5/Q23 shapes

- Q5: direct slots for the original branch, flip dimensions/branches, shared SiamSSM transformation, flip-back accumulation, and final quarter averaging; one aggregation obligation reconstructs the overall CSS flow and depends on those direct slots. No duplicate parent slot.
- Q23: direct slots for Table 1 values and the source statements in the Abstract/contribution section; ratio recomputation is an aggregation obligation; rounding confirmation is a qualification obligation. Table 1 values alone cannot satisfy the rounding obligation.

- [ ] **Step 1: Add RED decomposition tests**

Add exact semantic assertions for Q5 and Q23, plus one simple single-fact control that remains one direct slot and zero obligations. Assert sequential `S1..Sn`, sequential `O1..On`, and exact dependency IDs.

- [ ] **Step 2: Add RED planned-contract tests**

Cover both provider outcomes:

1. a correctly separated planner decision remains `slot_plan_source="llm_planner"`;
2. a planner decision that places ratio/rounding/reconstruction in `evidence_requirements` is not activated; the deterministic atomic decomposition is used with the existing degraded/fallback diagnostics and the original question remains represented.

Do not add another planner call or retry.

- [ ] **Step 3: Add RED schema invariants**

Extend `validate_active_atomic_contract()` coverage so every obligation ID is sequential `O1..On`, every obligation has at least one valid `depends_on_slot_ids`, and provider/deterministic assembly both call the same role validator before activation.

- [ ] **Step 4: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_requirement_decomposition.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_schemas.py -k "q5 or q23 or synthesis_role or derived_slot" -q
```

- [ ] **Step 5: Implement deterministic split and common validation**

- Add the small pure role classifier to `requirement_decomposition.py`.
- Teach `_classify_block()` to produce direct requirements plus derived obligations for the explicit patterns above.
- Preserve the current eight-slot/eight-obligation bounds and stable input order.
- In `contract_planner.py`, validate both `_build_slots_from_decision()` and `_build_slots_from_decomposition()` outputs with the same semantic rule.
- On a semantically invalid planned split, build the contract from the already prepared deterministic decomposition; do not call the provider again.
- Update the planner prompt to define direct source facts versus derived synthesis and keep its current JSON shape.
- Do not add a classifier model or modify route planning.

- [ ] **Step 6: Run GREEN and retrieval regressions**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_requirement_decomposition.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_schemas.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/requirement_decomposition.py data_base/agentic_v9/contract_planner.py data_base/agentic_v9/schemas.py tests/test_agentic_v9_requirement_decomposition.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_schemas.py
git diff --check
```

- [ ] **Step 7: Commit Task 2**

```powershell
git add data_base/agentic_v9/requirement_decomposition.py data_base/agentic_v9/contract_planner.py data_base/agentic_v9/schemas.py prompts/agentic_v9_contract_planner.json tests/test_agentic_v9_requirement_decomposition.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_schemas.py
git commit -m "fix(agentic-v9): separate evidence from synthesis"
```

## Wave 3A Checkpoint — Stop For Focused Deployment

- [ ] Deploy backend through Task 2 only.
- [ ] Use the user-selected Gemini 2.5 Flash-Lite configuration; do not substitute a model in code.
- [ ] Run Q5 and Q23 Agentic only and export all-run observability.
- [ ] Confirm new qualification responses cannot alter evidence statements.
- [ ] Confirm `qualification_statement_not_verbatim_count == 0` for the new identifier-only profile.
- [ ] Confirm Q5/Q23 contracts expose direct slots separately from obligations.
- [ ] Confirm provider-call counts remain within existing Wave 2 bounds.
- [ ] Stop and let the user decide whether to proceed to Wave 3B.

---

# Wave 3B — Grounded Finalization

## Task 3: Activate compact structured final synthesis and exact claims

**Repository:** `D:\flutterserver\pdftopng`

**Files:**

- Create: `data_base/agentic_v9/final_synthesis_context.py`
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `data_base/agentic_v9/provider_boundary.py`
- Modify: `data_base/agentic_v9/final_answer.py`
- Modify: `data_base/agentic_v9/claim_verifier.py`
- Modify: `data_base/agentic_v9/citation_renderer.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `prompts/agentic_rag_prompts.json`
- Create: `tests/test_agentic_v9_final_synthesis_context.py`
- Modify: `tests/test_agentic_v9_final_answer.py`
- Modify: `tests/test_agentic_v9_provider_boundary.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_rag_prompts.py`

### Exact provider and canonical models

Add export-independent canonical models:

```python
class SynthesizedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    obligation_id: str = Field(pattern=r"^O[1-8]$")
    statement: str = Field(min_length=1)
    premise_evidence_ids: list[str] = Field(min_length=1)

class UnresolvedObligation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    obligation_id: str = Field(pattern=r"^O[1-8]$")
    reason: str = Field(min_length=1)

class FinalAnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supported_findings: list[SupportedFinding] = Field(default_factory=list)
    synthesized_findings: list[SynthesizedFinding] = Field(default_factory=list)
    unresolved_requirements: list[UnresolvedRequirement] = Field(default_factory=list)
    unresolved_obligations: list[UnresolvedObligation] = Field(default_factory=list)
```

Extend `FinalClaim` with `obligation_id: str | None = None` and a model validator requiring exactly one of `slot_id` or `obligation_id`.

Derive synthesized support type in backend code, not provider output:

```python
_OBLIGATION_SUPPORT_TYPE = {
    "aggregation": "calculated",
    "comparison": "comparative_inference",
    "selection": "comparative_inference",
    "causal": "comparative_inference",
    "qualification": "qualified",
}
```

### Compact synthesis input

Create fixed models `FinalSynthesisSlot`, `FinalSynthesisEvidence`, and `FinalSynthesisContext`. The context contains only:

- original question;
- direct slot IDs, descriptions, and expected answer types;
- slot resolutions;
- synthesis obligations and response constraints;
- pre-generation unresolved direct requirements;
- exact packed qualified evidence with source document, locator, support type, and premise IDs;
- arbitration results.

Exclude route budgets, graph/visual policies, repeated authorized-document lists, rank diagnostics, trace payloads, provider payloads, and unselected candidates.

- [ ] **Step 1: Add RED compact-projection tests**

Assert exact serialized keys, contract-order slot ordering, pack-order evidence ordering, unique evidence IDs, qualified-only evidence, and rejection of unknown slot/evidence references. Assert the projection is materially smaller than serializing the full contract plus packet objects; use a deterministic fixture threshold, not production telemetry.

- [ ] **Step 2: Add RED draft/claim tests**

Cover:

1. direct finding creates a slot-bound claim;
2. synthesized finding creates an obligation-bound claim;
3. every obligation dependency must contribute at least one packed premise evidence ID;
4. unknown obligation, unknown evidence, missing premise closure, or direct finding with derived support type is rejected;
5. accepted claims alone determine ordered `used_evidence_ids`;
6. packed but unreferenced evidence is not used;
7. all high-risk claims are passed in one verifier batch; low-risk direct claims skip it;
8. a Q18-style rationale is rejected when evidence mentions the same entities but does not support the stated reason;
9. complete answers remain ordinary natural language without JSON or forced status headings, while partial answers add only a concise unresolved section.

- [ ] **Step 3: Add RED provider-boundary tests**

Add `final_synthesis_response_schema()` and `build_final_synthesis_provider()`. Bind it only for purpose `final_answer`. Reuse the existing Gemini-compatible schema projection before binding and keep full Pydantic validation after response decoding. Assert the setup-selected synthesizer remains authoritative.

- [ ] **Step 4: Add RED campaign integration tests**

Prove the campaign runtime:

- calls `FinalAnswerRenderer.render()` with contract, packed evidence, resolutions, sufficiency, and arbitration;
- no longer sends the free-text `Question: ... Evidence: ...` adapter;
- no longer wraps the whole answer as one direct claim;
- no longer marks all packed IDs used;
- produces natural-language output, never provider JSON;
- makes exactly one final synthesis provider call.

- [ ] **Step 5: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py -k "compact or synthesized or obligation or exact_claim or structured_final" -q
```

- [ ] **Step 6: Implement the compact projector and typed provider path**

- Implement `build_final_synthesis_context()` as a pure function.
- Add the final-synthesis prompt entry; keep it concise and require exactly the four draft collections.
- Bind structured output at `_provider_for_purpose("final_answer")`.
- Reuse `FinalAnswerRenderer`; do not add a campaign-specific renderer.
- Extend deterministic claim construction for obligations and premise closure.
- Reuse `ClaimVerifier` once for the batch of high-risk accepted candidates.
- Render user prose and citations through the existing citation renderer.
- On malformed response/provider failure, return a claim-free terminal result without regeneration.

- [ ] **Step 7: Run GREEN and full final-path regression**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_rag_prompts.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/final_synthesis_context.py data_base/agentic_v9/schemas.py data_base/agentic_v9/provider_boundary.py data_base/agentic_v9/final_answer.py data_base/agentic_v9/claim_verifier.py data_base/agentic_v9/citation_renderer.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py
git diff --check
```

- [ ] **Step 8: Commit Task 3**

```powershell
git add data_base/agentic_v9/final_synthesis_context.py data_base/agentic_v9/schemas.py data_base/agentic_v9/provider_boundary.py data_base/agentic_v9/final_answer.py data_base/agentic_v9/claim_verifier.py data_base/agentic_v9/citation_renderer.py evaluation/agentic_v9_campaign_runtime.py prompts/agentic_rag_prompts.json tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_rag_prompts.py
git commit -m "feat(agentic-v9): activate grounded final synthesis"
```

## Task 4: Enforce terminal status and publish the strict observability contract

### Task 4A — Backend

**Repository:** `D:\flutterserver\pdftopng`

**Files:**

- Modify: `data_base/agentic_v9/final_answer.py`
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `data_base/agentic_v9/execution_core.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `evaluation/campaign_schemas.py`
- Modify: `evaluation/export_schemas.py`
- Modify: `evaluation/export_service.py`
- Modify: `evaluation/research_analytics.py`
- Modify: `evaluation/retrieval_profiles.py`
- Modify: `evaluation/smoke_verification.py`
- Modify: `tests/test_agentic_v9_final_answer.py`
- Modify: `tests/test_agentic_v9_execution_core.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_smoke_runner.py`
- Modify: `tests/test_evaluation_retrieval_profiles.py`
- Modify: `tests/test_evaluation_export_v2_schemas.py`
- Modify: `tests/test_evaluation_export_redaction.py`
- Modify: `docs/BACKEND.md`
- Modify: `docs/agentic-v9-smoke-verification.md`
- Generate: `openapi.json`
- Generate: `contracts/openapi-contract.json`

### Pure reducer

Implement one pure function and make both `FinalAnswerRenderer` and campaign terminal projection use it:

```python
def reduce_terminal_status(
    *,
    contract: QueryContract,
    slot_resolutions: Sequence[SlotResolution],
    accepted_claims: Sequence[FinalClaim],
    unresolved_requirements: Sequence[UnresolvedRequirement],
    unresolved_obligations: Sequence[UnresolvedObligation],
) -> ResponseStatus:
    """Return complete, qualified_partial, or insufficient from verified output."""
```

Rules:

- `insufficient`: no accepted claim remains.
- `qualified_partial`: at least one renderable accepted claim remains, but any required direct slot lacks a supported resolution/direct claim, any obligation lacks an accepted synthesized claim, or any unresolved row remains. A rejected/qualified-reason candidate never satisfies coverage.
- `complete`: every direct slot has supported qualified evidence and an accepted slot claim; every obligation has an accepted obligation claim; all premise references are packed/qualified and closed; no unresolved rows remain.

`execution_core._prevent_response_status_upgrade()` may continue to cap a result against direct sufficiency, but it must not create `complete`; only the reducer can return `complete`.

### Typed observability additions

Add only missing data:

```python
class FinalAnswerResult(BaseModel):
    # existing fields...
    unresolved_requirements: list[UnresolvedRequirement] = Field(default_factory=list)
    unresolved_obligations: list[UnresolvedObligation] = Field(default_factory=list)
    claim_verifier_call_count: int = Field(default=0, ge=0, le=1)

class V9ExecutionMetrics(BaseModel):
    # existing fields...
    used_evidence_count: int | None = Field(default=None, ge=0)
    unresolved_requirement_count: int | None = Field(default=None, ge=0)
    claim_verifier_call_count: int | None = Field(default=None, ge=0, le=1)
```

`unresolved_requirement_count` is the combined number of unresolved direct slots and unresolved obligations. Final claims expose the distinction through XOR `slot_id`/`obligation_id`; do not add duplicate direct/synthesis claim counters.

- [ ] **Step 1: Add RED reducer matrix tests**

Cover all transitions independently:

1. packet-to-slot coverage without accepted claims is `insufficient`;
2. one valid direct claim plus one missing slot is `qualified_partial`;
3. all slots claimed but one obligation unresolved is `qualified_partial`;
4. all slots and obligations claimed with closed qualified premises is `complete`;
5. rejected/qualified-reason claims do not satisfy completion;
6. final provider failure with no claims is `insufficient`.

- [ ] **Step 2: Add RED persistence/export/smoke tests**

Assert:

- final claim serializes exactly one target (`slot_id` XOR `obligation_id`);
- unresolved direct and obligation rows survive runtime, canonical projection, export v2, and redaction;
- used/unresolved/verifier metrics are derived from typed final output;
- historical rows with missing new fields remain readable and appear unavailable, never fabricated zero in research analytics;
- smoke rejects `complete` when any slot/obligation/claim invariant fails;
- smoke accepts an exact complete fixture;
- profile becomes `finalpack_r1_active_atomic_contract_v3_grounded_completion_v1`.

- [ ] **Step 3: Run backend RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_retrieval_profiles.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py -k "terminal_status or unresolved_obligation or obligation_id or grounded_completion" -q
```

- [ ] **Step 4: Implement reducer and projections**

- Replace `_response_status()` with `reduce_terminal_status()`.
- Store typed unresolved rows and verifier call count in `FinalAnswerResult`.
- Derive the three new execution metrics in campaign runtime from that result.
- Extend canonical/export claim models with `obligation_id` and enforce XOR.
- Carry unresolved arrays through interactive and export v2 projections.
- Preserve redaction of claim statements while retaining IDs/status/reasons.
- Update smoke/profile/docs; do not add release score gates.

- [ ] **Step 5: Generate artifacts and run backend GREEN**

```powershell
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --write
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --check
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_requirement_decomposition.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_validator.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_retrieval_profiles.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9 evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_schemas.py evaluation/export_schemas.py evaluation/export_service.py evaluation/research_analytics.py evaluation/retrieval_profiles.py evaluation/smoke_verification.py
git diff --check
```

- [ ] **Step 6: Commit Task 4A**

```powershell
git add data_base/agentic_v9/final_answer.py data_base/agentic_v9/schemas.py data_base/agentic_v9/execution_core.py evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_schemas.py evaluation/export_schemas.py evaluation/export_service.py evaluation/research_analytics.py evaluation/retrieval_profiles.py evaluation/smoke_verification.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_retrieval_profiles.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py docs/BACKEND.md docs/agentic-v9-smoke-verification.md openapi.json contracts/openapi-contract.json
git commit -m "feat(agentic-v9): enforce grounded completion status"
```

### Task 4B — Frontend strict contract sync

**Repository:** `D:\flutterserver\Multimodal_RAG_System`

**Files:**

- Modify: `src/types/evaluation.ts`
- Modify: `src/services/evaluationExportSchema.ts`
- Modify: `src/services/evaluationExportSchema.test.ts`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `src/pages/EvaluationCenter.mappers.ts`
- Modify: `src/pages/EvaluationCenter.mappers.test.ts`
- Modify: `src/components/evaluation/ClaimEvidenceTab.tsx`
- Modify: `src/components/evaluation/ClaimEvidenceTab.test.tsx`
- Modify: `src/components/evaluation/AgenticV9Trace.tsx`
- Modify: `src/components/evaluation/RunTraceTab.test.tsx`
- Modify: `src/components/evaluation/AblationDashboardTab.test.tsx`
- Generate: `src/test/fixtures/agenticV9ApiContract.ts`

- [ ] **Step 7: Pin backend contract and capture frontend RED**

```powershell
cd D:\flutterserver\Multimodal_RAG_System
npm run contract:pin
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/RunTraceTab.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx
```

The strict fixture must contain full observability, non-null comparison, one direct claim, one obligation claim, one unresolved obligation, and populated new metrics. Add negative decoder tests for both-target and no-target claims, unknown status shapes, and missing required new-profile fields. Historical interactive fields may remain optional; export v2 new-profile fields are strict.

- [ ] **Step 8: Synchronize types, decoder, mapper, and minimal UI**

- Model claim target as a discriminated XOR union: slot target or obligation target.
- Decode unresolved direct/obligation rows and the three metrics.
- Preserve `obligation_id` in `EvaluationCenter.mappers.ts`.
- In Claim Evidence, label each row `Slot S#` or `Obligation O#`.
- In Agentic v9 trace, show combined unresolved count and verifier calls; do not duplicate claim totals already visible elsewhere.
- Keep export rejection sanitized and prevent download on malformed responses.

- [ ] **Step 9: Run frontend GREEN**

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/RunTraceTab.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx
npm run contract:check
npm run test:scripts
npm run lint:ci
npm run build
git diff --check
```

- [ ] **Step 10: Commit Task 4B**

```powershell
git add src/types/evaluation.ts src/services/evaluationExportSchema.ts src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/pages/EvaluationCenter.mappers.ts src/pages/EvaluationCenter.mappers.test.ts src/components/evaluation/ClaimEvidenceTab.tsx src/components/evaluation/ClaimEvidenceTab.test.tsx src/components/evaluation/AgenticV9Trace.tsx src/components/evaluation/RunTraceTab.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx src/test/fixtures/agenticV9ApiContract.ts
git commit -m "fix(evaluation-ui): decode grounded completion"
```

## Wave 3B Checkpoint — Stop For Deployment Study

- [ ] Confirm backend and frontend tracked worktrees are clean.
- [ ] Confirm backend OpenAPI artifacts are current and frontend `contract:check` pins the final backend hash.
- [ ] Deploy both repositories.
- [ ] Run Q5 and Q23 with Gemini 2.5 Flash-Lite first.
- [ ] Export summary and all-run observability.
- [ ] Confirm exact evidence text is backend-owned and used IDs equal accepted-claim references only.
- [ ] Confirm Q5 cannot be `complete` without original branch and final averaging evidence plus its reconstruction obligation.
- [ ] Confirm Q23 cannot be `complete` without direct Table/Abstract/contribution claims plus ratio and rounding obligations.
- [ ] Confirm natural-language answers contain no hidden JSON.
- [ ] Confirm provider calls remain bounded: qualification per admitted round, final `<=1`, verifier `<=1`.
- [ ] If Q5/Q23 semantics are correct, run paired Q1–Q32 Agentic/Naive and compare correctness, faithfulness, relevancy, status distribution, runtime tokens, and latency.
- [ ] Request one consolidated review after implementation and local verification, then correct only validated findings in separate commits.
- [ ] Do not add router/classifier/context-replacement work until this checkpoint identifies a remaining bottleneck.

## Rollback Boundary

- Wave 3A rollback range is exactly Task 1 plus Task 2 backend commits.
- Wave 3B backend rollback range is exactly Task 3 plus Task 4A commits.
- Wave 3B frontend rollback is exactly Task 4B.
- Roll back a whole failed checkpoint range. Do not restore the old free-text final adapter inside the new profile, mix profile fields, or add a runtime switch.

## Plan Completion Checklist

- [ ] Every approved design requirement maps to a task and focused behavior test.
- [ ] No task introduces model pinning, new routing, unbounded calls, or a parallel pipeline.
- [ ] Every production interface has a named owner and strict type.
- [ ] Every task has a RED command, GREEN command, and exact commit boundary.
- [ ] Both deployment checkpoints stop before subsequent work.
- [ ] Old pending Wave 3 Tasks 11–14 are explicitly marked superseded.
