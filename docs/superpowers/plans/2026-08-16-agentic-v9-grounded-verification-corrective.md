# Agentic v9 Grounded Verification Corrective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Wave 3B so packed qualified premises can support verified obligation synthesis, semantic claims use one batched verifier, explicit numeric forms cannot bypass checks, and qualification cannot claim support while missing explicit anchors.

**Architecture:** Keep the existing compact final synthesis, `FinalAnswerRenderer`, `ClaimVerifier`, budget controller, evidence qualification, and terminal reducer. Replace the ambiguous deterministic boolean with a three-way gate, reserve one verifier call, and add a conservative generic anchor check before provider-selected evidence becomes curated.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, FastAPI backend modules, JSON prompt registry, Ruff.

## Global Constraints

- Work in `D:\flutterserver\pdftopng` on the main worktree.
- Use TDD: capture a behavior RED before changing production code.
- Create exactly one commit per task.
- Keep all provider calls bounded: final synthesis `<=1`, claim verifier `<=1`, no per-claim/per-slot calls.
- Do not pin or branch on Gemini/model names.
- Do not add question IDs, benchmark answers, document names, or fixture-specific values to production code.
- Q5 and Q23 are regression fixtures only.
- Fail closed: unavailable verifier claims remain unresolved; never accept them to preserve completeness.
- Do not change frontend/export schemas unless an actual serialized contract change is discovered; stop and report before expanding scope.

---

## File Structure

- `data_base/agentic_v9/claim_verifier.py`: owns numeric token normalization, deterministic claim disposition, and the single semantic verifier batch.
- `data_base/agentic_v9/final_answer.py`: converts final draft findings into claims, applies the gate, invokes the verifier once, and derives accepted claims/used evidence/unresolved state.
- `data_base/agentic_v9/budget_feasibility.py`: reserves the optional verifier capacity during post-contract admission.
- `evaluation/agentic_v9_campaign_runtime.py`: passes the explicit verifier reservation into every post-contract feasibility attempt.
- `prompts/agentic_rag_prompts.json`: owns model instructions for supported versus unresolved output and source-stated versus derived conclusions.
- `data_base/agentic_v9/slot_constraints.py`: owns generic structured-locator and numeric-anchor extraction/matching.
- `data_base/agentic_v9/evidence_extractor.py`: applies generic hard-anchor eligibility before accepting provider source/slot rows.
- `tests/test_agentic_v9_claim_verifier.py`: focused gate, numeric, and one-batch semantics.
- `tests/test_agentic_v9_final_answer.py`: final renderer routing, accepted claims, unresolved rows, and honest status.
- `tests/test_agentic_v9_budget_feasibility.py`: exact verifier reservation ledger.
- `tests/test_agentic_v9_campaign_runtime.py`: runtime admission and at-most-one verifier call.
- `tests/test_agentic_v9_slot_constraints.py`: generic anchor extraction and matching.
- `tests/test_agentic_v9_evidence_extractor.py`: provider-row anchor enforcement.
- `tests/test_agentic_rag_prompts.py`: prompt registry behavior contract.

---

### Task 1: Replace ambiguous claim checks with a deterministic gate and one semantic batch

**Files:**

- Modify: `data_base/agentic_v9/claim_verifier.py`
- Modify: `data_base/agentic_v9/final_answer.py`
- Modify: `prompts/agentic_rag_prompts.json`
- Create: `tests/test_agentic_v9_claim_verifier.py`
- Modify: `tests/test_agentic_v9_final_answer.py`
- Modify: `tests/test_agentic_rag_prompts.py`

**Interfaces:**

- Consumes: `FinalClaim`, `EvidencePacket`, `QueryContract`, `SynthesisObligation`, `is_qualified_evidence()`.
- Produces: `ClaimGateResult`, `gate_claim_deterministically()`, and `ClaimVerifier.verify(..., contract=contract)`.

- [ ] **Step 1: Add RED numeric-token tests**

Create focused tests for semantic numeric normalization:

```python
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("33x fewer parameters", {("33", "ratio")}),
        ("33× fewer parameters", {("33", "ratio")}),
        ("33-fold fewer parameters", {("33", "ratio")}),
        ("12.50% reduction", {("12.5", "percent")}),
        ("Table 1", {("1", "scalar")}),
    ],
)
def test_numeric_tokens_preserve_ratio_and_percent_semantics(text, expected):
    assert numeric_tokens(text) == expected
```

Add a negative direct-claim test proving `33x` cannot be supported by evidence containing only `34x`, and another proving `33x` and `33×` normalize to the same ratio token.

- [ ] **Step 2: Run numeric RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_claim_verifier.py -k "numeric or ratio or percent" -q
```

Expected: collection/import failure because `numeric_tokens` does not exist, or assertion failures against the old `_NUMBERS` behavior.

- [ ] **Step 3: Add RED three-way gate tests**

Lock these generic dispositions:

```python
def test_direct_verbatim_span_is_accepted_without_verifier():
    result = gate_claim_deterministically(direct_claim("reported score is 91%"), packets())
    assert result.status == "accepted"

def test_direct_paraphrase_with_valid_provenance_is_sent_to_verifier():
    result = gate_claim_deterministically(direct_claim("the decoder has two stages"), packets())
    assert result.status == "verify"

def test_obligation_with_complete_direct_premises_is_sent_to_verifier():
    result = gate_claim_deterministically(obligation_claim(), premise_packets())
    assert result.status == "verify"

def test_unknown_or_unqualified_evidence_is_rejected_before_verifier():
    result = gate_claim_deterministically(claim_with_unknown_evidence(), packets())
    assert result.status == "rejected"
```

The obligation fixture must use only qualified direct premises and no `calculated` packet. Assert it is `verify`, not `calculated_claim_lacks_calculated_evidence`.

- [ ] **Step 4: Add RED renderer batch tests**

Extend `tests/test_agentic_v9_final_answer.py` with one run containing:

- one direct verbatim claim;
- one direct paraphrase;
- one arithmetic obligation;
- one rounding/qualification obligation.

Use `_RecordingInvoker` responses for final synthesis plus one verifier response. Assert:

```python
assert [call["purpose"] for call in invoker.calls] == ["final_answer", "claim_verifier"]
assert result.claim_verifier_call_count == 1
assert accepted_ids == {"direct-verbatim", "direct-paraphrase", "arithmetic"}
assert rounding_claim.qualified_reason == "rounding_method_not_stated"
assert result.response_status == "qualified_partial"
```

Add a verifier-unavailable case and assert all pending claims are unresolved and do not contribute to `used_evidence_ids`.

- [ ] **Step 5: Add RED final-prompt tests**

Assert the registered `final_synthesis` prompt contains requirements equivalent to all of:

```python
required_phrases = (
    "Evidence insufficiency belongs in unresolved",
    "Do not infer a rounding method",
    "Distinguish source-stated facts from derived conclusions",
    "all direct premise evidence IDs",
)
```

Also assert the verifier system message requires independent verdicts, rejects unsupported rounding assumptions, and permits arithmetic only when derivable from cited premises.

- [ ] **Step 6: Run complete Task 1 RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_claim_verifier.py tests/test_agentic_v9_final_answer.py tests/test_agentic_rag_prompts.py -k "gate or paraphrase or obligation or arithmetic or rounding or numeric or unresolved" -q
```

Expected: failures caused by the absent gate, obligation verifier bypass, old numeric regex, and missing prompt requirements.

- [ ] **Step 7: Implement semantic numeric tokens**

In `claim_verifier.py`, replace `_NUMBERS` with a public testable helper backed by `Decimal` normalization:

```python
_NUMERIC_TOKEN = re.compile(
    r"(?<![\w.])(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))"
    r"\s*(?P<suffix>%|percent|x|×|-?fold)?(?![\w.])",
    re.IGNORECASE,
)

def numeric_tokens(text: str) -> set[tuple[str, str]]:
    tokens: set[tuple[str, str]] = set()
    for match in _NUMERIC_TOKEN.finditer(text):
        value = _normalize_decimal(match.group("value"))
        suffix = (match.group("suffix") or "").lower()
        kind = "percent" if suffix in {"%", "percent"} else "ratio" if suffix in {"x", "×", "fold", "-fold"} else "scalar"
        tokens.add((value, kind))
    return tokens
```

Do not use floating-point conversion.

- [ ] **Step 8: Implement the three-way deterministic gate**

Add the exact owner types:

```python
ClaimGateStatus = Literal["accepted", "verify", "rejected"]

class ClaimGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(min_length=1)
    status: ClaimGateStatus
    reason: str | None = None
```

Implement `gate_claim_deterministically()` in this order:

1. collect and deduplicate evidence/premise IDs;
2. reject missing, unknown, unqualified, or non-closed evidence;
3. for obligation claims, return `verify` after the hard checks without requiring a calculated packet;
4. for direct claims, reject explicit numeric tokens not present with matching semantic kinds in cited evidence;
5. accept only when normalized claim text is a verbatim substring of one cited packet;
6. return `verify` for every remaining structurally valid direct paraphrase.

Remove `requires_prose_verification()` from renderer control flow. Keep a compatibility wrapper only if another production caller exists; it must delegate to the new gate rather than preserve the old obligation bypass.

- [ ] **Step 9: Strengthen the single verifier batch**

Change the signature to:

```python
async def verify(
    self,
    claims: Sequence[FinalClaim],
    packets_by_id: Mapping[str, EvidencePacket],
    *,
    contract: QueryContract,
) -> dict[str, ClaimVerdict]:
```

Build each batch row with `claim`, `target_kind`, `target_description`, and its cited `evidence_packets`. The system instruction must require:

- one verdict per claim;
- support only from supplied evidence;
- arithmetic recomputation from premises;
- no inferred rounding method;
- direct paraphrase accepted only when entailed;
- missing/ambiguous support rejected.

Keep the existing strict response model and fail-closed missing-verdict behavior.

- [ ] **Step 10: Rewire `FinalAnswerRenderer`**

For every claim:

```python
gate = gate_claim_deterministically(claim, packets_by_id)
if gate.status == "accepted":
    accepted.append(claim)
elif gate.status == "verify":
    pending_verification.append(claim)
else:
    accepted.append(qualify_failed_claim(claim, gate_as_verdict(gate)))
```

Invoke `ClaimVerifier.verify()` exactly once after the loop. Only claims with positive verifier verdicts enter `supported_claims`; rejected candidates retain provenance with `qualified_reason` and therefore become unresolved through the existing reducer.

- [ ] **Step 11: Update final synthesis and verifier prompts**

Update only `final_synthesis.version` and its template. Preserve the four-key structured response and add the approved rules. Do not add a model-specific prompt variant.

- [ ] **Step 12: Run Task 1 GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_claim_verifier.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_rag_prompts.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/claim_verifier.py data_base/agentic_v9/final_answer.py tests/test_agentic_v9_claim_verifier.py tests/test_agentic_v9_final_answer.py tests/test_agentic_rag_prompts.py
git diff --check
```

Expected: all selected tests pass and Ruff/diff checks report no errors.

- [ ] **Step 13: Commit Task 1**

```powershell
git add data_base/agentic_v9/claim_verifier.py data_base/agentic_v9/final_answer.py prompts/agentic_rag_prompts.json tests/test_agentic_v9_claim_verifier.py tests/test_agentic_v9_final_answer.py tests/test_agentic_rag_prompts.py
git commit -m "fix(agentic-v9): verify grounded final claims in one batch"
```

---

### Task 2: Reserve and enforce the single claim-verifier call

**Files:**

- Modify: `data_base/agentic_v9/budget_feasibility.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_budget_feasibility.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**

- Consumes: existing `validate_post_contract_feasibility()` and `RunBudgetController` phase ledger.
- Produces: explicit `claim_verifier_provider_calls: int = 0` feasibility input, always passed as `1` by the active grounded-completion runtime.

- [ ] **Step 1: Add RED ledger tests**

Add tests proving:

```python
result = validate_post_contract_feasibility(
    contract=contract,
    setup_snapshot=_setup(),
    remaining_token_budget=budget,
    remaining_llm_calls=3,
    evidence_qualification_provider_calls=1,
    claim_verifier_provider_calls=1,
)
assert result.required_provider_calls == {
    "evidence_extract": 1,
    "final_answer": 1,
    "claim_verifier": 1,
}
```

Also assert values `-1`, `2`, and `True` return `CONFIGURATION_INCOMPATIBLE` with reason `invalid_claim_verifier_provider_calls`, and a two-call budget is rejected when qualification, final synthesis, and verifier capacity require three calls.

- [ ] **Step 2: Add RED runtime wiring test**

Patch `validate_post_contract_feasibility` and assert every planner-admitted, planner-fallback, and deterministic contract call supplies:

```python
assert kwargs["evidence_qualification_provider_calls"] == 1
assert kwargs["claim_verifier_provider_calls"] == 1
```

Add one end-to-end fixture with two pending claims and assert the observer records exactly one `purpose="claim_verifier"` call.

- [ ] **Step 3: Run Task 2 RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py -k "claim_verifier and (feasibility or budget or provider_calls or at_most_once)" -q
```

Expected: unexpected keyword failures and missing `claim_verifier` ledger entries.

- [ ] **Step 4: Implement feasibility admission**

Extend the signature:

```python
def validate_post_contract_feasibility(
    *,
    # existing keyword-only arguments
    evidence_qualification_provider_calls: int = 0,
    claim_verifier_provider_calls: int = 0,
) -> FeasibilityResult:
```

Validate `claim_verifier_provider_calls` using the same strict integer/boolean rules as evidence qualification. When it equals `1`, add `pending_provider_calls["claim_verifier"] = 1` before computing required calls and token reservation.

- [ ] **Step 5: Wire the active runtime**

Pass `claim_verifier_provider_calls=1` to all three post-contract feasibility attempts in `plan_contract()`. Do not make this depend on the selected model or a benchmark question. Runtime still spends zero verifier calls when Task 1 produces no pending claims.

- [ ] **Step 6: Run Task 2 GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_budget_controller.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_v9_campaign_runtime.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/budget_feasibility.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py
git diff --check
```

- [ ] **Step 7: Commit Task 2**

```powershell
git add data_base/agentic_v9/budget_feasibility.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): reserve one final claim verification"
```

---

### Task 3: Reject qualification rows that miss explicit generic anchors

**Files:**

- Modify: `data_base/agentic_v9/slot_constraints.py`
- Modify: `data_base/agentic_v9/evidence_extractor.py`
- Modify: `tests/test_agentic_v9_slot_constraints.py`
- Modify: `tests/test_agentic_v9_evidence_extractor.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**

- Consumes: original question, `RequiredSlot`, canonical candidate `EvidencePacket`, existing locator metadata and source-slot authorization.
- Produces: `SlotHardAnchors`, `derive_slot_hard_anchors()`, and `candidate_satisfies_hard_anchors()`.

- [ ] **Step 1: Add RED generic anchor tests**

Use generic fixtures, not paper/question IDs:

```python
def test_structured_question_locator_is_inherited_when_slot_has_none():
    anchors = derive_slot_hard_anchors(
        question="According to Algorithm 2, explain the update flow.",
        slot=RequiredSlot(slot_id="S1", description="Explain the final update step"),
    )
    assert anchors.locators == ("algorithm 2",)

def test_slot_local_region_and_ratio_anchors_are_preserved():
    anchors = derive_slot_hard_anchors(
        question="Compare the efficiency statements.",
        slot=RequiredSlot(
            slot_id="S1",
            description="Extract the Abstract statement reporting 33x and 13× reductions",
        ),
    )
    assert anchors.regions == ("abstract",)
    assert set(anchors.numeric_tokens) == {("33", "ratio"), ("13", "ratio")}
```

Add matches proving a `contribution` candidate with `34×/13×` cannot satisfy an `Abstract` slot requiring `33×/13×`, while exact locator/ratio evidence can.

- [ ] **Step 2: Add RED provider-row enforcement tests**

Build one provider response selecting the same authorized source for two slots. Assert the row for a matching slot survives and the row missing its explicit anchor is skipped without invalidating the sibling row. Preserve existing row-tolerant counts.

Add a missing-structured-locator fixture where topically related text lacks the requested algorithm/table locator. Assert no curated packet is produced and sufficiency remains unresolved.

- [ ] **Step 3: Add sanitized real-export regressions**

In `tests/test_agentic_v9_campaign_runtime.py`, add two compact fixtures derived from the real contracts but rename entities and omit question IDs:

1. five direct mechanism slots plus one narrative aggregation obligation; source candidates omit the requested algorithm locator and detailed branch/fusion evidence; assert S2-S5 cannot all become supported and terminal status is not `complete`;
2. exact table premises plus two obligations; verifier accepts arithmetic recomputation but rejects an unstated rounding-method claim; assert the accepted obligation and unresolved obligation are both preserved and status is `qualified_partial`.

Do not load the exported JSON at runtime and do not copy campaign/document IDs into tests.

- [ ] **Step 4: Run Task 3 RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_campaign_runtime.py -k "hard_anchor or structured_locator or ratio_anchor or missing_algorithm or rounding_method" -q
```

Expected: missing-interface failures and provider rows incorrectly surviving anchor mismatches.

- [ ] **Step 5: Implement the anchor model and extraction**

Add a small frozen dataclass or strict Pydantic model:

```python
class SlotHardAnchors(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    locators: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    numeric_tokens: tuple[tuple[str, str], ...] = ()
    identifiers: tuple[str, ...] = ()
```

Derive only:

- `Algorithm|Table|Figure|Fig.|Section` plus explicit identifier;
- `Abstract`, `Contribution`, or `Method` when present in the slot description;
- semantic numeric tokens from Task 1;
- mixed-case/acronym technical identifiers of length at least two.

Normalize case/whitespace and preserve deterministic order. Slot-local structured locator wins; inherit a question locator only when the slot has none. Do not extract ordinary prose words.

- [ ] **Step 6: Implement candidate matching**

Match anchors against a canonical searchable projection consisting of packet statement plus normalized locator/section metadata. Require every explicit locator/region and numeric token. Require technical identifiers only when at least one was derived; compare case-insensitively without fuzzy matching.

When no anchors exist, return `True` so ordinary provider qualification remains unchanged.

- [ ] **Step 7: Apply the check before row coalescing**

In the provider-row parse path, after source existence and source-slot authorization but before merging slot IDs, call:

```python
if not candidate_satisfies_hard_anchors(
    question=question,
    slot=slot_by_id[slot_id],
    packet=source_packet,
):
    continue
```

Keep valid sibling slot IDs for the same source. Do not alter backend-owned statement copying, qualification batching, or deterministic Markdown table extraction.

- [ ] **Step 8: Run Task 3 GREEN and Wave corrective regression**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_campaign_runtime.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/slot_constraints.py data_base/agentic_v9/evidence_extractor.py tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_campaign_runtime.py
git diff --check
```

- [ ] **Step 9: Commit Task 3**

```powershell
git add data_base/agentic_v9/slot_constraints.py data_base/agentic_v9/evidence_extractor.py tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): enforce explicit qualification anchors"
```

---

## Final Verification and Deployment Checkpoint

- [ ] Run the complete affected backend suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_requirement_decomposition.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_budget_controller.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_validator.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_claim_verifier.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_rag_prompts.py tests/test_agentic_v9_full_rollback.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9 evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_claim_verifier.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_rag_prompts.py
git diff --check
git status --short
```

- [ ] Confirm the commit range contains exactly the three planned commits and the tracked worktree is clean.
- [ ] Deploy the backend only; no frontend contract change is expected.
- [ ] Re-run the two-question checkpoint with the user-selected model.
- [ ] Export full observability and confirm:
  - claim verifier calls are `0` or `1`, never greater;
  - valid direct paraphrases are verifier candidates rather than exact-string rejections;
  - arithmetic obligation can be accepted from complete packed direct premises;
  - an unstated rounding method is unresolved/rejected;
  - missing explicit locator evidence cannot make all detailed slots sufficient;
  - `used_evidence_ids` contains only accepted-claim references;
  - `complete` is impossible while any slot or obligation remains unresolved.
- [ ] Stop for user checkpoint. Do not start routing, context replacement, or new model work.

## Self-Review Record

- Spec coverage: all seven design decisions map to Tasks 1-3 and final verification.
- Placeholder scan: no TBD/TODO/future implementation placeholders remain.
- Type consistency: `ClaimGateResult`, `gate_claim_deterministically`, `ClaimVerifier.verify(..., contract=...)`, `claim_verifier_provider_calls`, `SlotHardAnchors`, `derive_slot_hard_anchors`, and `candidate_satisfies_hard_anchors` have one owner and consistent names.
- Scope: backend-only; Q5/Q23 appear only as sanitized regression intent, never as production branches.
