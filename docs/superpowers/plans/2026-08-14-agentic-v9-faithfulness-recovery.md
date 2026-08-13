# Agentic RAG v9 Faithfulness Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Evaluation Agentic v9 qualify evidence before sufficiency and generate slot-complete, evidence-bound final answers through the existing shared v9 components.

**Architecture:** Keep retrieval, route budgets, and persistence contracts intact. Strengthen the existing shared `EvidenceExtractor` and `FinalAnswerRenderer`, move qualification ahead of each sufficiency decision in `V9ExecutionCore`, then atomically replace the campaign adapter's no-op curator and synthetic final claim with those shared components.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, Ruff, existing Agentic v9 `RunBudgetController` and `BudgetedLlmInvoker`.

## Global Constraints

- Execute inline in the main worktree with `superpowers:executing-plans`; do not dispatch subagents.
- Use TDD for every behavior change: capture RED, implement the minimum GREEN change, run the task gate, self-review, then commit once.
- Create exactly one production commit per task. Documentation-only plan/spec commits remain separate.
- Stop after each Wave checkpoint so the user can push and verify the real system.
- Do not change Native, Advanced, Graph, v8, or the current general `/rag/agentic/stream` behavior.
- Do not change the model, embedding model, retriever, reranker, dataset, RAGAS evaluator, route `max_llm_calls`, runtime-token budgets, or Setup ceilings.
- Do not add an automatic Agentic/Naive token-ratio gate. The existing `<= 3.0` ratio remains a manual campaign acceptance measure.
- Keep final generation at zero or one provider call. No-evidence runs make zero final calls.
- Never admit a raw retrieved chunk as positive evidence solely because it has a `slot_id`.
- Never use expected answers, key points, golden evidence, or evaluation-only ground truth at runtime.
- Preserve the final-answer reserve. Budget or deadline exhaustion produces `qualified_partial` or `insufficient` rather than promoting unqualified evidence.
- Preserve current OpenAPI and frontend transport shapes; this plan changes internal v9 semantics and values, not response schemas.

## File Structure

No new production module is needed.

- `data_base/agentic_v9/schemas.py`: shared structured final-draft fields and truthful qualification-round metric bound.
- `data_base/agentic_v9/final_answer.py`: strict provider draft parsing, slot/evidence validation, status derivation, and deterministic partial construction.
- `data_base/agentic_v9/citation_renderer.py`: natural complete rendering and explicit confirmed/unresolved partial rendering.
- `data_base/agentic_v9/evidence_extractor.py`: retain already-qualified packets and qualify untrusted candidates fail-closed.
- `data_base/agentic_v9/execution_core.py`: run qualification before initial and post-repair sufficiency.
- `data_base/agentic_v9/phase_policy.py`: permit bounded evidence qualification across initial plus contract-bounded repair rounds without increasing the route's total call budget.
- `evaluation/agentic_v9_campaign_runtime.py`: mark raw candidates unqualified, wire the extractor and final renderer, propagate qualified evidence quality, and persist only qualified evidence as positive evidence.
- Existing focused test modules own all new regression coverage; no duplicate end-to-end harness is created.

## Wave 1: Correct Shared v9 Semantics

### Task 1: Enforce Structured Slot-Bound Final Synthesis

**Files:**
- Modify: `data_base/agentic_v9/schemas.py:477-503`
- Modify: `data_base/agentic_v9/final_answer.py:1-305`
- Modify: `data_base/agentic_v9/citation_renderer.py:51-79`
- Modify: `tests/test_agentic_v9_schemas.py:370-410`
- Modify: `tests/test_agentic_v9_final_answer.py:1-327`

**Interfaces:**
- Consumes:
  - `QueryContract.required_slots`
  - packed `EvidencePacket`s
  - authoritative `SlotResolution`s
  - existing `LlmInvoker`, `ClaimVerifier`, and deterministic claim verifier
- Produces:

```python
class SupportedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    support_type: ClaimSupportType = "direct"
    evidence_ids: list[str] = Field(default_factory=list)
    premise_evidence_ids: list[str] = Field(default_factory=list)


async def generate_final_answer(
    *,
    question: str,
    contract: QueryContract,
    packed_packets: Iterable[EvidencePacket] | PackedEvidenceProjection,
    slot_resolutions: Sequence[SlotResolution],
    llm_invoker: LlmInvoker,
    sufficiency_report: SufficiencyReport | None = None,
    arbitration: Any | None = None,
    citation_format_version: str = "1",
) -> FinalAnswerResult: ...
```

- The provider response is exactly:

```json
{
  "supported_findings": [
    {
      "slot_id": "slot-id",
      "statement": "source-bound finding",
      "support_type": "direct",
      "evidence_ids": ["evidence-id"],
      "premise_evidence_ids": []
    }
  ],
  "unresolved_requirements": [
    {"slot_id": "missing-slot", "reason": "not established by packed evidence"}
  ]
}
```

- Backend code, not the provider, derives `response_status`, unresolved requirements, used evidence IDs, and final prose.

- [ ] **Step 1: Add RED schema and renderer tests**

Extend `tests/test_agentic_v9_schemas.py` to prove the shared draft accepts support type and premise IDs but continues to reject provider-authored `answer`, `response_status`, unknown fields, and empty slot IDs.

Add focused cases to `tests/test_agentic_v9_final_answer.py`:

```python
@pytest.mark.asyncio
async def test_multi_slot_complete_requires_one_accepted_finding_per_required_slot():
    # The provider returns a valid finding for S1 and omits S2.
    # Both SlotResolutions claim supported, so this catches claim-coverage drift.
    result = await generate_final_answer(
        question="Compare A and B.",
        contract=_two_slot_contract(),
        packed_packets=[_packet("E1", "S1"), _packet("E2", "S2")],
        slot_resolutions=[
            SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
            SlotResolution(slot_id="S2", status="supported", evidence_ids=["E2"]),
        ],
        llm_invoker=_RecordingInvoker(
            {
                "supported_findings": [
                    {
                        "slot_id": "S1",
                        "statement": "Finding A.",
                        "support_type": "direct",
                        "evidence_ids": ["E1"],
                        "premise_evidence_ids": [],
                    }
                ],
                "unresolved_requirements": [],
            }
        ),
    )

    assert result.response_status == "qualified_partial"
    assert {claim.slot_id for claim in result.claims if not claim.qualified_reason} == {"S1"}
    assert "S2" in result.answer
```

Also add cases proving:

- a claim with no `slot_id` is rejected;
- an unknown slot is rejected;
- evidence authorized for `S1` cannot support an `S2` claim;
- a calculated claim requires complete premise IDs;
- a high-risk rejected/unavailable verifier result does not satisfy slot coverage;
- `used_evidence_ids` includes only accepted supported claims;
- invalid JSON returns fail-closed output without a second final generation;
- complete output has no empty unresolved heading;
- partial output contains confirmed findings and backend-derived unresolved slot descriptions.

- [ ] **Step 2: Run the final-answer RED gate**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_agentic_v9_schemas.py `
  tests/test_agentic_v9_final_answer.py `
  -q
```

Expected: new tests fail because `final_answer.py` still accepts the legacy `answer`/`claims` envelope, permits claims without slot coverage, and reports `complete` from SlotResolutions alone.

- [ ] **Step 3: Extend the shared structured draft minimally**

In `schemas.py`, add only `support_type` and `premise_evidence_ids` to `SupportedFinding`. Keep `FinalAnswerDraft` as the single strict provider-output model and keep `extra="forbid"`.

Remove the duplicate local `FinalAnswerDraft` and `_legacy_compatible_draft()` from `final_answer.py`. Import the shared `FinalAnswerDraft`, `SufficiencyReport`, and `UnresolvedRequirement` from `schemas.py`.

- [ ] **Step 4: Convert findings to claims and enforce slot authorization**

Add these private helpers in `final_answer.py`:

```python
def _claims_from_findings(
    draft: FinalAnswerDraft,
    *,
    contract: QueryContract,
    packets_by_id: Mapping[str, EvidencePacket],
) -> list[FinalClaim]:
    valid_slots = {slot.slot_id for slot in contract.required_slots}
    claims: list[FinalClaim] = []
    for index, finding in enumerate(draft.supported_findings, start=1):
        evidence_ids = list(
            dict.fromkeys([*finding.evidence_ids, *finding.premise_evidence_ids])
        )
        packets = [packets_by_id.get(evidence_id) for evidence_id in evidence_ids]
        if (
            finding.slot_id not in valid_slots
            or not evidence_ids
            or any(packet is None for packet in packets)
            or any(finding.slot_id not in packet.slot_ids for packet in packets if packet)
        ):
            continue
        claims.append(
            FinalClaim(
                claim_id=f"claim-{index}",
                slot_id=finding.slot_id,
                statement=finding.statement,
                support_type=finding.support_type,
                evidence_ids=finding.evidence_ids,
                premise_evidence_ids=finding.premise_evidence_ids,
            )
        )
    return claims
```

Keep the existing deterministic verification and one batched high-risk verifier. Failed or unavailable verifier results remain `qualified` diagnostics, but do not enter the supported coverage set or `used_evidence_ids`.

- [ ] **Step 5: Derive unresolved requirements and status**

Replace `_response_status()` with coverage derived from accepted, unqualified claims:

```python
def _supported_claim_slot_ids(claims: Sequence[FinalClaim]) -> set[str]:
    return {
        claim.slot_id
        for claim in claims
        if claim.slot_id is not None and claim.qualified_reason is None
    }


def _response_status(
    claims: Sequence[FinalClaim],
    contract: QueryContract,
    slot_resolutions: Sequence[SlotResolution],
) -> ResponseStatus:
    supported_claim_slots = _supported_claim_slot_ids(claims)
    required = {slot.slot_id for slot in contract.required_slots}
    resolution_by_slot = {resolution.slot_id: resolution for resolution in slot_resolutions}
    if not supported_claim_slots:
        return "insufficient"
    if required and required.issubset(supported_claim_slots) and all(
        resolution_by_slot.get(slot_id) is not None
        and resolution_by_slot[slot_id].status == "supported"
        for slot_id in required
    ):
        return "complete"
    return "qualified_partial"
```

Build unresolved requirements from required slots lacking an accepted claim or a supported resolution. Provider-proposed unresolved reasons may add detail only for a matching unresolved slot; they cannot hide, resolve, or invent slots.

- [ ] **Step 6: Render complete and partial answers without exposing JSON**

Update `render_verified_answer()`:

```python
if not unresolved:
    return "\n".join(rendered_claim_lines)
return "\n".join(
    [
        "## Confirmed from the supplied evidence",
        *rendered_claim_lines,
        "",
        "## Unable to confirm from the supplied evidence",
        *(f"- {item.slot_id}: {item.reason}" for item in unresolved),
    ]
)
```

An insufficient result uses a stable deterministic sentence and does not invoke another model.

- [ ] **Step 7: Run GREEN and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_agentic_v9_schemas.py `
  tests/test_agentic_v9_final_answer.py `
  -q
.\.venv\Scripts\python.exe -m ruff check `
  data_base/agentic_v9/schemas.py `
  data_base/agentic_v9/final_answer.py `
  data_base/agentic_v9/citation_renderer.py `
  tests/test_agentic_v9_schemas.py `
  tests/test_agentic_v9_final_answer.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 8: Self-review and commit Task 1**

Confirm the diff contains no campaign runtime changes and no compatibility parser for the old free-text envelope.

```powershell
git add -- `
  data_base/agentic_v9/schemas.py `
  data_base/agentic_v9/final_answer.py `
  data_base/agentic_v9/citation_renderer.py `
  tests/test_agentic_v9_schemas.py `
  tests/test_agentic_v9_final_answer.py
git commit -m "fix(agentic-v9): enforce slot-bound final synthesis"
```

---

### Task 2: Qualify Evidence Before Every Sufficiency Decision

**Files:**
- Modify: `data_base/agentic_v9/evidence_extractor.py:32-150`
- Modify: `data_base/agentic_v9/execution_core.py:55-340`
- Modify: `data_base/agentic_v9/phase_policy.py:35-60`
- Modify: `data_base/agentic_v9/schemas.py:613-623`
- Modify: `tests/test_agentic_v9_evidence_extractor.py`
- Modify: `tests/test_agentic_v9_execution_core.py`
- Modify: `tests/test_agentic_v9_phase_policy.py`
- Modify: `tests/test_agentic_v9_budget_controller.py`

**Interfaces:**
- Preserve the existing `V9ExecutionStages.prose_curate` callable signature to avoid a broad mechanical API rename:

```python
ProseCuratorStage = Callable[
    [str, QueryContract, tuple[EvidencePacket, ...]],
    _MaybeAwaitable[Sequence[EvidencePacket]],
]
```

- Its semantics become authoritative evidence qualification. It receives the accumulated candidate pool and returns only accepted positive packets.
- `V9ExecutionMetrics.prose_curator_call_count` remains a compatibility name. It records logical qualification-stage executions, not provider attempts. Actual provider attempts and tokens remain authoritative in the accounting ledger.

- [ ] **Step 1: Add RED extractor and core-order tests**

Add to `tests/test_agentic_v9_evidence_extractor.py`:

```python
@pytest.mark.asyncio
async def test_extractor_retains_prevalidated_packets_and_never_retains_invalid_raw_candidates():
    accepted = _item("E-valid", "Verified prose.", slot_ids=["S1"]).packet.model_copy(
        update={"validation_status": "quote_bound"}
    )
    raw = _item("E-raw", "Unverified prose.", slot_ids=["S2"]).packet.model_copy(
        update={"validation_status": "invalid"}
    )

    result = await EvidenceExtractor().extract(
        _contract(_slot("S1", "First"), _slot("S2", "Second")),
        [accepted, raw],
        repairs_complete=True,
    )

    assert [packet.evidence_id for packet in result] == ["E-valid"]
```

Add core cases proving this exact order:

```text
retrieve initial
deterministic candidates
prose_curate / qualify
sufficiency
plan repair only for still-missing slots
retrieve repair
prose_curate / qualify
sufficiency
pack/final
```

Assert raw invalid candidates never reach `evaluate_sufficiency`, a repair candidate is not supported until the second qualification result contains it, an empty qualification result produces deterministic insufficient output, and a no-task/no-evidence run does not invoke qualification or final generation.

- [ ] **Step 2: Run the qualification RED gate**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_agentic_v9_evidence_extractor.py `
  tests/test_agentic_v9_execution_core.py `
  tests/test_agentic_v9_phase_policy.py `
  tests/test_agentic_v9_budget_controller.py `
  -q
```

Expected: order tests fail because core evaluates sufficiency before curation; the extractor test fails because prevalidated packets are not explicitly retained; a second evidence-extraction phase attempt is rejected by the current per-phase capacity.

- [ ] **Step 3: Preserve accepted packets in `EvidenceExtractor`**

At the start of `extract()`, partition the pool:

```python
accepted = [
    item.packet
    for item in items
    if item.packet.validation_status
    in {"deterministic_valid", "quote_bound", "derived_non_evidence"}
    and item.packet.support_type != "contradictory"
]
packets = _deduplicate_packets([*accepted, *self.extract_deterministic(contract, items)])
```

Compute unresolved slots from this accepted set. Invalid raw candidates may be curator inputs, but they are returned only when deterministic or quote-bound validation creates a valid derived packet. On curator failure, return the accepted/deterministic packets rather than the raw pool.

- [ ] **Step 4: Move qualification ahead of sufficiency in the core**

Maintain separate lists:

```python
task_results: list[TaskRetrievalResult] = []
candidate_packets: list[EvidencePacket] = []
qualified_packets: list[EvidencePacket] = []
qualification_round_count = 0
```

After every non-empty retrieval round, call `prose_curate` with the accumulated candidate pool plus previously qualified packets, deduplicate its return by evidence ID, then evaluate sufficiency from `qualified_packets` only. Delete the single post-repair curation block.

Use a private helper with this exact behavior:

```python
async def _qualify_evidence(
    self,
    *,
    question: str,
    contract: QueryContract,
    candidates: Sequence[EvidencePacket],
    accepted: Sequence[EvidencePacket],
    deadline: ExecutionDeadline,
    cancellation: Any,
) -> tuple[EvidencePacket, ...]:
    combined = _deduplicate_packets([*accepted, *candidates])
    qualified = await self._run_stage(
        "llm",
        "evidence_extract",
        self._stages.prose_curate(question, contract, combined),
        deadline,
        cancellation,
    )
    return _deduplicate_packets(qualified)
```

The helper invocation is skipped when `combined` is empty. Qualification exceptions handled inside `EvidenceExtractor` preserve already-qualified packets and fail closed for raw candidates.

- [ ] **Step 5: Permit bounded repair-round qualification without changing route budgets**

Change only the evidence-extract phase capacity:

```python
MAX_PROVIDER_CALLS_BY_PHASE: dict[str, int] = {
    phase: (3 if phase == "evidence_extract" else 1)
    for phase in PHASE_POLICIES
}
```

Three equals one initial round plus the existing global maximum of two repair rounds. `RunBudgetController.max_llm_calls`, runtime-token budget, and final reserve still decide whether the second or third call is admitted. Do not alter route-planner budgets or feasibility requirements.

Update `V9ExecutionMetrics.prose_curator_call_count` to `Field(default=0, ge=0, le=3)` and set it to the logical qualification-round count. Add tests proving the phase permits at most three attempts while other phases remain capped at one, and the route call budget can still reject an extra attempt with `final_envelope_protected`.

- [ ] **Step 6: Run GREEN and subsystem regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_agentic_v9_evidence_extractor.py `
  tests/test_agentic_v9_execution_core.py `
  tests/test_agentic_v9_sufficiency_gate.py `
  tests/test_agentic_v9_repair.py `
  tests/test_agentic_v9_phase_policy.py `
  tests/test_agentic_v9_budget_controller.py `
  tests/test_agentic_v9_budget_feasibility.py `
  -q
.\.venv\Scripts\python.exe -m ruff check `
  data_base/agentic_v9/evidence_extractor.py `
  data_base/agentic_v9/execution_core.py `
  data_base/agentic_v9/phase_policy.py `
  data_base/agentic_v9/schemas.py `
  tests/test_agentic_v9_evidence_extractor.py `
  tests/test_agentic_v9_execution_core.py `
  tests/test_agentic_v9_phase_policy.py `
  tests/test_agentic_v9_budget_controller.py
git diff --check
```

Expected: all commands exit 0; the core still performs zero subtask generations and no more than one final generation.

- [ ] **Step 7: Self-review and commit Task 2**

Confirm sufficiency never receives `candidate_packets`, route budgets are unchanged, and additional evidence-extract attempts remain subject to both per-phase capacity and the existing run controller.

```powershell
git add -- `
  data_base/agentic_v9/evidence_extractor.py `
  data_base/agentic_v9/execution_core.py `
  data_base/agentic_v9/phase_policy.py `
  data_base/agentic_v9/schemas.py `
  tests/test_agentic_v9_evidence_extractor.py `
  tests/test_agentic_v9_execution_core.py `
  tests/test_agentic_v9_phase_policy.py `
  tests/test_agentic_v9_budget_controller.py
git commit -m "fix(agentic-v9): qualify evidence before sufficiency"
```

## Wave 1 Checkpoint

Stop after Tasks 1 and 2. Report both commit hashes and fresh focused test/Ruff results. The user may push and verify the shared core; do not start campaign runtime wiring until the user approves Wave 2.

---

## Wave 2: Atomically Wire Evaluation Campaign v9

### Task 3: Replace Campaign Shortcuts and Verify Observability

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py:20-100,350-960,1620-1695`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_full_rollback.py`
- Modify: `tests/test_evaluation_execution_observability.py`
- Modify: `docs/agentic-v9-smoke-verification.md`

**Interfaces:**
- Consumes:
  - Task 1 `generate_final_answer(...)`
  - Task 2 pre-sufficiency `prose_curate` qualification semantics
  - existing `BudgetedLlmInvoker`, controller, provider factory, observer, context packer, and normalized v9 trace
- Produces:
  - campaign answers derived from accepted structured claims;
  - `agentic_v9.evidence_packets` containing qualified positive evidence only;
  - `agentic_v9.final_claims`, used evidence, SlotResolutions, sufficiency, and completion status that agree with one another.

- [ ] **Step 1: Add production-shaped RED integration tests**

Create one provider double that distinguishes `evidence_extract`, `final_answer`, and `claim_verifier` by the prompt envelope and returns strict responses for each phase.

Add runtime tests proving:

1. Two generic required slots are qualified in one initial batch, included in the final payload with the full Query Contract and SlotResolutions, and persisted as two slot-bound claims.
2. A raw candidate with the correct `slot_id` but invalid/malformed qualification does not satisfy the slot, produces no final provider call, and returns `insufficient`.
3. One missing slot triggers the existing bounded repair; its new candidate becomes supported only after the repair qualification batch.
4. A repair qualification denied by the existing controller preserves prior qualified evidence, keeps the missing slot unresolved, and returns `qualified_partial` without increasing route limits.
5. A high-risk finding rejected by the verifier is not counted as supported or used evidence.
6. A visual packet already emitted with a usable validation status survives qualification, preventing the previous visual-evidence erasure regression.
7. `result.documents` contains only documents referenced by accepted final claims.

Use assertions shaped like:

```python
v9 = result.agent_trace["agentic_v9"]
assert result.agent_trace["response_status"] == "complete"
assert {item["slot_id"] for item in v9["final_claims"]} == {"S1", "S2"}
assert {item["validation_status"] for item in v9["evidence_packets"]} <= {
    "deterministic_valid",
    "quote_bound",
    "derived_non_evidence",
}
assert set(v9["slot_resolutions"][0]["evidence_ids"]) <= {
    item["evidence_id"] for item in v9["evidence_packets"]
}
```

- [ ] **Step 2: Run campaign RED tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_agentic_v9_campaign_runtime.py `
  tests/test_agentic_v9_full_rollback.py `
  tests/test_evaluation_execution_observability.py `
  -q
```

Expected: new tests fail because raw chunks are stamped `deterministic_valid`, `prose_curate` is a no-op, the final prompt omits the contract, and the adapter fabricates one complete claim using all packed IDs.

- [ ] **Step 3: Separate raw candidates from qualified evidence**

In `_evidence_packets_for_results()`, set raw internal candidate packets to:

```python
validation_status="invalid"
```

Do not append those packets to persisted `state["evidence_packets"]`. Keep them in a new internal `state["candidate_packets"]` collection for diagnostics and qualification input. Already-qualified visual packets retain their validator-issued status.

After qualification, replace `state["evidence_packets"]` with the deduplicated qualified return. The persisted `agentic_v9.evidence_packets`, sufficiency trace, context pack, and used-document selection must all read this qualified set.

- [ ] **Step 4: Wire `EvidenceExtractor` into `prose_curate`**

Construct the existing budgeted invoker from the current controller/provider/observer and call:

```python
extractor = EvidenceExtractor(budgeted_invoker)
qualified = await extractor.extract(
    contract,
    packets,
    repairs_complete=True,
    question=question,
)
```

Then apply the existing comparison balancing function to `qualified`, not raw candidates. Propagate `quality_by_evidence_id` from a candidate to each derived packet sharing the same `(doc_id, chunk_id, asset_id)` source identity so comparison packing does not lose reranker order.

Do not restore the July fail-soft fallback that promoted raw candidates when extraction was malformed. Empty extraction keeps only packets already carrying a usable validator-issued status.

- [ ] **Step 5: Replace synthetic final generation**

Replace the direct `BudgetedLlmInvoker.invoke()` call and whole-answer claim construction with:

```python
return await generate_final_answer(
    question=question,
    contract=contract,
    packed_packets=packed,
    slot_resolutions=resolutions,
    llm_invoker=budgeted_invoker,
    sufficiency_report=sufficiency_report,
    arbitration=arbitration,
)
```

Use the same controller, provider factory, observer, provider/model identity, and prompt capture policy already used by the campaign. Do not add a second final generation or legacy free-text parser.

Replace the hard-coded deterministic partial sentence with the Task 1 deterministic renderer so confirmed evidence and unresolved required slots remain visible without a provider call.

- [ ] **Step 6: Align observability with actual qualified state**

Ensure:

- `state["final_evidence_packets"]` is the qualified/balanced set passed to packing;
- trace `evidence_packets` excludes invalid raw candidates;
- SlotResolutions are exactly those used by final synthesis;
- `used_packets` and `documents` are selected from accepted `final.used_evidence_ids`;
- `final_claims` persists supported and qualified diagnostic claims without inventing one whole-answer claim;
- completion status equals the backend-derived final status;
- budget reservations remain the only authority for actual provider attempts and tokens.

Update `tests/test_evaluation_execution_observability.py` only where the corrected values are persisted; do not change normalized table schemas or API response models.

- [ ] **Step 7: Update affected existing fixtures and rollback protection**

Convert existing campaign provider doubles that return bare text into the strict `supported_findings`/`unresolved_requirements` envelope only where they exercise v9 final synthesis. Keep the full-rollback visual regression, but update its success response to the strict final envelope so it continues to prove visual capability gaps do not erase valid text evidence.

Do not add compatibility acceptance for legacy provider text; malformed legacy output must fail closed.

- [ ] **Step 8: Update the smoke verification guide**

In `docs/agentic-v9-smoke-verification.md`, add these manual checks:

- multi-slot runs contain multiple slot-bound accepted claims rather than one synthetic claim;
- raw retrieved chunks never appear as `deterministic_valid` without extractor/validator provenance;
- missing evidence produces partial/insufficient output and may trigger bounded repair;
- complete output has claim coverage for every required slot;
- compare Agentic and Naive normal runtime-token accounting and confirm the existing ratio target `<= 3.0` manually.

Do not describe an automatic ratio gate.

- [ ] **Step 9: Run Wave 2 GREEN verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_agentic_v9_campaign_runtime.py `
  tests/test_agentic_v9_full_rollback.py `
  tests/test_evaluation_execution_observability.py `
  tests/test_agentic_v9_evidence_extractor.py `
  tests/test_agentic_v9_execution_core.py `
  tests/test_agentic_v9_final_answer.py `
  -q
.\.venv\Scripts\python.exe -m pytest -k agentic_v9 -q
.\.venv\Scripts\python.exe -m ruff check `
  evaluation/agentic_v9_campaign_runtime.py `
  data_base/agentic_v9 `
  tests/test_agentic_v9_campaign_runtime.py `
  tests/test_agentic_v9_full_rollback.py `
  tests/test_evaluation_execution_observability.py
.\.venv\Scripts\python.exe -m compileall -q `
  evaluation/agentic_v9_campaign_runtime.py `
  data_base/agentic_v9
git diff --check
```

Expected: all focused and `agentic_v9` tests pass, Ruff and compileall exit 0, and no OpenAPI artifact changes are produced.

- [ ] **Step 10: Self-review and commit Task 3**

Review the complete Wave 2 diff against the approved design. Confirm no Native/v8/chat code, route budgets, evaluation ground truth, RAGAS evaluator, database schema, OpenAPI artifact, or frontend file changed.

```powershell
git add -- `
  evaluation/agentic_v9_campaign_runtime.py `
  tests/test_agentic_v9_campaign_runtime.py `
  tests/test_agentic_v9_full_rollback.py `
  tests/test_evaluation_execution_observability.py `
  docs/agentic-v9-smoke-verification.md
git commit -m "fix(evaluation): restore v9 evidence-grounded answers"
```

## Wave 2 Checkpoint

Stop after Task 3. Report the Task 3 commit hash, all fresh verification results, any baseline-only failures reproduced before the change, and the exact files changed. Do not start an additional reviewer or unrelated optimization unless the user explicitly requests it.

## Final Manual Campaign Verification

After the user pushes the Wave 2 checkpoint to the real environment, run the same paired Agentic/Naive campaign configuration used for the original 32-question investigation. Compare:

- faithfulness overall and by required-slot count;
- correctness and relevancy to ensure the fail-closed correction did not create an unacceptable answer-coverage regression;
- percentage of `complete`, `qualified_partial`, and `insufficient` Agentic runs;
- claim count versus required-slot count;
- repair execution for genuinely unresolved slots;
- accepted claim evidence IDs against packed evidence IDs;
- normal accounting Agentic/Naive runtime-token ratio against the existing `<= 3.0` target.

The expected first validation signal is structural: multi-slot runs no longer report one unconditional whole-answer claim and unsupported slots no longer appear complete. Quality scores are measured outcomes, not hard-coded implementation assertions.
