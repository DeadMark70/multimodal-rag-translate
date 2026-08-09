# Agentic v9 Retrieval and Evidence Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore Agentic v9 evidence recall without weakening source authorization, then improve generalization by reranking authorized chunks and binding them to atomic slots with one batched evidence-extraction call.

**Architecture:** Keep contract-v2 atomic question decomposition and the existing authorized document scope. Remove task-level locator rejection, treat only explicit structured locators as conditional slot constraints, rerank authorized hybrid-retrieval candidates with fail-soft fallback, and wire the existing `EvidenceExtractor` into the runtime as the single semantic slot-binding call.

**Tech Stack:** Python 3.11, Pydantic v2, LangChain `Document`, existing Jina `DocumentReranker`, asyncio/FastAPI threadpool, pytest.

## Global Constraints

- Preserve contract-v2 atomic slots and `RequiredSlot` source restrictions.
- Preserve source authorization as a hard gate; no out-of-scope document may become an `EvidencePacket`.
- Remove task locator matching as a hard gate.
- Ordinary text is never an exact locator constraint.
- Explicit `Table`, `Figure`, `Equation`/`Formula`, `Theorem`, `Appendix`, and `Section` locators are conditional slot constraints.
- Missing structured metadata produces `unavailable`, not evidence rejection.
- Explicit conflicting structured metadata produces `mismatched` and rejects that chunk for the affected slot.
- Reranking applies only after the authorized-document filter.
- Reranker unavailable, timeout, OOM, or exception preserves the original authorized hybrid ranking.
- Reuse the existing `evidence_extract` phase and make at most one batched evidence-extraction provider call per run; never add per-slot provider calls.
- Evidence-extraction failure remains fail-closed: no semantic slot support is fabricated.
- Do not add a dependency, database migration, frontend change, new semantic-matcher service, GraphRAG change, visual-pipeline change, or new repair-loop architecture in this recovery.
- Existing missing-slot repair remains bounded and fail-closed. Semantic multi-pass repair is deliberately deferred until the recovery smoke proves it is necessary.

---

## File Map

- `data_base/agentic_v9/slot_constraints.py`
  - Owns structured-locator parsing and the four-state match decision.
- `evaluation/agentic_v9_campaign_runtime.py`
  - Owns source-authorized candidate retrieval, fail-soft reranking, candidate projection, batched evidence extraction, and runtime trace projection.
- `data_base/agentic_v9/evidence_extractor.py`
  - Owns deterministic extraction for verified structured locations and one-call semantic binding for all remaining slots.
- `data_base/reranker.py`
  - Keeps legacy fail-soft behavior and exposes a strict v9 scoring entry point whose failures can be observed by the retrieval boundary.
- `tests/test_agentic_v9_slot_constraints.py`
  - New focused tests for structured, textual, missing, and conflicting locator behavior.
- `tests/test_agentic_v9_campaign_runtime.py`
  - Runtime regression tests for authorization, locator behavior, reranking, batched binding, and fail-closed extraction.
- `tests/test_agentic_v9_evidence_extractor.py`
  - Tests deterministic-versus-semantic slot selection and the one-call invariant.
- `tests/test_rag_filtering.py`
  - Existing retrieval-boundary metadata and fallback contract tests.
- `tests/test_reranker.py`
  - Existing local-model failure and device fallback tests.
- `scripts/run_agentic_v9_smoke.py`
  - Reuse unchanged unless the existing verifier cannot read the new retrieval diagnostics; no new runner is created.

---

### Task 1: Replace Exact Text Matching with Conditional Structured Locators

**Files:**
- Modify: `data_base/agentic_v9/slot_constraints.py:1-142`
- Create: `tests/test_agentic_v9_slot_constraints.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py:1344-1560`

**Interfaces:**
- Produces:

The exact public signatures are:

- `StructuredLocatorState = Literal["not_requested", "matched", "mismatched", "unavailable"]`
- `canonical_structured_locator(value: object) -> tuple[str, str] | None`
- `structured_locator_state(hints: Iterable[str], chunk: Mapping[str, Any]) -> StructuredLocatorState`

- `canonical_structured_locator()` returns `None` for ordinary prose, model names, numeric answers, and other non-structured hints.
- `structured_locator_state()` follows this exact precedence:
  1. no structured hint → `not_requested`;
  2. any exact type-and-identifier match → `matched`;
  3. relevant structured metadata exists but none matches → `mismatched`;
  4. the chunk has no metadata for the requested locator types → `unavailable`.
- Later tasks consume `structured_locator_state()` only at slot binding. Retrieval tasks may carry locator hints for query construction and observability, but may not reject candidates.
- `_evidence_packets_for_results(..., locator_diagnostics: list[dict[str, Any]] | None = None)` appends one bounded row per candidate/slot decision containing `task_id`, `chunk_id`, `slot_id`, `state`, and `accepted`.

- [ ] **Step 1: Write failing pure-policy tests**

Add tests containing these exact cases:

```python
def test_plain_text_hint_is_not_a_structured_locator() -> None:
    assert canonical_structured_locator("SwinUNETR long-range dependency") is None
    assert structured_locator_state(
        ["SwinUNETR long-range dependency"],
        {"text": "SwinUNETR uses shifted-window attention."},
    ) == "not_requested"


def test_matching_table_metadata_is_matched() -> None:
    assert structured_locator_state(
        ["Table 3"],
        {"table_id": "Table 3"},
    ) == "matched"


def test_explicit_wrong_table_metadata_is_mismatched() -> None:
    assert structured_locator_state(
        ["Table 3"],
        {"table_id": "Table 4"},
    ) == "mismatched"


def test_missing_table_metadata_is_unavailable_not_mismatched() -> None:
    assert structured_locator_state(
        ["Table 3"],
        {"text": "The requested Table 3 row is present in extracted PDF text."},
    ) == "unavailable"
```

- [ ] **Step 2: Run the policy tests and verify they fail**

Run:

```bash
pytest -q tests/test_agentic_v9_slot_constraints.py
```

Expected: FAIL because the new interface does not exist.

- [ ] **Step 3: Implement structured-only canonicalization**

Implement `canonical_structured_locator()` using the existing bounded locator pattern and aliases. Do not return `("text", value)` when the pattern does not match. Build actual chunk locators only from `figure_id`, `table_id`, `formula_id`, and `section`; normalize theorem/appendix/section hints through the same parser.

Keep `display_locator_hints()` capable of displaying ordinary hints, but do not feed ordinary hints into `canonical_structured_locator_set()` or any equality decision.

- [ ] **Step 4: Remove the task-level hard locator gate**

Delete this rejection from `_evidence_packets_for_results()`:

```python
if contract.contract_version == "2" and not locator_hints_match_chunk(
    task.locator_hints, chunk
):
    continue
```

Keep the task source-scope check unchanged:

```python
if doc_id not in task.source_scope.authorized_doc_ids:
    continue
```

Change the per-slot binding decision to:

```python
locator_state = structured_locator_state(slot.locator_hints, chunk)
if locator_state == "mismatched":
    continue
authorized.append(slot_id)
```

Thus `matched`, `unavailable`, and `not_requested` remain candidates for semantic evidence extraction.

Pass a runtime-owned diagnostics list into `_evidence_packets_for_results()` and persist it as `agent_trace["agentic_v9"]["locator_diagnostics"]`. Do not persist chunk text in this diagnostic.

- [ ] **Step 5: Replace the old v2 missing-metadata assertion**

Replace `test_v2_locator_hint_rejects_ordinary_retrieved_chunk_without_metadata` with:

```python
def test_v2_missing_structured_metadata_keeps_candidate_for_semantic_binding() -> None:
    packets = _evidence_packets_for_results(
        results=results,
        contract=contract,
        trace_id="trace",
        tasks_by_id={task.task_id: task},
    )
    assert [packet.slot_ids for packet in packets] == [["S1"]]
```

Retain and pass:

- wrong `Table 4` cannot satisfy `Table 3`;
- grouped `Table 3`/`Table 4` candidates bind only to the matching slot;
- a globally authorized document outside the slot source restriction remains rejected.

- [ ] **Step 6: Run focused locator and runtime tests**

Run:

```bash
pytest -q \
  tests/test_agentic_v9_slot_constraints.py \
  tests/test_agentic_v9_campaign_runtime.py::test_same_document_chunk_with_wrong_locator_cannot_support_slot \
  tests/test_agentic_v9_campaign_runtime.py::test_v2_missing_structured_metadata_keeps_candidate_for_semantic_binding \
  tests/test_agentic_v9_campaign_runtime.py::test_grouped_task_chunk_is_bound_only_to_its_matching_atomic_slot \
  tests/test_agentic_v9_campaign_runtime.py::test_text_evidence_outside_atomic_slot_authorized_ids_cannot_support_it
```

Expected: all PASS.

- [ ] **Step 7: Commit the locator recovery**

```bash
git add data_base/agentic_v9/slot_constraints.py \
  tests/test_agentic_v9_slot_constraints.py \
  tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): make structured locators conditional"
```

---

### Task 2: Enable Authorized Fail-Soft Reranking and Persist Diagnostics

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py:101-145,285-326,640-700,709-728`
- Modify: `data_base/rag_filtering.py:36-152`
- Modify: `data_base/reranker.py:380-455`
- Modify: `tests/test_rag_filtering.py:1-149`
- Modify: `tests/test_reranker.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**
- Produces:

The exact interfaces are:

- `V9RetrievalSelection.documents: tuple[Document, ...]`
- `V9RetrievalSelection.diagnostics: dict[str, Any]`
- `RetrievalAdapterResult = V9RetrievalSelection | list[Document]`
- `RetrievalAdapter = Callable[[str, str, list[str]], Awaitable[RetrievalAdapterResult]]`

- Default `_retrieve_documents()` returns `V9RetrievalSelection`.
- Existing injected test adapters returning `list[Document]` remain supported through one `_normalize_retrieval_selection()` compatibility boundary.
- `DocumentReranker.rerank_with_scores_strict(query, documents, top_k)` raises on model unavailability and scoring failure; legacy `rerank_with_scores()` remains fail-soft.
- Every `TaskRetrievalResult.retrieval.diagnostics` records:

The diagnostics contract contains:

- `source_filter.authorized_doc_ids: list[str]`
- `source_filter.pre_filter_count: int`
- `source_filter.post_filter_count: int`
- `reranking.enabled: bool`
- `reranking.available: bool | None`
- `reranking.fallback_reason: None | "not_initialized" | "timeout" | "exception"`
- `reranking.pre_rerank_ranks: list[dict[str, Any]]`
- `reranking.post_rerank_ranks: list[dict[str, Any]]`
- `reranking.selected_count: int`

- [ ] **Step 1: Write failing runtime reranker tests**

Add three tests:

```python
@pytest.mark.asyncio
async def test_v9_reranks_only_authorized_candidates() -> None:
    # Candidate from doc-b is filtered before the injected reranker sees input.
    # The injected reranker reverses the two doc-a chunks.
    # Assert selected order and persisted pre/post ranks.


@pytest.mark.asyncio
async def test_v9_reranker_exception_preserves_authorized_hybrid_order() -> None:
    # Inject a reranker that raises RuntimeError.
    # Assert the original authorized order remains and fallback_reason == "exception".


@pytest.mark.asyncio
async def test_v9_uninitialized_reranker_preserves_all_authorized_candidates() -> None:
    # Assert no candidate is removed merely because the local model is unavailable.
```

Use dependency injection; do not initialize or download a real reranker in unit tests.

- [ ] **Step 2: Run the new reranker tests and verify they fail**

Run:

```bash
pytest -q \
  tests/test_rag_filtering.py \
  tests/test_agentic_v9_campaign_runtime.py -k "rerank"
```

Expected: new tests FAIL because v9 passes `enable_reranking=False` and discards selection diagnostics.

- [ ] **Step 3: Make reranker failure metadata explicit**

Extend `filter_and_rerank_retrieval()` so the `reranking` metadata always contains `fallback_reason`. Catch exceptions around `rerank_with_scores` at the filtering boundary and return the original authorized candidate order with `score=None`; do not synthesize `0.0` relevance scores.

Keep the existing local reranker CUDA-OOM-to-CPU behavior. The filtering boundary is the final fail-soft guard for timeout and non-`RuntimeError` failures.

- [ ] **Step 4: Add a strict v9 reranker entry point without changing legacy callers**

Add `DocumentReranker.rerank_with_scores_strict()` and route it through `_run_rerank(..., raise_on_failure=True)`. In strict mode:

- missing model raises `RuntimeError("reranker_not_initialized")`;
- non-OOM model failures propagate;
- CUDA OOM still performs the existing single CPU retry;
- failure of the CPU retry propagates.

Keep `rerank_with_scores()` on `raise_on_failure=False` so other RAG modes retain current behavior.

- [ ] **Step 5: Wire the default v9 retrieval adapter**

In `_retrieve_documents()`:

1. run hybrid retrieval with HyDE and Multi-Query still disabled;
2. call `filter_and_rerank_retrieval()` in `run_in_threadpool`;
3. pass `doc_ids=authorized_doc_ids`;
4. pass `enable_reranking=True`;
5. pass `reranker_available=DocumentReranker.is_initialized()`;
6. use existing `RERANK_CANDIDATE_LIMIT=12` and `RERANK_TARGET_K=8`;
7. pass `DocumentReranker.get_instance().rerank_with_scores_strict` when available;
8. bound the threadpool operation with `asyncio.wait_for(..., timeout=8.0)`;
9. on timeout, rerun only the cheap authorized filter with reranking disabled and annotate `fallback_reason="timeout"`;
10. return selected documents plus source-filter/reranker diagnostics.

Do not introduce a score threshold. Ranking changes order and top-K selection only.

- [ ] **Step 6: Persist task-level retrieval diagnostics**

Normalize injected list adapters to `V9RetrievalSelection`, copy selection diagnostics into `RagRetrievalResult.diagnostics`, and add:

```python
"retrieval_diagnostics": [
    {
        "task_id": result.task_id,
        **result.retrieval.diagnostics,
    }
    for result in executed.task_results
],
```

to `agent_trace["agentic_v9"]`.

- [ ] **Step 7: Run retrieval and authorization tests**

Run:

```bash
pytest -q \
  tests/test_reranker.py \
  tests/test_rag_filtering.py \
  tests/test_agentic_v9_campaign_runtime.py -k "rerank or authorized or source_name"
```

Expected: all PASS; authorization rejection remains unchanged and fallback retains candidates.

- [ ] **Step 8: Commit reranker wiring**

```bash
git add evaluation/agentic_v9_campaign_runtime.py \
  data_base/rag_filtering.py \
  data_base/reranker.py \
  tests/test_reranker.py \
  tests/test_rag_filtering.py \
  tests/test_agentic_v9_campaign_runtime.py
git commit -m "feat(agentic-v9): rerank authorized retrieval candidates"
```

---

### Task 3: Wire One Batched Semantic Evidence-Binding Call

**Files:**
- Modify: `data_base/agentic_v9/evidence_extractor.py:55-125`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:121-145,493-499,640-700`
- Modify: `tests/test_agentic_v9_evidence_extractor.py:1-230`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**
- Consumes:
  - `structured_locator_state()` from Task 1.
  - Candidate `EvidencePacket`s that already passed effective source authorization.
- Produces:

```python
async def EvidenceExtractor.extract(
    self,
    contract: QueryContract,
    pool: Iterable[EvidencePacket | EvidencePoolItem | EvidencePoolEntry],
    *,
    repairs_complete: bool,
    question: str = "",
) -> list[EvidencePacket]:
    """Return only deterministic structured packets and LLM-curated quote-bound packets."""
```

- The existing JSON contract remains unchanged:

```json
{
  "packets": [
    {
      "source_evidence_id": "candidate-id",
      "slot_ids": ["S1", "S2"],
      "statement": "exact contiguous source span"
    }
  ]
}
```

- [ ] **Step 1: Add failing evidence-binding tests**

Add:

```python
@pytest.mark.asyncio
async def test_one_batch_binds_multiple_generic_slots_without_per_slot_calls() -> None:
    # Two generic atomic slots and multiple candidate chunks.
    # One response binds different source spans to S1 and S2.
    # Assert exactly one evidence_extract call and exact source provenance.


@pytest.mark.asyncio
async def test_structured_metadata_unavailable_falls_back_to_semantic_binding() -> None:
    # Slot requests Table 3; candidate has no table_id but contains the row text.
    # Assert it is presented to the one batched curator and may become quote-bound evidence.


@pytest.mark.asyncio
async def test_evidence_extract_failure_supports_no_generic_slot() -> None:
    # Invoker raises TimeoutError.
    # Assert [] rather than raw candidates or fabricated support.


@pytest.mark.asyncio
async def test_runtime_uses_one_batch_to_bind_multiple_atomic_slots() -> None:
    # Exercise AgenticV9CampaignRuntime rather than EvidenceExtractor directly.
    # Assert the production prose_curate stage invokes evidence_extract once.
```

- [ ] **Step 2: Run the extractor tests and verify they fail**

Run:

```bash
pytest -q \
  tests/test_agentic_v9_evidence_extractor.py \
  tests/test_agentic_v9_campaign_runtime.py::test_runtime_uses_one_batch_to_bind_multiple_atomic_slots
```

Expected: the extractor-level cases may already pass, while the production runtime case FAILS because `prose_curate()` is currently a no-op.

- [ ] **Step 3: Limit deterministic extraction to verified structured locations**

For each slot, construct locator metadata from both `EvidencePoolItem.metadata` and `EvidencePoolItem.packet.locator` (`table_id`, `figure_id`, and `section`) before calling the policy from Task 1. Then:

- run numeric/structured deterministic extraction only when at least one candidate has `structured_locator_state(slot.locator_hints, item.metadata) == "matched"`;
- send ordinary semantic slots and structured slots with `unavailable` metadata to the batch curator;
- never treat `mismatched` candidates as input for that slot;
- keep quote-bound validation and exact contiguous-span validation unchanged.

This avoids promoting arbitrary numbers from a merely authorized chunk while preserving cheap deterministic extraction when a real Table/Figure/Formula/Section locator matches.

- [ ] **Step 4: Replace the runtime no-op curator**

Replace the current `prose_curate` body that directly returns `packets` with one call to the existing `EvidenceExtractor`, using the existing `BudgetedLlmInvoker`:

```python
extractor = EvidenceExtractor(
    BudgetedLlmInvoker(
        controller=controller,
        provider_factory=self._provider_factory,
        observer=llm_call_observer,
        provider_name=provider_name,
        model_name=model_name,
        capture_policy=setup_snapshot.get("prompt_capture_policy"),
    )
)
curated = await extractor.extract(
    contract,
    packets,
    repairs_complete=True,
    question=question,
)
state["evidence_packets"] = list(curated)
return tuple(curated)
```

Use the already-admitted `evidence_extract` reservation. Do not change phase budgets or add an invocation inside a slot loop.

- [ ] **Step 5: Preserve fail-closed final synthesis**

Verify that:

- malformed/timeout evidence extraction returns no curated packets;
- final sufficiency is recomputed from curated packets;
- unsupported slots remain `not_found`;
- final synthesis receives only curated packets;
- `used_evidence_documents()` resolves documents from curated evidence IDs;
- the trace exports curated evidence rather than pre-curation candidates.

Do not redesign the repair loop in this task. Existing repairs may improve deterministic structured gaps; semantic multi-pass repair remains deferred.

- [ ] **Step 6: Add a production-shaped runtime regression**

Create one runtime test with:

- two generic atomic slots;
- one authorized document with two relevant chunks and one unauthorized chunk;
- task locator hints containing ordinary prose;
- an injected reranker that orders relevant chunks first;
- one evidence-extraction response binding both slots;
- one final-answer response using the curated evidence.

Assert:

```python
assert result.agent_trace["response_status"] != "insufficient"
assert len(result.agent_trace["agentic_v9"]["evidence_packets"]) >= 2
assert {packet["source"]["doc_id"] for packet in packets} == {"doc-authorized"}
assert evidence_extract_call_count == 1
assert result.documents
```

- [ ] **Step 7: Run focused v9 tests**

Run:

```bash
pytest -q \
  tests/test_agentic_v9_evidence_extractor.py \
  tests/test_agentic_v9_campaign_runtime.py
```

Expected: all PASS.

- [ ] **Step 8: Commit semantic binding**

```bash
git add data_base/agentic_v9/evidence_extractor.py \
  evaluation/agentic_v9_campaign_runtime.py \
  tests/test_agentic_v9_evidence_extractor.py \
  tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): bind atomic slots with one evidence batch"
```

---

### Task 4: Run the Minimum Verification Gate Before Spending on 16 Questions

**Files:**
- Modify only if required by an observed schema mismatch: `scripts/run_agentic_v9_smoke.py`
- Modify only if the runner changes: `tests/test_agentic_v9_smoke_runner.py`
- Create runtime artifact outside Git: `artifacts/agentic-v9-smoke/recovery-smoke-redacted.json`

**Interfaces:**
- Consumes a completed redacted campaign export.
- Produces a PASS/FAIL smoke manifest without modifying production data.

- [ ] **Step 1: Run the complete local focused suite**

Run:

```bash
pytest -q \
  tests/test_agentic_v9_slot_constraints.py \
  tests/test_reranker.py \
  tests/test_rag_filtering.py \
  tests/test_agentic_v9_evidence_extractor.py \
  tests/test_agentic_v9_campaign_runtime.py \
  tests/test_agentic_v9_smoke_runner.py
```

Expected: all PASS.

- [ ] **Step 2: Run the backend CI compile gate**

The current backend workflow declares this compile gate:

```bash
python -m compileall -q core data_base evaluation
```

Expected: exit code 0.

- [ ] **Step 3: Deploy once and run only the fixed five-question smoke**

Run Agentic v9 for:

```text
Q5, Q7, Q11, Q14, Q16
```

Use one repeat and no paired Naive arm for this gate. This is five runs, not 32.

- [ ] **Step 4: Apply the recovery smoke assertions**

The smoke fails if any condition is true:

- all five runs have zero contexts;
- any run contains an out-of-scope document ID;
- Q5, Q7, or Q11 has zero evidence packets solely because locator metadata is absent;
- the run records more than one `evidence_extract` provider call;
- reranker failure removes all otherwise authorized hybrid candidates;
- a positive final claim lacks slot and evidence provenance.

The smoke passes only when:

- every run completes without configuration incompatibility;
- at least Q5, Q7, and Q11 produce nonzero evidence packets;
- structured-locator questions Q14/Q16 record `matched`, `mismatched`, or `unavailable` honestly;
- token accounting and `evidence_extract` phase attribution are complete.

- [ ] **Step 5: Reuse the existing offline verifier**

Run:

```bash
python scripts/run_agentic_v9_smoke.py \
  --artifact artifacts/agentic-v9-smoke/recovery-smoke-redacted.json
```

Expected: PASS and a reproducible manifest. If the verifier cannot see retrieval diagnostics, make the smallest schema projection needed and add one focused test; do not build a second runner.

- [ ] **Step 6: Decide whether the full 16-question evaluation is justified**

Proceed to the 16-question paired Agentic-v9/Naive evaluation only if Task 4 passes. Compare:

- correctness;
- faithfulness;
- relevancy;
- total official tokens;
- P50/P95 latency;
- contexts and evidence packets per question;
- unauthorized-source count;
- reranker fallback count;
- evidence-extraction call count.

Do not claim an Agentic quality improvement unless the paired results support it. The primary recovery criterion is elimination of the 16/16 zero-context failure without introducing source leakage.

---

## Explicitly Deferred Work

The following work is intentionally excluded to minimize implementation time and token cost:

- per-slot LLM evidence calls;
- a second semantic evidence pass after corrective retrieval;
- embedding-based or custom semantic matcher service;
- new reranker model selection or training;
- HyDE or Multi-Query re-enablement;
- GraphRAG/visual-pipeline redesign;
- frontend visualization of retrieval diagnostics;
- database schema migration;
- formal benchmark release review.

These are reconsidered only if the five-question smoke demonstrates a concrete remaining failure that the current plan cannot explain.
