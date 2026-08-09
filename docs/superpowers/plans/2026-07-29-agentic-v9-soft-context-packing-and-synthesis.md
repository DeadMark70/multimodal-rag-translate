# Agentic v9 Soft Context Packing and Evidence-Aware Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Agentic v9 answer correctness without adding LLM calls or reintroducing locator/quote hard gates, first by selecting a better final evidence pack and only then, if evidence shows a residual synthesis problem, by adding a soft evidence-aware final-answer instruction.

**Architecture:** The current v9 retrieval path already performs Hybrid retrieve 8 → rerank 8 → task top-4. Wave 1 preserves that retrieval boundary and adds a versioned final-pack selection policy: reranker-derived quality is propagated into `EvidenceContextPacker`, exact duplicate packets remain the only hard removal, and source diversity / near-duplicate / typed visual signals only adjust ordering. Wave 2 is intentionally conditional on Wave 1 smoke evidence; it changes the one existing final LLM call's system instruction through the established prompt registry, with no verifier, retry, locator, or additional model invocation.

**Tech Stack:** Python 3.13, FastAPI evaluation runtime, Pydantic, LangChain `Document`, local Jina reranker, pytest, Ruff, JSON prompt registry.

## Global Constraints

- Keep Hybrid retrieve 8 → rerank candidates 8 → retrieval-task top-4 unchanged.
- Native/naive RAG stays unchanged and does not receive the Agentic-v9 reranker policy.
- Do not use `expected_sources`, Golden Dataset IDs, test-case key points, source authorization, atomic slots, text locator matching, or RAGAS labels as runtime ranking signals.
- Do not introduce an additional LLM call, per-slot LLM call, verifier call, retry, or external dependency.
- Exact source-chunk/span duplicates may be removed; all other packets remain eligible for final selection.
- A missing reranker score must fall back to the recorded post-rerank rank, then original retrieval order; it must never make a run insufficient or clear contexts.
- Visual evidence is recognized only from typed provenance (`asset_id`, `figure_id`, or `table_id`) that is already present in metadata. Do not classify visual content from Chinese/English prose prefixes or other brittle text heuristics.
- Any changed evaluation context policy must receive a new `context_policy_version` and `execution_profile`; historical campaigns must remain comparable only within their own profile.
- Preserve Setup authority for model name, temperature, thinking, and output limits. Evaluation code must not override those values.
- Keep each wave independently revertible with one focused commit. Do not start Wave 2 until the Wave 1 campaign evidence is reviewed.

## Current Root Cause

`EvidenceContextPacker.pack()` already accepts `quality_by_evidence_id`, but the evaluation runtime calls it without that mapping. Meanwhile `_chunk_projection()` drops `agentic_v9_reranking` metadata before `_evidence_packets_for_results()` creates packets. Every candidate therefore reaches the packer with quality `0.0`; existing source-count ordering dominates selection even when a reranker ran successfully. The Wave 1 fix must first preserve and consume the existing rerank outcome before adding any new selection signal.

---

## Wave 1 — Versioned Soft Final Context Pack

### Task 1: Preserve reranking and typed provenance through the evidence projection

**Files:**

- Modify: `evaluation/agentic_v9_campaign_runtime.py: retrieve(), _chunk_projection(), _evidence_packets_for_results()`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**

- Consumes: `Document.metadata["agentic_v9_reranking"]` created by `_annotate_rerank_selection()`.
- Produces: each `RagRetrievalResult.chunks` row carries a serializable `reranking` object and typed provenance fields (`asset_id`, `figure_id`, `table_id`) when they exist; `_evidence_packets_for_results()` returns both `list[EvidencePacket]` and `dict[str, float]` keyed by the emitted packet's `evidence_id`.
- Compatibility: chunks without reranking metadata remain valid and get a deterministic fallback quality. Existing callers that only consume evidence packets are updated in this runtime in the same task.

- [ ] **Step 1: Write failing runtime-projection tests**

Add tests that build a `Document` with the following metadata and assert that its chunk projection keeps only serializable fields needed downstream:

```python
metadata = {
    "doc_id": "doc-1",
    "chunk_id": "chunk-7",
    "asset_id": "asset-1",
    "figure_id": "Figure 2",
    "agentic_v9_reranking": {
        "status": "executed",
        "post_rerank_rank": 2,
        "rerank_score": 0.42,
    },
}
```

Assert that `_chunk_projection(...)["reranking"]` contains `status`, `post_rerank_rank`, and `rerank_score`; assert `asset_id` and `figure_id` remain present. Add a second test with no reranking metadata and assert the projected row has no fabricated score.

- [ ] **Step 2: Run the projection tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py -k "chunk_projection or rerank_quality" -v
```

Expected: FAIL because current `_chunk_projection()` only exports document ID, chunk ID, text, page, and section.

- [ ] **Step 3: Add the minimal projection and evidence-quality helper**

In `evaluation/agentic_v9_campaign_runtime.py`:

```python
def _chunk_reranking_projection(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    value = metadata.get("agentic_v9_reranking")
    if not isinstance(value, Mapping):
        return None
    return {
        "status": str(value.get("status") or "not_instrumented"),
        "post_rerank_rank": _positive_int_or_none(value.get("post_rerank_rank")),
        "rerank_score": _finite_float_or_none(value.get("rerank_score")),
    }

def _packet_quality(chunk: Mapping[str, Any], fallback_index: int) -> float:
    reranking = chunk.get("reranking")
    if isinstance(reranking, Mapping):
        rank = _positive_int_or_none(reranking.get("post_rerank_rank"))
        if rank is not None:
            return 1.0 / rank
    return 1.0 / (fallback_index + 1)
```

When `_evidence_packets_for_results()` creates an evidence ID, add `quality_by_evidence_id[evidence_id] = _packet_quality(chunk, index)`. Preserve typed `asset_id`, `figure_id`, and `table_id` in `EvidenceSource` / `SourceLocator` only when the source row actually contains them. Return a typed result:

```python
@dataclass(frozen=True, slots=True)
class EvidencePacketProjection:
    packets: list[EvidencePacket]
    quality_by_evidence_id: dict[str, float]
```

Update `deterministic_candidates()` to save the quality mapping in attempt-local `state` alongside `evidence_packets`.

- [ ] **Step 4: Run focused runtime tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py -v
```

Expected: PASS. Confirm existing fallback reranking and no-reranker tests still pass.

- [ ] **Step 5: Commit the data-flow repair**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): preserve rerank quality for context packing"
```

### Task 2: Add a conservative, soft selection policy to the existing packer

**Files:**

- Modify: `data_base/agentic_v9/context_packer.py`
- Modify: `tests/test_agentic_v9_context_packer.py`

**Interfaces:**

- Consumes: packet quality mapping from Task 1 plus existing `EvidencePacket` provenance.
- Produces: `EvidenceContextPacker.pack(..., selection_policy=...)` applies an optional, versioned policy and returns selection decisions with score components.
- Compatibility: `selection_policy=None` preserves the current packer behavior exactly. Mandatory slot / premise closure semantics remain unchanged and may exceed the preferred packet count rather than failing a run.

- [ ] **Step 1: Write failing policy tests**

Define a test-only `FinalContextSelectionPolicy` with:

```python
FinalContextSelectionPolicy(
    version="soft_final_pack_r1",
    preferred_max_packets=8,
    new_source_bonus=0.05,
    near_duplicate_penalty=0.08,
    visual_without_visual_intent_penalty=0.03,
)
```

Add these tests:

```python
def test_soft_policy_uses_rerank_quality_before_source_diversity() -> None:
    # A same-source candidate with quality 1.0 stays ahead of an unseen-source
    # candidate with quality 0.10; source diversity may not become a quota.

def test_soft_policy_only_removes_exact_source_duplicates() -> None:
    # Two near-duplicate statements with distinct chunk IDs remain eligible.
    # The lower-utility one is recorded as not selected only when the preferred
    # packet limit is reached; it is never recorded as a hard rejection.

def test_soft_policy_keeps_unique_numeric_or_structured_evidence() -> None:
    # A similar packet with a different raw value / table locator remains selected
    # when capacity permits and gets no near-duplicate penalty.

def test_soft_policy_downweights_typed_visual_packet_without_excluding_it() -> None:
    # An asset_id packet receives -0.03 for a text-only question but remains
    # selectable when it has the highest relevance.

def test_required_packet_closure_overrides_preferred_packet_limit() -> None:
    # Required evidence and premises are retained even if there are already
    # eight packets; the pack remains answerable rather than failing closed.
```

- [ ] **Step 2: Run the policy tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_context_packer.py -k "soft_policy" -v
```

Expected: FAIL because no selection policy or per-packet decision payload exists.

- [ ] **Step 3: Implement the soft policy without semantic hard filtering**

Add immutable dataclasses in `context_packer.py`:

```python
@dataclass(frozen=True, slots=True)
class FinalContextSelectionPolicy:
    version: str
    preferred_max_packets: int = 8
    new_source_bonus: float = 0.05
    near_duplicate_penalty: float = 0.08
    visual_without_visual_intent_penalty: float = 0.03

@dataclass(frozen=True, slots=True)
class ContextSelectionDecision:
    evidence_id: str
    selected: bool
    base_quality: float
    source_bonus: float
    redundancy_penalty: float
    visual_penalty: float
    utility: float
    reason: str
```

Use this policy only after existing required-slot and premise-closure selection:

```python
utility = (
    candidate.quality
    + (policy.new_source_bonus if source_is_new else 0.0)
    - redundancy_penalty
    - visual_penalty
)
```

Rules:

- Keep `_deduplicate()` for exact source chunk/span identity only.
- Calculate near-duplicate similarity with a deterministic character 5-gram Jaccard function. Apply `near_duplicate_penalty` only at similarity `>= 0.96`, cap it at the configured amount, and set it to `0` when packets differ in `raw_value`, `locator.table_id`, `locator.figure_id`, or `locator.section`.
- Set `visual_penalty` only if `packet.source.asset_id` is populated and no `RequiredSlot.visual_policy` is `preferred` or `required`. Missing metadata means zero penalty.
- `preferred_max_packets` stops only optional additions. Required slot evidence and transitive premises always win; emit a decision reason `required_evidence_over_preferred_limit` rather than failing or dropping them.
- Do not remove, reject, or mark a packet insufficient because of near similarity, source repetition, visual provenance, or absent reranker score.

Add `selection_policy_version` and `selection_decisions` to `PackedEvidenceContext` with defaults so existing callers remain compatible.

- [ ] **Step 4: Run packer and dependent v9 tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_context_packer.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py -v
```

Expected: PASS. Verify no test asserts an unversioned behavior accidentally changed.

- [ ] **Step 5: Commit the isolated packer policy**

```powershell
git add data_base/agentic_v9/context_packer.py tests/test_agentic_v9_context_packer.py
git commit -m "feat(agentic-v9): add soft final context selection"
```

### Task 3: Activate and persist the R1 policy only for the evaluation v9 profile

**Files:**

- Modify: `evaluation/agentic_v9_campaign_runtime.py: pack(), _context_pack_projection()`
- Modify: `evaluation/campaign_schemas.py: V9ContextPack`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_campaign_api_runs_and_streams_results.py` only if its expected v9 payload is affected

**Interfaces:**

- Consumes: attempt-local `quality_by_evidence_id` from Task 1 and `FinalContextSelectionPolicy` from Task 2.
- Produces: profile `agentic_eval_v9_open_corpus_hybrid8_rerank8_top4_finalpack_r1`, context-policy version `v5_final_context_soft_pack_r1`, and a durable `V9ContextPack` selection projection.
- Compatibility: campaign results produced under the previous profile retain their existing profile/version and can still be rendered.

- [ ] **Step 1: Write failing activation and trace-schema tests**

Add a runtime test that executes a fake v9 run with rerank ranks `1` and `4`, verifies that `EvidenceContextPacker.pack()` receives a non-empty quality mapping, and asserts these durable trace fields:

```python
context_pack = trace["agentic_v9"]["context_pack"]
assert context_pack["selection_policy_version"] == "soft_final_pack_r1"
assert context_pack["candidate_count"] >= len(context_pack["packed_evidence_ids"])
assert any(row["base_quality"] > 0 for row in context_pack["selection_decisions"])
```

Add a backward-compatibility schema test that validates a historical context pack with only `packed_evidence_ids`, `dropped_evidence_ids`, and `token_count`.

- [ ] **Step 2: Run activation tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_api_runs_and_streams_results.py -k "context_pack or soft_final" -v
```

Expected: FAIL because current runtime does not pass rerank quality or persist selection policy information.

- [ ] **Step 3: Activate the policy at the evaluation boundary**

In the runtime's `pack()` closure, call:

```python
packed = packer.pack(
    packets,
    required_slots=contract,
    quality_by_evidence_id=state["evidence_quality_by_id"],
    selection_policy=FinalContextSelectionPolicy(version="soft_final_pack_r1"),
)
```

Update the evaluation-only profile and policy constants, not the legacy v8 or user-facing native RAG paths. Extend `_context_pack_projection()` and `V9ContextPack` additively:

```python
{
    "packed_evidence_ids": [...],
    "dropped_evidence_ids": [...],
    "token_count": 123,
    "selection_policy_version": "soft_final_pack_r1",
    "candidate_count": 12,
    "selection_decisions": [...],
}
```

The projection must contain no raw prompt, full source text, or unsafe provider payload.

- [ ] **Step 4: Run focused regression and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_context_packer.py tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_api_runs_and_streams_results.py -v
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/context_packer.py evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_schemas.py tests/test_agentic_v9_context_packer.py tests/test_agentic_v9_campaign_runtime.py
```

Expected: PASS. Record the exact test counts before committing.

- [ ] **Step 5: Commit Wave 1 activation**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_schemas.py tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_api_runs_and_streams_results.py
git commit -m "feat(evaluation): version soft v9 context packing"
```

### Wave 1 Runtime Validation Gate

Run a paired smoke using the same saved Evaluation Setup preset, batch size `1`, repeat count `1`, and no concurrent campaign:

```text
Questions: Q2, Q5, Q7, Q14, Q16
Mode: agentic-v9 only
Profile: agentic_eval_v9_open_corpus_hybrid8_rerank8_top4_finalpack_r1
```

Review the exported raw trace before a full 16-question run:

- Each final context pack has a non-empty `selection_decisions` list and a non-zero `base_quality` for reranked packets.
- No candidate is dropped solely for near similarity, visual provenance, source repetition, missing locator metadata, or missing reranker score.
- Exact duplicates are the only `duplicate_source_identity` drops.
- `candidate_count`, `packed_evidence_ids`, and rerank telemetry can be joined for selected textual chunks.
- Q2 no longer admits clearly unrelated visual-derived context when more relevant typed text evidence has higher utility; if metadata does not identify a visual excerpt, record it as a retrieval limitation rather than adding a prose heuristic.
- Mean RAGAS correctness across the five questions is not lower than the immediately preceding baseline by more than 0.03; no single question loses more than 0.10 correctness without trace evidence explaining the loss; mean faithfulness does not drop by more than 0.05.
- No failed runs and no increased LLM-call count. Token and latency changes are recorded but are not release gates at this smoke size.

If the gate fails, revert the three Wave 1 commits in reverse order and stop. Do not begin Wave 2.

---

## Wave 2 — Conditional Evidence-Aware Synthesis

**Entry condition:** Start only after Wave 1 is retained and raw traces show that relevant, sufficiently diverse packets reach the final prompt, yet answers still make unsupported cross-study rankings, merge incompatible metrics, or fail to answer an explicit question dimension.

### Task 4: Externalize a soft final-answer instruction and add a no-extra-call contract test

**Files:**

- Modify: `prompts/agentic_rag_prompts.json`
- Modify: `evaluation/agentic_v9_campaign_runtime.py: generate_final()`
- Modify: `tests/test_agentic_rag_prompts.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**

- Consumes: `format_agentic_rag_prompt("evaluation_v9_evidence_aware_synthesis")`.
- Produces: one system message for the existing `final_answer` provider call, plus `synthesis_prompt_version="evidence_aware_r1"` in the v9 trace payload.
- Compatibility: no new provider phase, no changed setup config, no exact quote/locator/slot match requirement, and the profile is bumped again for the Wave 2 treatment.

- [ ] **Step 1: Write failing prompt and invocation tests**

Add a prompt-registry test:

```python
prompt = format_agentic_rag_prompt("evaluation_v9_evidence_aware_synthesis")
assert "Answer the user question directly" in prompt
assert "different datasets" in prompt
assert "exact quote" not in prompt.lower()
```

Add a v9 runtime test using a recording provider:

```python
assert len(provider.calls) == 1
assert provider.calls[0].phase == "final_answer"
assert "different datasets" in provider.calls[0].messages[0]["content"]
assert trace["agentic_v9"]["synthesis_prompt_version"] == "evidence_aware_r1"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_rag_prompts.py tests/test_agentic_v9_campaign_runtime.py -k "evidence_aware" -v
```

Expected: FAIL because the prompt key and trace version do not exist.

- [ ] **Step 3: Add the soft, general-purpose prompt**

Add this zero-variable prompt entry to `prompts/agentic_rag_prompts.json`:

```json
"evaluation_v9_evidence_aware_synthesis": {
  "version": 1,
  "description": "Soft evidence-aware final synthesis for the versioned evaluation Agentic v9 path.",
  "required_variables": [],
  "template": "Answer the user question directly using the supplied evidence. Cover the question's requested dimensions before optional background. Separate directly supported facts from cross-source interpretation. When evidence reports different datasets, protocols, model variants, prompt settings, or metrics, name the condition and give only a qualified directional comparison unless a like-for-like comparison is explicitly supported. State uncertainty or missing evidence plainly. Do not invent rankings, causal claims, values, or sources. This is a soft synthesis instruction: do not require exact quotes, locator matches, or one evidence packet per sentence. Cite no source outside the supplied evidence."
}
```

Import `format_agentic_rag_prompt` in the evaluation runtime and use this prompt as the existing final call's system content. Persist `synthesis_prompt_version` next to `context_policy_version`; bump the Wave 2 evaluation profile to `agentic_eval_v9_open_corpus_hybrid8_rerank8_top4_finalpack_r1_evidenceaware_r1`.

- [ ] **Step 4: Run prompt, runtime, and full impacted tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_rag_prompts.py tests/test_prompt_loader.py tests/test_agentic_v9_context_packer.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_execution_core.py -v
.\.venv\Scripts\python.exe -m ruff check evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_rag_prompts.py tests/test_agentic_v9_campaign_runtime.py
```

Expected: PASS. Confirm that all fake-provider tests still observe exactly one `final_answer` call.

- [ ] **Step 5: Commit Wave 2 independently**

```powershell
git add prompts/agentic_rag_prompts.json evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_rag_prompts.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "feat(agentic-v9): add evidence-aware final synthesis"
```

### Wave 2 Runtime Validation Gate

Repeat the same five-question smoke with the identical Setup preset and Wave 1 policy. Retain Wave 2 only when:

- The trace still has one final LLM call per completed run and no new provider phase.
- Context pack selection diagnostics are unchanged from Wave 1 except for normal retrieval/model variance.
- Q2-like answers distinguish per-paper numeric evidence from the qualified cross-paper conclusion.
- Mean correctness is non-decreasing versus retained Wave 1; mean faithfulness does not decrease by more than 0.03; no new failed run occurs.

If retained, run the complete 16-question Agentic-v9 campaign once with the same Setup and export raw traces. Compare only campaigns sharing the same model setup and explicit `execution_profile`; report correctness, faithfulness, relevancy, token accounting, P50/P95 latency, and failed-run count. If Wave 2 fails its gate, revert only its commit; Wave 1 remains independently usable.

## Documentation and Review Checklist

- [ ] Update `agent.md`'s continuous-learning note only after a Wave is retained by runtime validation; include the profile name and the fact that source diversity is a soft ordering signal, not a retrieval authorization gate.
- [ ] Do not update benchmark-release claims from a five-question smoke.
- [ ] Before reporting a Wave complete, run its focused tests and Ruff command again after the final diff, then report exact output.
- [ ] Preserve untracked `data/` and pre-existing `docs/superpowers/` drafts; do not stage them.
