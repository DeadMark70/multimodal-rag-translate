# Agentic v9 Observability and Capability Calibration

## Baseline

Campaign `a2a9d41b-e93f-4412-89e0-1984bdd7577d` completed all 16
Agentic v9 runs with no failures:

- correctness: 48.8%;
- faithfulness: 77.0%;
- relevancy: 66.7%;
- total runtime tokens: 34,107;
- mean latency: 15.4 seconds.

The raw export exposes several measurement gaps:

- all 85 recorded retrieval chunks have no rerank score and no observable rank
  change;
- the stored execution profile still says `rerank_off`, while the current v9
  runtime requests reranking;
- all 16 LLM usage rows have phase `unknown`;
- all expected-evidence matches are false;
- graph capability is recorded as unsatisfied for 10 questions and visual
  capability for five questions;
- Q2, Q9, Q14, Q15, and Q16 finish as `qualified_partial`.

The next work must remain incremental. Wave A changes telemetry only. Wave B is
a separate quality experiment and may start only after Wave A proves what the
runtime actually did.

## Goals

1. Make reranker execution and fallback behavior auditable per retrieval task.
2. Preserve authoritative total-token accounting while reporting phase
   attribution only when it can be reconciled.
3. Distinguish a real expected-source miss from a filename-to-UUID mapping gap.
4. Stop optional or unavailable Graph/Visual capabilities from unnecessarily
   downgrading text-grounded answers.
5. Validate the quality change on a five-question smoke set before any full
   benchmark.

## Non-goals

- Do not change hybrid retrieval candidate count, rerank target count, prompts,
  atomic-slot generation, corrective retrieval, or answer synthesis in Wave A.
- Do not build a new GraphRAG implementation or visual asset ingestion pipeline.
- Do not increase the number of LLM calls.
- Do not claim benchmark improvement from a single five-question smoke run.

## Wave A: Behavior-neutral telemetry

### Reranker diagnostics

The v9 runtime already creates one retrieval diagnostic projection per task.
Extend this projection with stable identifiers needed to join it to recorded
chunks:

- task ID;
- chunk ID and document ID;
- content hash;
- candidate and selected counts;
- reranker status: `executed`, `fallback`, or `not_available`;
- fallback reason;
- pre-rerank rank;
- post-rerank rank;
- rerank score when the reranker returned one.

The result-level observability recorder must consume this projection instead of
always assigning the final list index to both ranks. It must never synthesize a
score. When a diagnostic cannot be joined to a final context, the stored chunk
must remain `not_instrumented`.

The execution-profile identifier must describe the effective treatment:
Hybrid retrieve 8, rerank at most 8, select at most 4 per retrieval task,
fail-soft to Hybrid top 4.

### Token phase attribution

The aggregate provider usage remains the official accounting source. Phase
attribution may use v9 budget/usage records only when:

1. the record represents an executed provider call;
2. its phase is known;
3. its token usage is authoritative; and
4. the sum reconciles exactly with the official aggregate usage.

If reconciliation fails, total accounting remains complete while phase
attribution remains partial with an explicit reason. The implementation must
not duplicate the aggregate usage as an additional phase call or fabricate
zero-token phases.

### Expected-source identity

Expected-source matching must resolve test-case filenames to canonical document
UUIDs using the existing evaluation source resolver. The exported status must
distinguish:

- `matched`;
- `not_matched`;
- `identity_unresolved`.

An unresolved identity must not be reported as a retrieval miss and must not
affect runtime source authorization or retrieval.

### Export contract

Raw and redacted exports must expose the same diagnostic fields, except that
excerpts and prompts continue to obey the existing redaction controls. Export
must not require full prompt capture.

## Wave B: Capability requirement calibration

Wave B changes only whether an unavailable capability is required. It does not
change retrieval, reranking, or synthesis.

### Graph policy

- `required` only for an explicitly graph-relational contract with eligible
  graph source evidence.
- `optional` for multi-hop, comparison, multi-document, and structured lookup
  routes that can be answered from text evidence.
- `not_requested` when no graph relationship is requested.
- A missing optional graph result must preserve text evidence and must not
  cause `qualified_partial`.

### Visual policy

- `required` only when the question explicitly needs a figure/table visual and
  an eligible authorized asset exists.
- `optional` when parsed text already contains the requested table or figure
  content.
- `unavailable` when ingestion provides no eligible asset; this preserves text
  evidence and records a capability gap.
- Visual failure must never clear text contexts.

The final response may remain `qualified_partial` for missing answer evidence,
but not solely because an optional or unavailable capability produced no result.

## Validation

### Wave A

- Existing v9 answers and selected contexts remain behaviorally unchanged.
- Executed reranks show non-null scores and truthful rank transitions.
- Reranker fallback shows a safe reason and retains four Hybrid-ranked chunks.
- Official total tokens do not change because of observability recording.
- Phase status is complete only after exact reconciliation.
- Existing source authorization tests continue to pass.

### Wave B smoke

Run Q2, Q9, Q14, Q15, and Q16 once with batch size one and the same model setup.
Compare against the stored baseline:

- 5/5 runs complete without execution failure;
- no run has zero text contexts;
- Q9 is not graph-required or visual-required;
- unavailable Graph/Visual stages do not remove text evidence;
- `qualified_partial` is retained only when answer evidence is actually missing;
- correctness and faithfulness are treated as regression signals, not formal
  release claims.

Only after this smoke passes may a 16-question paired evaluation be proposed.

## Deferred Work

Targeted missing-subquestion corrective retrieval is the next possible quality
experiment. It is intentionally excluded from this design so that capability
calibration can be measured independently.

