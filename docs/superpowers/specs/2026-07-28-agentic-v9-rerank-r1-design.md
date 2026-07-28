# Agentic v9 Rerank R1 Design

## Goal

Reduce Agentic v9 evidence-context tokens without increasing retrieval breadth
or changing Naive behavior.

The production Agentic v9 retrieval path becomes:

```text
Hybrid retrieve 8 per retrieval task
→ Jina rerank 8
→ select 4 per retrieval task
→ evidence processing and generation
```

## Scope

- Change only the Agentic v9 retrieval adapter.
- Reuse the existing `filter_and_rerank_retrieval` boundary.
- Keep the open-user-corpus behavior introduced by Wave S.
- Keep Naive as plain FAISS retrieval with reranking disabled.
- Do not change routing, atomic slots, Graph, Visual, corrective retrieval,
  evidence synthesis, provider configuration, or evaluation scoring.
- Do not introduce a parallel execution profile or experiment flag. R1 replaces
  the current Agentic v9 retrieval selection behavior.

## Selection policy

Each retrieval task independently:

1. Uses the existing per-user hybrid retriever with `k=8`.
2. Sends at most eight retrieved chunks to the initialized local Jina reranker.
3. Selects the four highest-ranked chunks for downstream processing.
4. Never increases candidate breadth beyond eight.

The four-chunk limit is per retrieval task, not a global four-chunk limit for
the entire run. Multi-hop and multi-document routes may therefore retain four
chunks for each compiled task.

## Fail-soft policy

If the reranker is unavailable, times out, raises, returns no measured results,
or cannot complete after its existing device fallback:

- preserve the original Hybrid ranking;
- select the original top four chunks;
- do not return all eight chunks;
- do not fail the retrieval task or campaign unit;
- do not fabricate rerank scores.

The fallback path must continue enforcing the concrete user-authorized document
scope created by Wave S.

## Observability

The retrieval projection must make the following facts inspectable:

- candidate count;
- selected count;
- reranker status (`executed` or `fallback`);
- safe fallback reason when applicable;
- original retrieval rank;
- post-rerank rank;
- measured rerank score, or `null` when unavailable.

These fields must describe each retrieval task independently. Existing score
fields must remain `null` rather than displaying zero when the reranker did not
produce a measurement.

## Baseline behavior

Naive remains:

```text
Plain FAISS retrieve 6
→ no HyDE
→ no Multi-Query
→ no reranking
→ generation
```

No Naive preset or generic baseline behavior may be changed by R1.

## Verification

Automated tests must prove:

1. Successful reranking selects four from eight in reranked order.
2. An unavailable reranker selects the original Hybrid top four.
3. A reranker exception selects the original Hybrid top four.
4. Empty reranker output selects the original Hybrid top four.
5. Multiple retrieval tasks each enforce their own four-chunk limit.
6. Fallback telemetry contains no fabricated score.
7. Naive remains `enable_reranking=False` and `plain_mode=True`.

After local tests, deployment validation should begin with a small smoke run.
Only after the smoke output confirms executed/fallback status and four selected
chunks per task should the 16-question paired evaluation be run.

## Success criteria

- No Agentic v9 campaign unit fails because reranking is unavailable.
- Every Agentic v9 retrieval task supplies at most four vector chunks after the
  reranking boundary.
- Server trace proves whether reranking executed or fell back.
- Naive output and retrieval configuration remain unchanged.
- The following evaluation compares the new results with the completed Wave S
  run for correctness, faithfulness, relevancy, tokens, and latency.
