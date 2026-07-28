# Agentic v9 Wave S: Open-corpus evaluation

## Goal

Make official Agentic v9 evaluation search the evaluated user's full authorized
corpus. A test case's `source_docs`/`expected_sources` remain post-hoc labels
only and must not narrow runtime retrieval.

## Boundaries

- Keep the per-user document ACL as the runtime source boundary.
- Keep `expected_sources` in `BenchmarkExecutionResult`.
- Keep direct/oracle runtime calls with an explicit document list supported for
  diagnostics.
- Do not change Naive, reranking, Graph, Visual, slot planning, or synthesis.
- Give open-corpus v9 runs a distinct execution profile and trace policy.

## Runtime contract

- `authorized_doc_ids=None` means: resolve all document IDs owned by `user_id`.
- An explicit list means: retain the existing bounded/oracle source behavior.
- Retrieval and evidence admission continue to use a concrete authorized ID
  list, so downstream fail-closed behavior is unchanged.

## Acceptance

- Official v9 campaign execution passes no expected-source restriction.
- The result still persists the test case's expected sources.
- Open-corpus admission resolves only documents owned by the evaluated user.
- Existing explicit-source v9 behavior remains compatible.
- Naive remains unchanged and reranking remains disabled.
