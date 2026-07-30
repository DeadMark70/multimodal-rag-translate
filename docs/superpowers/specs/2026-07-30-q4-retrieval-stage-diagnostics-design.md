# Q4 Retrieval Stage Diagnostics Design

## Goal

Make a single v9 Q4 evaluation able to distinguish a missing/unindexed source
from a hybrid-retrieval recall miss and a reranker selection miss, without
changing retrieval, ranking, context packing, prompts, token use, or answers.

## Scope and constraints

- Runtime must not use evaluation `expected_sources` as a retrieval filter,
  authorization rule, or query hint.
- The diagnostic stores only stable document and chunk identifiers, counts, and
  the existing reranker status. It must not add document text to a new trace.
- Native RAG and every v9 decision outcome remain unchanged.
- The existing soft candidate-diversity policy remains observationally visible;
  it is not strengthened into a hard document-diversity gate.

## Design

The generic retrieval boundary already knows the capped hybrid candidates and
whether `tail_source_diversity_r1` replaced part of the candidate tail. It
will emit a compact `candidate_diversification` record containing the policy,
whether it applied, the document IDs represented before the tail, document
IDs admitted by the tail, and the ordered document IDs sent to the reranker.
It also records the ordered IDs returned by hybrid retrieval before candidate
capping. This record is copied into the existing
`agentic_v9_reranking` annotation for selected documents.

The v9 runtime's retrieval diagnostic projection will persist that same record
alongside the existing candidate count and selected reranker rows. The
campaign-export path can therefore show, for a task such as Q4:

1. which documents survived candidate capping;
2. whether an alternative document was deliberately admitted before reranking;
3. which documents the reranker ultimately selected.

## Interpretation after one Q4 run

- No EfficientMedNeXt document in hybrid/capped candidates: investigate
  ingestion/index coverage or general query recall.
- It appears before reranking but not among selected rows: investigate reranker
  calibration; do not add a hard gate yet.
- It appears in neither the corpus nor candidate records: upload/re-index the
  actual EfficientMedNeXt source and verify its chunks and metadata.

No interpretation may rely solely on the test-case expected source list,
because that list is benchmark metadata rather than runtime knowledge.

## Error handling and privacy

If a document lacks a stable ID, diagnostics omit it rather than inventing a
claim that a particular source was selected. Existing redacted exports keep
the normal document-ID policy and do not add prompts or source text.

## Testing

- A retrieval-boundary test proves the diversity record identifies the
  pre-tail and admitted alternate source.
- A v9 runtime test proves the record survives selected-document annotation
  and diagnostic projection.
- Existing tests prove disabled diversity and Native RAG behavior remain
  unchanged.
