# Question Comparison and v9 Phase Accounting Design

## Problem

Question Analysis currently selects the quality winner as the comparison target. When `naive` is that winner, it suppresses a real `agentic` versus `naive` comparison as `comparison_mode_missing`. Separately, v9 campaign execution exports only one aggregate token usage value, so durable accounting records the usage as `unclassified`.

## Decision

Question Analysis will retain `best_quality_mode` as descriptive metadata, but always calculate deltas from the explicit `agentic` target against the `naive` baseline. A missing agentic run remains `comparison_mode_missing`; incomplete quality or accounting remains fail-closed.

Every v9 provider invocation will propagate its real v9 phase through the accounting context. The execution worker already derives its accounting summary from those provider-call records; its later Campaign Engine observability projection is separate and does not feed research token accounting. Provider-call accounting remains authoritative; missing provider usage continues to produce partial accounting rather than invented totals.

## Non-goals

- Do not alter formal release-gate eligibility, benchmark identity, or v8-arm requirements.
- Do not infer graph traversal from the v9 phase accounting records.
- Do not change the existing explicit `best_quality_mode` display field.
