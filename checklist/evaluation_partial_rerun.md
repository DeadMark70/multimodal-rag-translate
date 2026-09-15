# Partial rerun verification

- Numeric source IDs resolve only to existing evidence.
- Invalid audit mappings retain a valid requested drill-down and keep diagnostics.
- Missing-only scoring preserves existing scores, including zero.
- Selected question/mode reruns do not schedule other modes for scoring.
- Different answer rows can share a batch without sharing score signatures.
- Missing execution/score coverage remains visible after a successful subset.
- A successful retry supersedes that work item's historical failure.

Implementation and request examples: [evaluation_partial_rerun.md](../agentlog/evaluation_partial_rerun.md).
