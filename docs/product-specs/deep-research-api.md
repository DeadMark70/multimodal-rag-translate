# Deep Research API

## User Outcomes

- Ask ordinary questions synchronously or through streamed progress.
- Plan and execute multi-step Deep Research.
- Request one-shot research answers where the backend owns the full plan/execute cycle.

## Acceptance Notes

- All chat modes use Gemini 3.1 Flash-Lite with Medium thinking, including nested
  rewriting/planning/graph/synthesis calls. Actual Gemini HTTP requests omit
  thinking_budget, temperature, top_p and top_k. Evaluation Center presets are
  independent of this chat policy.
- `rag_mode="agentic_v10"` selects the shared V10 pipeline for ask and ask/stream.
  It supports the same history, selected document scope, citations and cancellation.
  V10 owns its retrieval settings; ordinary strategy flags and Self-RAG scoring
  do not change or add scoring calls to this workflow.
- V10 replies identify `agentic_execution_version="v10"`. Each citation keeps
  its original `[Ref N]` in optional `reference_id`; retained evidence is never
  renumbered after a drill-down. These optional fields are absent from standard
  replies when not applicable.

- `/rag/ask` and `/rag/ask/stream` should stay behaviorally aligned.
- Planning and execution endpoints should keep their response contracts explicit.
- Long-running execution should surface incremental progress through dedicated stream/event paths rather than ad hoc logging only.

## Server acceptance for V10 chat

Rebuild both backend and frontend, preserving the installed PyTorch versions as
described in [deployment instructions](../dependency-upgrade-stage1.md). Ask one question
in Native, Advanced, Graph and the existing Agentic modes. Confirm a follow-up,
source citations, completion and cancellation. Verify request logs show
`gemini-3.1-flash-lite` / Medium and no deprecated sampling/budget values.
Evaluation Center presets should retain their previous settings. For Agentic
V10, ask a comparison question, then a pronoun-based follow-up; verify original
reference IDs and page links. Test stopping while it is searching/answering.
