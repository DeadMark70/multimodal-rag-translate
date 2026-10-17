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
- This first step changes the chat model configuration only; the separate V10
  chat workflow migration is pending server acceptance.

- `/rag/ask` and `/rag/ask/stream` should stay behaviorally aligned.
- Planning and execution endpoints should keep their response contracts explicit.
- Long-running execution should surface incremental progress through dedicated stream/event paths rather than ad hoc logging only.

## Server acceptance for the chat model step

Rebuild only the backend, preserving the installed PyTorch versions as described
in [deployment instructions](../dependency-upgrade-stage1.md). Ask one question
in Native, Advanced, Graph and the existing Agentic modes. Confirm a follow-up,
source citations, completion and cancellation. Verify request logs show
`gemini-3.1-flash-lite` / Medium and no deprecated sampling/budget values.
Evaluation Center presets should retain their previous settings. V10 chat
integration follows after this step is accepted.
