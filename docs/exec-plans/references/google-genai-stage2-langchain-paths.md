# Google GenAI Stage 2 LangChain Path Memo

## Scope
This memo records the Gemini runtime paths that still intentionally remain on `langchain-google-genai` after stage 1 removed direct `google-generativeai` usage.

## Current Remaining Paths
- `core/llm_factory.py`
  - Uses `ChatGoogleGenerativeAI` as the central Gemini chat/model factory.
  - Feeds most backend LLM purposes, including RAG QA, planning, GraphRAG extraction/community flows, and evaluation-time model overrides.
- `data_base/vector_store_manager.py`
  - Uses `GoogleGenerativeAIEmbeddings` for text embedding generation in the FAISS-backed retrieval store.
- `tests/test_llm_factory_override.py`
  - Patches `ChatGoogleGenerativeAI` and therefore documents the current abstraction boundary.

## Why Stage 1 Did Not Change These
- `langchain-google-genai==4.1.1` already depends on `google-genai` in the current environment.
- Replacing these paths would be a runtime-architecture change, not a direct deprecated-SDK removal.
- Keeping them stable avoided a much larger regression surface across GraphRAG, retrieval, evaluation, and streaming flows.

## Stage 2 Decision Inputs
- Keep LangChain if the current abstraction still provides needed prompt/model orchestration, structured output, and embeddings without blocking new Gemini features.
- Consider direct `google-genai` only if a concrete need appears, such as unsupported SDK features, lower-level transport control, or integration simplification in a specific subsystem.
- If stage 2 is approved, split it into separate tracks:
  - chat/model factory migration
  - embeddings/vector-store migration
  - GraphRAG-specific runtime migration

## Recommendation
Do not schedule stage 2 as a blind cleanup. Open it only when there is a concrete product or reliability reason to bypass `langchain-google-genai`.
