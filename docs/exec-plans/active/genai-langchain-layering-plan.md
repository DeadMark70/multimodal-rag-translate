# Google GenAI / LangChain Gemini Layering Plan

## Goal

Define a stable integration boundary for this repo so future Gemini work uses:

- direct `google-genai` for control-plane / platform-facing operations
- `langchain-google-genai` for execution-plane LLM and embedding flows that are already embedded in the LangChain-based RAG architecture

This is a handoff plan for a follow-up implementation/refactor agent. It is decision-complete enough to execute without re-deriving the boundaries.

## Current Repo Reality

### Direct `google-genai` today

- [evaluation/model_discovery.py](D:/flutterserver/pdftopng/evaluation/model_discovery.py)
  - Uses `genai.Client`
  - Performs model listing / capability filtering
  - Is control-plane code, not response-generation runtime

### `langchain-google-genai` today

- [core/llm_factory.py](D:/flutterserver/pdftopng/core/llm_factory.py)
  - Central `ChatGoogleGenerativeAI` factory
  - Purpose-specific config, runtime override, model-family-aware thinking controls
- [data_base/vector_store_manager.py](D:/flutterserver/pdftopng/data_base/vector_store_manager.py)
  - Central `GoogleGenerativeAIEmbeddings` initialization and usage
- Runtime call sites that already depend on LangChain message/document abstractions:
  - [data_base/RAG_QA_service.py](D:/flutterserver/pdftopng/data_base/RAG_QA_service.py)
  - [data_base/query_transformer.py](D:/flutterserver/pdftopng/data_base/query_transformer.py)
  - [data_base/context_enricher.py](D:/flutterserver/pdftopng/data_base/context_enricher.py)
  - [data_base/proposition_chunker.py](D:/flutterserver/pdftopng/data_base/proposition_chunker.py)
  - [pdfserviceMD/translation_chunker.py](D:/flutterserver/pdftopng/pdfserviceMD/translation_chunker.py)
  - [image_service/translation_service.py](D:/flutterserver/pdftopng/image_service/translation_service.py)
  - [multimodal_rag/image_summarizer.py](D:/flutterserver/pdftopng/multimodal_rag/image_summarizer.py)
  - [graph_rag/extractor.py](D:/flutterserver/pdftopng/graph_rag/extractor.py)
  - [graph_rag/community_builder.py](D:/flutterserver/pdftopng/graph_rag/community_builder.py)
  - [graph_rag/local_search.py](D:/flutterserver/pdftopng/graph_rag/local_search.py)
  - [graph_rag/global_search.py](D:/flutterserver/pdftopng/graph_rag/global_search.py)
  - [graph_rag/generic_mode.py](D:/flutterserver/pdftopng/graph_rag/generic_mode.py)
  - [agents/planner.py](D:/flutterserver/pdftopng/agents/planner.py)
  - [agents/evaluator.py](D:/flutterserver/pdftopng/agents/evaluator.py)
  - [agents/synthesizer.py](D:/flutterserver/pdftopng/agents/synthesizer.py)
  - [core/summary_service.py](D:/flutterserver/pdftopng/core/summary_service.py)

### Provider boundary today

- [core/providers.py](D:/flutterserver/pdftopng/core/providers.py) is the runtime abstraction boundary
  - fake-vs-real provider switching
  - test safety
  - current "real" provider delegates to `core.llm_factory.get_llm(...)`

This file is the right seam to preserve.

## Target Layering

### Layer 1: Platform / Control Plane

Use direct `google-genai` here.

Characteristics:

- lists models
- probes capabilities
- inspects account-visible Gemini features
- uses raw SDK clients or API-version-specific options
- may need low-level `http_options`, preview flags, or exact SDK response fields
- is not tightly coupled to LangChain messages/documents/chains

Modules that should be direct `google-genai`:

- [evaluation/model_discovery.py](D:/flutterserver/pdftopng/evaluation/model_discovery.py)
- any future feature such as:
  - model capability registry
  - account quota / model-availability probing
  - experimental Gemini feature toggles
  - direct file/upload/cache APIs if introduced later

### Layer 2: Runtime LLM Execution Plane

Keep `langchain-google-genai` here.

Characteristics:

- consumes or returns LangChain messages/documents
- participates in RAG pipelines, retrievers, graph extraction, prompt chains, async `ainvoke`, or structured-output wrappers already shaped around LangChain
- depends on existing response normalization, usage extraction, and provider test doubles

Modules that should stay on `langchain-google-genai`:

- [core/llm_factory.py](D:/flutterserver/pdftopng/core/llm_factory.py)
- all `core.providers.get_llm(...)` consumers
- all GraphRAG LLM execution modules
- all agent/planner/evaluator/synthesizer flows
- all translation and multimodal prompt flows

### Layer 3: Runtime Embedding Plane

Keep `langchain-google-genai` here for now.

Characteristics:

- tightly bound to FAISS + LangChain vector-store APIs
- currently uses `GoogleGenerativeAIEmbeddings` directly in vector store creation / loading
- changing this would be an architectural migration, not a package cleanup

Modules that should stay on `langchain-google-genai`:

- [data_base/vector_store_manager.py](D:/flutterserver/pdftopng/data_base/vector_store_manager.py)
- [graph_rag/entity_resolver.py](D:/flutterserver/pdftopng/graph_rag/entity_resolver.py) via `get_embeddings()`

## Concrete Module Decisions

### Keep direct `google-genai`

- [evaluation/model_discovery.py](D:/flutterserver/pdftopng/evaluation/model_discovery.py)
  - Keep as the canonical direct SDK integration
  - Allowed to use `genai.Client`, raw pager objects, and `google.genai.types`

### Keep `langchain-google-genai`

- [core/llm_factory.py](D:/flutterserver/pdftopng/core/llm_factory.py)
  - Remains the only place that instantiates `ChatGoogleGenerativeAI`
- [core/providers.py](D:/flutterserver/pdftopng/core/providers.py)
  - Remains the only runtime provider entrypoint used by business logic
- [data_base/vector_store_manager.py](D:/flutterserver/pdftopng/data_base/vector_store_manager.py)
  - Remains the only place that instantiates `GoogleGenerativeAIEmbeddings`
- All current `get_llm(...)` call sites
  - Do not bypass provider registry
  - Do not instantiate direct `genai.Client` inside business-logic modules

### Do not mix inside the same module

Disallow modules that do both of the following at once:

- instantiate direct `genai.Client`
- call `core.providers.get_llm(...)` / `ChatGoogleGenerativeAI`

If a module needs both control-plane and runtime behavior, split it into:

- a direct-SDK helper module
- a runtime LangChain-facing orchestration module

## Required Abstractions

### 1. Keep `core.providers.py` as the runtime facade

No follow-up agent should replace downstream imports of `get_llm(...)` with direct SDK calls.

Required rule:

- application/runtime code imports `get_llm` from [core/providers.py](D:/flutterserver/pdftopng/core/providers.py)
- only provider/factory internals know whether the real implementation is LangChain-backed

### 2. Introduce a small direct-SDK service layer if control-plane work grows

If more direct `google-genai` features are added, create:

- `core/google_genai_client.py`
  - cached client creation
  - env resolution
  - API version selection
  - shared timeout / http options
- `core/google_genai_capabilities.py`
  - model listing
  - capability normalization
  - future feature probing

Then move [evaluation/model_discovery.py](D:/flutterserver/pdftopng/evaluation/model_discovery.py) to depend on that helper instead of owning client lifecycle itself.

### 3. Keep embedding ownership centralized

Do not let multiple modules instantiate embeddings independently.

Required rule:

- [data_base/vector_store_manager.py](D:/flutterserver/pdftopng/data_base/vector_store_manager.py) remains the canonical embedding owner
- other modules obtain embeddings through `get_embeddings()`

## Implementation Plan for Another Agent

### Phase 1: Normalize boundaries without behavior change

1. Add a short architecture note to:
   - [core/providers.py](D:/flutterserver/pdftopng/core/providers.py)
   - [core/llm_factory.py](D:/flutterserver/pdftopng/core/llm_factory.py)
   - [data_base/vector_store_manager.py](D:/flutterserver/pdftopng/data_base/vector_store_manager.py)
   - [evaluation/model_discovery.py](D:/flutterserver/pdftopng/evaluation/model_discovery.py)
2. Document the boundary:
   - direct `google-genai` = control plane
   - `langchain-google-genai` = runtime generation + embeddings
3. Add a lightweight static regression test or grep-based test that fails if non-approved modules instantiate `genai.Client` directly.

### Phase 2: Extract shared direct-SDK helper

1. Create `core/google_genai_client.py`
2. Move:
   - API key resolution
   - client caching
   - timeout / `HttpOptions`
   out of [evaluation/model_discovery.py](D:/flutterserver/pdftopng/evaluation/model_discovery.py)
3. Keep `evaluation/model_discovery.py` focused on model normalization only

### Phase 3: Add enforcement tests

Add tests that assert:

- only approved control-plane modules import `from google import genai`
- runtime modules still go through `core.providers.get_llm(...)`
- embeddings are still sourced centrally

## Explicit Non-Goals

Another agent should **not** do these in this task:

- replace [core/llm_factory.py](D:/flutterserver/pdftopng/core/llm_factory.py) with raw `google-genai`
- replace [data_base/vector_store_manager.py](D:/flutterserver/pdftopng/data_base/vector_store_manager.py) embeddings with direct SDK embeddings
- rewrite GraphRAG extraction/search flows away from LangChain
- rewrite prompt/message code from `HumanMessage` / LangChain message abstractions to raw SDK request payloads

Those are separate architecture migrations.

## Acceptance Criteria

- Direct `google-genai` usage exists only in approved control-plane modules.
- All runtime LLM consumers still enter through [core/providers.py](D:/flutterserver/pdftopng/core/providers.py).
- All embedding creation still enters through [data_base/vector_store_manager.py](D:/flutterserver/pdftopng/data_base/vector_store_manager.py).
- No module mixes direct `genai.Client` creation with runtime `get_llm(...)` usage.
- Existing backend tests remain green, especially:
  - provider-registry tests
  - llm-factory override tests
  - evaluation model-discovery tests
  - GraphRAG tests

## Recommendation

For this repo, the correct architecture is not "pick one."

The correct architecture is:

- `google-genai` for platform/control-plane integration
- `langchain-google-genai` for runtime/orchestration integration

That gives access to the current official SDK without forcing a large, high-risk rewrite of the LangChain-based RAG system.
