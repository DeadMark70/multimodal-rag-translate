# Codebase Analysis & Optimization Report

> **Date:** 2025-12-29
> **Status:** Detailed Analysis
> **Scope:** Full Project (Core, RAG, PDF, GraphRAG, Agents)

## Executive Summary

The codebase is well-structured, modern, and follows robust engineering practices (Type Hints, Pydantic validation, Async IO). The implementation of **Phase 5 (GraphRAG)** is complete and integrated, not just planned.

Key strengths:
- **Architecture**: Clear separation of Routers, Services, and Stores.
- **Async Handling**: Correct usage of `run_in_threadpool` for CPU-bound tasks (OCR, PDF generation).
- **Modularity**: New features (GraphRAG) are well-isolated in their own modules.

Primary areas for optimization focus on **User Experience (UX) latency**, **caching strategies**, and **robustness of LLM outputs**.

---

## Detailed Analysis by Module

### 1. 📂 Core (`core/`)

#### **Current Status**
- **LLM Factory**: Uses `lru_cache` and purpose-based routing. Excellent.
- **Auth**: Centralized Supabase JWT validation.

#### **Optimizations**
- **[Config] Externalize Model Configuration**:
  - *Issue*: Models (`gemini-2.5-flash`, `gemma-3-27b-it`) are hardcoded in `llm_factory.py`.
  - *Fix*: Move model names to `config.env` or a `config.yaml` to allow changing models without code deployment.
- **[Auth] Caching User Validation**:
  - *Issue*: `get_current_user_id` calls Supabase API on *every* request.
  - *Fix*: Implement a short-lived memory cache (e.g., 60s) for validated tokens to reduce latency and API calls.

### 2. 📂 PDF Service (`pdfserviceMD/`)

#### **Current Status**
- **Pipeline**: Upload -> OCR (Sync) -> Translate (Async) -> PDF Gen (Sync) -> Return.
- **Post-processing**: RAG indexing and GraphRAG extraction run in background tasks.

#### **Optimizations**
- **[UX] Non-blocking Upload Flow**:
  - *Issue*: User waits for OCR + Translation + PDF Generation before getting a response. Large files causes timeouts.
  - *Fix*: Change `upload_pdf_md` to return `doc_id` immediately after file save. Let frontend poll `/status` or use SSE for progress. Move *all* heavy processing (OCR/Translation) to background tasks.
- **[Robustness] PDF Generation Retry**:
  - *Issue*: `markdown_to_pdf` is CPU heavy and can fail.
  - *Fix*: Add a retry mechanism for PDF generation specifically, as it's the final step and prone to formatting errors.

### 3. 📂 RAG Database (`data_base/`)

#### **Current Status**
- **Logic**: Handles HyDE, Multi-Query, Reranking, and GraphRAG context.
- **Store**: FAISS loading from disk.

#### **Optimizations**
- **[Performance] Vector Store Caching**:
  - *Issue*: `get_user_retriever` loads FAISS index from disk on every request.
  - *Fix*: Implement a global `VectorStoreCache` singleton that keeps the most recently used user indices in memory (LRU eviction).
- **[Refactor] Prompt Templates**:
  - *Issue*: Large f-strings in `RAG_QA_service.py` mix logic and presentation.
  - *Fix*: Extract prompts to `data_base/prompts.py` or `core/prompts.py` for better maintainability and versioning.
- **[Architecture] Database Repository Pattern**:
  - *Issue*: Direct `supabase.table(...).execute()` calls in services and routers.
  - *Fix*: Create a `repository/` layer to abstract Supabase calls. Makes testing easier (mocking repositories) and allows switching DBs later.

### 4. 📂 GraphRAG (`graph_rag/`)

#### **Current Status**
- **Implementation**: Fully implemented with NetworkX, Gemini Flash extraction, and Community detection.
- **Storage**: Pickle-based per-user graph.

#### **Optimizations**
- **[Robustness] Structured Output**:
  - *Issue*: `extractor.py` parses raw JSON strings from LLM, which is brittle.
  - *Fix*: Use LangChain's `.with_structured_output(PydanticModel)` if supported by the Google provider, or robust JSON repair parsers.
- **[Performance] Graph Lazy Loading**:
  - *Issue*: `GraphStore` loads the full pickle on initialization.
  - *Fix*: Verify if `GraphStore` is initialized per-request. If so, apply similar caching as Vector Store to avoid deserializing large graphs frequently.
- **[Feature] Incremental Updates**:
  - *Issue*: Rebuilding implies full reprocessing.
  - *Fix*: The current `add_node` logic supports merging, which is good. Ensure `modified_at` timestamps track when nodes were last verified.

### 5. 📂 Agents (`agents/`)

#### **Current Status**
- **Planner**: Uses Regex to parse sub-tasks.
- **Synthesizer**: Combines results.

#### **Optimizations**
- **[Robustness] Pydantic Parser**:
  - *Issue*: Regex parsing in `planner.py` (`r'^(\d+)[\.\)]...'`) fails if LLM changes numbering format.
  - *Fix*: Use Pydantic Output Parser or Structured Output for guaranteed schema compliance.

### 6. 📂 Conversations (`conversations/`)

#### **Current Status**
- **CRUD**: Standard endpoints.

#### **Optimizations**
- **[Error Handling] Global Exception Handler**:
  - *Issue*: Repeated `try...except PostgrestAPIError` in every endpoint.
  - *Fix*: Implement a FastAPI `exception_handler` for `PostgrestAPIError` in `main.py`. This removes boilerplate from all routers.

---

## Prioritized Action Plan

1.  **Refactor PDF Upload to Async**: This has the biggest impact on perceived performance and reliability for large files.
2.  **Implement Vector/Graph Store Caching**: Critical for scaling RAG to more users/documents without thrashing disk I/O.
3.  **Harden LLM Parsers**: Replace Regex/JSON-string parsing with Structured Outputs in `agents/planner.py` and `graph_rag/extractor.py`.
4.  **Extract Prompts**: Clean up `RAG_QA_service.py` before it grows too large.

