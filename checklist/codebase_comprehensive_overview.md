# Codebase Comprehensive Overview

> **Generated Date:** 2025-12-21
> **Version:** 2.1.0
> **Purpose:** Detailed architectural reference for autonomous agents and developers.

---

## 1. System Architecture

The project is a **FastAPI-based Multimodal RAG System** designed for processing academic PDFs, performing OCR, translating content, and enabling deep research capabilities through retrieval-augmented generation.

### High-Level Data Flow

1.  **Ingestion**: PDF Upload -> GPU-accelerated OCR (`marker-pdf`) -> Markdown Conversion.
2.  **Translation**: Page-based chunking -> LLM Translation (Gemini-3.0-Flash) -> PDF Reconstruction.
3.  **Indexing**:
    *   **Text**: Recursive/Semantic Chunking -> Gemini Embeddings -> FAISS Vector Store.
    *   **Visuals**: Image Summarization -> Multimodal Embeddings -> FAISS.
4.  **Retrieval**:
    *   **Hybrid**: Vector (FAISS) + Keyword (BM25) + Reranking (Cross-Encoder).
    *   **Advanced**: HyDE (Hypothetical Document Embeddings) & Multi-Query Fusion (RRF).
5.  **Generation**:
    *   **Simple QA**: Context-aware answering using Gemini-Pro/Gemma.
    *   **Deep Research**: Plan-and-Solve Agent (Planner -> Executor -> Synthesizer).

---

## 2. Directory Structure & Key Modules

### `main.py`
*   **Role**: Application entry point.
*   **Key Features**:
    *   FastAPI app initialization with CORS.
    *   Startup events: Directory creation, RAG component initialization, GPU warm-up.
    *   Router aggregation (`/pdfmd`, `/rag`, `/imagemd`, `/multimodal`, `/stats`).

### `core/` (Foundations)
*   **`llm_factory.py`**:
    *   Centralized LLM instantiation.
    *   **Routing**: Assigns specific models (`gemini-3.0-flash` vs `gemma-3-27b-it`) and configs (temperature, max_tokens) to tasks (`translation`, `rag_qa`, `planner`).
    *   **Caching**: Uses `lru_cache` for performance.
*   **`auth.py`**: Supabase JWT authentication logic.
*   **`summary_service.py`**: Document summarization logic.

### `data_base/` (RAG Engine)
*   **`vector_store_manager.py`**:
    *   Manages per-user FAISS indexes.
    *   Handles document addition/deletion.
    *   Supports `recursive` and `semantic` chunking strategies.
*   **`RAG_QA_service.py`**:
    *   Main orchestration for Question Answering.
    *   Implements the retrieval pipeline: `Retriever -> Filter -> Rerank -> Multimodal Prompt -> LLM`.
*   **`reranker.py`**: Cross-Encoder implementation for re-ranking retrieved documents.
*   **`query_transformer.py`**:
    *   **HyDE**: Generates hypothetical answers to improve semantic matching.
    *   **Multi-Query**: Generates query variations and fuses results using RRF (Reciprocal Rank Fusion).
*   **`parent_child_store.py`**: Implements hierarchical indexing (Small-to-Big retrieval).

### `agents/` (Autonomous Behaviors)
*   **`planner.py`**:
    *   **Role**: Decomposes complex research questions into 2-5 independent sub-tasks.
    *   **Logic**: Heuristic complexity check -> LLM Decomposition.
*   **`synthesizer.py`**:
    *   **Role**: Aggregates sub-task results into a coherent research report.
    *   **Logic**: Conflict resolution and multi-source citation.

### `pdfserviceMD/` (Document Processing)
*   **`translation_chunker.py`**:
    *   Advanced chunking for translation.
    *   Handles token limits by batching pages.
    *   Includes logic to preserve/repair `[[PAGE_N]]` markers.
*   **`local_marker_service.py`**: Integration with `marker-pdf` for high-quality OCR.

### `multimodal_rag/` (Visual Intelligence)
*   **`structure_analyzer.py`**: Extracts layout, tables, and figures.
*   **`image_summarizer.py`**: Generates textual descriptions for visual elements to enable semantic search over images.

---

## 3. Key Workflows

### A. Deep Research (Plan-and-Solve)
**Endpoint**: `POST /rag/research`

1.  **Input**: User asks "Compare Method A and Method B".
2.  **Planner**:
    *   Analyzes complexity.
    *   Generates sub-tasks: "What is Method A?", "What is Method B?", "Differences between A and B".
3.  **Execution** (Parallel/Sequential):
    *   Each sub-task triggers a standard RAG flow (`rag_answer_question`).
4.  **Synthesizer**:
    *   Collects all answers and sources.
    *   Generates a structured Markdown report with Summary and Detailed Analysis.

### B. Document Translation
**Endpoint**: `POST /pdfmd/upload_pdf_md`

1.  **OCR**: Convert PDF to Markdown + Images.
2.  **Chunking**: Split Markdown by pages (`[[PAGE_N]]`).
3.  **Batching**: Group pages to maximize context window usage (up to ~60k tokens).
4.  **Translation**: LLM translates text while preserving layout/images.
5.  **Reconstruction**: Reassemble translated Markdown -> Generate new PDF.

---

## 4. Configuration & Environment

*   **Config File**: `config.env` (loaded via `dotenv`).
*   **Key Variables**:
    *   `GOOGLE_API_KEY`: For Gemini LLM and Embeddings.
    *   `HF_TOKEN`: For HuggingFace models (if used locally).
    *   `SUPABASE_URL` / `SUPABASE_KEY`: Database and Auth.
    *   `DEV_MODE`: Toggles auth bypass.

## 5. Current Limitations (To be addressed in Phase 5)

1.  **Cross-Document Reasoning**: Standard RAG retrieves chunks independently. It struggles with "What are the common themes across all 10 papers?" (Requires Global Search).
2.  **Entity Consistency**: No shared understanding of entities (e.g., "BERT" in Paper A vs "Transformer" in Paper B).
3.  **Knowledge Graph**: Lack of explicit structural relationships between concepts.

---

## 6. Testing

*   **Framework**: `pytest`
*   **Coverage**: Core modules (`tests/test_*.py`).
*   **Status**: 104 tests passing (as of Dec 2025).
