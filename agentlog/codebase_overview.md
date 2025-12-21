# Codebase Overview & Architecture

## System Architecture

The Multimodal RAG System is a FastAPI-based application designed for:
1.  **PDF Processing**: OCR, translation, and recreation of PDFs.
2.  **RAG (Retrieval-Augmented Generation)**: Intelligent Q&A over document content using vector search and hybrid retrieval.
3.  **Agentic Workflows**: Complex query resolution using Planner, Executor (RAG), and Synthesizer agents.

### Core Components

*   **FastAPI Backend (`main.py`)**: Entry point, middleware, and router registration.
*   **PDF Service (`pdfserviceMD/`)**:
    *   **OCR**: Hybrid approach using **Local Marker** (default) or **Datalab API**.
    *   **Translation**: Uses `gemini-3.0-flash` for high-volume text translation.
    *   **PDF Generation**: Rebuilds PDFs maintaining layout using `markdown-pdf`.
*   **RAG Engine (`data_base/`)**:
    *   **Vector Store**: FAISS for fast similarity search.
    *   **Embeddings**: `models/gemini-embedding-001` (Google API).
    *   **Reranker**: `cross-encoder/ms-marco-MiniLM-L-12-v2` (Microsoft SBERT) for high-precision re-ranking.
    *   **Query Transformation**: Implements HyDE (Hypothetical Document Embeddings) and Multi-Query expansion.
*   **GraphRAG (`graph_rag/`)**:
    *   **Store**: NetworkX based graph storage.
    *   **Extraction**: LLM-based entity and relation extraction.
    *   **Community**: Leiden algorithm for community detection and summarization.
    *   **Search**: Local and Global search strategies.
*   **Agents (`agents/`)**:
    *   **Planner**: Decomposes complex user queries into sub-tasks.
    *   **Evaluator**: Self-RAG implementation to score retrieval quality and generation hallucinations.
    *   **Synthesizer**: Combines results from sub-tasks into a coherent final answer.
*   **LLM Factory (`core/llm_factory.py`)**: Centralized management of LLM instances with purpose-specific configurations (Temperature, Max Tokens).

## Data Flow

### 1. PDF Ingestion Pipeline
1.  **Upload**: User uploads PDF via `/pdfmd/upload_pdf_md`.
2.  **OCR**: System extracts text using Marker/Datalab.
3.  **Indexing (Background)**:
    *   Text is chunked (Semantic + Propositional).
    *   Images are summarized.
    *   Vectors are stored in FAISS.
4.  **Translation (Optional)**: Text is translated and a new PDF is generated.

### 2. RAG Query Pipeline (`/rag/ask`)
1.  **Query Analysis**: Query is rewritten (HyDE/Multi-Query).
2.  **Retrieval**: Vectors are fetched from FAISS.
3.  **Reranking**: Top-K results are re-ordered by relevance.
    *   **Generation**: LLM (`gemma-3-27b-it`) generates answer based on context.

### 3. GraphRAG Pipeline (Standalone)
1.  **Ingestion**: Extracts entities and relations from documents using `gemini-3.0-flash`.
2.  **Storage**: NetworkX graph stored in `uploads/{user_id}/rag_index/graph.pkl`.
3.  **Optimization**: Performs entity resolution and Leiden community detection.
4.  **Search**: Supports Local (entity-centric) and Global (community-summary) search modes.

### 4. Research Pipeline (`/rag/research`)
1.  **Planning**: `Planner` breaks down the query.
2.  **Execution**: Each sub-task is executed via the RAG pipeline.
3.  **Evaluation**: `Evaluator` checks quality.
4.  **Synthesis**: `Synthesizer` compiles the final report.

## Key Directories

| Directory | Purpose | Key Files |
| :--- | :--- | :--- |
| `core/` | Infrastructure | `llm_factory.py`, `auth.py`, `supabase_client.py` |
| `pdfserviceMD/` | PDF Logic | `PDF_OCR_services.py`, `ai_translate_md.py` |
| `data_base/` | RAG Logic | `vector_store_manager.py`, `RAG_QA_service.py` |
| `graph_rag/` | Knowledge Graph | `store.py`, `extractor.py`, `community_builder.py` |
| `agents/` | AI Agents | `planner.py`, `evaluator.py`, `synthesizer.py` |
| `multimodal_rag/` | Vision RAG | `image_summarizer.py` |

## Design Patterns

*   **Factory Pattern**: `llm_factory.py` for creating configured LLMs.
*   **Strategy Pattern**: Chunking strategies (`semantic`, `proposition`, `word`).
*   **Router Pattern**: FastAPI `APIRouter` for modular API definition.
*   **Asynchronous Processing**: Heavy tasks (OCR, Translation) run in threadpools or background tasks.
*   **Graph-based Modeling**: `graph_rag/store.py` uses NetworkX for knowledge representation.

## Model Configuration (Internal Defaults)

These are configured in `core/llm_factory.py` and are not currently exposed as env vars, but good to know:

*   **Translation Model**: `gemini-3.0-flash`
*   **Graph Extraction**: `gemini-3.0-flash`
*   **Community Summary**: `gemini-3.0-flash`
*   **General Model**: `gemma-3-27b-it`

## Roadmap Status

| Phase | Feature | Status |
| :--- | :--- | :--- |
| **Phase 1-3** | Basic RAG + Agents | ✅ Complete |
| **Phase 4** | Multimodal Features | ✅ Complete |
| **Phase 5** | GraphRAG (Core Modules) | ✅ Complete |
| **Phase 5.3** | GraphRAG Integration | ✅ Complete |
| **Phase 6** | ColPali (Visual Embeddings) | 📝 Planned |