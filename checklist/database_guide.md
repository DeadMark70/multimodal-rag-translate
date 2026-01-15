# RAG Database (data_base) Technical Documentation

## 1. Technical Implementation Details

### Core Logic
The `data_base` module forms the backbone of the project's Retrieval-Augmented Generation (RAG) system. It manages vector storage, semantic retrieval, and complex question-answering workflows.

1.  **Vector Store Management (`vector_store_manager.py`)**:
    -   **Engine**: Uses FAISS (local vector database) for fast similarity search.
    -   **Embeddings**: Utilizes Google Gemini Embedding API.
    -   **Chunking**: Supports both Recursive Character Chunking and Semantic Chunking strategies.
    -   **Hybrid Search**: Combines FAISS (vector) and BM25 (keyword) retrievers via `EnsembleRetriever`.
    -   **Parent-Child Indexing**: Maintains links between small chunks (for retrieval precision) and larger parent chunks (for context).

2.  **RAG QA Service (`RAG_QA_service.py`)**:
    -   **Workflow**:
        1.  **Retrieval**: Fetches relevant documents using the hybrid retriever.
        2.  **Query Transformation**: Optional HyDE (Hypothetical Document Embeddings) or Multi-Query expansion.
        3.  **Reranking**: Re-scores results using Cross-Encoders or reciprocal rank fusion.
        4.  **Context Enrichment**: Expands short chunks with parent context.
        5.  **Multimodal Synthesis**: Combines text and image data into a prompt for the LLM (Gemini).
        6.  **Visual Verification**: Optional Re-Act loop to verify image details if the initial summary is insufficient.
        7.  **Conflict Handling**: Explicit prompting strategies to handle conflicting information between documents.

3.  **Deep Research (`deep_research_service.py`)**:
    -   Implements a "Plan-and-Solve" agentic workflow.
    -   Decomposes complex questions into sub-tasks.
    -   Executes RAG on each sub-task and synthesizes the results.

### Algorithms
-   **Reciprocal Rank Fusion (RRF)**: Merges results from multiple retrievers.
-   **HyDE**: Generates hypothetical answers to improve retrieval semantic matching.
-   **Semantic Chunking**: Splits text based on semantic similarity rather than just character count.

## 2. Codebase Map

| File Path | Responsibility |
| :--- | :--- |
| `data_base/router.py` | Main API endpoints for QA, research, and planning. |
| `data_base/vector_store_manager.py` | Manages FAISS indices, embeddings, and chunking strategies. |
| `data_base/RAG_QA_service.py` | Core RAG pipeline logic (retrieval, ranking, synthesis). |
| `data_base/deep_research_service.py` | Orchestrates the agentic deep research workflow. |
| `data_base/reranker.py` | Implements Cross-Encoder reranking logic. |
| `data_base/query_transformer.py` | HyDE and Multi-Query transformation logic. |
| `data_base/parent_child_store.py` | Manages hierarchical document storage. |
| `data_base/context_enricher.py` | Enriches chunks with additional context. |

## 3. Usage Guide

**⚠️ IMPORTANT: All commands must be executed within the project's virtual environment (`.venv`).**

### API Usage
**Ask a Question:**
`POST /rag/ask`
-   **Body**: `{"question": "...", "history": [...]}`
-   **Features**: Supports `enable_evaluation`, `enable_graph_rag`, etc.

**Deep Research:**
`POST /rag/research`
-   **Body**: `{"question": "...", "max_subtasks": 5}`

**Streaming Execution:**
`POST /rag/execute/stream` (SSE)

### Standalone Testing
To test the vector store or RAG logic:

```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Example: Run RAG debug script
python debug_ragas_real.py
```

## 4. Dependencies

### Internal Modules
-   `core`: Authentication, LLM factory.
-   `graph_rag`: Knowledge graph integration for hybrid search.
-   `agents`: Planner, Evaluator, Synthesizer agents.
-   `supabase_client`: Database for logging and document metadata.

### External Libraries
-   `langchain`: Framework for RAG and chains.
-   `faiss-cpu`: Vector database.
-   `google-generativeai`: Embedding and LLM services.
-   `sentence-transformers`: Reranking models.
-   `rank_bm25`: Keyword retrieval.
