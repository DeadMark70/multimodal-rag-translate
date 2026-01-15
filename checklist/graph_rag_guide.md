# Graph RAG (graph_rag) Technical Documentation

## 1. Technical Implementation Details

### Core Logic
The `graph_rag` module implements a Graph Retrieval-Augmented Generation system. Unlike vector search which finds local similarities, GraphRAG builds a structured knowledge graph to understand global relationships and "connect the dots" across documents.

1.  **Extraction (`extractor.py`)**:
    -   **Engine**: Gemini Flash (fast & cost-effective).
    -   **Process**:
        -   **Entity Extraction**: Identifies concepts, methods, metrics, results, and authors.
        -   **Relation Extraction**: Identifies relationships like `uses`, `outperforms`, `proposes`, `cites`, etc.
    -   **Output**: JSON-structured entities and relations.

2.  **Storage (`store.py`)**:
    -   **Backend**: NetworkX (in-memory graph library), serialized to `graph.pkl`.
    -   **Data Structure**:
        -   **Nodes**: Entities with labels, types, and source document links.
        -   **Edges**: Relationships with weights and descriptions.
    -   **Per-User**: Each user has an isolated graph instance stored in `uploads/{user_id}/rag_index/`.

3.  **Local Search (`local_search.py`)**:
    -   **Strategy**: Entity-Centric.
    -   **Flow**:
        1.  Identify entities in the user's query.
        2.  Find matching nodes in the graph (fuzzy match).
        3.  Expand to 1-2 hops of neighbors.
        4.  Retrieve relationship context to answer specific questions.

4.  **Global Search (`global_search.py`)**:
    -   **Strategy**: Community-Based Map-Reduce.
    -   **Flow**:
        1.  **Community Detection**: Uses Leiden algorithm (via `community_builder.py`) to cluster nodes.
        2.  **Map**: Checks relevance of each community summary to the query.
        3.  **Reduce**: Synthesizes answers from relevant communities to provide a high-level overview.

5.  **Optimization (`entity_resolver.py`)**:
    -   **Entity Resolution**: Merges duplicate entities (e.g., "LLM" and "Large Language Model") using embedding similarity and LLM verification.

## 2. Codebase Map

| File Path | Responsibility |
| :--- | :--- |
| `graph_rag/router.py` | API endpoints for status, visualization, and maintenance (rebuild/optimize). |
| `graph_rag/store.py` | NetworkX wrapper for CRUD operations and persistence. |
| `graph_rag/extractor.py` | LLM-based entity and relation extraction logic. |
| `graph_rag/local_search.py` | Implements entity-neighbor expansion search. |
| `graph_rag/global_search.py` | Implements community-based map-reduce search. |
| `graph_rag/community_builder.py` | Detecting communities and generating summaries. |
| `graph_rag/entity_resolver.py` | Logic for deduplicating and merging graph nodes. |

## 3. Usage Guide

**⚠️ IMPORTANT: All commands must be executed within the project's virtual environment (`.venv`).**

### API Usage
**Get Graph Status:**
`GET /graph/status`

**Get Visualization Data:**
`GET /graph/data`
-   Returns nodes and links formatted for `react-force-graph`.

**Force Optimization:**
`POST /graph/optimize`
-   Triggers entity resolution and community detection.

### Standalone Testing
To test extraction or search logic:

```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Example: Run GraphRAG specific tests or debug script
python debug_graph_rag_package.py
```

## 4. Dependencies

### Internal Modules
-   `core`: LLM factory.
-   `data_base`: Integration for hybrid search.

### External Libraries
-   `networkx`: Graph data structure and algorithms.
-   `cdlib` / `leidenalg`: Community detection algorithms.
-   `pydantic`: Schema validation.
-   `langchain`: LLM interactions.
