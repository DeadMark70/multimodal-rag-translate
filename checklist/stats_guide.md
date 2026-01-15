# Dashboard Statistics (stats) Technical Documentation

## 1. Technical Implementation Details

### Core Logic
The `stats` module provides the backend logic for the user's analytical dashboard. It aggregates data from the `query_logs` table in Supabase to present a snapshot of the agent's performance and user engagement.

1.  **Metric Aggregation (`router.py`)**:
    -   **Source**: Queries the `query_logs` table via Supabase for the current user.
    -   **Metrics**:
        -   **Total Queries**: Raw count of interactions.
        -   **Accuracy / Faithfulness**: Counts logs tagged as `grounded`, `hallucinated`, or `uncertain` (populated by the Self-RAG Evaluator in `data_base`).
        -   **Confidence**: Computes the average confidence score across all evaluated queries.
        -   **Time Series**: Calculates a rolling 7-day query volume histogram.
        -   **Document Usage**: Parses the `doc_ids` array in logs to find the top 5 most referenced documents.

### Algorithms
-   **In-Memory Aggregation**: Currently pulls all logs for the user and computes stats in Python.
    -   *Note*: As data grows, this should move to a SQL/Postgres function (RPC) for performance.

## 2. Codebase Map

| File Path | Responsibility |
| :--- | :--- |
| `stats/router.py` | Contains the single endpoint `/stats/dashboard` and all aggregation logic. |

## 3. Usage Guide

**⚠️ IMPORTANT: All commands must be executed within the project's virtual environment (`.venv`).**

### API Usage
**Get Dashboard Data:**
`GET /stats/dashboard`
-   **Auth**: Requires JWT token.
-   **Response**: JSON object with `total_queries`, `accuracy_rate`, `queries_last_7_days`, etc.

### Standalone Testing
There is no dedicated script for stats. Testing usually involves generating some queries via RAG and then calling the endpoint to verify the numbers increment.

```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

## 4. Dependencies

### Internal Modules
-   `core`: Authentication.
-   `supabase_client`: Direct access to `query_logs` table.

### External Libraries
-   `fastapi`: API framework.
-   `pydantic`: Data validation for response models.
