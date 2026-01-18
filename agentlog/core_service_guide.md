# Core Components (core) Technical Documentation

## 1. Technical Implementation Details

### Core Logic
The `core` module provides centralized services used throughout the application, including authentication, LLM management, and document summarization.

1.  **Authentication (`auth.py`)**:
    -   **Mechanism**: Validates Supabase JWT tokens via the `Authorization: Bearer <token>` header.
    -   **Dependency**: Used as a FastAPI dependency (`get_current_user_id`) in all protected routers.
    -   **Dev Mode**: Supports a `DEV_MODE=true` environment variable to bypass authentication for local testing, using a fixed test user ID.
    -   **Error Handling**: Returns `401 Unauthorized` for missing or invalid tokens and `500 Internal Server Error` if Supabase is unavailable.

2.  **LLM Factory (`llm_factory.py`)**:
    -   **Cached Instances**: Uses `@lru_cache` to manage LLM instances, preventing redundant initializations.
    -   **Purpose-Specific Config**: Tailors LLM settings (temperature, max tokens) for different tasks (e.g., `rag_qa`, `translation`, `evaluator`, `planner`).
    -   **Model Selection**:
        -   Default: `gemma-3-27b-it` (high performance for reasoning).
        -   Fast/Cheap Tasks: `gemini-2.5-flash-lite` for translation and GraphRAG extraction.
    -   **Session Overrides**: Allows global model overrides for testing or experimentation via `set_session_model_override`.

3.  **Executive Summary Service (`summary_service.py`)**:
    -   **Structured Briefing**: Generates academic-style summaries (Core Problem, Methodology, Key Findings, Impact).
    -   **Async Processing**: Includes background task logic (`generate_summary_background`) to generate summaries after PDF upload without blocking the main response.
    -   **Persistence**: Saves and retrieves summaries from the `documents` table in Supabase.

## 2. Codebase Map

| File Path | Responsibility |
| :--- | :--- |
| `core/auth.py` | Centralized Supabase JWT authentication logic. |
| `core/llm_factory.py` | Configuration and factory for LangChain Google Generative AI instances. |
| `core/summary_service.py` | Service for generating and managing document executive summaries. |

## 3. Usage Guide

**⚠️ IMPORTANT: All commands must be executed within the project's virtual environment (`.venv`).**

### Authentication Example
In a FastAPI router:
```python
@router.get("/protected")
async def protected_route(user_id: str = Depends(get_current_user_id)):
    return {"user_id": user_id}
```

### LLM Usage Example
```python
from core.llm_factory import get_llm
llm = get_llm("rag_qa")
response = await llm.ainvoke("Hello")
```

## 4. Dependencies

### Internal Modules
-   `supabase_client`: Used by `auth.py` and `summary_service.py`.

### External Libraries
-   `fastapi`: Dependency injection for auth.
-   `langchain_google_genai`: Integration with Gemini models.
-   `postgrest`: Database error handling.
