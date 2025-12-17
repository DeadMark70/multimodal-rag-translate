# Exception Handling Refactoring Plan

> **Generated:** 2025-12-10
> **Goal:** Replace broad `except Exception` usage with specific exception types to improve code stability and debugging.

## 📊 Summary of Analysis

| Module                              | Count | Primary Domain             | Common Errors                                         |
| :---------------------------------- | :---- | :------------------------- | :---------------------------------------------------- |
| `data_base/vector_store_manager.py` | 12    | File I/O, FAISS, Pickling  | `OSError`, `RuntimeError`, `pickle.PickleError`       |
| `pdfserviceMD/router.py`            | 7     | HTTP API, DB, File I/O     | `httpx.RequestError`, `postgrest.APIError`, `OSError` |
| `data_base/router.py`               | 5     | API Routing, RAG Logic     | `httpx.RequestError`, `RuntimeError`                  |
| `data_base/RAG_QA_service.py`       | 3     | LLM Integration, Retrieval | `httpx.RequestError`, `ValueError`                    |

---

## 🛠️ Detailed Refactoring Map

### 1. `data_base/vector_store_manager.py`

| Line | Context                | Recommended Exceptions                             | Reasoning                               |
| :--- | :--------------------- | :------------------------------------------------- | :-------------------------------------- |
| 99   | Loading Embeddings     | `(RuntimeError, OSError)`                          | HuggingFace loading or device issues.   |
| 198  | Indexing to FAISS      | `(ValueError, RuntimeError, PickleError, OSError)` | FAISS operations, pickling, file write. |
| 254  | Loading User Index     | `(RuntimeError, PickleError, OSError)`             | Corrupt index file or missing file.     |
| 328  | Context Enrichment     | `(ImportError, RuntimeError)`                      | Optional module missing or LLM failure. |
| 349  | Loading Existing Index | `(RuntimeError, PickleError)`                      | Corrupt index file.                     |
| 361  | Saving Index           | `(OSError, IOError)`                               | Disk write permission/space issues.     |
| 423  | Parent Store Cleanup   | `(IOError, KeyError)`                              | File access or missing keys.            |
| 428  | Delete Document        | `(OSError, ValueError)`                            | File system or FAISS API errors.        |
| 604  | Load Index (Hierarchy) | `(RuntimeError, PickleError)`                      | Corrupt index file.                     |
| 614  | Save Index (Hierarchy) | `(OSError, IOError)`                               | Disk write issues.                      |

### 2. `pdfserviceMD/router.py`

| Line | Context                | Recommended Exceptions                     | Reasoning                                                         |
| :--- | :--------------------- | :----------------------------------------- | :---------------------------------------------------------------- |
| 84   | Update DB Status       | `(postgrest.APIError, httpx.RequestError)` | Supabase API failures.                                            |
| 152  | Create DB Record       | `(postgrest.APIError, httpx.RequestError)` | Supabase API failures.                                            |
| 175  | RAG Indexing           | `(RuntimeError, ValueError)`               | Non-fatal vector store errors.                                    |
| 248  | **Endpoint Top-Level** | `(HTTPException, OSError, RuntimeError)`   | **Keep `Exception` as last resort** but log with `exc_info=True`. |
| 301  | Get DB Record          | `(postgrest.APIError, httpx.RequestError)` | Supabase API failures.                                            |
| 318  | RAG Deletion           | `(RuntimeError, ValueError)`               | Non-fatal vector store errors.                                    |
| 333  | Delete DB Record       | `(postgrest.APIError, httpx.RequestError)` | Supabase API failures.                                            |

### 3. `data_base/router.py`

| Line | Context              | Recommended Exceptions                     | Reasoning                                           |
| :--- | :------------------- | :----------------------------------------- | :-------------------------------------------------- |
| 89   | Startup Init         | `(RuntimeError, ImportError)`              | Critical startup failures.                          |
| 150  | Chat Logging         | `(postgrest.APIError, httpx.RequestError)` | Non-fatal Supabase log.                             |
| 155  | `/ask` Endpoint      | `(HTTPException, RuntimeError)`            | RAG service failures.                               |
| 214  | Sub-task Execution   | `(RuntimeError, ValueError)`               | Single sub-task failure (non-fatal for whole plan). |
| 254  | `/research` Endpoint | `(HTTPException, RuntimeError)`            | Planner or Synthesizer failures.                    |

### 4. `data_base/RAG_QA_service.py`

| Line | Context        | Recommended Exceptions                        | Reasoning                         |
| :--- | :------------- | :-------------------------------------------- | :-------------------------------- |
| 115  | Get LLM        | `(RuntimeError, KeyError)`                    | API Key missing or factory error. |
| 150  | Retrieval      | `(RuntimeError, ValueError)`                  | FAISS or Retriever errors.        |
| 262  | LLM Generation | `(httpx.RequestError, httpx.HTTPStatusError)` | Network or API error from Google. |

### 5. `agents/evaluator.py`

| Line | Context           | Recommended Exceptions                                   | Reasoning                   |
| :--- | :---------------- | :------------------------------------------------------- | :-------------------------- |
| 142  | Retrieval Eval    | `(httpx.RequestError, ValueError, json.JSONDecodeError)` | LLM failure or bad parsing. |
| 191  | Faithfulness Eval | `(httpx.RequestError, ValueError, json.JSONDecodeError)` | LLM failure or bad parsing. |

---

## 💡 Implementation Guidelines

1.  **Imports:** You will need to import these exceptions at the top of the files.
    ```python
    from pickle import PickleError
    from httpx import RequestError, HTTPStatusError
    from postgrest import APIError  # Check supabase libs for exact path
    ```
2.  **Fallback:** For top-level API endpoints (routers), it is acceptable to keep `except Exception` **ONLY IF** it is the last block and performs a "500 Internal Server Error" response + full traceback logging.
3.  **Supabase:** Supabase client usually raises `postgrest.exceptions.APIError` or `gotrue.errors.AuthError`. Verify the library version.

## ✅ Action List (Completed 2025-12-10)

- [x] Update `data_base/vector_store_manager.py` (12 blocks - OSError, PickleError, RuntimeError)
- [x] Update `pdfserviceMD/router.py` (7 blocks - PostgrestAPIError, RuntimeError)
- [x] Update `data_base/RAG_QA_service.py` (3 blocks - RuntimeError, ValueError)
