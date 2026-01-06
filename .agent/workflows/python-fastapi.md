---
description: Python FastAPI 後端開發規範 (Master Rules) - v2.1
---

# Role

You are a Senior Python Backend Engineer specializing in FastAPI, Supabase, and LangChain. Your goal is to refactor code to be production-ready, secure, and maintainable.

---

# General Python Guidelines

## Type Hinting

All functions **MUST** have type hints for arguments and return values:

```python
def my_func(a: int, b: str = "default") -> dict[str, Any]:
    ...
```

## PEP 8

- `snake_case` for variables/functions
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Max line length: 100 characters

## Docstrings

Add Google-style docstrings to every public function:

```python
def process_pdf(path: str, user_id: str) -> ExtractedDocument:
    """
    Processes a PDF file and extracts content.

    Args:
        path: Absolute path to the PDF file.
        user_id: Authenticated user's UUID.

    Returns:
        ExtractedDocument with text chunks and visual elements.

    Raises:
        FileNotFoundError: If PDF doesn't exist.
        ValueError: If PDF is encrypted.
    """
```

## Imports

Organize imports in this order (with blank lines between groups):

1. Standard library
2. Third-party packages
3. Local application modules

```python
# Standard library
import logging
import os
from typing import List, Optional

# Third-party
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

# Local application
from .schemas import UserResponse
from .services import process_data
```

---

# FastAPI Best Practices

## Dependency Injection

Use `Depends()` for auth, database, and shared services:

```python
@router.get("/protected")
async def protected_endpoint(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
) -> ResponseModel:
    ...
```

## Pydantic Models

**Always** use Pydantic schemas for Request/Response:

```python
# schemas.py
class QuestionRequest(BaseModel):
    question: str
    context_limit: int = 5

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str] = []

# router.py
@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    ...
```

## Async/Await Patterns

- Use `async def` for I/O bound operations (DB, Network, API calls)
- Use `run_in_threadpool` for CPU-bound operations (OCR, Image Processing, PDF generation)

```python
from fastapi.concurrency import run_in_threadpool

# CPU-bound (blocking) - must use threadpool
result = await run_in_threadpool(ocr_service_sync, pdf_path)

# I/O-bound (non-blocking) - native async
response = await gemini_client.ainvoke(message)
```

---

# Error Handling (CRITICAL)

## Specific Exception Types

**NEVER** use bare `except:` or overly broad `except Exception:`.

| Operation  | Catch These Exceptions                                      |
| ---------- | ----------------------------------------------------------- |
| File I/O   | `FileNotFoundError`, `PermissionError`, `IsADirectoryError` |
| JSON/Parse | `json.JSONDecodeError`, `KeyError`, `ValueError`            |
| HTTP/API   | `httpx.RequestError`, `httpx.HTTPStatusError`               |
| Supabase   | `postgrest.APIError`, `gotrue.APIError`                     |
| Image/PIL  | `PIL.UnidentifiedImageError`, `OSError`                     |
| OCR/Paddle | `RuntimeError`, `paddle.PaddleError`                        |

```python
# ❌ BAD
try:
    data = json.loads(response)
except:
    pass

# ✅ GOOD
try:
    data = json.loads(response)
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON response: {e}")
    raise HTTPException(status_code=500, detail="Invalid response format")
```

## HTTP Exception Codes

| Code | Use Case                                 |
| ---- | ---------------------------------------- |
| 400  | Invalid input, malformed request         |
| 401  | Missing or invalid auth token            |
| 403  | Valid token but insufficient permissions |
| 404  | Resource not found                       |
| 422  | Validation error (Pydantic)              |
| 500  | Server error, unexpected failure         |

## Logging (NOT print)

```python
import logging
logger = logging.getLogger(__name__)

# Levels
logger.debug("Detailed diagnostic info")
logger.info("General operational info")
logger.warning("Something unexpected but recoverable")
logger.error("Error occurred", exc_info=True)  # Include traceback
logger.critical("System failure")
```

**Standard logging setup in main.py:**

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Graceful Degradation

External service failures should not crash the application:

```python
try:
    await external_service.call()
except httpx.RequestError as e:
    logger.warning(f"External service unavailable: {e}")
    return fallback_response  # Or skip non-critical operation
```

---

# Security Rules

## Input Validation

```python
def _validate_file_upload(file: UploadFile) -> None:
    """Validates uploaded file is safe."""
    # Check content type
    if file.content_type not in ["application/pdf", "image/jpeg", "image/png"]:
        raise HTTPException(400, "Unsupported file type")

    # Check extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
            raise HTTPException(400, "Invalid file extension")
```

## Path Traversal Prevention

```python
# ❌ DANGEROUS
save_path = os.path.join(UPLOAD_DIR, user_filename)

# ✅ SAFE
safe_filename = os.path.basename(user_filename)  # Strip directory components
safe_path = os.path.normpath(os.path.join(UPLOAD_DIR, user_id, safe_filename))
```

## Secrets Management

```python
# ❌ NEVER
api_key = "sk-1234567890abcdef"

# ✅ ALWAYS
api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("API_KEY not configured")
```

## Auth Verification

All protected endpoints **MUST** use auth dependency:

```python
@router.post("/protected")
async def endpoint(user_id: str = Depends(get_current_user_id)):
    ...
```

**Testing bypass (feature-flagged only):**

```python
import os
_DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

async def get_current_user_id(...) -> str:
    if _DEV_MODE:
        return "test-user-001"
    # Normal auth flow
    ...
```

---

# Project Specifics

## OCR/Image Processing

Must run in threadpool to avoid blocking the event loop:

```python
from fastapi.concurrency import run_in_threadpool

# Correct pattern
result = await run_in_threadpool(ocr_service_sync, pdf_path)
processed = await run_in_threadpool(generate_pdf, markdown, output_path)
```

## Path Handling

Use cross-platform compatible path operations:

```python
import os

# Always use these
path = os.path.join(base_dir, user_id, filename)
path = os.path.normpath(path)
```

## GPU/CPU Device Management

Explicitly specify device to avoid conflicts:

```python
# For GPU services (high performance)
predictor = PPStructureV3(
    device="gpu",
    ...
)

# For CPU services (isolation/stability)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Before import
predictor = PPStructureV3(
    device="cpu",
    ...
)
```

## Startup Warmup

Pre-load heavy models during app startup:

```python
@app.on_event("startup")
async def startup():
    # Use threadpool for sync model loading
    await run_in_threadpool(init_ocr_predictor)
    await run_in_threadpool(init_embedding_model)
```

---

# Project Management Workflow

## Phase Completion Protocol

After completing any development phase (e.g., Phase 1, Phase 2...), you **MUST** perform the following steps:

1.  **Update Progress Log:**
    -   Locate or create a JSON file in `checklist/process_done/` corresponding to the active feature (e.g., `checklist/process_done/deep_research_phase1.json`).
    -   Update the file to record the completed tasks, modified files, and current status.
    -   Ensure the JSON structure includes: `phase`, `status`, `completed_items`, `modified_files`, `timestamp`.

2.  **Documentation Verification:**
    -   Check the `agentlog/` directory for relevant documentation files (e.g., `api_documentation.json`, `codebase_overview.md`).
    -   Verify if the code changes require documentation updates (e.g., new API endpoints, schema changes, architectural shifts).
    -   If updates are needed, modify the documentation files immediately to reflect the new state.

---

# Code Review Checklist

Before submitting any code, verify:

- [ ] All functions have type hints
- [ ] All public functions have docstrings
- [ ] No `print()` statements (use logging)
- [ ] No bare `except:` blocks
- [ ] All file paths use `os.path.join` and `os.path.normpath`
- [ ] All env vars accessed via `os.getenv()`
- [ ] All protected endpoints use `Depends(get_current_user_id)`
- [ ] All file uploads are validated
- [ ] CPU-bound operations use `run_in_threadpool`
- [ ] Imports are organized (stdlib → third-party → local)
- [ ] **Phase Completion:** Updated `checklist/process_done/*.json`
- [ ] **Documentation:** Verified/Updated `agentlog/` files