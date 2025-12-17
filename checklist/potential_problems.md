# Project Health Checklist (Confirmed Findings)

Investigation completed on 2025-12-10.

## 🚨 Critical Issues (Must Fix)

- [ ] **Auth / Security:** `core/auth.py`
    - **Issue:** `_DEV_MODE` returns a hardcoded user ID `"test-user-id-001"` which is **not a valid UUID**.
    - **Impact:** Supabase database writes (which expect UUIDs) will fail in development mode.
    - **Fix:** Change to a valid static UUID (e.g., `00000000-0000-0000-0000-000000000000`).

- [ ] **Startup / Configuration:** `main.py`
    - **Issue:** Missing directory initialization.
    - **Impact:** App will crash when accessing `uploads/`, `output/imgs/`, or `faiss_index/`.
    - **Fix:** Add directory creation logic (`os.makedirs(..., exist_ok=True)`) to `startup_event`.

- [ ] **Environment:** `config.env`
    - **Issue:** File is missing.
    - **Fix:** Copy `config.env.example` to `config.env` and populate API keys.

## ⚠️ Code Quality & Maintenance

- [ ] **Error Handling:** Multiple files (e.g., `agents/evaluator.py`, `data_base/RAG_QA_service.py`)
    - **Issue:** Broad `except Exception as e:` usage.
    - **Impact:** Masks specific errors (like `KeyboardInterrupt` or system exits) and makes debugging harder.
    - **Fix:** Refactor to catch specific exceptions (`httpx.RequestError`, `ValueError`, `IOError`) where possible.

- [ ] **Cleanliness:** `pdfserviceMD/`
    - **Issue:** `PDF_OCR_config.null` exists and contains unused garbage.
    - **Fix:** Delete the file.

## ✅ Verified Good

- **Concurrency:** `run_in_threadpool` is correctly used in `image_service` and `pdfserviceMD` for CPU-bound tasks.
- **LLM Architecture:** `core/llm_factory.py` correctly routes `translation` vs `rag_qa` tasks.
- **Type Hints:** Core modules (`agents/`, `data_base/`) generally follow type hinting standards.

## 📋 Next Steps for Agent

1. Create `config.env`.
2. Fix `core/auth.py` UUID.
3. Add directory creation to `main.py`.
4. Refactor `except Exception` in `agents/evaluator.py`.