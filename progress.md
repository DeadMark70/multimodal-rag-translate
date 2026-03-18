# Progress Log

## 2026-03-16
- Created planning files (`task_plan.md`, `findings.md`, `progress.md`).
- Loaded required skills: `planning-with-files`, `senior-backend`, `senior-qa`.
- Confirmed implementation plan and prepared code/test changes.
- Implemented OCR local device policy with new `IMAGE_OCR_DEVICE` env (`cpu|auto|cuda`, default `cpu`) and removed module-level `CUDA_VISIBLE_DEVICES` mutation.
- Updated image router to lazily import `perform_ocr` inside request flow.
- Implemented reranker CUDA probe diagnostics and masked-env reason (`cuda_masked_by_env`).
- Added regression tests: `tests/test_image_ocr_service.py` and reranker masked-env case in `tests/test_reranker.py`.
- Executed targeted validation:
  - `python -m pytest tests/test_image_ocr_service.py tests/test_reranker.py` -> `20 passed`.
  - `ruff check` on changed Python files -> pass.
  - Runtime sanity snippet after importing `image_service.router` -> `_select_runtime_device("auto") == ('cuda', None)`.
