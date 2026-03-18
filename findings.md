# Findings

## 2026-03-16
- Root cause confirmed: `image_service/ocr_service.py` sets `CUDA_VISIBLE_DEVICES=""` at import time, which affects the whole process.
- Continuous learning record:
  - Mistake: image OCR module performed global CUDA env mutation at import time.
  - Root cause class: `streaming` (cross-component runtime state coupling via process-global env).
  - Prevention rule: never modify process-global hardware env vars during module import; enforce per-component runtime device placement.
- Import path matters: `core/app_factory.py` imports `image_service.router` during router registration, so the side effect can occur before reranker warmup.
- Reproduction: importing `image_service.ocr_service` before first CUDA probe causes `torch.cuda.device_count()` to become `0`, leading to reranker reason `cuda_unavailable`.
- Environment itself is GPU-capable: `.venv` has CUDA-enabled PyTorch and reports CUDA device when not globally masked.
- Post-fix validation: after importing `image_service.router`, `_select_runtime_device("auto")` now returns `("cuda", None)`, confirming OCR import no longer masks reranker GPU.
- Added explicit reranker observability for probe fields (`CUDA_VISIBLE_DEVICES`, `is_available`, `device_count`, and `cuda.init` error summary).
