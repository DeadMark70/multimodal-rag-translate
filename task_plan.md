# Task Plan: OCR CUDA Isolation + Reranker GPU Stability

## Goal
Implement the approved fix so image OCR no longer mutates global CUDA visibility, reranker can reliably use GPU, and diagnostics/tests cover the failure mode.

## Phases
| Phase | Status | Description |
|---|---|---|
| 1 | completed | Add planning files and confirm implementation scope |
| 2 | completed | Refactor image OCR device policy to local/per-service control |
| 3 | completed | Add reranker CUDA probe diagnostics and reason refinement |
| 4 | completed | Add regression tests for OCR policy and reranker env-masking |
| 5 | completed | Run targeted tests and summarize results |

## Decisions
- Device strategy: **Reranker priority**.
- `IMAGE_OCR_DEVICE` default: `cpu`.
- Keep `RERANKER_DEVICE` and `RERANKER_MIN_GPU_GB` behavior compatible.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| `test_get_ocr_engine_respects_cpu_policy` failed because mocked `.to()` returned a new mock object | 1 | Set `mock_engine.to.return_value = mock_engine` to mirror `torch.nn.Module.to()` behavior. |
