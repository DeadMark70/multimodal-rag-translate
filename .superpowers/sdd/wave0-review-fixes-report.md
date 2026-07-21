# Wave 0 Review Fixes Report

## Commit

`fix(agentic-v9): harden wave zero contracts`

## Files

- `evaluation/golden/agentic_v9_questions_v2.json`
- `evaluation/golden/agentic_v9_baseline_manifest.json`
- `data_base/agentic_v9/schemas.py`
- `tests/test_agentic_v9_golden_dataset.py`
- `tests/test_agentic_v9_schemas.py`

## Changes

- Q14 records unsupported original-SAM and SegmentAnyBone facts as explicit
  unavailable slot resolutions; only SegVol remains positive expected evidence.
- Golden artifact and complete baseline-manifest snapshots use code-pinned SHA-256
  constants. Route fixtures enforce the exact six-route, one-synthetic-graph,
  non-formal integrity contract.
- Slot and sufficiency cross-field validators reject evidence/absence and complete
  state contradictions. Contract coverage includes request authorization fields,
  route defaults, claim support typing, and v8 serialized trace round-tripping.

## Verification

- `D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_golden_dataset.py tests/test_agentic_v9_schemas.py tests/test_evaluation_test_case_schema.py -q`
  - 37 passed; 23 pre-existing third-party Pydantic deprecation warnings.
- `D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/schemas.py tests/test_agentic_v9_golden_dataset.py tests/test_agentic_v9_schemas.py`
  - passed.

## Residual Risks

- The golden artifact hashes are intentionally byte-sensitive; future approved
  fixture updates must refresh the manifest and code-pinned constants together.
- The focused suite does not exercise later Wave runtime modules, which are out of
  scope for this contract-only correction.
