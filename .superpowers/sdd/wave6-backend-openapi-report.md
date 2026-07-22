# Wave 6 backend OpenAPI contract repair

- Scope: generated `openapi.json` only; no runtime modules were changed.
- Source revision: `cfcf54d` (`fix(evaluation): close r6 identity and redaction gaps`).
- Generator command:
  ```powershell
  $env:TEST_MODE='true'
  & 'D:\flutterserver\pdftopng\.venv\Scripts\python.exe' -c "import json; from pathlib import Path; from main import app; Path('openapi.json').write_text(json.dumps(app.openapi(), ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')"
  ```
- Generated snapshot SHA-256: `7C9516DED44E516F6275A939E4B8F0A7836A880F7CF872AA38FC4FCFA2123486`.

## Contract evidence

The regenerated static snapshot is structurally equal to `main.app.openapi()` and exposes both campaign-control fields in `CampaignCreateRequest` and `CampaignConfig`:

- `agentic_execution_version`: enum `["v8", "v9"]`, default `"v8"`
- `shadow_evaluation_policy`: nullable enum `["operational", "research"]`

The regeneration also synchronizes existing evaluation paths and schemas which had previously been present in the live FastAPI application but absent from the tracked snapshot.

## Verification

- Live-versus-snapshot equality plus field assertions: passed.
- `pytest tests/test_evaluation_task14.py tests/test_api_contracts_v3.py -q -p no:cacheprovider`: **7 passed** (24 third-party/config deprecation warnings).
- `ruff check evaluation/campaign_schemas.py evaluation/router.py tests/test_evaluation_task14.py tests/test_api_contracts_v3.py`: passed.
- `git diff --check`: passed.
