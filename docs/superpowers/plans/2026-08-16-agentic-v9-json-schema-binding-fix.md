# Agentic v9 JSON Schema Binding Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route raw JSON Schema dictionaries through LangChain Google GenAI's `response_json_schema` field so contract planning and evidence qualification no longer receive Gemini HTTP 400 responses.

**Architecture:** Keep the existing shared `bind_json_schema()` boundary and raw `AIMessage` return contract. Change only the Google GenAI binding keyword; do not adopt `with_structured_output()`, bypass LangChain, or change downstream parsers.

**Tech Stack:** Python 3.13, `langchain-google-genai==4.1.1`, `google-genai==1.55.0`, pytest, Ruff.

## Global Constraints

- Preserve existing provider callbacks, accounting, budget admission, and raw response parsing.
- Do not change planner, evidence qualification, sufficiency, repair, or final-answer behavior.
- Keep the patch limited to the shared provider boundary and its focused regression test.

---

### Task 1: Correct the raw JSON Schema binding

**Files:**
- Modify: `tests/test_agentic_v9_provider_boundary.py:185`
- Modify: `tests/test_agentic_v9_campaign_runtime.py:180`
- Modify: `core/providers.py:288`

**Interfaces:**
- Consumes: `bind_json_schema(llm: Any, *, schema: dict[str, Any]) -> Any`
- Produces: the same raw bound provider, configured with `response_mime_type="application/json"` and `response_json_schema=schema`

- [x] **Step 1: Change the focused test to require the JSON Schema field**

```python
assert captured == {
    "response_mime_type": "application/json",
    "response_json_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    },
}
assert "response_schema" not in captured
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py::test_bind_json_schema_uses_native_json_configuration -q
```

Expected: FAIL because production still passes `response_schema`.

- [x] **Step 3: Apply the minimal production fix**

```python
return binder(
    response_mime_type="application/json",
    response_json_schema=schema,
)
```

- [x] **Step 4: Run focused and impacted verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_contract_planner_canary.py tests/test_agentic_v9_evidence_qualification_canary.py -q
.\.venv\Scripts\python.exe -m ruff check core/providers.py tests/test_agentic_v9_provider_boundary.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 5: Commit the independently deployable fix**

```powershell
git add core/providers.py tests/test_agentic_v9_provider_boundary.py docs/superpowers/plans/2026-08-16-agentic-v9-json-schema-binding-fix.md
git commit -m "fix(agentic-v9): bind raw JSON schemas correctly"
```

- [ ] **Step 6: Verify on the real server after deployment**

Run the existing `current`, `minimal`, and evidence `--invoke` canaries. Expected: exit code `0`, `response_received=true`, and evidence `qualified_packet_count >= 1`.
