# Agentic v9 Provider Schema Compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the production-equivalent `current` contract-planner canary and campaign planner bind a Gemini-compatible compact schema without weakening canonical response validation.

**Architecture:** Keep `atomic_contract_planner_response_schema()` unchanged as the strict acceptance contract. Add one pure projection in the shared provider boundary and apply it only inside `build_contract_planning_provider(...)`, so campaign execution and both canary modes share the same binding path while evidence qualification remains untouched.

**Tech Stack:** Python 3.13, Pydantic 2.12, LangChain Google GenAI 4.1, Google GenAI 1.55, pytest, Ruff.

## Global Constraints

- Keep exactly one contract-planning provider call and do not add retries.
- Do not change `_PlannerDecision`, `QueryContract`, planner prompts, routing, budget admission, or semantic validation.
- Remove only `additionalProperties`, `title`, `default`, `minLength`, `maxLength`, `minItems`, `maxItems`, `minimum`, and `maximum` from the provider projection.
- Preserve `$defs`, `$ref`, `properties`, `required`, `items`, `enum`, `anyOf`, nullability, and all field names.
- Evidence qualification must continue binding its existing schema byte-for-byte.
- Use type hints on every modified function signature.

---

## File Map

- Modify `data_base/agentic_v9/provider_boundary.py`: own the pure compact projection and apply it only to contract-planning provider construction.
- Modify `tests/test_agentic_v9_provider_boundary.py`: lock recursive keyword removal, structural preservation, campaign binding, and evidence-qualification isolation.
- Modify `tests/test_agentic_v9_campaign_runtime.py`: assert the actual LangChain bind receives the compact schema while returning the raw provider response unchanged.
- Verify `tests/test_agentic_v9_contract_planner_canary.py`: retain the existing proof that current/minimal modes use the shared provider builder exactly once.
- Modify `docs/agentic-v9-smoke-verification.md`: explain provider projection versus canonical post-response validation.

### Task 1: Compact the contract-planner provider schema

**Files:**
- Modify: `data_base/agentic_v9/provider_boundary.py`
- Modify: `tests/test_agentic_v9_provider_boundary.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `docs/agentic-v9-smoke-verification.md`

**Interfaces:**
- Consumes: `build_contract_planning_provider(*, response_schema: Mapping[str, Any]) -> Any` and `atomic_contract_planner_response_schema() -> dict[str, Any]`.
- Produces: `project_contract_planner_provider_schema(schema: Mapping[str, Any]) -> dict[str, Any]`.

- [ ] **Step 1: Write the recursive projection RED test**

Add a test to `tests/test_agentic_v9_provider_boundary.py` that calls the wished-for API with nested dictionaries and lists:

```python
def test_contract_planner_provider_schema_removes_only_generation_limits() -> None:
    canonical = atomic_contract_planner_response_schema()

    projected = provider_boundary_module.project_contract_planner_provider_schema(
        canonical
    )

    serialized = json.dumps(projected, sort_keys=True)
    for keyword in (
        "additionalProperties",
        "title",
        "default",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
    ):
        assert f'"{keyword}"' not in serialized

    assert projected["properties"].keys() == canonical["properties"].keys()
    assert projected["required"] == canonical["required"]
    assert projected["properties"]["comparison"]["anyOf"][0]["$ref"] == (
        canonical["properties"]["comparison"]["anyOf"][0]["$ref"]
    )
    assert canonical["additionalProperties"] is False
    assert canonical["properties"]["confidence"]["maximum"] == 1.0
```

Add `import json` to the test module. This test also proves the projection does not mutate the canonical schema.

- [ ] **Step 2: Update provider-binding RED assertions**

Change `test_shared_contract_planning_provider_owns_model_and_schema_binding` so its fixture contains a removable key and expects the projected copy at `observed["schema"]`. Keep `test_shared_evidence_qualification_provider_owns_model_and_schema_binding` unchanged so it continues expecting its original `additionalProperties: false` values.

In `tests/test_agentic_v9_campaign_runtime.py`, change the real bind assertion to:

```python
assert captured["response_json_schema"] == (
    provider_boundary_module.project_contract_planner_provider_schema(
        atomic_contract_planner_response_schema()
    )
)
assert atomic_contract_planner_response_schema()["additionalProperties"] is False
```

- [ ] **Step 3: Run RED and confirm the missing projection is the cause**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py -k "provider_schema or contract_planning_provider_binds_schema" -q
```

Expected: failure because `project_contract_planner_provider_schema` does not exist and the shared builder still binds the canonical schema unchanged. No production file may be edited before this failure is observed.

- [ ] **Step 4: Implement the minimal pure projection**

In `data_base/agentic_v9/provider_boundary.py`, add:

```python
_CONTRACT_PROVIDER_OMITTED_SCHEMA_KEYS = frozenset(
    {
        "additionalProperties",
        "title",
        "default",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
    }
)


def _project_contract_schema_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _project_contract_schema_value(item)
            for key, item in value.items()
            if key not in _CONTRACT_PROVIDER_OMITTED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [_project_contract_schema_value(item) for item in value]
    return value


def project_contract_planner_provider_schema(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Return Gemini-compatible generation guidance without weakening validation."""
    return _project_contract_schema_value(schema)
```

Apply it only in `build_contract_planning_provider(...)`:

```python
return bind_json_schema(
    get_llm("synthesizer"),
    schema=project_contract_planner_provider_schema(response_schema),
)
```

Do not modify `build_evidence_qualification_provider(...)`.

- [ ] **Step 5: Run focused GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py -k "provider_schema or contract_planning_provider_binds_schema" -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Lock canonical rejection behavior**

Add one parameterized regression to `tests/test_agentic_v9_contract_planner.py` using a fully valid planner response as the base. Mutate it in four cases: add a root `extra_field`, set the requirement description to 513 characters, create nine evidence requirements, and set `confidence` to `1.1`. Assert that `contract_planner_module._parse_decision(...)` raises `PlannerSchemaValidationError` in every case:

```python
@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"extra_field": "forbidden"}),
        lambda payload: payload["evidence_requirements"][0].update(
            {"description": "x" * 513}
        ),
        lambda payload: payload.update(
            {"evidence_requirements": payload["evidence_requirements"] * 9}
        ),
        lambda payload: payload.update({"confidence": 1.1}),
    ],
)
def test_canonical_planner_validation_remains_strict_after_provider_projection(
    mutate: Any,
) -> None:
    payload = {
        "evidence_requirements": [{"description": "Find the answer."}],
        "synthesis_obligations": [],
        "response_constraints": [],
        "comparison": None,
        "confidence": 1.0,
    }
    mutate(payload)

    with pytest.raises(contract_planner_module.PlannerSchemaValidationError):
        contract_planner_module._parse_decision(
            SimpleNamespace(content=json.dumps(payload))
        )
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_contract_planner.py -q
```

Expected: all tests pass with the canonical strict schema unchanged.

- [ ] **Step 7: Document the provider/acceptance boundary**

Under the canary section in `docs/agentic-v9-smoke-verification.md`, add a short paragraph stating that the current-schema canary and campaign use the same compact generation schema, while `_PlannerDecision` and semantic validation remain authoritative. Do not change the documented commands or success matrix.

- [ ] **Step 8: Run the complete affected verification gate**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_contract_planner_canary.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_qualification_canary.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/provider_boundary.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py
git diff --check
```

Expected: pytest and Ruff exit 0; `git diff --check` emits no errors. Existing third-party deprecation warnings may be reported separately but must not be attributed to this change.

- [ ] **Step 9: Review and commit the implementation**

Confirm `git diff --stat` contains only the four Task 1 files, then commit:

```powershell
git add -- data_base/agentic_v9/provider_boundary.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py docs/agentic-v9-smoke-verification.md
git commit -m "fix(agentic-v9): compact planner provider schema"
git status --short
```

Expected: exactly one implementation commit and a clean tracked worktree.

- [ ] **Step 10: Deployment checkpoint**

After rebuilding the backend image, run:

```powershell
docker compose exec -T -w /app backend python scripts/agentic_v9_contract_planner_canary.py --schema current --model-config-json /tmp/agentic-v9-model-config-3.1.json
docker compose exec -T -w /app backend python scripts/agentic_v9_evidence_qualification_canary.py --model-config-json /tmp/agentic-v9-model-config-3.1.json --invoke
```

Expected: both commands return `"success": true`. Do not claim real-server completion from local mocked tests alone.
