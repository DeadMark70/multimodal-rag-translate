# Agentic v9 Comparison Planner Structured Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Agentic v9 comparison planner produce a compact, schema-constrained Gemini response with thinking disabled while preserving one-call budgeting, token accounting, semantic validation, and fail-soft base retrieval.

**Architecture:** Replace the verbose untrusted planner payload with a compact Pydantic transport model and deterministically promote it into the unchanged trusted `ComparisonPlan`. Move the planner wording into the standard Agentic RAG prompt registry, then bind Gemini native JSON schema only for purpose `agentic_v9_comparison_plan` while retaining the raw LangChain response and `usage_metadata`. Version the Agentic v9 execution profile after all focused tests pass.

**Tech Stack:** Python 3.13, Pydantic v2, LangChain `ChatGoogleGenerativeAI`, Gemini native structured output, pytest/pytest-asyncio, Ruff.

## Global Constraints

- Do not add an LLM call, retry, parser-repair call, or thinking requirement.
- Do not change Hybrid retrieval, reranking, per-task top-k, final context packing, source authorization, graph/visual execution, or final synthesis.
- Keep `BudgetedLlmInvoker` as the only Agentic v9 provider-attempt boundary.
- Preserve raw LangChain response objects so `usage_metadata` remains available to token accounting.
- Keep the existing trusted `ComparisonPlan` and `ComparisonSubject` schemas unchanged.
- Planner failures must remain fail-soft and must preserve base retrieval.
- Do not add frontend, API, database, or historical-result migrations.
- Do not create an OpenRouter abstraction in this change.
- Use type hints on every new or modified function signature.
- Use `core/prompt_loader.py` for production prompt access.
- Stage and commit only the files named by the current task; preserve unrelated untracked workspace files.

---

## File Map

- `data_base/agentic_v9/comparison_planner.py`: compact provider transport, response schema export, deterministic subject promotion, and existing guards.
- `prompts/agentic_rag_prompts.json`: registered semantic-only comparison planner system/user prompts.
- `prompts/agentic_v9_comparison_planner.json`: delete after the registered prompts replace it.
- `core/providers.py`: provider-facade helper that binds a JSON schema without bypassing the configured provider.
- `evaluation/agentic_v9_campaign_runtime.py`: select schema-bound provider only for comparison planning and stop passing source names into planner classification.
- `evaluation/retrieval_profiles.py`: version the changed Agentic v9 benchmark condition.
- `tests/test_agentic_v9_comparison_planner.py`: compact transport, deterministic promotion, diagnostics, prompt messages, and fail-soft unit coverage.
- `tests/test_agentic_rag_prompts.py`: prompt registry contract and formatting coverage.
- `tests/test_agentic_v9_provider_boundary.py`: purpose-specific schema binding and raw usage preservation.
- `tests/test_agentic_v9_campaign_runtime.py`: end-to-end planner overlay/fallback behavior and profile persistence.
- `tests/test_evaluation_retrieval_profiles.py`: exact profile-version assertions.

---

### Task 1: Compact Transport Schema and Deterministic Promotion

**Files:**
- Modify: `data_base/agentic_v9/comparison_planner.py:77-128`
- Modify: `data_base/agentic_v9/comparison_planner.py:297-406`
- Test: `tests/test_agentic_v9_comparison_planner.py`

**Interfaces:**
- Consumes: existing `ComparisonPlan`, `ComparisonSubject`, `_contains_explicit_span()`, `_reject_invented_numbers()`, and sanitized fallback diagnostics.
- Produces: `comparison_planner_response_schema() -> dict[str, Any]` and a compact provider payload `{is_comparison, subjects[{name, query}], dimensions, qualification}`.
- Preserves: `ComparisonPlanner.plan(...) -> ComparisonPlannerOutcome` and the trusted domain models.

- [ ] **Step 1: Rewrite the test payload helpers for the compact provider contract**

Replace the provider helper in `tests/test_agentic_v9_comparison_planner.py` with:

```python
def _planner_subject(
    name: str,
    *,
    retrieval_query: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "query": retrieval_query or name,
    }


def _payload(
    *,
    subjects: list[dict[str, object]] | None = None,
    dimensions: list[object] | None = None,
) -> str:
    return json.dumps(
        {
            "is_comparison": True,
            "subjects": subjects
            or [
                _planner_subject(
                    "nnMamba",
                    retrieval_query="nnMamba parameters FLOPs",
                ),
                _planner_subject(
                    "EfficientMedNeXt-L",
                    retrieval_query="EfficientMedNeXt-L parameters FLOPs",
                ),
            ],
            "dimensions": dimensions
            or ["parameters", "FLOPs", "computational efficiency"],
            "qualification": "cross-paper relative comparison",
        }
    )
```

Update existing test call sites so they no longer pass provider-generated IDs,
aliases, roles, or question spans. Replace the old object-normalization success
test with a rejection test: object-valued `dimensions` must now produce
`schema_violation` at `transport_schema`.

- [ ] **Step 2: Add failing schema and deterministic-promotion tests**

Add `import re`, import `comparison_planner_response_schema`, and add:

```python
def test_comparison_planner_response_schema_requires_string_dimensions() -> None:
    schema = comparison_planner_response_schema()

    assert schema["properties"]["dimensions"]["items"]["type"] == "string"
    subject_ref = schema["properties"]["subjects"]["items"]["$ref"]
    subject_key = subject_ref.rsplit("/", 1)[-1]
    subject_schema = schema["$defs"][subject_key]
    assert set(subject_schema["properties"]) == {"name", "query"}


@pytest.mark.asyncio
async def test_compact_subjects_receive_stable_backend_ids() -> None:
    first = await ComparisonPlanner(llm_invoker=_Invoker(_payload())).plan(
        question=Q4,
        authorized_source_names=[],
        timeout_seconds=1,
    )
    second = await ComparisonPlanner(llm_invoker=_Invoker(_payload())).plan(
        question=Q4,
        authorized_source_names=[],
        timeout_seconds=1,
    )

    assert first.status == second.status == "planned"
    assert first.plan is not None and second.plan is not None
    first_ids = [subject.subject_id for subject in first.plan.subjects]
    second_ids = [subject.subject_id for subject in second.plan.subjects]
    assert first_ids == second_ids
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", value) for value in first_ids)
```

Also retain focused cases for two subjects, three subjects, duplicate names,
unanchored names, numeric invention, and malformed payload diagnostics. For the
single-entity Q3 contract, make the fake provider return
`{"is_comparison": false, "subjects": [], "dimensions": []}` and assert the
existing `not_comparison` fallback; do not add lexical entity heuristics.

- [ ] **Step 3: Run the new tests and verify the expected failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_comparison_planner.py -q
```

Expected: FAIL because `comparison_planner_response_schema` does not exist and
the existing transport model still requires `subject_id`, `display_name`,
`retrieval_query`, and `subject_role`.

- [ ] **Step 4: Implement the compact transport models and schema export**

Replace the untrusted subject fields with:

```python
class _PlannerSubjectPayload(BaseModel):
    """Untrusted compact subject emitted by the provider."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        min_length=1,
        max_length=160,
        description="Exact independent entity name copied from the question.",
    )
    query: str = Field(
        min_length=1,
        max_length=512,
        description="Retrieval query that explicitly names this entity.",
    )


class _PlannerPayload(BaseModel):
    """Provider JSON before deterministic comparison-plan promotion."""

    model_config = ConfigDict(extra="ignore")

    is_comparison: bool = Field(
        description="True only for two or more independent named entities."
    )
    subjects: list[_PlannerSubjectPayload] = Field(default_factory=list, max_length=4)
    dimensions: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Comparison dimensions explicitly requested by the question.",
    )
    qualification: str | None = Field(
        default=None,
        max_length=512,
        description="Optional scope qualification copied or summarized from the question.",
    )


def comparison_planner_response_schema() -> dict[str, Any]:
    """Return the exact native provider schema for comparison planning."""
    return _PlannerPayload.model_json_schema()
```

Delete `_DIMENSION_LABEL_KEYS`, `_normalize_planner_dimension()`, and the
`dimensions` pre-validator. Native structured output owns transport shape; the
Pydantic validator remains the application boundary if malformed data still
arrives.

- [ ] **Step 5: Implement stable subject IDs and compact promotion**

Add `hashlib` to the standard-library imports and implement:

```python
def _stable_subject_id(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).strip().casefold()
    ascii_slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    if ascii_slug:
        return ascii_slug[:80]
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"subject-{digest}"
```

Replace `_validated_subjects()` with promotion based only on explicit names:

```python
def _validated_subjects(
    question: str,
    candidates: Sequence[_PlannerSubjectPayload],
) -> list[ComparisonSubject]:
    accepted: list[ComparisonSubject] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not _contains_explicit_span(question, candidate.name):
            continue
        identity = _normalized_identity(candidate.name)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        try:
            accepted.append(
                ComparisonSubject(
                    subject_id=_stable_subject_id(candidate.name),
                    display_name=candidate.name,
                    aliases=[],
                    retrieval_query=candidate.query,
                )
            )
        except ValidationError:
            continue
    return accepted
```

Update `_reject_invented_numbers()` and all downstream references only where
field names changed; do not alter the numeric policy or trusted plan.

- [ ] **Step 6: Run focused planner tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_comparison_planner.py -q
```

Expected: PASS, including explicit rejection of object-valued dimensions and
stable promotion into unchanged domain types.

- [ ] **Step 7: Commit Task 1**

```powershell
git add data_base/agentic_v9/comparison_planner.py tests/test_agentic_v9_comparison_planner.py
git commit -m "refactor(agentic-v9): simplify comparison planner transport"
```

---

### Task 2: Registry-Managed Semantic-Only Planner Prompt

**Files:**
- Modify: `prompts/agentic_rag_prompts.json`
- Delete: `prompts/agentic_v9_comparison_planner.json`
- Modify: `data_base/agentic_v9/comparison_planner.py:28-32`
- Modify: `data_base/agentic_v9/comparison_planner.py:145-166`
- Modify: `data_base/agentic_v9/comparison_planner.py:259-277`
- Test: `tests/test_agentic_rag_prompts.py`
- Test: `tests/test_agentic_v9_comparison_planner.py`

**Interfaces:**
- Consumes: `format_agentic_rag_prompt(key: str, **variables: Any) -> str`.
- Produces: prompt keys `comparison_planner_system` and `comparison_planner_user`.
- Changes: `ComparisonPlanner.plan(*, question: str, timeout_seconds: float)` no longer accepts source names because source authorization is not a planner input.

- [ ] **Step 1: Add failing prompt-registry contract tests**

Extend `EXPECTED_KEYS` and `EXPECTED_REQUIRED_VARIABLES` in
`tests/test_agentic_rag_prompts.py`:

```python
EXPECTED_KEYS = {
    "planner",
    "graph_planner",
    "followup",
    "refine_query",
    "intent_classifier",
    "conflict_arbitration",
    "synthesizer",
    "academic_report",
    "retrieval_eval",
    "faithfulness_eval",
    "detailed_eval",
    "pure_llm_eval",
    "fact_state",
    "evidence_extract",
    "comparison_planner_system",
    "comparison_planner_user",
}

EXPECTED_REQUIRED_VARIABLES = {
    "planner": ["question"],
    "graph_planner": ["question"],
    "followup": ["original_question", "current_findings", "existing_questions"],
    "refine_query": ["original_question", "evaluation_reason", "failed_answer"],
    "intent_classifier": ["question"],
    "conflict_arbitration": ["sub_results"],
    "synthesizer": ["original_question", "sub_results"],
    "academic_report": ["original_question", "sub_results"],
    "retrieval_eval": ["question", "documents"],
    "faithfulness_eval": ["question", "documents", "answer"],
    "detailed_eval": ["question", "documents", "answer"],
    "pure_llm_eval": ["question", "answer", "ground_truth"],
    "fact_state": ["question", "source_doc_ids", "answer"],
    "evidence_extract": ["question", "unresolved_slots", "source_evidence"],
    "comparison_planner_system": [],
    "comparison_planner_user": ["question"],
}
```

Extend the format smoke:

```python
comparison_system = format_agentic_rag_prompt("comparison_planner_system")
comparison_user = format_agentic_rag_prompt(
    "comparison_planner_user",
    question="Compare nnMamba and EfficientMedNeXt-L.",
)

assert "independent named entities" in comparison_system
assert "Compare nnMamba and EfficientMedNeXt-L." in comparison_user
assert "authorized source" not in comparison_user.casefold()
```

- [ ] **Step 2: Run prompt tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_rag_prompts.py -q
```

Expected: FAIL because both prompt keys are absent.

- [ ] **Step 3: Add the registered semantic-only prompts**

Add these entries inside `prompts/agentic_rag_prompts.json` under `prompts`:

```json
"comparison_planner_system": {
  "version": 1,
  "description": "Classify explicit independent comparison entities for Agentic v9 retrieval planning.",
  "required_variables": [],
  "template": "Analyze retrieval planning only. Set is_comparison=true only when the question compares, relates, or jointly judges at least two independent named entities such as models, methods, datasets, or documents. Two claims, metrics, capabilities, conditions, or prompt types about one entity are not independent comparison subjects. For each independent entity, copy its name from the question and create one subject-specific retrieval query. List only comparison dimensions explicitly requested by the question. Do not answer the question, choose a winner, name source files, or invent values."
},
"comparison_planner_user": {
  "version": 1,
  "description": "Supply the original question to the Agentic v9 comparison planner.",
  "required_variables": ["question"],
  "template": "Question: {question}"
}
```

Do not include an inline JSON example. The native response schema added in Task
3 owns serialization.

- [ ] **Step 4: Rewire planner message construction and remove source names**

In `comparison_planner.py`:

- remove `Path` and `_PROMPT_PATH`;
- import `format_agentic_rag_prompt` from `core.prompt_loader`;
- remove `authorized_source_names` from `ComparisonPlanner.plan()` and
  `_planner_messages()`;
- build messages with:

```python
def _planner_messages(*, question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": format_agentic_rag_prompt("comparison_planner_system"),
        },
        {
            "role": "user",
            "content": format_agentic_rag_prompt(
                "comparison_planner_user",
                question=question,
            ),
        },
    ]
```

Update every planner unit-test invocation to omit
`authorized_source_names=[]`. Replace the missing-file test by monkeypatching
`comparison_planner_module.format_agentic_rag_prompt`. Import
`PromptConfigError` from `core.prompt_loader` and use:

```python
def fail_prompt(*args: object, **kwargs: object) -> str:
    del args, kwargs
    raise PromptConfigError("comparison prompt unavailable")


monkeypatch.setattr(
    comparison_planner_module,
    "format_agentic_rag_prompt",
    fail_prompt,
)
```

Assert `provider_error` and zero invoker calls.

Delete `prompts/agentic_v9_comparison_planner.json` only after registry tests
pass.

- [ ] **Step 5: Run prompt and planner tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_rag_prompts.py tests/test_agentic_v9_comparison_planner.py -q
```

Expected: PASS. Planner messages contain the question but no source-name list or
inline JSON sample.

- [ ] **Step 6: Commit Task 2**

```powershell
git add prompts/agentic_rag_prompts.json prompts/agentic_v9_comparison_planner.json data_base/agentic_v9/comparison_planner.py tests/test_agentic_rag_prompts.py tests/test_agentic_v9_comparison_planner.py
git commit -m "refactor(agentic-v9): simplify comparison planner prompt"
```

---

### Task 3: Purpose-Specific Gemini Native Schema Binding

**Files:**
- Modify: `core/providers.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:23-42`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:280-300`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:1370-1371`
- Test: `tests/test_agentic_v9_provider_boundary.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**
- Consumes: `comparison_planner_response_schema() -> dict[str, Any]` from Task 1 and configured LLM instances from `core.providers.get_llm()`.
- Produces: `bind_json_schema(llm: Any, *, schema: dict[str, Any]) -> Any` in the provider facade.
- Preserves: one `BudgetedLlmInvoker` attempt, phase-policy scope, raw response object, and `usage_metadata`.

- [ ] **Step 1: Add failing provider-facade binding tests**

In `tests/test_agentic_v9_provider_boundary.py`, import `ProviderError` and
`bind_json_schema` from `core.providers`, then add:

```python
def test_bind_json_schema_uses_native_json_configuration() -> None:
    captured: dict[str, object] = {}

    class _Bindable:
        def bind(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return "bound-provider"

    result = bind_json_schema(
        _Bindable(),
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    assert result == "bound-provider"
    assert captured == {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
    }


def test_bind_json_schema_rejects_provider_without_bind() -> None:
    with pytest.raises(ProviderError, match="native JSON schema"):
        bind_json_schema(object(), schema={"type": "object"})
```

- [ ] **Step 2: Add failing purpose-selection and usage-preservation tests**

In `tests/test_agentic_v9_campaign_runtime.py`, monkeypatch the runtime module's
`get_llm` and verify `_provider_for_purpose()`:

```python
@pytest.mark.asyncio
async def test_comparison_provider_binds_schema_without_replacing_raw_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = SimpleNamespace(
        content='{"is_comparison":false,"subjects":[],"dimensions":[]}',
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )
    captured: dict[str, object] = {}

    class _Provider:
        def bind(self, **kwargs: object) -> "_Provider":
            captured.update(kwargs)
            return self

        async def ainvoke(self, messages: object) -> object:
            del messages
            return raw

    monkeypatch.setattr(runtime_module, "get_llm", lambda purpose: _Provider())

    provider = runtime_module._provider_for_purpose("agentic_v9_comparison_plan")
    response = await provider.ainvoke([])

    assert captured["response_mime_type"] == "application/json"
    assert captured["response_schema"] == comparison_planner_response_schema()
    assert response is raw
    assert response.usage_metadata["total_tokens"] == 10
```

Add a second test asserting `_provider_for_purpose("agentic_v9_final_answer")`
does not call `bind()`.

- [ ] **Step 3: Run provider tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py -q
```

Expected: FAIL because `bind_json_schema` is absent and
`_provider_for_purpose()` currently ignores its purpose.

- [ ] **Step 4: Add the provider-facade helper**

In `core/providers.py` add:

```python
def bind_json_schema(llm: Any, *, schema: dict[str, Any]) -> Any:
    """Bind native JSON output while preserving the provider response object."""
    binder = getattr(llm, "bind", None)
    if not callable(binder):
        raise ProviderError("LLM provider does not support native JSON schema binding")
    return binder(
        response_mime_type="application/json",
        response_schema=schema,
    )
```

Do not call Google SDK classes directly and do not return `response.parsed`.

- [ ] **Step 5: Bind only the comparison-plan provider**

In `evaluation/agentic_v9_campaign_runtime.py` import
`bind_json_schema` and `comparison_planner_response_schema`, then replace the
default factory with:

```python
def _provider_for_purpose(purpose: str) -> Any:
    provider = get_llm("synthesizer")
    if purpose != "agentic_v9_comparison_plan":
        return provider
    return bind_json_schema(
        provider,
        schema=comparison_planner_response_schema(),
    )
```

Update the runtime planner call to the Task 2 signature:

```python
outcome = await planner.plan(
    question=question,
    timeout_seconds=min(64.0, deadline.remaining_seconds()),
)
```

Do not catch binding errors in `_provider_for_purpose()`. The existing
`ComparisonPlanner.plan()` provider-error boundary must convert them into a
single fail-soft fallback without retry.

- [ ] **Step 6: Run provider, accounting, and runtime tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_token_cost.py -q
```

Expected: PASS. Existing token tests confirm the response remains observable and
the comparison planner still produces exactly one terminal provider attempt.

- [ ] **Step 7: Commit Task 3**

```powershell
git add core/providers.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "feat(agentic-v9): enforce native comparison planner schema"
```

---

### Task 4: Runtime Regression Coverage and Benchmark Identity

**Files:**
- Modify: `evaluation/retrieval_profiles.py:21-27`
- Modify: `tests/test_evaluation_retrieval_profiles.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_smoke_runner.py` only if its exact profile fixture requires the new suffix.

**Interfaces:**
- Consumes: compact planner, registered prompt, and purpose-specific binding from Tasks 1-3.
- Produces: distinct Agentic v9 execution profiles ending in `comparison_structured_v2` for open-corpus and explicitly scoped runs.
- Preserves: context policy version `v5_final_context_soft_pack_r1` and all retrieval/reranking parameters in the profile.

- [ ] **Step 1: Add failing profile-version assertions**

Update `tests/test_evaluation_retrieval_profiles.py`:

```python
from evaluation.retrieval_profiles import (
    AGENTIC_V9_EXPLICIT_SCOPE_PROFILE,
    AGENTIC_V9_OPEN_CORPUS_PROFILE,
    agentic_v9_execution_profile,
)


def test_agentic_v9_profiles_identify_structured_comparison_planner() -> None:
    assert AGENTIC_V9_OPEN_CORPUS_PROFILE.endswith("comparison_structured_v2")
    assert AGENTIC_V9_EXPLICIT_SCOPE_PROFILE.endswith("comparison_structured_v2")
    assert agentic_v9_execution_profile(open_user_corpus=True) == (
        AGENTIC_V9_OPEN_CORPUS_PROFILE
    )
    assert agentic_v9_execution_profile(open_user_corpus=False) == (
        AGENTIC_V9_EXPLICIT_SCOPE_PROFILE
    )
```

Retain the existing prefix text so Hybrid 8, rerank 8, diverse tail 2, task
top-4, and final pack R1 remain identifiable.

- [ ] **Step 2: Add runtime integration cases for overlay and fail-soft fallback**

Update planner fake responses in `tests/test_agentic_v9_campaign_runtime.py` to
the compact payload. Keep or add assertions that:

```python
assert v9["comparison_planner"]["status"] == "planned"
assert len(v9["query_contract"]["comparison_plan"]["subjects"]) == 2
assert all(task["subject_id"] for task in v9["query_contract"]["retrieval_tasks"])
```

For malformed/schema-binding failure, assert:

```python
assert v9["comparison_planner"]["status"] == "fallback"
assert "comparison_plan" not in v9["query_contract"]
assert result.documents
```

Use the existing injected provider factory; do not invoke Gemini in unit tests.

- [ ] **Step 3: Run profile/runtime tests and verify profile failure first**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_retrieval_profiles.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_smoke_runner.py -q
```

Expected before profile modification: FAIL on the new structured-comparison
suffix. Runtime behavior tests should otherwise exercise the compact payload.

- [ ] **Step 4: Version both Agentic v9 profiles**

In `evaluation/retrieval_profiles.py`, keep the existing profile body and append
the same suffix:

```python
AGENTIC_V9_OPEN_CORPUS_PROFILE = (
    "agentic_eval_v9_open_corpus_hybrid8_rerank8_diverse_tail2_top4_"
    "finalpack_r1_comparison_structured_v2"
)
AGENTIC_V9_EXPLICIT_SCOPE_PROFILE = (
    "agentic_eval_v9_explicit_scope_hybrid8_rerank8_diverse_tail2_top4_"
    "finalpack_r1_comparison_structured_v2"
)
```

Do not change `AGENTIC_V9_CONTEXT_POLICY_VERSION`; context packing did not
change.

- [ ] **Step 5: Run the complete focused verification set**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_comparison_planner.py tests/test_agentic_rag_prompts.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_smoke_runner.py tests/test_agentic_v9_budget_feasibility.py tests/test_evaluation_retrieval_profiles.py tests/test_evaluation_token_cost.py tests/test_evaluation_analytics_context.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/comparison_planner.py core/providers.py evaluation/agentic_v9_campaign_runtime.py evaluation/retrieval_profiles.py tests/test_agentic_v9_comparison_planner.py tests/test_agentic_rag_prompts.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_retrieval_profiles.py
```

Expected: all selected tests PASS and Ruff reports no new violations. If a
legacy unrelated test baseline fails, record the exact test and error rather
than weakening the new assertions.

- [ ] **Step 6: Inspect the final diff for forbidden scope changes**

Run:

```powershell
git diff --check
git diff --stat
git status --short
```

Verify that no retrieval, reranker, source authorization, context packer,
synthesis, frontend, migration, or database file appears in the diff.

- [ ] **Step 7: Commit Task 4**

```powershell
git add evaluation/retrieval_profiles.py tests/test_evaluation_retrieval_profiles.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_smoke_runner.py
git commit -m "test(agentic-v9): version structured comparison planner"
```

If `tests/test_agentic_v9_smoke_runner.py` did not require modification, omit it
from `git add`.

---

## Post-Implementation Server Smoke Gate

This gate uses the deployed Gemini provider and is intentionally not simulated
by unit tests.

- [ ] Run Q3, Q4, and Q14 with Agentic v9, thinking disabled, repeat 3, batch 1.
- [ ] Confirm all nine work items produce usable results or an explicit
  planner-only fail-soft fallback.
- [ ] Confirm `dimensions/value_error` count is zero.
- [ ] Confirm Q4 records two independent comparison subjects.
- [ ] Confirm Q14 records three independent comparison subjects.
- [ ] Confirm Q3 does not promote claims/conditions about one entity into an
  entity comparison overlay.
- [ ] Confirm each eligible run records at most one `comparison_plan` LLM call.
- [ ] Confirm token accounting and phase attribution are complete.
- [ ] Confirm fallback runs retain base retrieval documents/evidence.

Do not run the full sixteen-question paired evaluation until this smoke gate
passes. If the smoke fails semantically while schema compliance is clean, keep
the transport change isolated and diagnose the prompt classification separately;
do not add parser exceptions or extra calls.

## Completion Handoff

Report:

- the four implementation commit IDs;
- focused pytest pass/fail counts;
- Ruff result;
- whether token usage metadata survived native schema binding;
- any files intentionally omitted from Task 4 staging;
- the exact server smoke configuration the user should run.
