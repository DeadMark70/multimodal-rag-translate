# RAG QA Prompt Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move production RAG question-answering prompts out of Python source into validated JSON config, while preserving current runtime behavior and test coverage.

**Architecture:** Add a small prompt-loading boundary in `core/` that reads versioned JSON prompt definitions from `prompts/`. `data_base/RAG_QA_service.py` will request templates by key and continue formatting them with the existing variables, so retrieval, LLM invocation, visual verification, and response contracts stay unchanged.

**Tech Stack:** FastAPI backend, Python 3.13, stdlib `json` and `string.Formatter`, Pydantic-style validation via plain typed dataclasses or small functions, pytest, ruff.

---

## Current Workspace Confirmation

### Backend cleanup changes currently present

The working tree is dirty and contains the completed backend cleanup batch:

- Modified: `graph_rag/router.py`
- Added: `graph_rag/maintenance.py`
- Modified: `pdfserviceMD/router.py`
- Added: `pdfserviceMD/indexing_tasks.py`
- Modified: `pdfserviceMD/service.py`
- Modified: `pytest.ini`
- Modified tests:
  - `tests/test_graph_router_copy.py`
  - `tests/test_graph_router_rebuild_full.py`
  - `tests/test_pdfservice_background_processing.py`
  - `tests/test_pdfservice_manual_translation.py`
  - `tests/test_router_boundaries.py`
- Added plan directory content under `docs/superpowers/`

Behavioral summary:

- GraphRAG maintenance jobs now live in `graph_rag/maintenance.py`.
- `graph_rag/router.py` now mainly handles API I/O, authorization dependency, active-job checks, response shaping, and background task scheduling.
- PDF post-processing and retry-index background work now lives in `pdfserviceMD/indexing_tasks.py`.
- Retry-index eligibility and scheduling payload preparation now lives in `pdfserviceMD/service.py` as `prepare_retry_index_context`.
- Router boundary tests now block router-to-router imports, service/import-router regressions, and router-defined `*_task` background jobs.
- `pytest.ini` now redirects pytest cache to `output/test_tmp/.pytest_cache`, removing the previous `.pytest_cache` permission warning.

Verification already run after the cleanup:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_router_boundaries.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py tests/test_pdfservice_manual_translation.py tests/test_pdfservice_background_processing.py tests/test_rag_startup.py -q
```

Expected current result:

```text
39 passed, 24 warnings
```

```powershell
.\.venv\Scripts\python.exe -m ruff check graph_rag\maintenance.py graph_rag\router.py pdfserviceMD\router.py pdfserviceMD\service.py pdfserviceMD\indexing_tasks.py tests\test_graph_router_copy.py tests\test_graph_router_rebuild_full.py tests\test_router_boundaries.py tests\test_pdfservice_background_processing.py tests\test_pdfservice_manual_translation.py
```

Expected current result:

```text
All checks passed!
```

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -c "import main; print(main.app.title)"
```

Expected current result:

```text
PDF Translation & RAG API
```

Known broader baseline issues not caused by this cleanup:

- Full `pytest -q` collection currently fails on legacy `experiments.evaluation_pipeline` / RAGAS imports because `tiktoken` tries to download `o200k_base.tiktoken` and local SSL certificate verification fails.
- A broader production/backend run excluding legacy experiments/RAGAS collection blockers produced `434 passed, 4 failed`.
- The 4 known failures are:
  - `tests/test_gemini_layering.py::test_business_logic_does_not_import_get_llm_from_llm_factory`
  - `tests/test_migration_metadata.py::test_conversations_has_metadata_column`
  - `tests/test_planner_intent_hardset.py::test_classify_question_intent_on_ragas_hardset_v2_samples`
  - `tests/test_reranker.py::TestRerankerSingleton::test_singleton_uses_cpu_when_gpu_memory_is_too_small`

### Prompt externalization scope decision

First implementation batch should only externalize production RAG QA answer prompts in `data_base/RAG_QA_service.py`:

- `PLAIN_RAG_PROMPT_TEMPLATE`
- `ADVANCED_RAG_PROMPT_TEMPLATE`
- `VISUAL_TOOL_INSTRUCTION`
- Visual verification synthesis prompt currently built inside `_execute_visual_verification_loop`

Do not include these in the first batch:

- Graph extraction prompts in `graph_rag/extractor.py`
- Community-building prompts in `graph_rag/community_builder.py`
- GraphRAG local/global/router prompts in `graph_rag/local_search.py`, `graph_rag/global_search.py`, `graph_rag/generic_mode.py`
- Image analysis prompts in `multimodal_rag/image_summarizer.py`
- Summary prompts in `core/summary_service.py`
- Query transformation/chunking prompts in `data_base/query_transformer.py`, `data_base/context_enricher.py`, `data_base/proposition_chunker.py`
- Evaluation or benchmarking prompts under `experiments/`, `evaluation/`, or `bergen/`

Reason: the first batch should be behavior-preserving and easy to verify. RAG answer prompts are the most user-facing and easiest to validate through existing `rag_answer_question` focused tests.

---

## File Structure

Create:

- `prompts/rag_qa_prompts.json`
  - Owns configurable prompt text.
  - Stores prompt keys, versions, descriptions, required variables, and templates.

- `core/prompt_loader.py`
  - Owns reading, caching, validating, and formatting prompt templates.
  - Does not know RAG business logic.

- `tests/test_prompt_loader.py`
  - Unit tests for JSON loading, key lookup, required-variable validation, and formatting.

- `tests/test_rag_qa_prompts.py`
  - Focused tests that production RAG QA prompt keys exist and include required variables.

Modify:

- `data_base/RAG_QA_service.py`
  - Replace module-level long prompt constants with calls to `core.prompt_loader`.
  - Keep current formatting variables:
    - `context_text`
    - `graph_section`
    - `history_section`
    - `question`
    - `intent_constraints`
    - `visual_instruction`
    - visual synthesis fields used in `_execute_visual_verification_loop`

- `tests/test_rag_retrieval_logic.py`
  - Add one behavior-level assertion that `rag_answer_question(..., return_docs=True)` still exposes a fully formatted prompt in `thought_process`.

Optional documentation:

- `agent.md`
  - Only update if the team decides prompt JSON becomes a permanent convention.
  - Do not update in the first implementation unless requested.

---

### Task 1: Prompt Loader Unit

**Files:**

- Create: `core/prompt_loader.py`
- Create: `tests/test_prompt_loader.py`

- [ ] **Step 1: Write failing loader tests**

Create `tests/test_prompt_loader.py`:

```python
from pathlib import Path

import pytest

from core.prompt_loader import PromptConfigError, PromptRegistry


def test_prompt_registry_formats_prompt_from_json(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        """
{
  "prompts": {
    "plain_rag_answer": {
      "version": 1,
      "description": "Plain RAG answer prompt.",
      "required_variables": ["question", "context_text"],
      "template": "Q: {question}\\nC: {context_text}"
    }
  }
}
""",
        encoding="utf-8",
    )

    registry = PromptRegistry(path)

    assert registry.format(
        "plain_rag_answer",
        question="What is RAG?",
        context_text="Retrieved chunk",
    ) == "Q: What is RAG?\nC: Retrieved chunk"


def test_prompt_registry_rejects_missing_required_variable(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        """
{
  "prompts": {
    "plain_rag_answer": {
      "version": 1,
      "description": "Plain RAG answer prompt.",
      "required_variables": ["question", "context_text"],
      "template": "Q: {question}\\nC: {context_text}"
    }
  }
}
""",
        encoding="utf-8",
    )

    registry = PromptRegistry(path)

    with pytest.raises(PromptConfigError, match="context_text"):
        registry.format("plain_rag_answer", question="What is RAG?")


def test_prompt_registry_rejects_template_variables_not_declared(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        """
{
  "prompts": {
    "broken": {
      "version": 1,
      "description": "Broken prompt.",
      "required_variables": ["question"],
      "template": "Q: {question}\\nC: {context_text}"
    }
  }
}
""",
        encoding="utf-8",
    )

    with pytest.raises(PromptConfigError, match="context_text"):
        PromptRegistry(path)


def test_prompt_registry_rejects_unknown_prompt_key(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        """
{
  "prompts": {}
}
""",
        encoding="utf-8",
    )

    registry = PromptRegistry(path)

    with pytest.raises(PromptConfigError, match="missing_prompt"):
        registry.format("missing_prompt")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_prompt_loader.py -q
```

Expected:

```text
ERROR tests/test_prompt_loader.py
ModuleNotFoundError: No module named 'core.prompt_loader'
```

- [ ] **Step 3: Implement prompt loader**

Create `core/prompt_loader.py`:

```python
"""Prompt configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Formatter
from typing import Any


class PromptConfigError(RuntimeError):
    """Raised when prompt configuration is missing or invalid."""


@dataclass(frozen=True)
class PromptDefinition:
    """One prompt template loaded from JSON."""

    key: str
    version: int
    description: str
    required_variables: tuple[str, ...]
    template: str


class PromptRegistry:
    """Loads prompt templates from a JSON config file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._prompts = self._load(path)

    def get(self, key: str) -> PromptDefinition:
        try:
            return self._prompts[key]
        except KeyError as exc:
            raise PromptConfigError(f"Prompt key not found: {key}") from exc

    def format(self, key: str, **variables: object) -> str:
        prompt = self.get(key)
        missing = sorted(set(prompt.required_variables) - set(variables))
        if missing:
            raise PromptConfigError(
                f"Prompt {key} missing required variables: {', '.join(missing)}"
            )
        return prompt.template.format(**variables)

    def _load(self, path: Path) -> dict[str, PromptDefinition]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise PromptConfigError(f"Could not read prompt config: {path}") from exc
        except json.JSONDecodeError as exc:
            raise PromptConfigError(f"Invalid prompt JSON: {path}") from exc

        raw_prompts = payload.get("prompts")
        if not isinstance(raw_prompts, dict):
            raise PromptConfigError("Prompt config must contain an object field: prompts")

        prompts: dict[str, PromptDefinition] = {}
        for key, raw in raw_prompts.items():
            if not isinstance(key, str) or not isinstance(raw, dict):
                raise PromptConfigError("Prompt entries must be objects keyed by string")
            prompts[key] = self._parse_prompt(key, raw)
        return prompts

    def _parse_prompt(self, key: str, raw: dict[str, Any]) -> PromptDefinition:
        version = raw.get("version")
        description = raw.get("description")
        required_variables = raw.get("required_variables")
        template = raw.get("template")

        if not isinstance(version, int) or version < 1:
            raise PromptConfigError(f"Prompt {key} must define positive integer version")
        if not isinstance(description, str) or not description.strip():
            raise PromptConfigError(f"Prompt {key} must define description")
        if not isinstance(required_variables, list) or not all(
            isinstance(item, str) and item for item in required_variables
        ):
            raise PromptConfigError(f"Prompt {key} must define required_variables")
        if not isinstance(template, str) or not template.strip():
            raise PromptConfigError(f"Prompt {key} must define template")

        required_tuple = tuple(required_variables)
        template_variables = {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name
        }
        undeclared = sorted(template_variables - set(required_tuple))
        if undeclared:
            raise PromptConfigError(
                f"Prompt {key} template variables are not declared: {', '.join(undeclared)}"
            )

        return PromptDefinition(
            key=key,
            version=version,
            description=description,
            required_variables=required_tuple,
            template=template,
        )


def default_prompt_config_path() -> Path:
    """Return the default production prompt config path."""
    return Path(__file__).resolve().parents[1] / "prompts" / "rag_qa_prompts.json"


@lru_cache(maxsize=1)
def get_default_prompt_registry() -> PromptRegistry:
    """Return the cached default prompt registry."""
    return PromptRegistry(default_prompt_config_path())


def format_prompt(key: str, **variables: object) -> str:
    """Format one prompt from the default registry."""
    return get_default_prompt_registry().format(key, **variables)
```

- [ ] **Step 4: Run loader tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_prompt_loader.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Run lint for new loader**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check core\prompt_loader.py tests\test_prompt_loader.py
```

Expected:

```text
All checks passed!
```

---

### Task 2: Production RAG QA Prompt JSON

**Files:**

- Create: `prompts/rag_qa_prompts.json`
- Create: `tests/test_rag_qa_prompts.py`

- [ ] **Step 1: Write failing prompt config tests**

Create `tests/test_rag_qa_prompts.py`:

```python
from core.prompt_loader import get_default_prompt_registry


def test_rag_qa_prompt_config_contains_required_prompt_keys() -> None:
    registry = get_default_prompt_registry()

    assert registry.get("plain_rag_answer").version >= 1
    assert registry.get("advanced_rag_answer").version >= 1
    assert registry.get("visual_tool_instruction").version >= 1
    assert registry.get("visual_verification_synthesis").version >= 1


def test_plain_rag_answer_prompt_requires_current_variables() -> None:
    prompt = get_default_prompt_registry().get("plain_rag_answer")

    assert prompt.required_variables == (
        "context_text",
        "graph_section",
        "history_section",
        "question",
        "intent_constraints",
    )


def test_advanced_rag_answer_prompt_requires_current_variables() -> None:
    prompt = get_default_prompt_registry().get("advanced_rag_answer")

    assert prompt.required_variables == (
        "context_text",
        "graph_section",
        "history_section",
        "question",
        "intent_constraints",
        "visual_instruction",
    )


def test_visual_synthesis_prompt_requires_current_variables() -> None:
    prompt = get_default_prompt_registry().get("visual_verification_synthesis")

    assert prompt.required_variables == (
        "context",
        "question",
        "initial_response",
        "verification_results",
    )
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_rag_qa_prompts.py -q
```

Expected:

```text
core.prompt_loader.PromptConfigError: Could not read prompt config
```

- [ ] **Step 3: Generate prompt JSON from current source prompts**

Create `prompts/rag_qa_prompts.json` by generating the JSON from the current Python constants. This avoids hand-escaping prompt newlines and preserves the current prompt wording byte-for-byte for the three existing module constants.

Run this one-time generation command during implementation:

```python
from __future__ import annotations

import json
from pathlib import Path

from data_base import RAG_QA_service as rag

visual_synthesis_template = """???????????????????????????

??????
{initial_response}

????????
{verification_results}

???????
{context}

???????
{question}

??????????????????????????????????????????"""

payload = {
    "schema_version": 1,
    "prompts": {
        "plain_rag_answer": {
            "version": 1,
            "description": "Plain production RAG answer prompt used when images are not included.",
            "required_variables": [
                "context_text",
                "graph_section",
                "history_section",
                "question",
                "intent_constraints",
            ],
            "template": rag.PLAIN_RAG_PROMPT_TEMPLATE,
        },
        "advanced_rag_answer": {
            "version": 1,
            "description": "Advanced production RAG answer prompt used for multimodal or advanced answer generation.",
            "required_variables": [
                "context_text",
                "graph_section",
                "history_section",
                "question",
                "intent_constraints",
                "visual_instruction",
            ],
            "template": rag.ADVANCED_RAG_PROMPT_TEMPLATE,
        },
        "visual_tool_instruction": {
            "version": 1,
            "description": "Instruction block that teaches the answer model how to request visual verification.",
            "required_variables": [],
            "template": rag.VISUAL_TOOL_INSTRUCTION,
        },
        "visual_verification_synthesis": {
            "version": 1,
            "description": "Prompt used to synthesize the initial answer with visual verification results.",
            "required_variables": [
                "context",
                "question",
                "initial_response",
                "verification_results",
            ],
            "template": visual_synthesis_template,
        },
    },
}

path = Path("prompts/rag_qa_prompts.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

Then inspect the generated file and confirm there are no placeholder markers such as unfinished copy instructions or temporary task notes. Use an editor search against the generated JSON before committing.

Expected: no placeholder marker text is found.

- [ ] **Step 4: Run prompt config tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_prompt_loader.py tests/test_rag_qa_prompts.py -q
```

Expected:

```text
8 passed
```

---

### Task 3: Wire RAG QA Service To Prompt Registry

**Files:**

- Modify: `data_base/RAG_QA_service.py`
- Modify: `tests/test_rag_retrieval_logic.py`

- [ ] **Step 1: Add behavior test for formatted prompt trace**

Append to `tests/test_rag_retrieval_logic.py`:

```python
@pytest.mark.asyncio
async def test_rag_answer_question_formats_prompt_from_registry() -> None:
    retriever = Mock()
    retriever.invoke.return_value = [
        Document(page_content="retrieved context", metadata={"doc_id": "doc-1"})
    ]
    llm = Mock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="ok"))

    with (
        patch("data_base.RAG_QA_service.get_llm", return_value=llm),
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={}),
        patch(
            "data_base.RAG_QA_service.get_user_retriever",
            new=AsyncMock(return_value=retriever),
        ),
        patch(
            "data_base.RAG_QA_service.fetch_document_filenames",
            new=AsyncMock(return_value={"doc-1": "demo.pdf"}),
        ),
    ):
        result = await rag_answer_question(
            question="What does the document say?",
            user_id="user-1",
            return_docs=True,
        )

    assert "What does the document say?" in result.thought_process
    assert "retrieved context" in result.thought_process
```

- [ ] **Step 2: Run test before implementation**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_logic.py::test_rag_answer_question_formats_prompt_from_registry -q
```

Expected before wiring may already pass because the Python constants still exist. If it passes, continue; this is a characterization test that protects final behavior rather than a red test.

- [ ] **Step 3: Replace direct constants with prompt loader calls**

In `data_base/RAG_QA_service.py`, add import:

```python
from core.prompt_loader import format_prompt
```

Replace plain prompt formatting:

```python
prompt_text = PLAIN_RAG_PROMPT_TEMPLATE.format(
    context_text=context_text,
    graph_section=graph_section,
    history_section=history_section,
    question=question,
    intent_constraints=intent_constraints,
)
```

with:

```python
prompt_text = format_prompt(
    "plain_rag_answer",
    context_text=context_text,
    graph_section=graph_section,
    history_section=history_section,
    question=question,
    intent_constraints=intent_constraints,
)
```

Replace advanced prompt formatting:

```python
visual_instruction = (
    VISUAL_TOOL_INSTRUCTION if enable_visual_verification and image_paths else ""
)
prompt_text = ADVANCED_RAG_PROMPT_TEMPLATE.format(
    context_text=context_text,
    graph_section=graph_section,
    history_section=history_section,
    question=question,
    intent_constraints=intent_constraints,
    visual_instruction=visual_instruction,
)
```

with:

```python
visual_instruction = (
    format_prompt("visual_tool_instruction")
    if enable_visual_verification and image_paths
    else ""
)
prompt_text = format_prompt(
    "advanced_rag_answer",
    context_text=context_text,
    graph_section=graph_section,
    history_section=history_section,
    question=question,
    intent_constraints=intent_constraints,
    visual_instruction=visual_instruction,
)
```

Replace the visual verification synthesis f-string:

```python
synthesis_prompt = f"""..."""
```

with:

```python
synthesis_prompt = format_prompt(
    "visual_verification_synthesis",
    context=context,
    question=question,
    initial_response=initial_response,
    verification_results=chr(10).join(verification_results),
)
```

Keep the old module constants for one commit only if needed for JSON generation. Remove them before the final task if no production or test imports still use them.

- [ ] **Step 4: Run RAG prompt tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_prompt_loader.py tests/test_rag_qa_prompts.py tests/test_rag_retrieval_logic.py -q
```

Expected:

```text
passed
```

---

### Task 4: Remove Old Prompt Constants And Guard Against Regression

**Files:**

- Modify: `data_base/RAG_QA_service.py`
- Modify: `tests/test_rag_qa_prompts.py`

- [ ] **Step 1: Add static regression test**

Append to `tests/test_rag_qa_prompts.py`:

```python
from pathlib import Path


def test_rag_qa_service_no_long_prompt_constants() -> None:
    source = Path("data_base/RAG_QA_service.py").read_text(encoding="utf-8")

    assert "PLAIN_RAG_PROMPT_TEMPLATE = " not in source
    assert "ADVANCED_RAG_PROMPT_TEMPLATE = " not in source
    assert "VISUAL_TOOL_INSTRUCTION = " not in source
```

- [ ] **Step 2: Run static test and verify failure**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_rag_qa_prompts.py::test_rag_qa_service_no_long_prompt_constants -q
```

Expected:

```text
FAILED
```

- [ ] **Step 3: Remove old constants from `data_base/RAG_QA_service.py`**

Delete the old module-level definitions:

```python
VISUAL_TOOL_INSTRUCTION = """..."""
PLAIN_RAG_PROMPT_TEMPLATE = """..."""
ADVANCED_RAG_PROMPT_TEMPLATE = """..."""
```

Keep `MAX_VISUAL_ITERATIONS` and the rest of the runtime code unchanged.

- [ ] **Step 4: Run static test again**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_rag_qa_prompts.py::test_rag_qa_service_no_long_prompt_constants -q
```

Expected:

```text
1 passed
```

---

### Task 5: Verification Batch

**Files:**

- No new code files.

- [ ] **Step 1: Run focused prompt and RAG tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_prompt_loader.py tests/test_rag_qa_prompts.py tests/test_rag_retrieval_logic.py tests/test_rag_graph_evidence_docs.py tests/test_rag_ask_stream.py tests/test_rag_modes_agentic.py -q
```

Expected:

```text
passed
```

- [ ] **Step 2: Run touched-file lint**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check core\prompt_loader.py data_base\RAG_QA_service.py tests\test_prompt_loader.py tests\test_rag_qa_prompts.py tests\test_rag_retrieval_logic.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 3: Run FastAPI import smoke**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -c "import main; print(main.app.title)"
```

Expected:

```text
PDF Translation & RAG API
```

- [ ] **Step 4: Run current backend cleanup focused suite**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_router_boundaries.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py tests/test_pdfservice_manual_translation.py tests/test_pdfservice_background_processing.py tests/test_rag_startup.py -q
```

Expected:

```text
passed
```

---

## Rollback Plan

If prompt loading fails in production or tests:

1. Revert `data_base/RAG_QA_service.py` to use the old inline constants.
2. Keep `core/prompt_loader.py` and `prompts/rag_qa_prompts.json` only if tests still pass and they are unused.
3. Re-run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_logic.py -q
```

Expected:

```text
passed
```

---

## Follow-Up Plan Candidates

After this first batch is stable:

- Move GraphRAG answer/search prompts from:
  - `graph_rag/global_search.py`
  - `graph_rag/local_search.py`
  - `graph_rag/generic_mode.py`
- Move multimodal image analysis prompts from:
  - `multimodal_rag/image_summarizer.py`
- Move preprocessing prompts from:
  - `data_base/query_transformer.py`
  - `data_base/context_enricher.py`
  - `data_base/proposition_chunker.py`
- Add optional prompt config version metadata to persisted evaluation traces only after a separate evaluation compatibility plan.

---

## Self-Review

Spec coverage:

- Current backend changes are documented before the new plan.
- First implementation scope is limited to production RAG QA prompts.
- JSON config, loader, service wiring, tests, and verification are covered.
- Evaluation/RAGAS and GraphRAG extraction prompts are intentionally excluded from this first batch.

Placeholder scan:

- No unfinished copy instructions or temporary task-note placeholders remain in the plan.

Type consistency:

- `PromptRegistry.format(key: str, **variables: object) -> str` is used consistently.
- Prompt keys are consistent across tests and implementation steps:
  - `plain_rag_answer`
  - `advanced_rag_answer`
  - `visual_tool_instruction`
  - `visual_verification_synthesis`

