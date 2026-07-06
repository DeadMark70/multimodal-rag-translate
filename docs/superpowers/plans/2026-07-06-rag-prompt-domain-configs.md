# RAG Prompt Domain Configs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the next three RAG prompt domains out of Python constants and into JSON prompt config files without changing runtime behavior.

**Architecture:** Keep the existing `core.prompt_loader.PromptRegistry` validation model, but add domain-specific registries so each RAG area can evolve independently. The first pass preserves current prompt wording byte-for-byte where practical and only changes how prompts are loaded and formatted.

**Tech Stack:** Python 3.13, FastAPI backend, LangChain `HumanMessage`, `pytest`, `ruff`, JSON prompt files under `prompts/`.

---

## Scope And Boundaries

### In Scope

This plan covers exactly these prompt domains:

1. Agentic RAG / Deep Research prompts
   - `agents/planner.py`
   - `agents/synthesizer.py`
   - `agents/evaluator.py`
   - `data_base/research_execution_core.py`

2. GraphRAG prompts
   - `graph_rag/extractor.py`
   - `graph_rag/global_search.py`
   - `graph_rag/local_search.py`
   - `graph_rag/community_builder.py`
   - `graph_rag/generic_mode.py`

3. RAG pipeline supporting prompts
   - `data_base/query_transformer.py`
   - `data_base/proposition_chunker.py`
   - `data_base/context_enricher.py`

### Out Of Scope

Do not move these in this plan:

- `data_base/RAG_QA_service.py` prompts, already moved to `prompts/rag_qa_prompts.json`.
- `core/summary_service.py` `_SUMMARY_PROMPT`.
- `multimodal_rag/image_summarizer.py` generated prompt methods.
- `image_service/translation_service.py` inline translation template.
- `pdfserviceMD/translation_chunker.py` `STRICT_TRANSLATION_PROMPT`.
- `evaluation/agentic_evaluation_service.py` `_DUPLICATE_FOLLOWUP_PROMPT`.
- Test-only prompts under `tests/`.

### Behavior Boundary

- Preserve current prompt text and variable substitution behavior.
- Do not rewrite prompt wording, fix mojibake, change output schemas, or adjust LLM instructions in this pass.
- Do not change graph retrieval, agent planning, synthesis, evaluation, chunking, or query transformation algorithms.
- Do not introduce YAML, database-backed prompts, hot reload, prompt editing API, or admin UI.
- Do not install dependencies.

### File Boundary

Create:

- `prompts/agentic_rag_prompts.json`
- `prompts/graph_rag_prompts.json`
- `prompts/rag_pipeline_prompts.json`
- `tests/test_agentic_rag_prompts.py`
- `tests/test_graph_rag_prompts.py`
- `tests/test_rag_pipeline_prompts.py`

Modify:

- `core/prompt_loader.py`
- `agents/planner.py`
- `agents/synthesizer.py`
- `agents/evaluator.py`
- `data_base/research_execution_core.py`
- `graph_rag/extractor.py`
- `graph_rag/global_search.py`
- `graph_rag/local_search.py`
- `graph_rag/community_builder.py`
- `graph_rag/generic_mode.py`
- `data_base/query_transformer.py`
- `data_base/proposition_chunker.py`
- `data_base/context_enricher.py`
- Existing focused tests only if their imports assert old constants directly.

Do not modify:

- Router files.
- Provider setup.
- Requirements files.
- Docker files.
- Frontend files.

---

## Prompt Key Map

### `prompts/agentic_rag_prompts.json`

- `planner`
  - Source: `agents/planner.py::_PLANNER_PROMPT`
  - Required variables: `question`
- `graph_planner`
  - Source: `agents/planner.py::_GRAPH_PLANNER_PROMPT`
  - Required variables: `question`
- `followup`
  - Source: `agents/planner.py::_FOLLOWUP_PROMPT`
  - Required variables: `original_question`, `current_findings`, `existing_questions`
- `refine_query`
  - Source: `agents/planner.py::_REFINE_QUERY_PROMPT`
  - Required variables: `original_question`, `evaluation_reason`, `failed_answer`
- `intent_classifier`
  - Source: `agents/planner.py::_INTENT_CLASSIFIER_PROMPT`
  - Required variables: `question`
- `conflict_arbitration`
  - Source: `agents/synthesizer.py::_CONFLICT_ARBITRATION_PROMPT`
  - Required variables: `sub_results`
- `synthesizer`
  - Source: `agents/synthesizer.py::_SYNTHESIZER_PROMPT`
  - Required variables: `original_question`, `sub_results`
- `academic_report`
  - Source: `agents/synthesizer.py::_ACADEMIC_REPORT_PROMPT`
  - Required variables: `original_question`, `sub_results`
- `retrieval_eval`
  - Source: `agents/evaluator.py::_RETRIEVAL_EVAL_PROMPT`
  - Required variables: `question`, `documents`
- `faithfulness_eval`
  - Source: `agents/evaluator.py::_FAITHFULNESS_EVAL_PROMPT`
  - Required variables: `question`, `documents`, `answer`
- `detailed_eval`
  - Source: `agents/evaluator.py::_DETAILED_EVAL_PROMPT`
  - Required variables: `question`, `documents`, `answer`
- `pure_llm_eval`
  - Source: `agents/evaluator.py::_PURE_LLM_EVAL_PROMPT`
  - Required variables: `question`, `answer`, `ground_truth`
- `fact_state`
  - Source: `data_base/research_execution_core.py::_FACT_STATE_PROMPT`
  - Required variables: `question`, `source_doc_ids`, `answer`

### `prompts/graph_rag_prompts.json`

- `entity_extraction`
  - Source: `graph_rag/extractor.py::_ENTITY_EXTRACTION_PROMPT`
  - Required variables: `text`
- `relation_extraction`
  - Source: `graph_rag/extractor.py::_RELATION_EXTRACTION_PROMPT`
  - Required variables: `text`, `entities`
- `one_pass_extraction`
  - Source: `graph_rag/extractor.py::_ONE_PASS_EXTRACTION_PROMPT`
  - Required variables: `text`
- `relevance_check`
  - Source: `graph_rag/global_search.py::_RELEVANCE_CHECK_PROMPT`
  - Required variables: `question`, `title`, `summary`
  - Note: Current runtime uses lexical relevance and does not call this prompt. Move it to JSON for source cleanup, but do not introduce a new LLM call.
- `community_answer`
  - Source: `graph_rag/global_search.py::_COMMUNITY_ANSWER_PROMPT`
  - Required variables: `question`, `title`, `summary`, `entities`
- `global_synthesis`
  - Source: `graph_rag/global_search.py::_SYNTHESIS_PROMPT`
  - Required variables: `question`, `community_answers`
- `entity_identification`
  - Source: `graph_rag/local_search.py::_ENTITY_IDENTIFICATION_PROMPT`
  - Required variables: `question`
- `community_summary`
  - Source: `graph_rag/community_builder.py::_COMMUNITY_SUMMARY_PROMPT`
  - Required variables: `entities_and_relations`
- `parent_community`
  - Source: `graph_rag/community_builder.py::_PARENT_COMMUNITY_PROMPT`
  - Required variables: `child_summaries`
- `generic_router`
  - Source: `graph_rag/generic_mode.py::_ROUTER_PROMPT`
  - Required variables: `question`, `stage_hint`, `task_type_hint`, `prefer_global`, `prefer_local`, `has_communities`

### `prompts/rag_pipeline_prompts.json`

- `hyde`
  - Source: `data_base/query_transformer.py::_HYDE_PROMPT`
  - Required variables: `question`
- `multi_query`
  - Source: `data_base/query_transformer.py::_MULTI_QUERY_PROMPT`
  - Required variables: `question`
- `proposition`
  - Source: `data_base/proposition_chunker.py::_PROPOSITION_PROMPT`
  - Required variables: `text`
- `context_enrichment`
  - Source: `data_base/context_enricher.py::_CONTEXT_PROMPT_TEMPLATE`
  - Required variables: `document_title`, `chunk_content`

---

## Task 1: Add Domain Registry Accessors

**Files:**

- Modify: `core/prompt_loader.py`
- Test: `tests/test_prompt_loader.py`

- [ ] **Step 1: Write tests for domain-specific config loading**

Append tests that prove each new domain accessor loads its own JSON file and that the existing RAG QA accessor still works:

```python
from core.prompt_loader import (
    format_agentic_rag_prompt,
    format_graph_rag_prompt,
    format_rag_pipeline_prompt,
    get_agentic_rag_prompt_registry,
    get_graph_rag_prompt_registry,
    get_rag_pipeline_prompt_registry,
)


def test_domain_prompt_registries_are_distinct() -> None:
    assert get_agentic_rag_prompt_registry() is not get_graph_rag_prompt_registry()
    assert get_graph_rag_prompt_registry() is not get_rag_pipeline_prompt_registry()


def test_domain_prompt_format_helpers_load_expected_files() -> None:
    agentic = format_agentic_rag_prompt("planner", question="What is RAG?")
    graph = format_graph_rag_prompt("entity_identification", question="What is RAG?")
    pipeline = format_rag_pipeline_prompt("hyde", question="What is RAG?")

    assert "What is RAG?" in agentic
    assert "What is RAG?" in graph
    assert "What is RAG?" in pipeline
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_prompt_loader.py::test_domain_prompt_registries_are_distinct tests/test_prompt_loader.py::test_domain_prompt_format_helpers_load_expected_files -q
```

Expected:

- Fails because `format_agentic_rag_prompt`, `format_graph_rag_prompt`, and `format_rag_pipeline_prompt` do not exist yet.

- [ ] **Step 3: Implement minimal domain accessors**

Add these helpers to `core/prompt_loader.py` after `get_default_prompt_registry()`:

```python
def _prompt_config_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / filename


@lru_cache(maxsize=1)
def get_rag_qa_prompt_registry() -> PromptRegistry:
    return PromptRegistry(_prompt_config_path("rag_qa_prompts.json"))


@lru_cache(maxsize=1)
def get_agentic_rag_prompt_registry() -> PromptRegistry:
    return PromptRegistry(_prompt_config_path("agentic_rag_prompts.json"))


@lru_cache(maxsize=1)
def get_graph_rag_prompt_registry() -> PromptRegistry:
    return PromptRegistry(_prompt_config_path("graph_rag_prompts.json"))


@lru_cache(maxsize=1)
def get_rag_pipeline_prompt_registry() -> PromptRegistry:
    return PromptRegistry(_prompt_config_path("rag_pipeline_prompts.json"))
```

Change `get_default_prompt_registry()` to delegate to `get_rag_qa_prompt_registry()`:

```python
@lru_cache(maxsize=1)
def get_default_prompt_registry() -> PromptRegistry:
    return get_rag_qa_prompt_registry()
```

Add format helpers:

```python
def format_agentic_rag_prompt(key: str, **variables: Any) -> str:
    return get_agentic_rag_prompt_registry().format(key, **variables)


def format_graph_rag_prompt(key: str, **variables: Any) -> str:
    return get_graph_rag_prompt_registry().format(key, **variables)


def format_rag_pipeline_prompt(key: str, **variables: Any) -> str:
    return get_rag_pipeline_prompt_registry().format(key, **variables)
```

- [ ] **Step 4: Create temporary minimal JSON files so Task 1 tests can pass**

Create only the prompt keys used by the new tests. Later tasks replace these with the full extracted files.

`prompts/agentic_rag_prompts.json`:

```json
{
  "prompts": {
    "planner": {
      "version": 1,
      "description": "Agentic RAG planning prompt.",
      "required_variables": ["question"],
      "template": "Question: {question}"
    }
  }
}
```

`prompts/graph_rag_prompts.json`:

```json
{
  "prompts": {
    "entity_identification": {
      "version": 1,
      "description": "GraphRAG entity identification prompt.",
      "required_variables": ["question"],
      "template": "Question: {question}"
    }
  }
}
```

`prompts/rag_pipeline_prompts.json`:

```json
{
  "prompts": {
    "hyde": {
      "version": 1,
      "description": "HyDE query expansion prompt.",
      "required_variables": ["question"],
      "template": "Question: {question}"
    }
  }
}
```

- [ ] **Step 5: Run tests and lint**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_prompt_loader.py -q
.\.venv\Scripts\python.exe -m ruff check core\prompt_loader.py tests\test_prompt_loader.py
```

Expected:

- `tests/test_prompt_loader.py` passes.
- Ruff passes.

---

## Task 2: Move Agentic RAG And Deep Research Prompts

**Files:**

- Create or replace: `prompts/agentic_rag_prompts.json`
- Create: `tests/test_agentic_rag_prompts.py`
- Modify: `agents/planner.py`
- Modify: `agents/synthesizer.py`
- Modify: `agents/evaluator.py`
- Modify: `data_base/research_execution_core.py`
- Modify existing tests only if they import prompt constants directly.

- [ ] **Step 1: Record runtime variable names from source**

Use this command to audit every `.format(...)` call against the Prompt Key Map above:

```powershell
rg -n "_PLANNER_PROMPT|_GRAPH_PLANNER_PROMPT|_FOLLOWUP_PROMPT|_REFINE_QUERY_PROMPT|_INTENT_CLASSIFIER_PROMPT|_CONFLICT_ARBITRATION_PROMPT|_SYNTHESIZER_PROMPT|_ACADEMIC_REPORT_PROMPT|_RETRIEVAL_EVAL_PROMPT|_FAITHFULNESS_EVAL_PROMPT|_DETAILED_EVAL_PROMPT|_PURE_LLM_EVAL_PROMPT|_FACT_STATE_PROMPT|\\.format\\(" agents data_base\\research_execution_core.py -S
```

Expected:

- The discovered variables match the Prompt Key Map exactly before editing source files.

- [ ] **Step 2: Write prompt config tests**

Create `tests/test_agentic_rag_prompts.py`:

```python
from pathlib import Path

from core.prompt_loader import get_agentic_rag_prompt_registry


def test_agentic_rag_prompt_config_contains_required_keys() -> None:
    registry = get_agentic_rag_prompt_registry()

    for key in (
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
    ):
        assert registry.get(key).version >= 1


def test_agentic_rag_prompt_runtime_variables() -> None:
    registry = get_agentic_rag_prompt_registry()

    assert registry.get("planner").required_variables == ("question",)
    assert registry.get("graph_planner").required_variables == ("question",)
    assert registry.get("intent_classifier").required_variables == ("question",)
    assert registry.get("followup").required_variables == (
        "original_question",
        "current_findings",
        "existing_questions",
    )
    assert registry.get("refine_query").required_variables == (
        "original_question",
        "evaluation_reason",
        "failed_answer",
    )
    assert registry.get("conflict_arbitration").required_variables == ("sub_results",)
    assert registry.get("synthesizer").required_variables == (
        "original_question",
        "sub_results",
    )
    assert registry.get("academic_report").required_variables == (
        "original_question",
        "sub_results",
    )
    assert registry.get("retrieval_eval").required_variables == (
        "question",
        "documents",
    )
    assert registry.get("faithfulness_eval").required_variables == (
        "question",
        "documents",
        "answer",
    )
    assert registry.get("detailed_eval").required_variables == (
        "question",
        "documents",
        "answer",
    )
    assert registry.get("pure_llm_eval").required_variables == (
        "question",
        "answer",
        "ground_truth",
    )
    assert registry.get("fact_state").required_variables == (
        "question",
        "source_doc_ids",
        "answer",
    )


def test_agentic_source_no_long_prompt_constants() -> None:
    files = (
        Path("agents/planner.py"),
        Path("agents/synthesizer.py"),
        Path("agents/evaluator.py"),
        Path("data_base/research_execution_core.py"),
    )
    forbidden = (
        "_PLANNER_PROMPT = ",
        "_GRAPH_PLANNER_PROMPT = ",
        "_FOLLOWUP_PROMPT = ",
        "_REFINE_QUERY_PROMPT = ",
        "_INTENT_CLASSIFIER_PROMPT = ",
        "_CONFLICT_ARBITRATION_PROMPT = ",
        "_SYNTHESIZER_PROMPT = ",
        "_ACADEMIC_REPORT_PROMPT = ",
        "_RETRIEVAL_EVAL_PROMPT = ",
        "_FAITHFULNESS_EVAL_PROMPT = ",
        "_DETAILED_EVAL_PROMPT = ",
        "_PURE_LLM_EVAL_PROMPT = ",
        "_FACT_STATE_PROMPT = ",
    )

    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

    for name in forbidden:
        assert name not in combined
```
- [ ] **Step 3: Run tests and confirm failure**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_agentic_rag_prompts.py -q
```

Expected:

- Fails because the JSON does not yet contain all keys and source constants still exist.

- [ ] **Step 4: Generate `prompts/agentic_rag_prompts.json` from current constants**

Use an extraction script or a careful manual copy. The JSON must follow this shape for every key:

```json
{
  "version": 1,
  "description": "Short maintenance description.",
  "required_variables": ["question"],
  "template": "Original prompt text with {question} placeholders preserved."
}
```

Important:

- Preserve `{...}` placeholders exactly.
- Preserve literal double braces if the original prompt uses JSON examples.
- Keep UTF-8 encoding.
- Do not rewrite wording.

- [ ] **Step 5: Replace planner constants with registry calls**

In `agents/planner.py`, import:

```python
from core.prompt_loader import format_agentic_rag_prompt
```

Replace:

```python
_INTENT_CLASSIFIER_PROMPT.format(question=question)
```

with:

```python
format_agentic_rag_prompt("intent_classifier", question=question)
```

Replace graph/non-graph planning selection with:

```python
prompt = format_agentic_rag_prompt(
    "graph_planner" if enable_graph_planning else "planner",
    question=question,
)
```

Replace follow-up formatting with:

```python
prompt = format_agentic_rag_prompt(
    "followup",
    original_question=original_question,
    current_findings=current_findings,
    existing_questions=existing_list,
)
```

Replace refine query formatting with:

```python
prompt = format_agentic_rag_prompt(
    "refine_query",
    original_question=original_question,
    evaluation_reason=truncated_reason,
    failed_answer=truncated_answer,
)
```

- [ ] **Step 6: Replace synthesizer constants with registry calls**

In `agents/synthesizer.py`, import:

```python
from core.prompt_loader import format_agentic_rag_prompt
```

Replace conflict arbitration:

```python
prompt = format_agentic_rag_prompt(
    "conflict_arbitration",
    sub_results=self._format_arbitration_input(sub_results),
)
```

Replace synthesis template selection:

```python
prompt_key = "academic_report" if use_academic_template else "synthesizer"
prompt = format_agentic_rag_prompt(
    prompt_key,
    original_question=original_question,
    sub_results=formatted_results,
)
```

Preserve all existing extra string appends for conflict arbitration guidance and citation requirements.

- [ ] **Step 7: Replace evaluator constants with registry calls**

In `agents/evaluator.py`, import:

```python
from core.prompt_loader import format_agentic_rag_prompt
```

Replace each evaluator prompt format call with:

```python
prompt = format_agentic_rag_prompt("retrieval_eval", question=question, documents=doc_text)
prompt = format_agentic_rag_prompt("faithfulness_eval", question=question, documents=doc_text, answer=answer[:1000])
prompt = format_agentic_rag_prompt("detailed_eval", question=question, documents=doc_text, answer=answer[:2000])
prompt = format_agentic_rag_prompt("pure_llm_eval", question=question, answer=pure_llm_answer[:2000], ground_truth=ground_truth[:2000])
```

Use the argument names shown in this task.

- [ ] **Step 8: Replace fact-state prompt**

In `data_base/research_execution_core.py`, import:

```python
from core.prompt_loader import format_agentic_rag_prompt
```

Replace `_FACT_STATE_PROMPT.format(...)` with:

```python
prompt = format_agentic_rag_prompt("fact_state", question=result.question, source_doc_ids=", ".join(source_ids) if source_ids else "(none)", answer=result.answer[:2400])
```

Use the argument names shown in this task.

- [ ] **Step 9: Run focused agentic tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_agentic_rag_prompts.py tests/test_synthesizer.py tests/test_evaluator_audit.py tests/test_planner_visual_requirement.py tests/test_rag_modes_agentic.py -q
.\.venv\Scripts\python.exe -m ruff check agents\\planner.py agents\\synthesizer.py agents\\evaluator.py data_base\\research_execution_core.py tests\\test_agentic_rag_prompts.py
```

Expected:

- Prompt config tests pass.
- Existing planner/synthesizer/evaluator tests pass.
- Ruff passes.

---

## Task 3: Move GraphRAG Prompts

**Files:**

- Create or replace: `prompts/graph_rag_prompts.json`
- Create: `tests/test_graph_rag_prompts.py`
- Modify: `graph_rag/extractor.py`
- Modify: `graph_rag/global_search.py`
- Modify: `graph_rag/local_search.py`
- Modify: `graph_rag/community_builder.py`
- Modify: `graph_rag/generic_mode.py`

- [ ] **Step 1: Record runtime variable names from source**

Run:

```powershell
rg -n "_ENTITY_EXTRACTION_PROMPT|_RELATION_EXTRACTION_PROMPT|_ONE_PASS_EXTRACTION_PROMPT|_RELEVANCE_CHECK_PROMPT|_COMMUNITY_ANSWER_PROMPT|_SYNTHESIS_PROMPT|_ENTITY_IDENTIFICATION_PROMPT|_COMMUNITY_SUMMARY_PROMPT|_PARENT_COMMUNITY_PROMPT|_ROUTER_PROMPT|\\.format\\(" graph_rag -S
```

Expected:

- The discovered variables match the GraphRAG Prompt Key Map exactly before editing source files.

- [ ] **Step 2: Write GraphRAG prompt tests**

Create `tests/test_graph_rag_prompts.py`:

```python
from pathlib import Path

from core.prompt_loader import get_graph_rag_prompt_registry


def test_graph_rag_prompt_config_contains_required_keys() -> None:
    registry = get_graph_rag_prompt_registry()

    for key in (
        "entity_extraction",
        "relation_extraction",
        "one_pass_extraction",
        "relevance_check",
        "community_answer",
        "global_synthesis",
        "entity_identification",
        "community_summary",
        "parent_community",
        "generic_router",
    ):
        assert registry.get(key).version >= 1


def test_graph_rag_prompt_runtime_variables() -> None:
    registry = get_graph_rag_prompt_registry()

    assert registry.get("entity_extraction").required_variables == ("text",)
    assert registry.get("relation_extraction").required_variables == ("text", "entities")
    assert registry.get("one_pass_extraction").required_variables == ("text",)
    assert registry.get("relevance_check").required_variables == (
        "question",
        "title",
        "summary",
    )
    assert registry.get("community_answer").required_variables == (
        "question",
        "title",
        "summary",
        "entities",
    )
    assert registry.get("global_synthesis").required_variables == (
        "question",
        "community_answers",
    )
    assert registry.get("entity_identification").required_variables == ("question",)
    assert registry.get("community_summary").required_variables == (
        "entities_and_relations",
    )
    assert registry.get("parent_community").required_variables == ("child_summaries",)
    assert registry.get("generic_router").required_variables == (
        "question",
        "stage_hint",
        "task_type_hint",
        "prefer_global",
        "prefer_local",
        "has_communities",
    )


def test_graph_rag_source_no_long_prompt_constants() -> None:
    files = (
        Path("graph_rag/extractor.py"),
        Path("graph_rag/global_search.py"),
        Path("graph_rag/local_search.py"),
        Path("graph_rag/community_builder.py"),
        Path("graph_rag/generic_mode.py"),
    )
    forbidden = (
        "_ENTITY_EXTRACTION_PROMPT = ",
        "_RELATION_EXTRACTION_PROMPT = ",
        "_ONE_PASS_EXTRACTION_PROMPT = ",
        "_RELEVANCE_CHECK_PROMPT = ",
        "_COMMUNITY_ANSWER_PROMPT = ",
        "_SYNTHESIS_PROMPT = ",
        "_ENTITY_IDENTIFICATION_PROMPT = ",
        "_COMMUNITY_SUMMARY_PROMPT = ",
        "_PARENT_COMMUNITY_PROMPT = ",
        "_ROUTER_PROMPT = ",
    )

    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

    for name in forbidden:
        assert name not in combined
```
- [ ] **Step 3: Run tests and confirm failure**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_rag_prompts.py -q
```

Expected:

- Fails because JSON is incomplete and constants still exist.

- [ ] **Step 4: Create full `prompts/graph_rag_prompts.json`**

Replace the temporary file from Task 1 with all GraphRAG prompt entries. Preserve current templates exactly.

Every entry must use:

```json
{
  "version": 1,
  "description": "Short maintenance description.",
  "required_variables": ["text"],
  "template": "Original prompt text."
}
```

- [ ] **Step 5: Wire GraphRAG modules**

Add this import to each modified GraphRAG module:

```python
from core.prompt_loader import format_graph_rag_prompt
```

Replace prompt formatting calls:

```python
format_graph_rag_prompt("entity_extraction", text=text[:4000])
format_graph_rag_prompt("relation_extraction", text=text[:4000], entities=entity_str)
format_graph_rag_prompt("one_pass_extraction", text=text[:4000])
format_graph_rag_prompt("relevance_check", question=question, title=community.title or f"社群 {community.id}", summary=community.summary)
format_graph_rag_prompt("community_answer", question=question, title=community.title or f"社群 {community.id}", summary=community.summary, entities=entity_str)
format_graph_rag_prompt("global_synthesis", question=question, community_answers=answers_str)
format_graph_rag_prompt("entity_identification", question=question)
format_graph_rag_prompt("community_summary", entities_and_relations=context)
format_graph_rag_prompt("parent_community", child_summaries="\n".join(child_summary_lines))
format_graph_rag_prompt("generic_router", question=question, stage_hint=hints.stage_hint or "none", task_type_hint=hints.task_type_hint or "none", prefer_global=str(hints.prefer_global).lower(), prefer_local=str(hints.prefer_local).lower(), has_communities=str(has_communities).lower())
```

Use the argument names shown in this task.

- [ ] **Step 6: Run focused GraphRAG tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_rag_prompts.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py tests/test_rag_graph_evidence_docs.py -q
.\.venv\Scripts\python.exe -m ruff check graph_rag\\extractor.py graph_rag\\global_search.py graph_rag\\local_search.py graph_rag\\community_builder.py graph_rag\\generic_mode.py tests\\test_graph_rag_prompts.py
```

Expected:

- Prompt config tests pass.
- Existing GraphRAG focused tests pass.
- Ruff passes.

---

## Task 4: Move RAG Pipeline Supporting Prompts

**Files:**

- Create or replace: `prompts/rag_pipeline_prompts.json`
- Create: `tests/test_rag_pipeline_prompts.py`
- Modify: `data_base/query_transformer.py`
- Modify: `data_base/proposition_chunker.py`
- Modify: `data_base/context_enricher.py`

- [ ] **Step 1: Record runtime variable names from source**

Run:

```powershell
rg -n "_HYDE_PROMPT|_MULTI_QUERY_PROMPT|_PROPOSITION_PROMPT|_CONTEXT_PROMPT_TEMPLATE|\\.format\\(" data_base\\query_transformer.py data_base\\proposition_chunker.py data_base\\context_enricher.py -S
```

Expected variable names:

- `hyde`: `question`
- `multi_query`: `question`
- `proposition`: `text`
- `context_enrichment`: `document_title`, `chunk_content`

- [ ] **Step 2: Write RAG pipeline prompt tests**

Create `tests/test_rag_pipeline_prompts.py`:

```python
from pathlib import Path

from core.prompt_loader import get_rag_pipeline_prompt_registry


def test_rag_pipeline_prompt_config_contains_required_keys() -> None:
    registry = get_rag_pipeline_prompt_registry()

    for key in ("hyde", "multi_query", "proposition", "context_enrichment"):
        assert registry.get(key).version >= 1


def test_rag_pipeline_prompt_runtime_variables() -> None:
    registry = get_rag_pipeline_prompt_registry()

    assert registry.get("hyde").required_variables == ("question",)
    assert registry.get("multi_query").required_variables == ("question",)
    assert registry.get("proposition").required_variables == ("text",)
    assert registry.get("context_enrichment").required_variables == (
        "document_title",
        "chunk_content",
    )


def test_rag_pipeline_source_no_long_prompt_constants() -> None:
    files = (
        Path("data_base/query_transformer.py"),
        Path("data_base/proposition_chunker.py"),
        Path("data_base/context_enricher.py"),
    )
    forbidden = (
        "_HYDE_PROMPT = ",
        "_MULTI_QUERY_PROMPT = ",
        "_PROPOSITION_PROMPT = ",
        "_CONTEXT_PROMPT_TEMPLATE = ",
    )

    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

    for name in forbidden:
        assert name not in combined
```

The `context_enrichment` tuple is explicitly asserted above.

- [ ] **Step 3: Run tests and confirm failure**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_rag_pipeline_prompts.py -q
```

Expected:

- Fails because JSON is incomplete and constants still exist.

- [ ] **Step 4: Create full `prompts/rag_pipeline_prompts.json`**

Replace the temporary file from Task 1 with all RAG pipeline prompt entries. Preserve current templates exactly.

- [ ] **Step 5: Wire RAG pipeline modules**

Add this import to each modified module:

```python
from core.prompt_loader import format_rag_pipeline_prompt
```

Replace prompt formatting calls:

```python
format_rag_pipeline_prompt("hyde", question=question)
format_rag_pipeline_prompt("multi_query", question=question)
format_rag_pipeline_prompt("proposition", text=text[:2000])
format_rag_pipeline_prompt("context_enrichment", document_title=document_title, chunk_content=chunk.page_content[:1000])
```

Use the argument names shown in this task.

- [ ] **Step 6: Run focused RAG pipeline tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_rag_pipeline_prompts.py tests/test_rag_retrieval_logic.py -q
.\.venv\Scripts\python.exe -m ruff check data_base\\query_transformer.py data_base\\proposition_chunker.py data_base\\context_enricher.py tests\\test_rag_pipeline_prompts.py
```

Expected:

- Prompt config tests pass.
- Existing RAG retrieval logic tests pass.
- Ruff passes.

---

## Task 5: Boundary Tests For Prompt Externalization

**Files:**

- Modify: `tests/test_router_boundaries.py` or create `tests/test_prompt_boundaries.py`

- [ ] **Step 1: Add a production prompt boundary test**

Create `tests/test_prompt_boundaries.py`:

```python
from pathlib import Path


PRODUCTION_DIRS = (
    Path("agents"),
    Path("graph_rag"),
    Path("data_base"),
)

ALLOWED_INLINE_PROMPT_FILES = {
    Path("data_base/RAG_QA_service.py"),
}

FORBIDDEN_MARKERS = (
    "_PLANNER_PROMPT = ",
    "_GRAPH_PLANNER_PROMPT = ",
    "_FOLLOWUP_PROMPT = ",
    "_REFINE_QUERY_PROMPT = ",
    "_INTENT_CLASSIFIER_PROMPT = ",
    "_CONFLICT_ARBITRATION_PROMPT = ",
    "_SYNTHESIZER_PROMPT = ",
    "_ACADEMIC_REPORT_PROMPT = ",
    "_RETRIEVAL_EVAL_PROMPT = ",
    "_FAITHFULNESS_EVAL_PROMPT = ",
    "_DETAILED_EVAL_PROMPT = ",
    "_PURE_LLM_EVAL_PROMPT = ",
    "_FACT_STATE_PROMPT = ",
    "_ENTITY_EXTRACTION_PROMPT = ",
    "_RELATION_EXTRACTION_PROMPT = ",
    "_ONE_PASS_EXTRACTION_PROMPT = ",
    "_RELEVANCE_CHECK_PROMPT = ",
    "_COMMUNITY_ANSWER_PROMPT = ",
    "_SYNTHESIS_PROMPT = ",
    "_ENTITY_IDENTIFICATION_PROMPT = ",
    "_COMMUNITY_SUMMARY_PROMPT = ",
    "_PARENT_COMMUNITY_PROMPT = ",
    "_ROUTER_PROMPT = ",
    "_HYDE_PROMPT = ",
    "_MULTI_QUERY_PROMPT = ",
    "_PROPOSITION_PROMPT = ",
    "_CONTEXT_PROMPT_TEMPLATE = ",
)


def test_rag_prompt_constants_are_externalized() -> None:
    offenders: list[str] = []

    for directory in PRODUCTION_DIRS:
        for path in directory.rglob("*.py"):
            if path in ALLOWED_INLINE_PROMPT_FILES:
                continue
            source = path.read_text(encoding="utf-8")
            for marker in FORBIDDEN_MARKERS:
                if marker in source:
                    offenders.append(f"{path}: {marker}")

    assert offenders == []
```

- [ ] **Step 2: Run the boundary test**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_prompt_boundaries.py -q
```

Expected:

- Passes after Tasks 2-4.

---

## Task 6: Final Verification

**Files:**

- No source edits unless verification finds a regression.

- [ ] **Step 1: Run all prompt config tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_prompt_loader.py tests/test_rag_qa_prompts.py tests/test_agentic_rag_prompts.py tests/test_graph_rag_prompts.py tests/test_rag_pipeline_prompts.py tests/test_prompt_boundaries.py -q
```

Expected:

- All pass.

- [ ] **Step 2: Run focused RAG and backend suites**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_logic.py tests/test_rag_graph_evidence_docs.py tests/test_rag_ask_stream.py tests/test_rag_modes_agentic.py tests/test_synthesizer.py tests/test_evaluator_audit.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py tests/test_router_boundaries.py -q
```

Expected:

- All pass, except only pre-existing warnings.

- [ ] **Step 3: Run ruff**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check core\\prompt_loader.py agents graph_rag data_base\\query_transformer.py data_base\\proposition_chunker.py data_base\\context_enricher.py data_base\\research_execution_core.py tests\\test_prompt_loader.py tests\\test_agentic_rag_prompts.py tests\\test_graph_rag_prompts.py tests\\test_rag_pipeline_prompts.py tests\\test_prompt_boundaries.py
```

Expected:

- `All checks passed!`

- [ ] **Step 4: Run FastAPI import smoke**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -c "import main; print(main.app.title)"
```

Expected:

- Prints `PDF Translation & RAG API`.

---

## Rollback Plan

If any task causes regressions:

1. Revert only the files touched by that task.
2. Keep `core/prompt_loader.py` changes only if no module has been rewired to new domain helpers.
3. Restore the Python prompt constants from git for that domain.
4. Remove the domain JSON file if it is unused.
5. Re-run the focused tests from the failed task.

Do not use `git reset --hard`.

---

## Execution Order

1. Task 1: Domain registry accessors.
2. Task 2: Agentic RAG / Deep Research prompts.
3. Task 3: GraphRAG prompts.
4. Task 4: RAG pipeline supporting prompts.
5. Task 5: Boundary tests.
6. Task 6: Final verification.

Commit recommendation:

- Commit after Task 1.
- Commit after Task 2.
- Commit after Task 3.
- Commit after Task 4.
- Commit after Task 5-6 verification.

---

## Self-Review

- Spec coverage: Covers requested priorities 1, 2, and 3 only. Translation, summary, image, and evaluation-service duplicate-followup prompts are intentionally excluded.
- Placeholder scan: No implementation placeholder is left for code mechanics. All required variable lists are explicit in the Prompt Key Map; source audit steps are verification gates, not design placeholders.
- Type consistency: All new APIs follow existing `PromptRegistry.format(key, **variables)` and return `str`.
- Behavior preservation: The plan requires preserving prompt text and existing string appends.
- Verification: Each domain has config tests, source constant removal tests, focused behavioral tests, ruff, and FastAPI import smoke.




