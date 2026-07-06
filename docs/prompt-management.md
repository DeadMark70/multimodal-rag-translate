# Prompt Management Guide

Production RAG prompts are now managed through JSON config files under `prompts/` instead of long Python constants. This keeps prompt edits reviewable and separates prompt text from orchestration code.

## Loader

Use `core/prompt_loader.py`.

Available helpers:

- `format_prompt(...)`
  - Default RAG QA registry, backed by `prompts/rag_qa_prompts.json`.
- `format_agentic_rag_prompt(...)`
  - Backed by `prompts/agentic_rag_prompts.json`.
- `format_graph_rag_prompt(...)`
  - Backed by `prompts/graph_rag_prompts.json`.
- `format_rag_pipeline_prompt(...)`
  - Backed by `prompts/rag_pipeline_prompts.json`.

Each helper validates:

- The prompt key exists.
- Required variables are supplied.
- Template placeholders are declared in `required_variables`.
- The template can be formatted.

## Prompt Files

### `prompts/rag_qa_prompts.json`

Main RAG answer path:

- `plain_rag_answer`
- `advanced_rag_answer`
- `visual_tool_instruction`
- `visual_verification_synthesis`

Used mainly by `data_base/RAG_QA_service.py`.

### `prompts/agentic_rag_prompts.json`

Agentic RAG and Deep Research prompts:

- planner prompts
- graph-aware planner prompt
- follow-up and query refinement prompts
- evaluator prompts
- synthesizer and academic report prompts
- fact-state extraction prompt

Used by:

- `agents/planner.py`
- `agents/synthesizer.py`
- `agents/evaluator.py`
- `data_base/research_execution_core.py`

### `prompts/graph_rag_prompts.json`

GraphRAG prompts:

- entity/relation extraction
- one-pass extraction
- global search community answer/synthesis
- local entity identification
- community summary and parent community summary
- generic graph router prompt

Used by `graph_rag/*`.

Note: `relevance_check` is present in JSON for source cleanup, but current runtime still uses lexical relevance in `graph_rag/global_search.py`. Do not add a new LLM call just because the prompt exists.

### `prompts/rag_pipeline_prompts.json`

Supporting RAG pipeline prompts:

- `hyde`
- `multi_query`
- `proposition`
- `context_enrichment`

Used by:

- `data_base/query_transformer.py`
- `data_base/proposition_chunker.py`
- `data_base/context_enricher.py`

## JSON Entry Contract

Every prompt entry must have:

```json
{
  "version": 1,
  "description": "Short maintenance description.",
  "required_variables": ["question"],
  "template": "Prompt text with {question} placeholders."
}
```

Rules:

- `version` is an integer.
- `description` is a short maintenance note.
- `required_variables` must list every Python `.format(...)` placeholder used by `template`.
- `template` is UTF-8 text.
- Literal JSON examples inside a prompt must keep escaped/doubled braces as needed so Python formatting does not treat them as variables.

## Edit Workflow

When changing prompt text:

1. Edit the relevant `prompts/*.json` file.
2. Keep variable names stable unless you also update the runtime call.
3. Run the prompt domain test.
4. Run at least one affected behavior test.
5. Do not combine a wording change with a loader/refactor change unless there is a clear reason.

When adding a new prompt:

1. Add the JSON entry.
2. Add a registry test that checks `required_variables`.
3. Add a format smoke test with representative variables.
4. Wire the source module through the correct domain helper.
5. Add or update a boundary test if the new prompt replaces a Python constant.

## Tests

Prompt tests:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_prompt_loader.py tests/test_rag_qa_prompts.py tests/test_agentic_rag_prompts.py tests/test_graph_rag_prompts.py tests/test_rag_pipeline_prompts.py tests/test_prompt_boundaries.py -q
```

Focused RAG tests:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_logic.py tests/test_rag_graph_evidence_docs.py tests/test_rag_ask_stream.py tests/test_rag_modes_agentic.py tests/test_synthesizer.py tests/test_evaluator_audit.py -q
```

Lint:

```powershell
.\.venv\Scripts\python.exe -m ruff check core\prompt_loader.py agents graph_rag data_base\query_transformer.py data_base\proposition_chunker.py data_base\context_enricher.py data_base\research_execution_core.py tests\test_prompt_loader.py tests\test_agentic_rag_prompts.py tests\test_graph_rag_prompts.py tests\test_rag_pipeline_prompts.py tests\test_prompt_boundaries.py
```

## Current State

The 2026-07-06 externalization pass preserved existing prompt wording. The purpose was maintainability, not prompt quality tuning.

Future prompt wording cleanup should be done in small, labeled commits with before/after intent and focused behavior checks.
