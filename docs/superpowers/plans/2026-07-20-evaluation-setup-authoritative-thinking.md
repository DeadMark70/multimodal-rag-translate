# Evaluation Setup-authoritative thinking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Make the Evaluation Setup model and thinking controls authoritative for every GraphRAG LLM call, preventing Gemini 2.5/3 parameter mismatches and restoring Graph token accounting.

**Architecture:** Carry an explicit \`thinking_enabled\` flag and the selected model through the request-scoped LLM override context. GraphRAG derives its parameter family from that effective model only; when thinking is disabled it clears both thinking parameter families before constructing the SDK client. Existing non-evaluation Graph defaults remain unchanged when no request context is active.

**Tech Stack:** Python, \`contextvars\`, \`ChatGoogleGenerativeAI\`, pytest, Pydantic model capability normalization.

## Global Constraints

- Evaluation Setup is the only authority for model and thinking configuration during evaluation execution.
- \`thinking_mode=false\` must send neither \`thinking_budget\` nor \`thinking_level\`.
- Gemini 2.5 uses only \`thinking_budget\`; Gemini 3 uses only \`thinking_level\`.
- Existing non-evaluation GraphRAG defaults and fallback behavior remain compatible.
- Do not rewrite existing campaign/database rows; verification uses a fresh campaign.

---

### Task 1: Add failing runtime-context and model-capability tests

**Files:**
- Modify: \`tests/test_model_capabilities.py\`
- Modify: \`tests/test_llm_factory_override.py\`

**Interfaces:**
- \`normalize_model_config_for_runtime()\` will expose \`thinking_enabled\` for nested consumers.
- \`graph_rag_llm_runtime_override()\` will derive final parameters from the active request context.

- [x] **Step 1: Write the failing tests**

Add this model capability test:

    def test_runtime_normalization_exposes_explicit_thinking_disabled_state() -> None:
        runtime = normalize_model_config_for_runtime({
            "model_name": "gemini-2.5-flash-lite",
            "thinking_mode": False,
            "thinking_budget": 8192,
            "thinking_level": "high",
        })
        assert runtime["thinking_enabled"] is False
        assert "thinking_budget" not in runtime
        assert "thinking_level" not in runtime

Add Graph factory tests for these cases:
- outer model \`gemini-2.5-flash-lite\`, thinking disabled: constructor receives no thinking key;
- outer model \`gemini-2.5-flash-lite\`, enabled budget 4096: constructor receives only \`thinking_budget=4096\`;
- outer model \`gemini-3.1-flash-lite\`, enabled level high: constructor receives only \`thinking_level=high\`;
- all three cases preserve the outer model name.

Use the existing \`clear_llm_cache\`, \`llm_runtime_override\`, \`graph_rag_llm_runtime_override\`, \`get_llm\`, \`patch("core.llm_factory.ChatGoogleGenerativeAI")\` test pattern.

- [x] **Step 2: Run the tests and verify RED**

Run:

    .venv\Scripts\python.exe -m pytest tests/test_model_capabilities.py tests/test_llm_factory_override.py -q

Expected: the new \`thinking_enabled\` assertion and the nested Graph tests fail because the production context does not yet preserve Setup authority.

### Task 2: Implement Setup-authoritative runtime context

**Files:**
- Modify: \`evaluation/model_capabilities.py\`
- Modify: \`core/llm_factory.py\`

**Interfaces:**
- \`normalize_model_config_for_runtime()\` returns \`thinking_enabled: bool\`.
- \`llm_runtime_override()\` accepts a \`clear\` keyword containing keys to remove before applying nested overrides.
- \`graph_rag_llm_runtime_override()\` uses the active \`model_name\` and \`thinking_enabled\` when present.

- [x] **Step 1: Preserve explicit thinking state**

Initialize normalized runtime values with:

    runtime = {
        key: model_config.get(key)
        for key in _BASE_RUNTIME_KEYS
        if model_config.get(key) is not None
    }
    runtime["thinking_enabled"] = bool(model_config.get("thinking_mode"))

Keep existing capability-specific normalization only when this flag is true.

- [x] **Step 2: Add safe nested key clearing**

Change the context manager signature to:

    def llm_runtime_override(*, clear: tuple[str, ...] = (), **overrides: Any):

Copy the current context, remove each key in \`clear\`, then apply non-None overrides. Existing callers without \`clear\` must behave identically.

- [x] **Step 3: Make GraphRAG use the active Setup model**

Inside \`graph_rag_llm_runtime_override\`:
- use the explicit \`model_name\` argument, then current context \`model_name\`, then the non-evaluation Graph default;
- if current context has \`thinking_enabled=False\`, clear both thinking keys and add only \`include_thoughts=False\`;
- if current context has \`thinking_enabled=True\`, keep only the capability-compatible Setup key;
- if no Setup context exists, retain the existing Graph defaults;
- clear stale conflicting keys before applying the final values.

The effective model passed by \`get_llm\` must remain the Setup model.

- [x] **Step 4: Run the new tests and verify GREEN**

Run the Task 1 command. Expected: all new tests pass.

### Task 3: Add exact Evaluation Setup integration coverage

**Files:**
- Modify: \`tests/test_llm_factory_override.py\`

- [x] **Step 1: Test the actual evaluation normalization path**

Create a setup dictionary matching the report (\`gemini-2.5-flash-lite\`, temperature 0.7, top_p 0.95, top_k 64, max output 8192, thinking_mode false, budget 8192), pass it through \`normalize_model_config_for_runtime\`, enter \`llm_runtime_override(**setup)\`, then enter \`graph_rag_llm_runtime_override("community_summary")\` and call \`get_llm\`. Assert the constructor model is \`gemini-2.5-flash-lite\) and neither thinking key is present.

- [x] **Step 2: Run focused regression tests**

Run:

    .venv\Scripts\python.exe -m pytest tests/test_model_capabilities.py tests/test_llm_factory_override.py tests/test_llm_usage_callback.py -q

Expected: all tests pass.

### Task 4: Verify and commit

**Files:**
- Modify: \`docs/evaluation-center.md\` only if the Setup-authority rule is not already documented.

- [x] **Step 1: Run backend evaluation regression tests**

    .venv\Scripts\python.exe -m pytest tests/test_model_capabilities.py tests/test_llm_factory_override.py tests/test_llm_usage_callback.py tests/test_evaluation_research_end_to_end.py -q

Expected: all pass.

- [x] **Step 2: Run the full backend suite**

    .venv\Scripts\python.exe -m pytest -q

Expected: no new failures. Keep any pre-existing local environment-key mismatch separate from this fix.

- [x] **Step 3: Review and commit only implementation files**

    git diff --check
    git add core/llm_factory.py evaluation/model_capabilities.py tests/test_llm_factory_override.py tests/test_model_capabilities.py tests/test_llm_usage_callback.py docs/evaluation-center.md
    git commit -m "fix(evaluation): honor setup thinking controls in graph rag"

Do not stage the existing untracked \`data/\` snapshots or planning documents.
