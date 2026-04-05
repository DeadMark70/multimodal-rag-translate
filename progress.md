# Progress Log

## Session: 2026-04-06 (Chat Dual Research Integration)

### Phase 1: Contract and runtime mapping
- **Status:** complete
- Actions taken:
  - Re-read backend docs and runtime surface (`/rag`, `/api/evaluation`).
  - Verified chat currently has no benchmark-agentic stream endpoint.
  - Verified benchmark agentic trace payload shape and persistence model from evaluation subsystem.
  - Confirmed compatibility requirement: keep deep-research execution path unchanged.
  - Updated planning files (`task_plan.md`, `findings.md`, `progress.md`) for this new implementation track.

### Phase 2: Backend implementation
- **Status:** in_progress
- Next actions:
  - Add chat agentic stream schema + execution service.
  - Wire `/rag/agentic/stream`.
  - Persist `research_engine=agentic_benchmark` and full trace in conversation metadata.

## Session: 2026-04-04 (Correctness-First Plan Implementation)

### Phase 1: Scope & Baseline Refresh
- **Status:** complete
- Actions taken:
  - 重新讀取 `agent.md`、skills、既有測試。
  - 解析 `campaign-dd619...metrics.json` 與 `ragas_hardset_v2.json`。
  - 從 `data/evaluation.db` 交叉檢視 `campaign_results + agent_traces`，確認 `Q5/Q2/Q6` 失分根因。
  - 將 `task_plan.md / findings.md / progress.md` 切換為本輪 correctness-first 目標。

### Phase 2: Core Implementation
- **Status:** complete
- Actions taken:
  - `evaluation/agentic_evaluation_service.py`
    - benchmark numeric detection 改為「指標詞 + 數值語境」且排除 `Dice supervision` 等方法詞誤觸發。
    - benchmark routing 固化：非 numeric 一律 `hybrid_compare`；`generic_graph` 僅限 numeric + 關係/跨節點任務。
    - figure-flow planner 加入原題錨定（第 1 子任務固定原題）與最多 1 個輔助子任務過濾。
    - 新增 retrieval quality gate 與 gap-targeted follow-up 過濾。
    - synthesis 改為 single-task 也走 normalize（`force_llm_for_single=True`）。
    - `execution_profile` bump：`agentic_eval_v5_correctness`。
  - `agents/planner.py`
    - intent 分類改為 benchmark 訊號更嚴格，避免 `Q6` 這類方法描述被誤判為 benchmark。
  - `data_base/RAG_QA_service.py`
    - benchmark/figure_flow prompt constraints 強化（格式與「資料不足」規則）。
  - `agents/synthesizer.py`
    - 新增 `force_llm_for_single`，支援單任務 normalization。
  - 同步更新 docs 與 `agent.md` learned note。

### Phase 3: Tests
- **Status:** complete
- Actions taken:
  - 更新 `tests/test_agentic_evaluation_service.py`（routing、intent、figure-flow anchor、quality gate、single-task synthesis-lite）。
  - 新增 `tests/test_planner_intent_hardset.py`（Q1~Q8 固定樣本 intent 斷言）。
  - 更新 `tests/test_rag_modes_agentic.py` 與 `tests/test_campaign_engine.py` 的 execution profile 斷言。
  - 驗證：
    - `pytest tests/test_agentic_evaluation_service.py tests/test_rag_modes_agentic.py tests/test_planner_intent_hardset.py tests/test_campaign_engine.py` -> `26 passed`
    - `ruff check ...`（修改檔案）-> pass

### Phase 4: Verification & Delivery
- **Status:** in_progress
- Remaining:
  - 回報本輪變更摘要、驗證結果、以及 campaign 驗收下一步命令。

## Errors Encountered
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-04-04 | 遞迴掃描觸發 `.pytest_cache` 權限拒絕 | 1 | 改掃指定子目錄（`data/`, `evaluation/`, `conversations/`） |
| 2026-04-04 | `ruff` 報 `unused import` | 1 | 移除 `test_agentic_evaluation_service.py` 的未使用 `RAGResult` import |
