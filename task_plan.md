# Task Plan: Agentic Eval Correctness-First 實作 (Evaluation-only)

## Goal (2026-04-06) - Chat Dual Research Integration

Implement backend support for chat dual research modes:
- Keep existing user Deep Research pipeline unchanged.
- Add chat-facing benchmark agentic stream endpoint (`POST /rag/agentic/stream`).
- Emit rich SSE events for plan/task/drilldown/evaluation/trace/synthesis lifecycle.
- Persist `research_engine=agentic_benchmark`, `result`, and full `agent_trace` in conversation metadata for restore/review.

## Current Phase (2026-04-06)
Phase 1 complete, entering Phase 2 backend implementation.

## Phases (2026-04-06)

### Phase 1: Contract and runtime mapping
- [x] Locate reusable evaluation agentic runtime (`AgenticEvaluationService`) and trace schema.
- [x] Confirm existing `/rag` router lacks chat benchmark agentic endpoint.
- [x] Confirm deep-research endpoints and persistence path to keep unchanged.
- **Status:** complete

### Phase 2: Backend implementation
- [ ] Add chat agentic stream schemas/events and service.
- [ ] Add `/rag/agentic/stream` endpoint in `data_base/router.py`.
- [ ] Persist agentic chat trace + result into conversation metadata with `research_engine`.
- **Status:** in_progress

### Phase 3: Backend tests + verification
- [ ] Add stream order/payload tests.
- [ ] Add metadata persistence tests.
- [ ] Run targeted pytest suites.
- **Status:** pending

### Phase 4: Docs + closeout
- [ ] Update backend docs (`BACKEND.md`, generated API surface, specs if touched).
- [ ] Execute continuous-learning protocol if incident/correction occurs.
- **Status:** pending

## Goal
在 `D:\flutterserver\pdftopng` 針對 evaluation-only agentic baseline 實作 correctness-first 優化：
- intent/routing 修正（避免 `Q2/Q6` 被錯誤導向高成本路徑）
- planner fidelity 修正（`Q5` 錨定原題，避免 broad rewrite）
- single-task synthesis-lite（避免輸出格式漂移）
- retrieval quality gate + coverage-targeted drilldown（CRAG/Self-RAG 輕量化）
- 題型 prompt 約束加強（benchmark / figure_flow）
- 新增回歸測試並維持同模型公平比較

## Current Phase
Phase 4

## Phases

### Phase 1: Scope & Baseline Refresh
- [x] 重新讀取目前程式與測試狀態
- [x] 釐清 `campaign-dd619...` 與 `ragas_hardset_v2` 問題分佈
- [x] 確認只動 evaluation agentic flow
- **Status:** complete

### Phase 2: Core Implementation
- [x] 更新 `evaluation/agentic_evaluation_service.py`：
  - numeric benchmark 判定收斂
  - benchmark route policy 固化
  - figure_flow plan anchor 與輔助任務過濾
  - retrieval quality gate + gap-targeted drilldown
  - single-task synthesis-lite 入口
  - execution_profile bump
- [x] 更新 `agents/planner.py` intent 分類優先序與 benchmark 判斷條件
- [x] 更新 `data_base/RAG_QA_service.py` 題型約束（correctness-first）
- [x] 更新 `agents/synthesizer.py` 以支援單任務 LLM 正規化
- **Status:** complete

### Phase 3: Tests
- [x] 更新 `tests/test_agentic_evaluation_service.py`
- [x] 新增/更新 planner intent 與 routing regression tests
- [x] 執行 targeted pytest 並修正回歸
- **Status:** complete

### Phase 4: Verification & Delivery
- [x] 匯總變更、風險、測試結果
- [x] 更新 `agent.md` learned rule（依 Continuous Learning Protocol）
- **Status:** in_progress

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 僅修改 evaluation agentic baseline，不碰線上 chat path | 降低回歸風險並維持公平比較 |
| 保持模型與參數不變 | 讓分數變化可歸因於流程優化 |
| token 預算上限採 +60% | 依使用者指定成本邊界 |
