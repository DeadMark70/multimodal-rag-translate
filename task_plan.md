# Task Plan: Agentic Eval 微優化實作 (Faithfulness 優先 + Correctness 提升)

## Goal
在 `D:\flutterserver\pdftopng` 實作你指定的 evaluation-only 微優化方案：
- 路由策略調整（`hybrid_graph` 改名 `generic_graph`、benchmark/figure_flow 分流、drilldown 收斂）
- Full Prompt Basis（graph evidence 納入 `RAGResult.documents`）
- 輕量 correctness boost（benchmark/figure_flow 輸出約束 + grounding）
- 相容層（舊 trace `hybrid_graph` 與新值 `generic_graph` 聚合一致）
- 補上對應測試

## Current Phase
Phase 4

## Phases

### Phase 1: Scope & Baseline
- [x] 讀取 `agent.md` / `AGENTS.md` / skills 規範
- [x] 盤點目標檔案與現行行為
- [x] 確認測試覆蓋點
- **Status:** complete

### Phase 2: Implementation
- [x] 更新 `evaluation/agentic_evaluation_service.py`
- [x] 更新 `data_base/RAG_QA_service.py`
- [x] 更新 `agents/synthesizer.py`
- [x] 更新 `evaluation/db.py` 相容層
- **Status:** complete

### Phase 3: Tests
- [x] 新增/調整單元測試（route rename、routing policy、drilldown 限制、graph evidence documents）
- [x] 執行目標 pytest
- **Status:** complete

### Phase 4: Verification & Delivery
- [x] 檢視差異與風險
- [ ] 回報變更摘要與測試結果
- **Status:** in_progress

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 僅改 evaluation flow 相關模組，不動線上 chat API 流程 | 遵守 user 指示與可比性要求 |
| 在 `RAG_QA_service` 讓 `_get_graph_context` 新增可選 evidence 回傳模式 | 保持既有測試與呼叫相容（預設仍回傳字串） |
| 以 route rename + DB 正規化做相容層 | 避免歷史 trace 指標斷裂 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `rg.exe` 在此環境啟動遭拒 (Access denied) | 1 | 改用 PowerShell `Select-String` + 分段讀檔 |
| 查找 `evaluation/metrics_aggregator.py` 路徑不存在 | 1 | 改為只檢索實際存在檔案 |

## Notes
- 目標驗收基準：`correctness delta >= +0.01`, `faithfulness delta >= -0.02`（需重跑 campaign 驗證）。
