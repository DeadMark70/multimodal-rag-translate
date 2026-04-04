# Findings & Decisions

## Current Baseline (campaign `dd619059-6512-44f4-99e5-127dc0a4b912`)
- Overall: agentic `answer_correctness` 與 `faithfulness` 皆低於 naive，且平均 token 顯著增加。
- 問題級觀察：
  - `Q5`（圖文交叉重建）correctness 明顯劣化，agentic 回答常改寫成「整體架構介紹」，偏離題幹要求的 CSS 流程重建。
  - `Q2/Q6` 出現 route/intent 偏移：`3D`、`Dice` 等詞觸發 benchmark-heavy 行為，導致成本升高且答案不穩。
  - 部分單任務路徑直接回傳子任務風格答案，格式未經 synthesis 正規化，容易出現標題/題目回顯。

## Root Causes Mapped to Code
- `evaluation/agentic_evaluation_service.py`
  - `_is_numeric_benchmark_subtask` 過於寬鬆：只要出現 metric 關鍵詞就可能走 numeric graph 路徑。
  - `generate_agentic_plan` 對 `figure_flow` 沒有「原題錨定」保護。
  - drilldown 僅靠長度與失敗字詞，缺少 retrieval quality gate 與 coverage-targeted followup 過濾。
  - `_synthesize_execution_results` 單任務不走 synthesis 正規化（`enabled=len(sub_results)>1`）。
- `agents/planner.py`
  - `classify_question_intent` 對 benchmark 判定過寬，`Dice supervision` 類方法描述也容易被誤判。
- `data_base/RAG_QA_service.py`
  - benchmark / figure_flow 題型約束仍可再收斂，缺少「避免題目回顯/標題化輸出」與「不確定時收斂」指令。

## Design Decisions for This Patch
1. 只改 evaluation agentic flow，不改線上 chat path。
2. 保持模型與參數設定不變，確保公平比較。
3. `benchmark_data` 非 numeric 子任務固定走 `hybrid_compare`。
4. `generic_graph` 僅保留給「numeric + 明確關係/跨節點」任務。
5. `figure_flow` 第一子任務強制原題錨定；輔助任務最多 1 個且需命中流程缺口。
6. 單任務結果強制 synthesis-lite，避免風格漂移。

## Risk Watchlist
- Intent 規則收緊後，少數真 benchmark 題可能被降級為 comparison 路徑；需靠測試樣本覆蓋。
- Drilldown 過濾過嚴可能降低召回；需檢查是否過早停止。
- 新 `execution_profile` 必須同步測試與 trace 斷言，避免歷史資料解讀混淆。

## Implemented Outcomes
- `execution_profile` 已升級為 `agentic_eval_v5_correctness`，測試與文件同步完成。
- benchmark numeric 判定已改為雙條件（metric + numeric context），並排除方法型詞彙誤判。
- figure-flow 已加原題錨定與輔助任務過濾，避免 planner 產生 broad architecture 偏題子任務。
- drilldown 新增 retrieval quality gate + coverage-gap targeted follow-up，避免無關追問。
- single-task 路徑已強制走 synthesis-lite，降低輸出標題化/題目回顯漂移。
