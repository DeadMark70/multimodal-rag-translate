# 🚀 Deep Research 最終優化計畫書 (Final Optimization Plan)

本計畫書針對目前 Deep Research 系統在處理大規模檢索與多文檔推理時發現的關鍵瓶頸，提出三項核心改進措施：**預設 GraphRAG 增強**、**信心度校準** 與 **對抗性強制鑽取**。

---

## 1. 改進目標 (Objectives)

1.  **提升抗噪能力**: 在海量文檔 (50+) 環境下，確保能精準抓出關鍵文檔，體現與普通 LLM (Context Window 限制) 的差異。
2.  **校準信心度**: 解決「錯誤回答卻給 100% 信心」的問題，引入衝突懲罰機制。
3.  **強化邏輯深度**: 透過強制 Drill-down 和自我質疑 (Counter-Query)，強迫系統進行辯證思考，避免表面化回答。

---

## 2. 實作細節 (Implementation Details)

### Phase 6.1: 強制 GraphRAG 與海量檢索策略 (Scaling)

> **目標**: 讓 Deep Research 預設具備全域視角，利用 GraphRAG 處理普通 Vector Search 遺漏的隱藏關聯。

#### [MODIFY] `data_base/deep_research_service.py`

*   **預設開啟 GraphRAG**:
    在 `execute_plan` 方法中，將傳遞給 `rag_answer_question` 的 `enable_graph_rag` 參數預設值改為 `True`。
    
*   **搜尋模式調整**:
    將 `graph_search_mode` 預設為 `"hybrid"` (混合模式)，確保同時利用 Vector 的精準度與 Graph 的廣度。

#### [MODIFY] `data_base/RAG_QA_service.py` (Path Retrieval)

*   **實作路徑檢索**:
    修改 `_get_graph_context` 方法，當問題中偵測到兩個實體 (Entities) 時，執行 NetworkX 的 `shortest_path` 或 `all_simple_paths` 算法，找出兩者之間的關聯路徑，並將其轉化為自然語言注入 Prompt。

---

### Phase 6.2: 信心度校準 (Confidence Calibration)

> **目標**: 讓信心分數真實反映回答的可靠度，特別是在有衝突觀點時。

#### [MODIFY] `agents/synthesizer.py`

*   **加權平均演算法**:
    修改 `_parse_report` 或 `synthesize` 方法。最終報告的 `confidence` 不再預設為 1.0。
    
    ```python
    # 偽代碼
    base_confidence = sum(r.confidence for r in sub_results) / len(sub_results)
    
    # 衝突懲罰 (透過檢查 <think> 標籤或 LLM 輸出)
    if "衝突" in report_content or "不一致" in report_content:
        conflict_penalty = 0.8
    else:
        conflict_penalty = 1.0
        
    final_confidence = base_confidence * conflict_penalty
    ```

---

### Phase 6.3: 強制 Drill-down 與對抗性機制 (Adversarial Drill-down)

> **目標**: 打破「一次檢索」的侷限，強迫系統進行「正反辯證」。

#### [MODIFY] `data_base/deep_research_service.py`

*   **強制迭代 (Forced Iteration)**:
    修改 `_should_skip_drilldown` 邏輯。
    若 `max_iterations > 0` 且當前是 `iteration 0`，則 **強制返回 False** (不跳過)，確保至少執行一次 Drill-down。

#### [MODIFY] `agents/planner.py`

*   **對抗性查詢生成 (Counter-Query Generation)**:
    修改 `create_followup_tasks` 或 `plan` 方法。
    在生成查詢時，利用 Prompt 指示 LLM：「針對每個核心論點，生成一個反面或限制性的查詢」。
    
    *   *Prompt 範例*: "For every main topic 'X', also generate a query about 'Limitations of X' or 'Arguments against X'."

---

## 3. 驗證計畫 (Verification)

### 實驗 A: 抗噪壓力測試
*   **Setup**: 上傳 1 篇目標論文 (e.g., nnU-Net) + 49 篇無關論文 (食譜、旅遊、其他領域)。
*   **Query**: "nnU-Net 的核心架構是什麼？"
*   **Pass**: 系統能準確回答，且 Source 僅包含那篇目標論文。

### 實驗 B: 信心度校準測試
*   **Setup**: 上傳兩篇衝突論文。
*   **Query**: "A 與 B 哪個好？" (已知有衝突)
*   **Pass**: 最終報告的 Confidence Score 低於 0.9 (例如 0.7-0.8)，反映出不確定性。

### 實驗 C: 辯證深度測試
*   **Setup**: Deep Research 流程。
*   **Check**: 檢查 Log，確認是否在 Iteration 1 (Drill-down) 中出現了「缺點」、「限制」或「反對意見」相關的查詢。

---

## 4. 執行順序

1.  **Phase 6.1 (Service)**: 修改 `DeepResearchService` 預設開啟 GraphRAG 與強制 Drill-down。
2.  **Phase 6.2 (Synthesizer)**: 實作信心度加權平均與衝突懲罰。
3.  **Phase 6.3 (Planner)**: 優化 Prompt 加入對抗性查詢指令。
4.  **Phase 6.1 (Graph)**: 實作 `shortest_path` 路徑檢索 (較複雜，放最後)。
