# Product Guidelines

## 1. 語調與溝通風格 (Tone & Voice)
*   **專業且嚴謹 (Professional & Academic)**：
    *   模擬資深學術研究員的語氣，用詞精確、客觀。
    *   強調論證的邏輯性與證據來源，避免模糊不清的陳述。
    *   在處理不確定的資訊時，明確指出信賴度或資料缺口，而非強行解釋。
    *   回應應包含適當的學術引用格式。

## 2. 視覺與排版原則 (Visual & Layout)
*   **可讀性優先 (Readability First)**：
    *   在翻譯與重製 PDF 時，首要目標是確保翻譯後的繁體中文內容流暢易讀。
    *   允許適度調整原始排版結構，以適應中文的文字密度與閱讀習慣。
*   **結構化數據 (Structured Data)**：
    *   確保所有輸出的中間產物（尤其是 Markdown）具有嚴謹的語義結構。
    *   Markdown 轉 PDF 的流程需針對學術格式（標題層級、圖表說明、公式顯示）進行持續優化。

## 3. 系統透明度與互動 (Transparency & Interaction)
*   **完全透明模式 (Research Mode)**：
    *   預設為實驗用途，完整揭露 Agentic Workflow 的內部運作。
    *   API 應回傳詳細的日誌，包含 Planner 的決策樹、Executor 的檢索來源、Evaluator 的評分細節與修正建議。
    *   這有助於驗證「Agentic RAG」相較於「單純 LLM」在推理過程上的差異。
*   **階段摘要模式 (Production Mode)**：
    *   保留未來切換至此模式的架構設計。
    *   僅向終端用戶展示高層級的進度狀態（如：規劃中 -> 檢索中 -> 撰寫中），以提升使用者體驗。
