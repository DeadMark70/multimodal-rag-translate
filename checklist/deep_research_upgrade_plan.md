# 🚀 Deep Research Agent 系統升級計畫書 (v1.0)

這份計畫書融合了論文 *《Deep Research Agents: A Systematic Examination And Roadmap》* 的理論架構，以及您現有 Codebase（`multimodal-rag-translate`）的實際狀況，並嚴格遵守「**僅限本地資料、不聯網**」的限制。

---

## 1. 專案核心目標

打造一個 **「本地端、多模態、具備自我修正能力」** 的學術研究 Agent。它不只是回答問題，而是能像研究生一樣，針對使用者的難題進行「廣度檢索 -> 深度挖掘 -> 視覺查證 -> 辯證總結」。

## 2. 系統架構升級藍圖 (基於論文架構)

我們將系統分為四個核心層級進行優化：**工具層 (Tools)**、**大腦層 (Reasoning & Planning)**、**評估層 (Evaluation)** 與 **表現層 (Synthesis)**。

### 📊 第一層：工具層 (The Toolbox) - 讓 Agent 手眼通天

> *對應論文概念：Iterative Tool Use & Modular Tool Framework*

目標：賦予 Agent 四種核心能力，使其在不聯網的情況下也能徹底挖掘 PDF。

| 工具名稱 | 功能描述 | 對應 Codebase 模組 | 優化行動 (Action Item) |
| --- | --- | --- | --- |
| **Fact Finder** | **精確檢索**：尋找具體的定義、數據、語句。 | `vector_store_manager.py` | [已確認] Metadata 已包含 `page` 欄位，可直接用於精確引用。 |
| **Concept Graph** | **關聯分析**：尋找跨文檔的趨勢、作者關係、方法論演變。 | `graph_rag/` | 在 Prompt 中強化「尋找 A 與 B 的隱藏關聯」指令。 |
| **Context Expander** | **上下文擴充**：當檢索片段斷章取義時，擴大閱讀前後文。 | `data_base/context_enricher.py` | **[新功能]** 實作 `expand_context(chunk_id, window=2)` 工具供 Agent 主動呼叫。 |
| **Figure Inspector** | **視覺查證**：讀取圖表中的數據趨勢。 | `image_summarizer.py` | **[新功能]** 實作 `inspect_figure(fig_id, specific_question)` 作為深層查證備用工具。 |

### 🧠 第二層：大腦層 (Reasoning) - 批判性修正迴圈

> *對應論文概念：Adaptive Long-horizon Planning & Dynamic Reasoning*

目標：從「線性執行」升級為「動態修正」。Agent 不應一條路走到黑，發現資料不足或錯誤應自動轉向。

**優化後的「Deep Research 迴圈」邏輯：**

1. **Plan:** 生成初始研究步驟。
2. **Execute:** 執行檢索（Vector/Graph/Vision）。
3. **Evaluate (關鍵新增):** 針對「當下發現」進行微型評估（Evaluator）。
    * *資料夠嗎？* -> 不夠 -> **Branch:** 新增「追問任務 (Follow-up)」。
    * *有衝突嗎？* -> 有 -> **Branch:** 新增「驗證任務 (Verification)」，比對兩篇論文的實驗設置。
    * *走錯路嗎？* -> 是 -> **Correct:** 取消後續無效任務，重新規劃。
4. **Synthesize:** 彙整資訊。

### ⚖️ 第三層：評估層 (Evaluation) - 學術級嚴謹度

> *對應論文概念：Fact-checking & Hallucination Reduction*

目標：確保輸出的每一句話都有憑有據（Groundedness）。

* **即時檢查 (Runtime Check):** 在 `evaluator.py` 中，當 `groundedness` 分數低於閾值（如 4.0/5.0）時，**拒絕**該次回答，並強制 Agent 重新檢索，而不是硬著頭皮生成。
* **引用驗證 (Citation Validation):** 新增一個正則表達式（Regex）檢查器，確保生成的 Markdown 報告中，所有的 `[Source ID]` 確實存在於 Context 中。

### 📝 第四層：表現層 (Synthesis) - 結構化學術報告

> *對應論文概念：Structured Analytical Reports*

目標：產出「文獻回顧」與「深度問答」混合的高品質報告。

**報告模板結構 (Markdown):**

1. **Executive Summary:** 30秒快速結論。
2. **Key Findings (Deep QA):** 針對使用者問題的直接回答（最核心部分）。
3. **Detailed Analysis:** 包含圖表引用 (`![Figure 1]`)、數據比較表格。
4. **Research Gaps:** 指出目前本地資料庫中缺少的拼圖。
5. **References:** 嚴格的引用列表。

---

## 3. 實作建議清單 (Action Checklist)

請依照以下順序執行優化，優先級由高至低：

### ✅ Phase 1: 基礎工具強化 (The Foundation)

- [x] **Context Enricher 整合**: 修改 `DeepResearchService`，當檢索回來的 chunks 過短 (< 100 字) 時，自動呼叫 `context_enricher.py` 獲取前後文。
    *   *實作細節*: 透過 `metadata['parent_id']` 呼叫 `ParentDocumentStore.get_parent(parent_id)` 來獲取完整的 Parent Chunk (約 2000 字) 作為擴充上下文，不需重新讀取原始 Markdown 檔案。
    *   *未來優化*: 可考慮由 LLM 判斷 `needs_more_context`，但在 MVP 階段先採用字數閾值策略。
- [x] **圖表索引優化**: 檢查 `vector_store_manager.py`，確保 `image_path` 和 `type` 欄位在 `RAG_QA_service` 和 `reranker` 的流程中被**完整保留**，沒有被意外過濾或遺失。
    *   *關鍵修正*: 在 `RAG_QA_service.py` 組合 Prompt 時，必須將圖片路徑顯式寫入文字中，例如：`[圖片摘要] (Path: {img_path}) {content}`，否則 LLM 無法得知圖片路徑進而無法在報告中生成圖片連結。
- [x] **報告模板設計**: 在 `synthesizer.py` 中建立一個新的 Prompt Template，專門用於生成上述的「混合式學術報告」。

### ✅ Phase 2: 動態規劃核心 (The Brain)

- [x] **實作評估驅動迴圈**: 修改 `planner.py` 的 `drill_down_loop`。在每次 `_execute_single_task` 後，插入 `evaluator.evaluate_detailed`。
- [x] **動態任務分支邏輯**: 寫 code 處理：如果 `completeness` 分數低，則自動生成一個新的 search query 並 append 到任務隊列中。
    *   **⚠️ 安全防護**: 必須設定 `max_retries` (如 2 次) 與 `max_depth`，防止 Agent 因為找不到答案而陷入無窮迴圈。若達上限仍失敗，強制終止並標記「資料不足」。

### ✅ Phase 3: 多模態與介面 (The Interface)

- [x] **圖文並茂輸出**: 修改 `synthesizer.py`，指示 LLM 在提到相關數據時，嘗試輸出 `![Figure X](path/to/image)` (路徑從 Metadata 取得)。
- [ ] **(選修) 進階視覺查證**: 修改 `image_summarizer.py`，增加 `re_examine_image(image_data, specific_question)` 函數，以備不時之需。

---

## 4. 給開發者的具體 Prompt 建議 (用於修改程式碼)

您可以直接使用以下 Prompt 來請 AI 協助修改您的程式碼：

**針對 Context Enricher 整合：**

> "請修改 `data_base/deep_research_service.py`。在檢索階段，如果取回的 text chunk 字數少於 100 字，請利用 `contentFetchId` 自動呼叫 `context_enricher.py` 獲取該 chunk 的前後文，並將擴充後的內容合併進 context。"

**針對動態修正迴圈：**

> "請參考論文中的 Adaptive Planning 概念，修改 `agents/planner.py`。在 `drill_down_loop` 中，執行完任務後，請呼叫 `evaluator.py` 檢查結果的完整性。如果分數低於 3 分，請讓 LLM 根據當前的失敗原因，生成一個新的 Search Query 來重試，而不是直接進入下一步。"

**針對多模態報告：**

> "請修改 `agents/synthesizer.py` 的 System Prompt。要求模型在撰寫報告時，必須參考 Context 中提供的圖片摘要。如果引用了圖片摘要的內容，請務必在報告中以 Markdown 圖片格式 `![Figure Name](image_path)` 插入圖片。"
