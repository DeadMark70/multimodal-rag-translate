# 📚 前端整合指南：學術評估引擎與多模態顯示 (v3.0)

> **建立日期**: 2026-01-06  
> **適用版本**: Backend v3.0.0+ (1-10 分制評估系統)  
> **狀態**: 🚀 Ready for Implementation

---

## 1. 核心變動總覽 (Executive Summary)

本次後端升級引入了 **1-10 分制學術評估引擎**，前端需要配合調整以下三點：
1.  **API 請求**：在問答時主動開啟評估開關。
2.  **資料接收**：處理新的 `DetailedEvaluationResult` 結構 (不再是 1-5 分)。
3.  **UI 呈現**：新增「評分雷達圖」與「Markdown 圖片渲染」。

---

## 2. API 整合細節

### A. 普通問答 (Basic RAG)
*   **Endpoint**: `POST /rag/ask`
*   **關鍵參數**: `enable_evaluation: true`

```json
// Request 範例
{
  "question": "Transformer 是什麼?",
  "doc_ids": ["uuid..."],
  "enable_evaluation": true 
}
```

```json
// Response (EnhancedAskResponse)
{
  "answer": "Transformer 是一種...",
  "sources": [...],
  "metrics": {
    "accuracy": 9.0,        // 精確度 (1-10)
    "completeness": 8.0,    // 完整性 (1-10)
    "clarity": 9.5,         // 清晰度 (1-10)
    "weighted_score": 8.8,  // 加權總分
    "is_passing": true,     // 是否及格
    "suggestion": ""        // 改進建議 (若不及格會有值)
  }
}
```

### B. 深度研究 (Deep Research)
*   **Endpoint**: `POST /rag/execute`
*   **新參數**: `enable_deep_image_analysis: boolean` (預設為 false)

---

## 3. UI 設計指南 (展示「差異化」)

建議在展示頁面中實作以下三種視覺模式：

### 模式 1: 品質標籤 (Answer Bubble)
在每個 AI 回答的氣泡下方，顯示一個精簡的「品質信賴度」。

*   **及格 (Score >= 7)**: 🟢 🛡️ 9.0/10 (高信賴)
*   **普通 (Score 6-7)**: 🟡 ⚠️ 6.5/10 (建議查證)
*   **不及格 (Score < 6)**: 🔴 ❌ 4.0/10 (可能包含幻覺)

### 模式 2: 對照實驗競技場 (Arena Mode)
用於向評審/教授展示 RAG 的優越性。

1.  **操作**: 使用者輸入問題。
2.  **執行**: 前端同時發送兩個請求：
    *   **左側 (Pure LLM)**: 呼叫 `/rag/ask` 但**不帶** `doc_ids`。
    *   **右側 (Deep RAG)**: 呼叫 `/rag/ask` 並帶上文件 ID。
3.  **比較**: 並排顯示兩者的「雷達圖」。通常右側的 `Accuracy` 會顯著高於左側。

### 模式 3: 圖文並茂報告 (Multimodal Rendering)
Deep Research 輸出的 Markdown 現在包含圖片引用。

*   **格式**: `![Figure 1](/uploads/user_id/rag_index/img.png)`
*   **處理**: 前端 Markdown 元件需處理圖片路徑，建議補上後端 Base URL。
*   **點擊**: 支援點擊圖片放大預覽。

---

## 4. 遷移檢查表 (Frontend Checklist)

- [ ] **TypeScript**: 更新 `metrics` 的 Interface 為 1-10 分制欄位。
- [ ] **API Client**: `askQuestion` 函數新增 `enable_evaluation` 布林參數。
- [ ] **UI Component**: 實作 `RadarChart` (可使用 Recharts 或 Flutter Chart 套件)。
- [ ] **Error Handling**: 當 `is_passing` 為 false 時，顯示 `suggestion` 中的修正建議。
- [ ] **Image Handling**: 檢查 Markdown 圖片渲染是否能正確載入 `/uploads` 下的資源。

---

## 5. 常見問題 (FAQ)

**Q: 為什麼 Accuracy 分數特別低？**  
A: 通常是因為模型檢索到了無關資料，或是出現了與文獻不符的幻覺。請查看 `reason` 欄位以獲取 AI 裁判的詳細分析。

**Q: 如何在不影響速度的情況下使用？**  
A: 評估會增加約 2-3 秒的延遲。建議在 UI 上提供一個切換開關，讓使用者決定是否需要「品質分析」。
