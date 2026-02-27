# 前端實作指南：深層研究歷史詳情 (Deep Research History Details)

> **交接對象：** 前端開發 Agent
> **背景：** 後端已完成資料持久化邏輯，現在深層研究的完整結果（報告、子任務、信心分數）會自動儲存在 `conversations` 表的 `metadata` 欄位中。

---

## 1. 後端變動總結 (Backend Changes)

1.  **資料庫更新**：`conversations` 表現在擁有 `metadata (JSONB)` 欄位。
2.  **API 參數更新**：`POST /rag/execute` 和 `POST /rag/execute/stream` 的請求體 (`ExecutePlanRequest`) 現在接受可選的 `conversation_id`。
3.  **自動持久化**：當執行研究時若帶入 `conversation_id`，執行完成後後端會自動：
    *   將完整的 `ExecutePlanResponse` 存入該會話的 `metadata`。
    *   自動根據研究問題更新該會話的 `title`（不再是 "新對話"）。
4.  **讀取介面**：`GET /api/conversations/{id}` 現在會返回包含完整研究數據的 `metadata`。

---

## 2. 前端實作任務 (Frontend Tasks)

### 階段 1：類型與 API 對接
- **更新類型定義**：在 `Conversation` 介面中加入 `metadata: any` (或定義更具體的 `ResearchMetadata` 類型)。
- **傳遞 ID**：在呼叫執行研究的 Hook/Service 時，請務必先建立會話取得 `id`，並在 `/execute` 請求中帶入 `conversation_id`。

### 階段 2：UI 組件開發
- **ResearchDetailModal**：
    *   當使用者點擊歷史紀錄中的「研究」類型項目時，彈出此 Modal。
    *   數據來源：從 `conversation.metadata` 中提取。
    *   內容包含：
        *   **Summary**：顯示行政摘要。
        *   **Detailed Answer**：使用 Markdown 渲染完整報告。
        *   **Sub-tasks (Accordion)**：使用摺疊面板顯示每個子任務的 QA 與引用來源。
- **側邊欄優化**：在歷史列表項中，若 `type === 'research'`，顯示一個「查看詳情」的按鈕或特定的研究圖示。

---

## 3. Metadata 資料結構參考

`metadata` 欄位中的 JSON 結構如下：

```json
{
  "summary": "...",
  "detailed_answer": "...",
  "confidence": 0.95,
  "total_iterations": 2,
  "all_sources": ["doc_id_1", "..."],
  "sub_tasks": [
    {
      "id": 1,
      "question": "...",
      "answer": "...",
      "sources": ["..."],
      "thought_process": "...",
      "iteration": 0
    }
  ],
  "question": "原始問題"
}
```

---

## 4. 驗收標準 (Acceptance Criteria)
- [ ] 點擊左側歷史清單中的研究任務，能開啟 Modal 並正確渲染上述所有資訊。
- [ ] 子任務應預設摺疊，點擊可展開查看細節。
- [ ] 完整報告需支援 Markdown 格式（包含代碼塊、表格等）。
