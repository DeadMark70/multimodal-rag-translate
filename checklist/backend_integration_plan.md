# Multimodal RAG System - 後端整合與功能開關計畫書

這份文件詳細說明了將 FastAPI 後端服務整合至 React 前端的策略。重點在於**全面將後端的可配置參數（開關）暴露給前端使用者**，同時確保類型安全、可維護性與安全性。

## 1. 專案目標

-   **全功能介面化**: 所有後端支援的參數（如 RAG 策略、圖譜模式、評估模式）都必須在前端有對應的控制項（開關、滑桿、選單）。
-   **完整的後端整合**: 涵蓋 PDF 處理、RAG 問答、深度研究 (Deep Research) 與儀表板數據。
-   **類型安全 (Type Safety)**: 前後端共用一致的 TypeScript 類型定義。
-   **使用者體驗**: 針對耗時操作（如 OCR、生成）提供明確的進度回饋與錯誤處理。

## 2. 架構與標準

### 2.1 目錄結構優化

我們將標準化 `src` 目錄下的服務與類型定義：

```text
src/
├── services/
│   ├── api.ts              # 核心 Axios 實例（攔截器處理 Token）
│   ├── ragApi.ts           # RAG 問答與深度研究 API
│   ├── pdfApi.ts           # PDF 上傳、狀態查詢、管理 API
│   ├── graphApi.ts         # (新增) GraphRAG 相關 API
│   ├── statsApi.ts         # (新增) 儀表板數據 API
│   └── supabase.ts         # Supabase 客戶端設定
├── types/
│   ├── api.ts              # 通用 API 回應介面
│   ├── rag.ts              # RAG 相關參數與回應定義 (含功能開關介面)
│   ├── document.ts         # 文件狀態與列表定義
│   ├── graph.ts            # (新增) 圖譜狀態定義
│   └── stats.ts            # (新增) 統計數據定義
├── hooks/
│   ├── queries/            # 讀取資料 (useDocuments, useGraphStatus)
│   └── mutations/          # 修改資料 (useUploadPdf, useAskQuestion)
└── contexts/
    ├── AuthContext.tsx     # 既有的認證 Context
    └── SettingsContext.tsx # (新增) 全域功能開關設定 Context
```

### 2.2 開發規範
-   **React Query**: 全面使用 TanStack Query 取代 useEffect 進行資料獲取與快取管理。
-   **錯誤邊界 (Error Boundaries)**: 為主要區塊添加錯誤攔截，避免單一錯誤導致白屏。
-   **環境變數**: 僅透過 `import.meta.env` 存取設定。

## 3. 功能開關與 UI 對應 (Feature Toggles)

這是本計畫的核心，確保使用者能控制後端行為。我們將建立一個 `SettingsContext` 或在聊天介面增加「進階設定面板」。

### 3.1 RAG 問答設定 (`/rag/ask`)
這些設定應以「側邊欄」或「彈出式設定選單」呈現於聊天室中。

| 參數名稱 | 類型 | UI 元件 | 說明 |
| :--- | :--- | :--- | :--- |
| `enable_hyde` | Switch (開關) | 假設性文件增強 (HyDE) | 讓 AI 先生成假設性答案來幫助檢索，提升召回率。 |
| `enable_multi_query` | Switch (開關) | 多重查詢融合 | 將問題拆解為多個角度進行檢索。 |
| `enable_reranking` | Switch (開關) | 精準重排序 (Rerank) | **預設開啟**。使用 Cross-Encoder 對檢索結果進行二次排序，大幅提升準確度。 |
| `enable_evaluation` | Switch (開關) | AI 自我評估 (Self-RAG) | 讓 AI 評估回答的「忠實度」與「信心分數」，並顯示於對話氣泡旁。 |
| `enable_graph_rag` | Switch (開關) | 知識圖譜增強 | 啟用 GraphRAG 檢索，捕捉跨文件的關聯性。 |
| `graph_search_mode` | Select (下拉選單) | 圖譜搜尋模式 | 選項：`Local` (細節), `Global` (宏觀), `Hybrid` (混合), `Auto` (自動)。 |
| `enable_graph_planning` | Switch (開關) | 圖譜輔助規劃 | 針對複雜問題，使用圖譜結構進行推理規劃。 |

### 3.2 深度研究設定 (`/rag/research`)
這些設定應位於「深度研究模式」的啟動畫面。

| 參數名稱 | 類型 | UI 元件 | 說明 |
| :--- | :--- | :--- | :--- |
| `max_subtasks` | Slider (滑桿) | 最大子任務數 | 設定 1-10，決定研究的廣度與深度。 |
| `enable_reranking` | Switch (開關) | 啟用重排序 | 研究模式下是否啟用精準排序。 |

### 3.3 知識圖譜管理 (`/graph/*`)
位於「知識庫」頁面的進階操作區。

| 參數名稱 | 類型 | UI 元件 | 說明 |
| :--- | :--- | :--- | :--- |
| `force` (Rebuild) | Checkbox | 強制重建 | 即使圖譜已存在，仍強制刪除並重新建立。 |
| `regenerate_communities` | Switch (開關) | 重生社群摘要 | 優化圖譜時，是否重新讓 LLM 總結社群資訊。 |

## 4. 整合階段規劃

### 第一階段：基礎建設與類型定義 (立即執行)
1.  **安裝依賴**: 安裝 `@tanstack/react-query`。
2.  **類型同步**: 建立 `src/types/*.ts`，完整定義上述所有參數介面。
3.  **Axios 優化**: 確保 `src/services/api.ts` 的攔截器能正確處理 401 錯誤與 Token 刷新。

### 第二階段：核心功能與開關實作
1.  **文件管理**:
    -   實作 PDF 上傳（顯示詳細進度條）。
    -   實作文件列表與刪除。
2.  **聊天介面改造**:
    -   實作 `SettingsPanel` 元件，放入上述 3.1 節的所有開關。
    -   更新 `useChat` Hook，將這些設定值傳遞給 `askQuestion` API。
    -   在對話氣泡中顯示 `EvaluationMetrics`（信心分數、忠實度標籤）。

### 第三階段：進階功能 (GraphRAG & Dashboard)
1.  **知識圖譜頁面**:
    -   新增圖譜狀態顯示（節點數、邊數）。
    -   實作「重建圖譜」與「優化圖譜」的操作按鈕。
2.  **儀表板**:
    -   串接 `/stats/dashboard`，使用圖表庫顯示查詢趨勢與準確率。
3.  **深度研究模式**:
    -   獨立的 UI 視圖，顯示 Plan-and-Solve 的執行步驟 (`sub_tasks`)。

## 5. 狀態管理策略

使用 **React Query** 取代手動 fetch：

```typescript
// 範例：獲取文件列表
export const useDocuments = () => {
  return useQuery({
    queryKey: ['documents'],
    queryFn: pdfApi.listDocuments,
    staleTime: 60 * 1000, // 1 分鐘快取
  });
};

// 範例：聊天設定 (使用 React Context 或 LocalStorage 持久化)
export const useChatSettings = () => {
  // 儲存使用者偏好的 RAG 設定，避免每次重整都要重設
  // ...
}
```

## 6. 安全性檢查清單
- [ ] **JWT 驗證**: 確保所有 API 請求（除健康檢查外）都帶有 `Authorization` header。
- [ ] **輸入驗證**: 前端需先驗證檔案類型 (PDF) 與大小限制。
- [ ] **錯誤處理**: API 回傳 500 時，不可讓應用程式崩潰，應顯示友善提示。

## 7. 測試策略
-   **單元測試**: 測試 Hooks 與工具函式。
-   **整合測試**: 模擬 API 回傳，測試開關切換是否正確改變了送出的 Request Body。