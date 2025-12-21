# Backend API Reference

> 完整的後端 API 文件，供前端 agent 整合使用

---

## 認證

所有端點需要在 Header 帶入 Supabase JWT：

```
Authorization: Bearer <supabase-jwt-token>
```

開發環境設定 `DEV_MODE=true` 可跳過認證。

---

## RAG 問答端點 (`/rag`)

### GET `/rag/ask` - 基本問答

```http
GET /rag/ask?question=什麼是機器學習&doc_ids=uuid1,uuid2
```

| 參數       | 類型   | 必填 | 說明                             |
| ---------- | ------ | ---- | -------------------------------- |
| `question` | string | ✅   | 使用者問題                       |
| `doc_ids`  | string | ❌   | 逗號分隔的文件 ID (留空查詢全部) |

**Response:**

```json
{
  "question": "什麼是機器學習",
  "answer": "機器學習是人工智慧的一個分支...",
  "sources": ["doc-uuid-1", "doc-uuid-2"]
}
```

---

### POST `/rag/ask` - 上下文感知問答 (推薦)

支援對話歷史與進階檢索策略。**新增實驗室評估模式**。

**Request:**

```json
{
  "question": "這份文件的結論是什麼？",
  "doc_ids": ["doc-uuid-123"],
  "history": [
    { "role": "user", "content": "這份研究報告的主題是什麼？" },
    {
      "role": "assistant",
      "content": "這份研究報告探討機器學習在醫療診斷中的應用..."
    }
  ],
  "enable_hyde": false,
  "enable_multi_query": false,
  "enable_reranking": true,
  "enable_evaluation": false
}
```

| 欄位                 | 類型          | 預設  | 說明                      |
| -------------------- | ------------- | ----- | ------------------------- |
| `question`           | string        | -     | 使用者問題 (1-2000 字)    |
| `doc_ids`            | string[]      | null  | 限定查詢的文件 ID         |
| `history`            | ChatMessage[] | null  | 對話歷史 (最多 10 條)     |
| `enable_hyde`        | boolean       | false | 假設性文件增強檢索        |
| `enable_multi_query` | boolean       | false | 多重查詢融合檢索          |
| `enable_reranking`   | boolean       | true  | Cross-Encoder 重排序      |
| `enable_evaluation`  | boolean       | false | 🆕 啟用 Self-RAG 評估模式 |

**Response (enable_evaluation=false):**

```json
{
  "question": "...",
  "answer": "...",
  "sources": ["doc-id-1", "doc-id-2"]
}
```

**Response (enable_evaluation=true):** 🆕

```json
{
  "question": "...",
  "answer": "...",
  "sources": [
    {
      "doc_id": "doc-id-1",
      "filename": "paper_a.pdf",
      "page": 3,
      "snippet": "相關段落內容...",
      "score": 0.85
    }
  ],
  "metrics": {
    "faithfulness": "grounded",
    "confidence_score": 0.82,
    "evaluation_reason": "答案完全根據文檔內容，包含具體數據支撑"
  }
}
```

| metrics.欄位        | 說明                   |
| ------------------- | ---------------------- |
| `faithfulness`      | 忠實度等級             |
| `confidence_score`  | 加權信心分數 (0.2-1.0) |
| `evaluation_reason` | 評估結果說明 (新增)    |

| faithfulness 值     | 說明                                       |
| ------------------- | ------------------------------------------ |
| `grounded`          | 答案完全有據 (groundedness ≥ 4) ✅         |
| `uncertain`         | 部分有據 (groundedness = 3)                |
| `hallucinated`      | 答案可能包含編造內容 (groundedness ≤ 2) ⚠️ |
| `evaluation_failed` | LLM 評估失敗 ❌                            |

---

### POST `/rag/research` - 深度研究

複雜問題分解與綜合分析 (Plan-and-Solve)。

**Request:**

```json
{
  "question": "比較 Python 和 JavaScript 的優缺點",
  "max_subtasks": 5,
  "enable_reranking": true
}
```

**Response:**

```json
{
  "question": "比較 Python 和 JavaScript 的優缺點",
  "summary": "兩種語言各有優勢...",
  "detailed_answer": "## Python\n優點：...\n## JavaScript\n優點：...",
  "sub_tasks": [
    {
      "id": 1,
      "question": "Python 的主要優點是什麼？",
      "answer": "...",
      "sources": []
    },
    {
      "id": 2,
      "question": "JavaScript 的主要優點是什麼？",
      "answer": "...",
      "sources": []
    }
  ],
  "all_sources": ["doc-1", "doc-2"],
  "confidence": 0.85
}
```

---

## PDF 處理端點 (`/pdfmd`)

### GET `/pdfmd/list` - 取得文件列表 🆕

```http
GET /pdfmd/list
```

**Response:**

```json
{
  "documents": [
    {
      "id": "uuid-1",
      "filename": "paper_a.pdf",
      "created_at": "2024-12-19T10:00:00Z",
      "status": "completed",
      "processing_step": "indexed"
    }
  ],
  "total": 1
}
```

| 欄位              | 類型     | 說明         |
| ----------------- | -------- | ------------ |
| `id`              | string   | 文件 UUID    |
| `filename`        | string   | 原始檔名     |
| `created_at`      | datetime | 上傳時間     |
| `status`          | string   | 處理狀態     |
| `processing_step` | string   | 詳細處理步驟 |

> 📌 **限制**: 最多返回 50 筆，依上傳時間降序排序

---

### POST `/pdfmd/upload_pdf_md` - 上傳並翻譯 PDF

```http
POST /pdfmd/upload_pdf_md
Content-Type: multipart/form-data
```

| 欄位   | 類型 | 說明     |
| ------ | ---- | -------- |
| `file` | File | PDF 檔案 |

**Response:** 直接返回翻譯後的 PDF 檔案 (FileResponse)

**處理流程:**

1. OCR → 2. 翻譯 → 3. 生成 PDF → 4. (背景) RAG 索引 → 5. (背景) 摘要生成

---

### GET `/pdfmd/file/{doc_id}/status` - 取得處理狀態

前端輪詢用端點。

```http
GET /pdfmd/file/{doc_id}/status
```

**Response:**

```json
{
  "step": "translating",
  "step_label": "翻譯中",
  "is_pdf_ready": false,
  "is_fully_complete": false
}
```

| step 值          | 說明        |
| ---------------- | ----------- |
| `uploading`      | 上傳中      |
| `ocr`            | OCR 辨識中  |
| `translating`    | 翻譯中      |
| `generating_pdf` | 生成 PDF 中 |
| `completed`      | 翻譯完成    |
| `indexing`       | 建立索引中  |
| `indexed`        | 全部完成    |
| `failed`         | 處理失敗    |

---

### GET `/pdfmd/file/{doc_id}` - 下載翻譯 PDF

```http
GET /pdfmd/file/{doc_id}
```

**Response:** PDF 檔案 (FileResponse)

---

### DELETE `/pdfmd/file/{doc_id}` - 刪除文件

刪除文件及相關的 RAG 索引。

```http
DELETE /pdfmd/file/{doc_id}
```

**Response:**

```json
{ "status": "success", "message": "Document deleted successfully" }
```

---

### GET `/pdfmd/file/{doc_id}/summary` - 取得摘要

```http
GET /pdfmd/file/{doc_id}/summary
```

**Response:**

```json
{
  "status": "ready",
  "summary": "本文探討了..."
}
```

| status 值       | 說明       |
| --------------- | ---------- |
| `ready`         | 摘要已就緒 |
| `generating`    | 正在生成中 |
| `not_available` | 尚未生成   |

---

### POST `/pdfmd/file/{doc_id}/summary/regenerate` - 重新生成摘要

```http
POST /pdfmd/file/{doc_id}/summary/regenerate
```

**Response:**

```json
{ "status": "started", "message": "Summary regeneration scheduled" }
```

---

## 圖片翻譯端點 (`/imagemd`)

### POST `/imagemd/translate_image` - 圖片文字翻譯

```http
POST /imagemd/translate_image
Content-Type: multipart/form-data
```

| 欄位   | 類型 | 說明                |
| ------ | ---- | ------------------- |
| `file` | File | 圖片 (jpg/png/webp) |

**Response:** 翻譯後的圖片 (JPEG)

---

## 多模態端點 (`/multimodal`)

### POST `/multimodal/extract` - 擷取文字與視覺元素

```http
POST /multimodal/extract
Content-Type: multipart/form-data
```

| 欄位   | 類型 | 說明     |
| ------ | ---- | -------- |
| `file` | File | PDF 檔案 |

**Response:**

```json
{
  "doc_id": "uuid",
  "user_id": "user-id",
  "text_chunks": [
    {"page_number": 1, "content": "...", "chunk_id": "..."}
  ],
  "visual_elements": [
    {
      "id": "uuid",
      "type": "figure",
      "page_number": 1,
      "image_path": "path/to/img.jpg",
      "bbox": [x1, y1, x2, y2],
      "summary": "這是一張流程圖，顯示..."
    }
  ]
}
```

---

### DELETE `/multimodal/file/{doc_id}` - 刪除多模態文件

```http
DELETE /multimodal/file/{doc_id}
```

**Response:**

```json
{ "status": "success", "message": "Multimodal document deleted successfully" }
```

---

## 統計端點 (`/stats`) 🆕

### GET `/stats/dashboard` - 儀表板統計

```http
GET /stats/dashboard
```

**Response:**

```json
{
  "total_queries": 42,
  "accuracy_rate": 0.85,
  "grounded_count": 36,
  "hallucinated_count": 4,
  "uncertain_count": 2,
  "avg_confidence": 0.78,
  "queries_last_7_days": [5, 8, 6, 7, 4, 6, 6],
  "top_documents": [
    { "doc_id": "uuid-1", "filename": "paper_a.pdf", "query_count": 15 }
  ]
}
```

| 欄位                  | 類型  | 說明                          |
| --------------------- | ----- | ----------------------------- |
| `total_queries`       | int   | 總查詢次數                    |
| `accuracy_rate`       | float | 準確率 (grounded / evaluated) |
| `grounded_count`      | int   | 有據回答數                    |
| `hallucinated_count`  | int   | 幻覺回答數                    |
| `uncertain_count`     | int   | 無法判斷數                    |
| `avg_confidence`      | float | 平均信心分數                  |
| `queries_last_7_days` | int[] | 近 7 天查詢趨勢               |
| `top_documents`       | array | 最常查詢的文件                |

> 📌 需先執行 `001_create_query_logs.sql` migration

---

## Schemas Reference

### ChatMessage

```typescript
interface ChatMessage {
  role: "user" | "assistant"; // 注意：不支援 "system"
  content: string;
}
```

### AskRequest

```typescript
interface AskRequest {
  question: string; // 1-2000 字
  doc_ids?: string[] | null;
  history?: ChatMessage[] | null; // 最多 10 條
  enable_hyde?: boolean; // default: false
  enable_multi_query?: boolean; // default: false
  enable_reranking?: boolean; // default: true
  enable_evaluation?: boolean; // default: false 🆕
}
```

### AskResponse (基本回應)

```typescript
interface AskResponse {
  question: string;
  answer: string;
  sources: string[]; // 引用的文件 ID
}
```

### EnhancedAskResponse (評估模式回應) 🆕

```typescript
interface EnhancedAskResponse {
  question: string;
  answer: string;
  sources: SourceDetail[];
  metrics: EvaluationMetrics | null;
}
```

### SourceDetail 🆕

```typescript
interface SourceDetail {
  doc_id: string;
  filename: string | null;
  page: number | null;
  snippet: string; // 引用段落 (前 200 字)
  score: number; // 相關性分數 0.0-1.0
}
```

### EvaluationMetrics 🆕

```typescript
interface EvaluationMetrics {
  faithfulness: "grounded" | "hallucinated" | "uncertain" | "evaluation_failed";
  confidence_score: number; // 0.2-1.0 (加權計算)
  evaluation_reason: string | null; // 評估結果說明
}
```

> **信心分數計算**: `(相關性×0.3 + 依據性×0.5 + 完整性×0.2) / 5`

### DocumentItem 🆕

```typescript
interface DocumentItem {
  id: string;
  filename: string;
  created_at: string; // ISO 8601
  status: string | null;
  processing_step: string | null;
}
```

### DashboardStats 🆕

```typescript
interface DashboardStats {
  total_queries: number;
  accuracy_rate: number;
  grounded_count: number;
  hallucinated_count: number;
  uncertain_count: number;
  avg_confidence: number;
  queries_last_7_days: number[];
  top_documents: DocumentStat[];
}

interface DocumentStat {
  doc_id: string;
  filename: string | null;
  query_count: number;
}
```

---

## 錯誤處理

| HTTP Status | 說明                                |
| ----------- | ----------------------------------- |
| 400         | 無效輸入 (檔案類型錯誤、歷史過長等) |
| 401         | 未認證或 Token 過期                 |
| 404         | 文件不存在                          |
| 500         | 伺服器內部錯誤                      |

**錯誤回應格式:**

```json
{ "detail": "錯誤訊息" }
```

---

## 實作狀態

### 已完成 ✅

- [x] GET `/rag/ask` 基本問答
- [x] POST `/rag/ask` 上下文感知問答 (含評估模式)
- [x] POST `/rag/research` 深度研究
- [x] GET `/pdfmd/list` 文件列表
- [x] POST `/pdfmd/upload_pdf_md` 上傳翻譯
- [x] GET `/pdfmd/file/{doc_id}/status` 處理狀態
- [x] GET `/pdfmd/file/{doc_id}` 下載檔案
- [x] DELETE `/pdfmd/file/{doc_id}` 刪除文件
- [x] GET `/pdfmd/file/{doc_id}/summary` 取得摘要
- [x] POST `/pdfmd/file/{doc_id}/summary/regenerate` 重新生成摘要
- [x] POST `/imagemd/translate_image` 圖片翻譯
- [x] POST `/multimodal/extract` 多模態擷取
- [x] DELETE `/multimodal/file/{doc_id}` 刪除多模態文件
- [x] GET `/stats/dashboard` 儀表板統計

---

## 版本記錄

| 版本  | 日期       | 變更                                             |
| ----- | ---------- | ------------------------------------------------ |
| 2.2.0 | 2024-12-20 | 🆕 評估優化：1-5 分制信心計算、evaluation_reason |
| 2.1.0 | 2024-12-19 | 新增 `/pdfmd/list`, `/stats/dashboard`, 評估模式 |
| 2.0.0 | 2024-12-01 | 初始 API 規格                                    |
