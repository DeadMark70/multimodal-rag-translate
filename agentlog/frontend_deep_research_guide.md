# Deep Research 前端整合指南

> 本文件指導前端開發如何整合 Interactive Deep Research API

---

## 概述

系統提供三種深度研究方式：

| 端點                                          | 用途              |
| --------------------------------------------- | ----------------- |
| `POST /rag/research`                          | 一鍵到底（舊版）  |
| `POST /rag/plan` + `POST /rag/execute`        | 兩階段確認式      |
| `POST /rag/plan` + `POST /rag/execute/stream` | 兩階段 + SSE 串流 |

**推薦使用第三種方式**，提供最佳用戶體驗。

---

## API 端點

### 1. `POST /rag/plan` - 生成研究計畫

```typescript
// Request
interface ResearchPlanRequest {
  question: string; // 研究問題
  doc_ids?: string[]; // 限定文件 (可選)
  enable_graph_planning?: boolean; // 啟用圖譜規劃
}

// Response
interface ResearchPlanResponse {
  status: "waiting_confirmation";
  original_question: string;
  sub_tasks: EditableSubTask[];
  estimated_complexity: "simple" | "medium" | "complex";
  doc_ids: string[] | null;
}

interface EditableSubTask {
  id: number;
  question: string;
  task_type: "rag" | "graph_analysis";
  enabled: boolean; // 使用者可切換
}
```

### 2. `POST /rag/execute` - 執行計畫（非串流）

```typescript
// Request
interface ExecutePlanRequest {
  original_question: string;
  sub_tasks: EditableSubTask[]; // 來自 /plan 回傳，可修改
  doc_ids?: string[];
  max_iterations?: number; // 1-5，預設 2
  enable_reranking?: boolean;
  enable_drilldown?: boolean; // 啟用遞迴深入
  enable_deep_image_analysis?: boolean; // 🆕 啟用進階圖片查證 (預設 false)
}

// Response: ExecutePlanResponse (完整結果)
```

### 3. `POST /rag/execute/stream` - 執行計畫（SSE 串流）

請求格式同 `/execute`，回傳 SSE 串流。

---

## SSE 事件類型

```typescript
type SSEEventType =
  | "plan_confirmed" // 計畫確認，開始執行
  | "task_start" // 子任務開始
  | "task_done" // 子任務完成
  | "drilldown_start" // 遞迴深入開始
  | "drilldown_task_start"
  | "drilldown_task_done"
  | "synthesis_start" // 綜合報告生成中
  | "complete" // 完成，包含完整結果
  | "error"; // 錯誤
```

### 事件資料結構

```typescript
// plan_confirmed
{ task_count: number; enabled_count: number }

// task_start / drilldown_task_start
{ id: number; question: string; task_type: string; iteration: number }

// task_done / drilldown_task_done
{ id: number; question: string; answer: string; sources: string[]; contexts: string[]; iteration: number }

// drilldown_start
{ iteration: number; new_task_count: number }

// synthesis_start
{ total_tasks: number }

// complete
// 完整的 ExecutePlanResponse

// error
{ message: string; task_id?: number }
```

---

## React 整合範例

### 1. Hook: `useDeepResearch`

```typescript
import { useState, useCallback } from "react";

interface UseDeepResearchReturn {
  // 狀態
  plan: ResearchPlanResponse | null;
  isPlanning: boolean;
  isExecuting: boolean;
  progress: TaskProgress[];
  result: ExecutePlanResponse | null;
  error: string | null;

  // 方法
  generatePlan: (question: string, docIds?: string[]) => Promise<void>;
  executePlan: (plan: ExecutePlanRequest) => void;
  updateTask: (taskId: number, updates: Partial<EditableSubTask>) => void;
}

interface TaskProgress {
  id: number;
  question: string;
  status: "pending" | "running" | "done" | "error";
  answer?: string;
  iteration: number;
}

export function useDeepResearch(): UseDeepResearchReturn {
  const [plan, setPlan] = useState<ResearchPlanResponse | null>(null);
  const [progress, setProgress] = useState<TaskProgress[]>([]);
  const [result, setResult] = useState<ExecutePlanResponse | null>(null);
  // ... 實作略
}
```

### 2. SSE 連接

```typescript
async function executePlanWithSSE(
  request: ExecutePlanRequest,
  accessToken: string,
  onEvent: (event: SSEEvent) => void
) {
  const response = await fetch(`${API_BASE}/rag/execute/stream`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split("\n");

    let eventType = "";
    let eventData = "";

    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        eventData = line.slice(5).trim();
      } else if (line === "" && eventType && eventData) {
        onEvent({ type: eventType, data: JSON.parse(eventData) });
        eventType = "";
        eventData = "";
      }
    }
  }
}
```

---

## UI 元件建議

### 1. 計畫預覽 (PlanPreview)

```
┌─────────────────────────────────────────┐
│ 研究計畫預覽                             │
├─────────────────────────────────────────┤
│ 原始問題：比較 Transformer 和 RNN...     │
│ 預估複雜度：medium                       │
├─────────────────────────────────────────┤
│ ☑ 1. Transformer 的核心架構？  [RAG]     │
│ ☑ 2. RNN 的優缺點？            [RAG]     │
│ ☐ 3. 兩者的關係               [GRAPH]   │
│   └─ [編輯] [刪除]                       │
│                                          │
│ [+ 新增子任務]                           │
├─────────────────────────────────────────┤
│        [取消]     [開始研究 →]           │
└─────────────────────────────────────────┘
```

### 2. 執行進度 (ExecutionProgress)

```
┌─────────────────────────────────────────┐
│ 研究進行中...                            │
├─────────────────────────────────────────┤
│ ✓ 任務 1: Transformer 架構    (完成)     │
│ ● 任務 2: RNN 優缺點          (執行中)   │
│ ○ 任務 3: 關係分析            (待執行)   │
├─────────────────────────────────────────┤
│ 第 1 輪深入挖掘                          │
│ ○ 追加任務: Attention 機制細節           │
├─────────────────────────────────────────┤
│ [=====>              ] 40%               │
└─────────────────────────────────────────┘
```

---

## 錯誤處理

```typescript
// SSE 錯誤事件
eventSource.addEventListener("error", (event) => {
  const data = JSON.parse(event.data);
  toast.error(`研究失敗: ${data.message}`);
});

// HTTP 錯誤
if (response.status === 400) {
  // 無效請求 (如：沒有啟用的子任務)
}
if (response.status === 401) {
  // 未授權，需重新登入
}
if (response.status === 500) {
  // 伺服器錯誤
}
```

---

## 依賴安裝

後端需要安裝：

```bash
pip install sse-starlette
```

前端無需額外依賴，使用原生 `fetch` + `ReadableStream`。
