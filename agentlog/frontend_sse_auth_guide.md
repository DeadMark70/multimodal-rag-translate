# 🔧 SSE 串流端點認證修復指南

## 問題描述

`POST /rag/execute/stream` SSE 端點回傳 **401 Unauthorized**。

**根本原因**：瀏覽器原生 `EventSource` API **無法設定自定義 headers**，導致 Authorization header 未傳遞。

---

## 解決方案：使用 `fetch` + `ReadableStream`

### TypeScript 實作

```typescript
interface SSEEvent {
  event: string;
  data: any;
}

async function executeResearchStream(
  request: ExecutePlanRequest,
  token: string,
  onEvent: (event: SSEEvent) => void,
  onError: (error: Error) => void
): Promise<void> {
  const response = await fetch("/rag/execute/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`, // ← 關鍵：手動加入 header
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // 解析 SSE 格式
    const lines = buffer.split("\n");
    buffer = lines.pop() || ""; // 保留未完成的行

    let currentEvent = "";
    let currentData = "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7);
      } else if (line.startsWith("data: ")) {
        currentData = line.slice(6);
      } else if (line === "" && currentEvent && currentData) {
        try {
          onEvent({
            event: currentEvent,
            data: JSON.parse(currentData),
          });
        } catch (e) {
          console.error("Failed to parse SSE data:", e);
        }
        currentEvent = "";
        currentData = "";
      }
    }
  }
}
```

### React Hook 範例

```typescript
function useDeepResearch() {
  const [status, setStatus] = useState<string>("idle");
  const [progress, setProgress] = useState<number>(0);
  const [result, setResult] = useState<ExecutePlanResponse | null>(null);
  const { session } = useSupabase();

  const execute = async (request: ExecutePlanRequest) => {
    setStatus("running");

    await executeResearchStream(
      request,
      session?.access_token || "",
      (event) => {
        switch (event.event) {
          case "task_start":
            setProgress((prev) => prev + 10);
            break;
          case "task_done":
            setProgress((prev) => prev + 10);
            break;
          case "complete":
            setResult(event.data);
            setStatus("complete");
            break;
          case "error":
            setStatus("error");
            break;
        }
      },
      (error) => {
        console.error(error);
        setStatus("error");
      }
    );
  };

  return { execute, status, progress, result };
}
```

---

## SSE 事件類型

| 事件              | 說明         | data 欄位                                  |
| ----------------- | ------------ | ------------------------------------------ |
| `plan_confirmed`  | 開始執行     | `total_tasks`                              |
| `task_start`      | 子任務開始   | `task_id`, `question`                      |
| `task_done`       | 子任務完成   | `task_id`, `answer`, `sources`, `contexts` |
| `drilldown_start` | 深度探索開始 | `iteration`, `gap_count`                   |
| `synthesis_start` | 合成報告開始 | -                                          |
| `complete`        | 執行完成     | 完整 `ExecutePlanResponse`                 |
| `error`           | 錯誤         | `message`                                  |

---

## 注意事項

1. **Token 來源**：從 Supabase session 取得 `access_token`
2. **錯誤處理**：需處理網路斷線、401 過期等情況
3. **進度計算**：可根據 `total_tasks` 和完成數計算百分比
