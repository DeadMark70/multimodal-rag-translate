# 前端遷移指南：評估引擎升級 (Phase 4)

> **版本**: v3.0.0  
> **日期**: 2026-01-07  
> **狀態**: 🟢 後端 API 已更新，前端需配合遷移

---

## 1. Breaking Changes 總覽

### 1.1 `DetailedEvaluationResult` Schema 變更

| 舊欄位 (v2.x)              | 新欄位 (v3.0)           | 變更說明           |
| :------------------------- | :---------------------- | :----------------- |
| `relevance_score` (1-5)    | ❌ **已移除**           | 被 `accuracy` 取代 |
| `groundedness_score` (1-5) | ❌ **已移除**           | 被 `accuracy` 取代 |
| `completeness_score` (1-5) | `completeness` (1-10)   | 量表擴展           |
| -                          | `accuracy` (1-10)       | 🆕 新欄位          |
| -                          | `clarity` (1-10)        | 🆕 新欄位          |
| -                          | `weighted_score` (1-10) | 🆕 加權總分        |
| -                          | `suggestion` (string)   | 🆕 改進建議        |
| -                          | `is_passing` (boolean)  | 🆕 是否通過門檻    |
| `reason`                   | `reason`                | ✅ 無變更          |
| `confidence` (0-1)         | `confidence` (0-1)      | ✅ 無變更          |
| `evaluation_failed`        | `evaluation_failed`     | ✅ 無變更          |

### 1.2 新 Schema TypeScript 定義

```typescript
// types/evaluation.ts

/** 1-10 分制多維度評估結果 */
export interface DetailedEvaluationResult {
  /** D1: 數據精確度 (權重 50%) */
  accuracy: number; // 1-10

  /** D2: 完整覆蓋率 (權重 30%) */
  completeness: number; // 1-10

  /** D3: 邏輯表達清晰度 (權重 20%) */
  clarity: number; // 1-10

  /** 加權總分 (0.5*accuracy + 0.3*completeness + 0.2*clarity) */
  weighted_score: number; // 1-10

  /** 詳細評分理由 */
  reason: string;

  /** 改進建議 (用於前端顯示或 Debug) */
  suggestion: string;

  /** 是否通過門檻 (accuracy >= 7) */
  is_passing: boolean;

  /** 信心分數 (保留用於舊邏輯相容) */
  confidence: number; // 0.0-1.0

  /** 評估是否失敗 */
  evaluation_failed: boolean;
}
```

---

## 2. 受影響的 API Endpoints

### 2.1 `/rag/ask` (POST)

當 `enable_evaluation: true` 時，Response 中的 `metrics` 欄位變更：

**舊格式 (v2.x)**

```json
{
  "metrics": {
    "faithfulness": "grounded",
    "confidence_score": 0.85
  }
}
```

**新格式 (v3.0)**

```json
{
  "metrics": {
    "accuracy": 8.5,
    "completeness": 7.0,
    "clarity": 9.0,
    "weighted_score": 8.15,
    "is_passing": true,
    "suggestion": "",
    "faithfulness": "grounded",
    "confidence_score": 0.815
  }
}
```

### 2.2 `/rag/execute` 與 `/rag/execute/stream`

Deep Research 的評估結果現在包含更詳細的分數，可用於：

- 顯示各維度雷達圖
- 標記低分回答 (紅色警示)
- 顯示改進建議 (tooltip)

---

## 3. 前端 UI 建議更新

### 3.1 評分顯示元件

**舊版 (簡單百分比)**

```tsx
<ConfidenceBar value={confidence} /> // 0-100%
```

**新版 (多維度雷達圖)**

```tsx
<EvaluationRadar
  accuracy={8.5}
  completeness={7.0}
  clarity={9.0}
  weighted={8.15}
/>

// 或簡化版顯示
<ScoreBadge score={weighted_score} passing={is_passing} />
```

### 3.2 Passing/Failing 視覺提示

```tsx
// 根據 is_passing 顯示不同樣式
<div className={is_passing ? "bg-green-100" : "bg-red-100"}>
  <span>加權分數: {weighted_score.toFixed(1)}/10</span>
  {!is_passing && <WarningIcon />}
</div>
```

### 3.3 Suggestion 顯示

```tsx
// 當 suggestion 非空時顯示改進建議
{
  suggestion && (
    <Tooltip content={suggestion}>
      <InfoIcon className="text-yellow-500" />
    </Tooltip>
  );
}
```

---

## 4. 向後相容性映射 (Optional)

如果前端暫時無法更新，後端可提供相容層：

```typescript
// 從新分數換算回舊格式
const legacyRelevance = Math.round(accuracy / 2); // 1-10 → 1-5
const legacyGroundedness = Math.round(accuracy / 2); // 1-10 → 1-5
const legacyCompleteness = Math.round(completeness / 2); // 1-10 → 1-5
```

> ⚠️ **注意**: 向後相容層會在 v3.1 移除，請盡快遷移。

---

## 5. 遷移檢查清單

- [ ] 更新 TypeScript 型別定義
- [ ] 更新評分顯示元件 (移除舊欄位依賴)
- [ ] 新增雷達圖/多維度顯示 (optional)
- [ ] 處理 `is_passing` 視覺提示
- [ ] 處理 `suggestion` 顯示 (optional)
- [ ] 測試 `/rag/ask?enable_evaluation=true`
- [ ] 測試 Deep Research 流程

---

## 6. 聯絡與支援

如有問題，請參考：

- [API 文件](api_documentation.md)
- [Codebase Overview](codebase_overview.md)
