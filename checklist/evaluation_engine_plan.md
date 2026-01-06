# 🎯 Phase 4 特別計畫：次世代學術評估引擎實作計畫書 (v1.0)

> **核心目標**：建立一套基於 **1-10 分制** 與 **多維度指標 (Radar Metrics)** 的自動化評估系統。此系統將作為本專案的核心「裁判」，用於量化 Deep Research 的優越性，並驅動系統的自我修正。

---

## 1. 系統架構設計 (System Architecture)

### 1.1 評分維度定義 (The Rubric)

我們將評分拆解為三個獨立維度，並計算加權總分。

| 維度代號 | 名稱 | 權重 | 評分範圍 | 定義與評分標準 (Rubric) |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | **Accuracy (精確度)** | **50%** | 1-10 | • **10**: 數據精確至小數點，引用無誤，或正確指出文獻無資料。<br>• **7-9**: 數據大致正確，無明顯幻覺。<br>• **5-6**: 使用模糊詞彙(大約/很多)，缺乏精確度。<br>• **1-4**: 嚴重幻覺，數據錯誤，或文獻無資料卻瞎掰。 |
| **D2** | **Completeness (完整性)** | 30% | 1-10 | • **10**: 完整覆蓋問題的所有子面向。<br>• **5-9**: 漏掉次要觀點。<br>• **1-4**: 遺漏關鍵論點或斷章取義。 |
| **D3** | **Clarity (邏輯表達)** | 20% | 1-10 | • **10**: 結構清晰，推論連貫，無冗言。<br>• **1-4**: 邏輯混亂，語句不通，答非所問。 |

### 1.2 資料結構 (Data Schemas)

在 `agents/evaluator.py` 與 `data_base/schemas_deep_research.py` 中定義：

```python
from pydantic import BaseModel, Field

class EvaluationMetrics(BaseModel):
    accuracy: float = Field(..., ge=1, le=10, description="D1: 數據精確度")
    completeness: float = Field(..., ge=1, le=10, description="D2: 完整覆蓋率")
    clarity: float = Field(..., ge=1, le=10, description="D3: 表達與邏輯")
    weighted_score: float = Field(..., description="加權總分 (0.5*Acc + 0.3*Comp + 0.2*Clar)")

class FineGrainedEvaluationResult(BaseModel):
    metrics: EvaluationMetrics
    reason: str = Field(..., description="詳細評分理由")
    suggestion: str = Field(..., description="改進建議 (用於 Retry)")
    is_passing: bool = Field(..., description="是否通過門檻 (通常 Accuracy >= 7)")
```

### 1.3 程式碼框架 (Code Skeleton)

**`agents/evaluator.py`** 將新增以下方法：

```python
class RAGEvaluator:
    # ... 原有代碼 ...

    async def evaluate_fine_grained(
        self,
        question: str,
        answer: str,
        documents: List[Document],  # 若為 None，則進入 Pure LLM 評估模式
    ) -> FineGrainedEvaluationResult:
        """
        執行多維度細粒度評估。
        """
        # 1. 準備 Context (將 Documents 轉為 String)
        context_text = self._format_docs(documents) if documents else "無參考文獻 (Pure LLM Mode)"
        
        # 2. 選擇 Prompt (有無文獻 Prompt 略有不同)
        prompt = _FINE_GRAINED_EVAL_PROMPT.format(
            question=question,
            context=context_text,
            answer=answer
        )
        
        # 3. 呼叫 LLM (強制 JSON 輸出)
        response = await self.llm.ainvoke(prompt)
        
        # 4. 解析 JSON 並計算加權分
        result_dict = self._parse_json(response)
        
        # 計算加權分
        weighted = (
            result_dict['accuracy'] * 0.5 + 
            result_dict['completeness'] * 0.3 + 
            result_dict['clarity'] * 0.2
        )
        
        return FineGrainedEvaluationResult(...)
```

---

## 2. 關鍵 Prompt 設計 (Prompt Engineering)

這是系統靈魂所在。我們需要一個 **Chain-of-Thought (CoT)** Prompt 來確保評分公正。

**Prompt Template 草案：**

```text
你是一位嚴格的學術論文評審。請針對以下「使用者問題」與「參考文獻」，評估「AI 回答」的品質。

## 評估資料
問題：{question}
參考文獻：
{context}
AI 回答：{answer}

## 評分標準 (1-10分)
1. Accuracy (50%): 數據/引用是否精確？有無幻覺？(文獻沒寫但AI說不知道 => 給滿分；文獻沒寫但AI瞎掰 => 給1分)
2. Completeness (30%): 是否回答了所有子問題？
3. Clarity (20%): 邏輯是否清晰？

## 評估步驟 (Chain of Thought)
請先進行「錯誤分析」，列出回答中的具體事實錯誤或遺漏點。
然後再根據上述標準給出分數。

## 輸出格式 (JSON Only)
{{
  "analysis": "分析過程...",
  "accuracy": <1-10>,
  "completeness": <1-10>,
  "clarity": <1-10>,
  "reason": "簡短評語",
  "suggestion": "如何改進"
}}
```

---

## 3. 介面串接方案 (Integration)

### 3.1 與 Deep Research Service 串接
*   **用途**：驅動 Retry Loop。
*   **整合點**：`DeepResearchService._drill_down_loop`
*   **邏輯**：
    ```python
    eval_result = await evaluator.evaluate_fine_grained(q, ans, docs)
    if eval_result.metrics.accuracy < 8.0:  # 設定較高標準
        # 觸發 Retry，傳入 eval_result.suggestion
        await planner.refine_query(..., eval_result.suggestion)
    ```

### 3.2 與實驗腳本串接 (Auto-Arena)
*   **用途**：產出實驗圖表。
*   **整合點**：新腳本 `tests/run_arena.py` (獨立運行，不影響主服務)。

---

## 4. 潛在問題與風險 (Risks & Mitigations)

| 風險 | 描述 | 解決方案 (Mitigation) |
| :--- | :--- | :--- |
| **純 LLM 評估困難** | 沒有文獻時，Evaluator 無法判斷 Accuracy。 | **比較模式**：在評估 Pure LLM 時，**偷看** Deep Research 檢索到的文獻作為 Ground Truth。 |
| **分數通膨** | LLM 傾向給高分 (7-8分)。 | 在 Prompt 中加入 **Few-shot Examples**（給一個瞎掰的例子並標註為 2 分），校準模型標準。 |
| **文獻過長** | Context 超過 LLM Context Window。 | 評估時只擷取 Top-5 或 Top-10 最相關的 chunks，或使用長 Context 模型 (Gemini 1.5 Pro)。 |
| **JSON 解析失敗** | LLM 輸出格式錯誤。 | 使用 `try-except` 與 `Regex` 補救，若失敗則回傳預設低分並 Log 錯誤。 |

---

## 5. 驗證與測試計畫 (Verification Plan)

### 5.1 單元測試 (Unit Tests)
在 `tests/test_evaluator.py` 新增：
*   **Case 1: 完美回答** (Mock 數據吻合 -> 預期 Accuracy > 9)
*   **Case 2: 嚴重幻覺** (Mock 數據衝突 -> 預期 Accuracy < 3)
*   **Case 3: 誠實的不知道** (Mock 文獻無資料，回答不知道 -> 預期 Accuracy > 8)

### 5.2 整合測試 (Integration Test)
*   實際跑一次 Deep Research 流程，觀察 Log：
    *   確認 Evaluator 被呼叫。
    *   確認低分時有觸發 Retry。
    *   確認 Retry 時使用了 `suggestion` 中的建議。

### 5.3 實驗驗證 (Manual Golden Set)
*   準備 3 個真實 PDF 與 3 個高難度問題。
*   人工判斷分數 vs 系統評分。
*   目標：**人工與系統評分誤差在 ±1 分以內**。
