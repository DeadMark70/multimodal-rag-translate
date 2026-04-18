# Agentic RAG 2026 架構升級計畫書 (Architecture Optimization Plan)

本計畫書旨在將現有的「Plan-and-Execute」深度研究系統，升級至 2026 年業界主流的 Agentic RAG 模式（整合 CRAG, Adaptive RAG, 與進階 Self-RAG）。目標是降低幻覺、減少無效 Token 消耗、增強長文本推理能力，並提升處理文獻衝突的準確度。

本文件作為實作 Agent 的指導方針，包含具體的程式碼修改細節與邏輯注入點。

此計畫都已全部完成

---

## 優化 A：實施「檢索守衛」(Corrective Retrieval Guard / CRAG Pattern)

### 1. 概念與目標
目前的系統在 `rag_answer_question` 中檢索文獻後，直接交由 LLM 生成答案，隨後才在 `ResearchExecutionCore` 的 Drill-down loop 中進行評估 (`evaluator.evaluate_detailed`)。這會導致「Garbage In, Garbage Out」並浪費大量 Token。
**CRAG 模式**要求在「生成前」先對檢索到的 Document 進行輕量級評估 (Grader)。若相關性過低，應觸發 Query Rewrite（查詢重寫）並重新檢索，或者直接回傳「需擴充檢索」，避免強制模型基於無關上下文作答。

### 2. 修改範圍
- `pdftopng/agents/evaluator.py`
- `pdftopng/data_base/RAG_QA_service.py`

### 3. 實作細節

**步驟 A1：增強檢索評估器**
在 `evaluator.py` 中，優化 `evaluate_retrieval` 方法，使其更快速且支援 Batch 處理。
```python
# pdftopng/agents/evaluator.py 建議修改
async def grade_documents(self, question: str, documents: List[Document]) -> bool:
    """輕量級文件相關性評分 (CRAG Grader)"""
    if not documents: return False
    
    # 使用較小、較快的模型進行二元分類 (Relevant / Irrelevant)
    llm = get_llm("evaluator") # 建議配置為 gemini-2.5-flash 或等效快速模型
    
    doc_text = "\n\n".join([f"Doc {i}: {doc.page_content[:300]}" for i, doc in enumerate(documents[:5])])
    prompt = f"判斷以下文檔是否包含回答問題 '{question}' 的必要資訊。只需回答 'yes' 或 'no'。\n\n文檔：{doc_text}"
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return "yes" in response.content.lower()
```

**步驟 A2：在 RAG Service 注入 CRAG 邏輯**
修改 `rag_answer_question`，在檢索 (Step 4/5) 與生成 (Step 8/9) 之間插入守衛邏輯。
```python
# pdftopng/data_base/RAG_QA_service.py 建議修改 (Step 5.5 之後)
    # [新增] Step 5.7: Corrective Retrieval Guard (CRAG)
    if enable_reranking and docs:
        from agents.evaluator import RAGEvaluator
        from data_base.query_transformer import transform_query_with_hyde
        
        evaluator = RAGEvaluator()
        is_relevant = await evaluator.grade_documents(question, docs)
        
        if not is_relevant:
            logger.info(f"CRAG Guard: Retrieved docs irrelevant for '{question}'. Triggering rewrite.")
            await _emit_progress(progress_callback, "crag_correction", {"status": "rewriting_query"})
            
            # Query Rewrite (可複用 HYDE 或專用改寫邏輯)
            refined_query = await transform_query_with_hyde(question, enabled=True)
            
            # Re-retrieval
            new_batches = await invoke_retriever_queries_async(retriever, [refined_query])
            if new_batches and new_batches[0]:
                # 替換為新檢索的結果
                docs = new_batches[0][:target_k]
                logger.info("CRAG Guard: Re-retrieval complete.")
```

---

## 優化 B：結構化事實持久化 (Fact-State Persistence)

### 1. 概念與目標
目前的 Drill-down loop (`_drill_down_loop`) 只是將每次的 `SubTaskExecutionResult.answer` 拼接成字串（`_build_findings_summary`）餵給下一輪。當深度研究達到 3-4 輪時，Context 會極度膨脹，發生 "Lost in the middle"。
**解法**：不傳遞完整的長篇回答，而是由 Planner / Synthesizer 在每一輪結束後，提取出「原子事實 (Atomic Facts)」，並以結構化列表維護狀態。

### 2. 修改範圍
- `pdftopng/data_base/schemas_deep_research.py`
- `pdftopng/data_base/research_execution_core.py`

### 3. 實作細節

**步驟 B1：定義 Atomic Fact Schema**
```python
# pdftopng/data_base/schemas_deep_research.py 建議新增
from pydantic import BaseModel, Field
from typing import List

class AtomicFact(BaseModel):
    claim: str = Field(description="單一、具體的事實陳述")
    source_doc_ids: List[str] = Field(description="支持此事實的文件 ID")
```

**步驟 B2：在執行核心中維護 Fact State**
修改 `ResearchExecutionCore`，引入事實提取步驟。
```python
# pdftopng/data_base/research_execution_core.py 建議修改
    async def _extract_atomic_facts(self, result: SubTaskExecutionResult) -> List[AtomicFact]:
        """從子任務回答中提取原子事實"""
        # 調用 LLM 將長篇大論轉化為 1-3 條精煉事實
        # 實作：可新增一個 LLM prompt，要求輸出符合 AtomicFact schema 的 JSON list
        pass

    async def _drill_down_loop(self, ...):
        global_facts: List[AtomicFact] = [] # 維護全局事實庫
        
        for iteration in range(1, max_iterations + 1):
            # 提取上一輪的事實 (為節省時間，可異步並行處理 current_results)
            for result in current_results[-len(followup_tasks) if iteration > 1 else None:]:
                facts = await self._extract_atomic_facts(result)
                global_facts.extend(facts)
            
            # 使用結構化事實來生成追問，而非使用長篇 summary
            formatted_facts = "\n".join([f"- {f.claim} (來源: {', '.join(f.source_doc_ids)})" for f in global_facts])
            
            followup_tasks = await planner.create_followup_tasks(
                original_question=original_question,
                current_findings=formatted_facts, # 替換原本的 _build_findings_summary
                ...
            )
```

---

## 優化 C：多模態深度整合 (Unified Multimodal Reasoning)

### 1. 概念與目標
目前的視覺查證 (Visual Verification) 是在單一節點 (`rag_answer_question` 內的 Re-Act 迴圈) 處理。如果圖片揭示了新的實體（例如圖表中的某個特定縮寫 "XYZ"），系統不會自動在下一輪針對 "XYZ" 進行文本檢索。
**解法**：將視覺查證的結果更明確地反饋給 Planner，讓視覺發現能夠驅動下一輪的文本或圖譜檢索。

### 2. 修改範圍
- `pdftopng/agents/planner.py`

### 3. 實作細節

**步驟 C1：強化 Planner 對視覺發現的感知**
```python
# pdftopng/agents/planner.py 建議修改
# 在 _FOLLOWUP_PROMPT 中新增針對視覺發現的指示
_FOLLOWUP_PROMPT = """你正在協助研究以下問題：
{original_question}

目前已找到的資訊（若包含圖片查證結果，請特別留意其中的新名詞或數據）：
{current_findings}

...

## 多模態追問指示 (Multimodal Follow-up)
- 如果目前的發現中透過「視覺查證（圖片）」發現了新的專有名詞、未解釋的縮寫或異常數據，**必須**生成一個 [RAG] 子任務來專門檢索該名詞的文字定義或背景脈絡。
..."""
```

---

## 優化 D：強化衝突仲裁引擎 (Conflict Arbitration Engine)

> 實作狀態（2026-04-18）：✅ 已完成，且已在 **Deep Research** 與 **Agentic benchmark** 兩條 synthesis 路徑啟用。  
> 驗證：`pytest tests/test_synthesizer.py tests/test_research_execution_core_generic.py tests/test_agentic_evaluation_service.py` → `36 passed`

### 1. 概念與目標
雖然目前有 Phase 5 衝突感知檢查，但仍依賴 Synthesizer Prompt 自發處理。對於嚴謹的 Agentic RAG，需要一個獨立的**仲裁節點 (Arbitrator)**，當多份文獻結論相左時，強制執行證據權重評估（例如：Benchmark > 單篇聲明）。

### 2. 修改範圍
- `pdftopng/agents/synthesizer.py`

### 3. 實作細節

**步驟 D1：實作衝突檢測與仲裁**
在 `synthesizer.py` 中，在生成最終報告前掃描子任務結果。
```python
# pdftopng/agents/synthesizer.py 建議修改
async def detect_and_arbitrate_conflicts(sub_results: List['SubTaskResult']) -> str:
    """
    分析所有子任務結果，檢測邏輯衝突。
    如果發現衝突，輸出「仲裁聲明 (Arbitration Statement)」。
    """
    from core.providers import get_llm
    from langchain_core.messages import HumanMessage
    
    llm = get_llm("synthesizer")
    results_text = "\n".join([f"Task {r.task_id}: {r.answer}" for r in sub_results])
    
    prompt = f"""請分析以下多個檢索結果是否存在矛盾或衝突的觀點：
{results_text}
    
如果存在衝突，請遵循以下仲裁原則：
1. 基準測試 (Benchmark) 數據優於單一論文的宣稱。
2. 有具體數據支持的論點優於模糊的定性描述。

請輸出仲裁結論，說明應採信哪一方及原因（若無衝突請回覆 "NO_CONFLICT"）。"""
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return "NO_CONFLICT"

# 修改 synthesize_results
async def synthesize_results(...):
    # 1. 衝突檢測
    arbitration = await detect_and_arbitrate_conflicts(sub_results)
    
    # 2. 注入仲裁結果到 Synthesizer Prompt
    synth_prompt = _SYNTHESIZER_PROMPT # 假設這是原本的 prompt
    if arbitration and "NO_CONFLICT" not in arbitration:
        synth_prompt += f"\n\n## 衝突仲裁指示\n分析發現文獻間存在衝突。請務必遵循以下仲裁結論來撰寫最終報告：\n{arbitration}"
    
    # ... 繼續原本的生成流程 ...
```

---

## 執行順序與測試策略 (Execution & Testing Strategy)

為確保系統穩定性與向後相容性，建議依序實作：

1. **Phase 1 (優化 A - CRAG)**：
   - 實作難度最低，防幻覺效益最高。
   - **測試**：編寫 `tests/test_crag_guard.py`，模擬給定不相關文檔時，系統能成功觸發 Query Rewrite 而不中斷。
2. **Phase 2 (優化 D - 衝突仲裁)**：
   - 提升學術回答品質的關鍵。
   - **測試**：在 `test_deep_research.py` 中注入結論矛盾的 mock documents，驗證最終報告是否正確採納 Benchmark 數據並包含仲裁聲明。
3. **Phase 3 (優化 B - Fact-State Persistence)**：
   - 涉及執行核心架構修改與 Schema 變更。
   - **測試**：驗證 `ResearchExecutionCore` 經過 3 次 Iteration 後的 Context Token 消耗是否低於舊版，且 JSON 解析穩定。
4. **Phase 4 (優化 C - 多模態聯動)**：
   - 依賴 `SubTaskExecutionResult` 的 Meta 傳遞。
   - **測試**：提供一張包含未知縮寫的圖表，驗證 Planner 是否能基於視覺解析結果，自動生成針對該縮寫的文本檢索追問。

**安全護欄 (Guardrails)**：
- 所有新節點（CRAG Grader、Arbitrator、Fact Extractor）必須使用 `try-except` 包覆。若 LLM 解析失敗或 Timeout，應 Fallback 至原始流程，確保主要問答不斷線。
- 所有 Prompt 變更請嚴格遵守：主要思考與摘要可用繁體中文，但**檢索用的子問題 (Sub-questions) 必須強制輸出英文**，以維持學術文獻的檢索命中率。

此計畫都已全部完成
