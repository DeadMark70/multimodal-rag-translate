# 🛡️ Phase 5 計畫書：多文檔衝突仲裁與信心校準系統

> **核心目標**：解決 Deep Research 在面對多篇觀點衝突的文獻時，容易產生「結論和稀泥 (Averaging Hallucination)」以及「錯誤報告卻高信心度 (Overconfidence)」的問題。

---

## 1. 問題定義 (Problem Definition)

### 痛點 1: 衝突平均化 (Conflict Averaging)
當 RAG 檢索到 A 論文說「有效」、B 論文說「無效」時，目前的 LLM 傾向於回答「綜合來看效果不錯」，導致具有「反駁性質」的關鍵證據（通常是最新的 Benchmark 論文）被淹沒。

### 痛點 2: 信心度失準 (Miscalibrated Confidence)
目前的 Evaluator 僅檢查「回答是否有依據 (Groundedness)」。如果回答引用了 A 論文（但忽略了 B 論文的反駁），Evaluator 仍會認為它「有憑有據」而給出滿分，這在學術研究中是危險的。

---

## 2. 解決方案架構 (Architecture)

我們將引入三個層級的防護機制：

### 層級 1：Metadata 增強 (Evidence Grading)
*   **目標**：讓 LLM 具備「時間感」與「證據權重感」。
*   **策略**：在 Prompt 中顯式標註來源的屬性（如年份、類型）。

### 層級 2：Synthesizer 的「對抗性合成」 (Adversarial Synthesis)
*   **目標**：教導 Synthesizer 主動尋找並呈現衝突。
*   **策略**：
    *   在合成前增加 **Conflict Detection** 步驟。
    *   強制報告格式：當發現衝突時，必須使用「對照式陳述」（On one hand... On the other hand...），並根據證據強度（Benchmark > Single Paper）下結論。

### 層級 3：Evaluator 的「一致性懲罰」 (Consistency Penalty)
*   **目標**：當報告選擇性忽略反面證據時，大幅扣分。
*   **策略**：在評分標準中加入 **Conflict Awareness** 維度。若 Context 含有 "However", "Contrary to" 等轉折語意卻未被反映在答案中，Accuracy 扣分。

---

## 3. 實作步驟 (Action Plan)

### ✅ Step 1: Synthesizer 升級 (核心)
*   **Target**: `agents/synthesizer.py`
*   **Action**:
    1.  更新 `_SYNTHESIS_PROMPT`，加入 **「衝突處理守則」**。
    2.  要求 LLM 在總結前，先列出「檢索到的主要觀點」及其「支持/反對」關係。

### ✅ Step 2: RAG Prompt 優化
*   **Target**: `data_base/RAG_QA_service.py`
*   **Action**:
    1.  在 `_format_docs_for_prompt` 中，注入文件 Metadata (Year, Title)。
    2.  Prompt 指令增加：「若不同來源有衝突，優先採信『基準測試 (Benchmark)』或『較新發表』的結論。」

### ✅ Step 3: Evaluator 信心校準
*   **Target**: `agents/evaluator.py`
*   **Action**:
    1.  更新 `_DETAILED_EVAL_PROMPT`，新增檢查點：「回答是否完整反映了文獻中的爭議點？」。
    2.  調整評分權重：未處理衝突視為 `Completeness` 的重大缺失。

---

## 4. 驗證與測試 (Validation)

### 黃金測試案例 (The "nnU-Net vs SwinUNETR" Case)
*   **Input**: 上傳 `SwinUNETR.pdf` (Claiming Superiority) 與 `2404.09556v2.pdf` (Rebuttal/Benchmark)。
*   **Expected Output**:
    *   **報告**: 明確指出「雖然 SwinUNETR 原文宣稱優於 nnU-Net，但最新的大規模基準測試 (2404...) 顯示 nnU-Net 在多數資料集上仍勝出。」
    *   **分數**: Accuracy > 8.0。
    *   **失敗指標**: 若報告只說「兩者互有優劣」或「SwinUNETR 較強」，則視為失敗。

---

## 5. 預期成果 (Deliverables)
1.  更具批判性思維的學術報告生成器。
2.  能正確處理「打臉文」與「翻案文」的 RAG 系統。
3.  具備自我懷疑能力的信心評分系統（不會盲目給 100%）。
