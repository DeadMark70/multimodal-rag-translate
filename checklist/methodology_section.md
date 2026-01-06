# (四) 研究方法及步驟 (Methodology)

本研究旨在建構一套結合多模態理解與圖譜推論的智慧文檔處理系統。本節將詳細闡述系統的理論基礎、整體架構設計、核心演算法邏輯以及驗證策略。

## 1. 理論基礎 (Theoretical Basis) 

本系統的核心技術建立在向量空間模型、檢索融合演算法與圖論社群檢測之上。

### 向量嵌入與餘弦相似度 (Vector Embeddings & Cosine Similarity)
為了捕捉文本的深層語義，本研究利用高維向量空間模型將非結構化文本轉換為數值向量。語義相似度透過計算查詢向量 $A$ 與文檔向量 $B$ 之間的餘弦相似度來衡量，其公式如下：

$$ \text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} $$

此方法能有效解決傳統關鍵字匹配無法處理同義詞或語義多樣性的問題。本系統採用 `models/gemini-embedding-001` 作為嵌入模型。

### 倒數排名融合 (Reciprocal Rank Fusion, RRF)
為了整合來自不同視角（如 HyDE 假設性文檔、多重查詢）的檢索結果，本系統採用 RRF 演算法進行多路檢索融合。該演算法不依賴具體的分數值，而是基於排名進行加權，能穩健地提升檢索的魯棒性。其計算公式為：

$$ RRFscore(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)} $$

其中 $d$ 為文檔，$R$ 為檢索器集合，$k$ 為平滑常數（通常設為 60），$rank_r(d)$ 為文檔在第 $r$ 個檢索器中的排名。

### Leiden 社群檢測演算法 (Leiden Algorithm)
在 GraphRAG 模組中，為了理解文檔的全域結構並回答宏觀問題，本研究採用 **Leiden 演算法** 進行社群檢測。相較於傳統的 Louvain 演算法，Leiden 演算法透過迭代優化模組度 (Modularity) 來識別圖譜中緊密連結的節點聚類，並保證了社群的連通性，能有效避免弱連結社群的產生，從而生成更高質量的層次化知識摘要。

---

## 2. 系統架構 (System Architecture)

本系統採用微服務架構，以前後端分離與異步任務隊列為基礎，實現從文檔攝入到智慧問答的端到端處理。

> **[在此插入系統架構圖：展示從 PDF 輸入、雙路索引構建到 RAG 檢索生成的完整流程]**

### 資料攝入層 (Ingestion Layer)
系統首先透過 `/pdfmd/ocr` 端點接收 PDF 文件。針對複雜排版，採用 **Hybrid OCR** 策略：優先使用 **Marker** 模型在本地進行佈局感知 (Layout-Aware) 的文字與表格提取；若需更高精度，則切換至 **Datalab API**。提取後的內容轉換為標準 Markdown，並透過語義分塊 (Semantic Chunking) 技術切分為適合嵌入的片段。

### 索引層 (Indexing Layer)
本研究設計了雙重索引機制以支援混合檢索：
1.  **向量索引 (Vector Store)**：使用 `models/gemini-embedding-001` 模型將文本片段轉換為向量，並存儲於 **FAISS** 索引中，以支援高效的相似度搜尋。
2.  **知識圖譜 (Knowledge Graph)**：利用 LLM (`gemini-2.5-flash`) 自動抽取實體與關係，構建基於 **NetworkX** 的圖譜結構，並執行實體解析 (Entity Resolution) 與 Leiden 社群檢測，為全域檢索建立結構化基礎。

### 檢索與生成層 (Retrieval & Generation Layer)
當用戶發起查詢時，Router 模組根據問題性質選擇檢索策略。檢索結果經過 **Cross-Encoder** (`ms-marco-MiniLM-L-12-v2`) 進行重排序 (Reranking)，篩選出 Top-K 最相關的上下文，最後輸入至生成模型 (`gemma-3-27b-it` 或 Gemini) 產出答案。

---

## 3. 演算法邏輯與實作 (Algorithm Logic & Implementation)

本研究在標準 RAG 的基礎上，創新性地整合了圖譜推理與代理人機制。

### 雙路混合檢索策略 (Dual-Path Hybrid Retrieval)
系統結合了 **Dense Retrieval** 與 **Graph Traversal**。
*   針對具體事實型問題（如「某參數數值為何？」），利用向量檢索精確定位。
*   針對全域性問題（如「這些文件的主要論點為何？」），則啟動 GraphRAG 的 **Global Search** 模式，透過 Map-Reduce 機制彙總各社群 (Community) 的摘要資訊，生成具備宏觀視野的回答。

### 代理人驅動的深度研究 (Agentic Deep Research)
針對複雜任務，引入 "Plan-and-Solve" 代理架構，邏輯流程如下：
1.  **任務分解 (Decomposition)**：**Planner Agent** (`agents/planner.py`) 分析用戶查詢，將其拆解為一系列邏輯依賴的子任務 (Sub-tasks)。
2.  **迭代執行 (Execution)**：系統依序執行子任務，每個子任務均可觸發獨立的 RAG 檢索流程。
3.  **綜合生成 (Synthesis)**：**Synthesizer Agent** (`agents/synthesizer.py`) 彙整所有子任務的結果，解決潛在資訊衝突，生成最終報告。

> **[在此插入 Agentic Deep Research 流程圖：Plan-Execute-Synthesize 循環]**

### Self-RAG 評估與反幻覺 (Self-RAG Evaluation)
為降低模型幻覺，系統內建 **Evaluator Agent** (`agents/evaluator.py`)。該模組在生成回應前，會對檢索內容進行「相關性評分」(Relevance Score)，並對生成答案進行「忠實度檢查」(Faithfulness Check)。若評分低於閾值，系統將自動觸發重寫或重新檢索機制。

---

## 4. 驗證策略 (Validation Strategy)

為確保系統的有效性與可靠性，本研究將採用定量與定性相結合的驗證方法。

### 定量指標 (Quantitative Metrics)
利用內部的 Evaluator 模組與人工標註數據集，針對不同檢索模式進行評測：
*   **檢索精確率與召回率 (Precision & Recall)**：評估向量檢索與圖譜檢索覆蓋關鍵資訊的能力。
*   **忠實度評分 (Faithfulness Score)**：採用 1-5 分制，評量生成答案是否嚴格基於檢索到的上下文，以量化幻覺程度。

### 案例研究 (Case Studies)
選取具有代表性的學術論文集合進行測試。比較系統生成的「全域摘要」(Global Summary) 與論文原作者撰寫的「摘要」(Abstract) 及「結論」(Conclusion)，分析系統在跨文檔歸納與趨勢分析上的準確性與邏輯連貫性。

### 效能評估 (Performance)
在標準硬體環境下（配置 8GB+ VRAM, CUDA 11.8+），測量系統的延遲 (Latency) 與資源消耗。重點比較純向量檢索模式與啟用 GraphRAG 模式下的回應時間差異，以及在處理不同規模文檔集時的系統穩定性。
