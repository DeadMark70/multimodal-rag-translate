# **多模態 RAG 系統深度架構優化與科研級分析能力提升研究報告**

## **1\. 執行摘要**

隨著大型語言模型（LLM）技術的飛速發展，檢索增強生成（Retrieval-Augmented Generation, RAG）已成為解決模型幻覺與知識截止問題的標準範式。然而，針對高強度的學術研究、法律分析與複雜的商業情報場景，現有的基礎 RAG 架構往往難以滿足對「深度推理」、「跨文檔關聯分析」以及「細粒度多模態理解」的需求。

本研究報告基於對現有程式碼庫的深度審計，特別是針對核心檢索邏輯 data\_base/RAG\_QA\_service.py 1、向量存儲管理 data\_base/vector\_store\_manager.py 1 以及多模態處理模組 multimodal\_rag 1 的分析，提出了一套系統性的架構升級藍圖。

現有系統建立在 FastAPI 與 Google Gemini 模型之上 1，採用了 PaddleOCR 進行文檔結構化 1，並實現了基礎的混合檢索（Hybrid Retrieval）機制 1。雖然該系統已具備處理圖文混排 PDF 的基本能力，但在面對需要多步推理（Multi-hop Reasoning）與全域視角（Global Understanding）的研究型任務時，仍存在顯著的架構瓶頸。本報告將從資料攝取層的語義結構化、檢索層的上下文感知與重排序、推理層的代理人工作流（Agentic Workflow），以及知識層的圖譜增強（GraphRAG）四個維度，詳盡闡述如何將當前系統轉型為具備科研級分析能力的智慧平台。

## ---

**2\. 現有系統架構審計與瓶頸分析**

在探討進階技術之前，必須先對現有的系統架構進行嚴謹的技術審計，以識別限制分析深度的關鍵因素。

### **2.1 數據攝取與分塊策略的語義斷裂**

數據是 RAG 系統的基石。現有系統在 data\_base/word\_chunk\_strategy.py 中實現了基於字符長度的遞歸切分策略（RecursiveCharacterTextSplitter）1。具體而言，系統設定了 chunk\_size=1000 與 overlap=150 的參數，並依據 \[\[PAGE\_N\]\] 標記進行物理頁面的強制分割 1。

這種「機械式切分」策略雖然保證了上下文視窗的安全性，但在語義連貫性上存在嚴重缺陷。學術論文或研究報告中的論證邏輯往往跨越多個段落甚至跨頁。當一個完整的論點被物理頁碼或固定的字符限制強制切斷時，該論點的語義向量（Embedding）將變得破碎且不完整。例如，若一個結論段落中的代詞「它」指代的是上一頁的某個實驗設置，切分後的 Chunk 將失去該指代關係，導致檢索時無法通過「實驗設置」的關鍵詞召回該結論。

此外，在 multimodal\_rag/structure\_analyzer.py 中，系統使用 PaddleOCR (PP-StructureV3) 進行版面分析 1。雖然代碼中嘗試區分 figure, table, formula 等元素，但在最終的 TextChunk 生成階段，這些結構化資訊往往被簡化為線性的文本流，丟失了文檔原有的層級結構（Hierarchy）與邏輯區塊（Logical Sections）資訊。

### **2.2 多模態資訊的「有損壓縮」**

現有的多模態處理邏輯集中在 multimodal\_rag/image\_summarizer.py 1。系統採用了一種「摘要索引法」（Caption-based Indexing），即利用 Gemini 模型將圖像轉化為文字描述，然後對該描述進行向量化 1。

Python

\# : image\_summarizer.py  
prompts \= {  
    VisualElementType.TABLE: "Analyze this table image. Extract the data...",  
    VisualElementType.FIGURE: "Analyze this figure/chart/diagram..."  
}

這種方法本質上是一種「有損壓縮」。複雜的科學圖表（如熱力圖、多變量散點圖）或高密度的財務報表包含成百上千個數據點。一個簡短的文字摘要（Summary）根本無法承載如此豐富的資訊量。當使用者詢問圖表中某個具體的異常值或趨勢轉折點時，由於摘要中未提及，檢索器將無法定位到該圖片。這直接限制了系統在數據密集型任務中的研究能力。

### **2.3 檢索路徑的線性與單薄**

在 data\_base/vector\_store\_manager.py 中，系統實現了由 FAISS（向量檢索）與 BM25（關鍵字檢索）組成的 EnsembleRetriever 1。

Python

\# : vector\_store\_manager.py  
ensemble\_retriever \= EnsembleRetriever(  
    retrievers=\[faiss\_retriever, bm25\_retriever\],  
    weights=\[0.5, 0.5\]  
)

雖然混合檢索優於單一檢索，但固定的權重分配（0.5/0.5）無法適應多變的查詢意圖。更關鍵的是，當前架構缺乏「重排序」（Reranking）機制。向量檢索（Bi-Encoder）基於餘弦相似度，往往會召回大量「語義相關但邏輯不符」的文檔。例如，查詢「A 模型比 B 模型好在哪裡？」，向量檢索可能會召回大量單純介紹 A 模型或 B 模型的片段，而將真正的「比較分析」片段排在後面。缺乏 Reranker 導致送入 LLM 的上下文噪聲過大，進而引發幻覺或回答膚淺。

### **2.4 推理模式的被動性**

檢視 data\_base/router.py 1 與 data\_base/RAG\_QA\_service.py 1，系統的運作邏輯是線性的：接收問題 \-\> 檢索 \-\> 生成 \-\> 返回。

這是一個典型的「被動問答」模式，而非「主動研究」模式。對於複雜的研究問題（例如：「綜合分析這十份報告中關於供應鏈韌性的演變趨勢」），單次檢索往往無法獲取足夠的資訊。系統缺乏自我規劃（Planning）、多步檢索（Multi-hop Retrieval）以及自我反思（Self-Reflection）的能力，無法像人類研究員那樣，在發現資訊不足時主動調整查詢方向或進行更深層的挖掘。

## ---

**3\. 數據攝取層的重構：語義感知與結構化**

為了提升分析深度，我們必須從源頭改善數據的品質。建議將現有的「分塊（Chunking）」策略升級為「解析（Parsing）」策略。

### **3.1 引入語義分塊 (Semantic Chunking)**

傳統的固定長度分塊破壞了語義。建議引入基於 Embedding 變化的語義分塊機制。

實作邏輯：  
不再依賴固定的 chunk\_size=1000 1，而是先將文檔按句子切分，然後計算相鄰句子的 Embedding 相似度。當相似度低於某個閾值（Breakpoint）時，意味著話題發生了轉移，此時進行切分。這確保了每個 Chunk 都包含一個完整的、語義自洽的論點。  
此外，應實作「命題索引（Proposition Indexing）」。將複雜的複合句分解為多個原子命題（Atomic Propositions），每個命題獨立索引。這能顯著提高檢索的精確度，特別是針對那些包含多重條件限制的法律或技術條款。

### **3.2 層級化索引結構 (Hierarchical Indexing)**

為了在檢索時兼顧「精確匹配」與「豐富上下文」，建議採用父子文檔索引策略（Parent-Child Indexing）。

* **子塊（Child Chunks）：** 較小的文本單元（如 200-400 字符），用於高精度的向量匹配。  
* **父塊（Parent Chunks）：** 包含子塊的完整段落或章節（如 2000+ 字符）。  
* **檢索機制：** 系統對「子塊」進行檢索，但在構建 Prompt 上下文時，返回該子塊所屬的「父塊」。

這需要在 data\_base/vector\_store\_manager.py 的 index\_extracted\_document 函數 1 中進行改造：

Python

\# 擬議的架構變更示意  
def index\_with\_hierarchy(doc):  
    \# 存儲父文檔  
    doc\_store.mset(parent\_chunks)  
    \# 索引子文檔，但在 metadata 中記錄 parent\_id  
    vector\_store.add\_documents(child\_chunks)

這樣，當使用者詢問細節時，LLM 能夠獲得該細節周圍完整的邏輯論證，從而生成更具深度的分析，而非斷章取義。

### **3.3 增強型多模態解析**

針對 multimodal\_rag 1 流程，應深化對非文本元素的處理：

* **表格處理（Table Understanding）：** 放棄單純的 Markdown 轉換。對於複雜表格，應保留其 HTML 結構，並利用 LLM 生成一段「分析性摘要」（例如描述表格中的最大值、最小值、趨勢變化），將「原始 HTML」與「分析摘要」同時存入索引。  
* **公式處理（Formula Reasoning）：** StructureAnalyzer 已能識別 VisualElementType.FORMULA 1。應將提取的 LaTeX 公式進一步轉化為語義描述（例如：「這是 Black-Scholes 期權定價模型」），以便使用者能通過概念而非數學符號進行檢索。

## ---

**4\. 檢索層的進化：上下文感知與重排序**

檢索層是連接數據與推理的橋樑。為了實現科研級的準確度，必須引入更先進的檢索範式。

### **4.1 引入 Cross-Encoder 重排序 (Reranking)**

現有的 EnsembleRetriever 1 僅完成了召回（Recall）步驟。為了過濾噪聲，必須在召回後加入重排序步驟。

**技術架構：**

1. **第一階段（召回）：** 使用現有的 BAAI/bge-m3 1 檢索 Top-50 文檔。此時追求的是廣度，寧可多抓不可漏抓。  
2. **第二階段（重排序）：** 引入 Cross-Encoder 模型（如 BGE-Reranker-v2-M3）。不同於 Bi-Encoder 分別編碼 Query 和 Doc，Cross-Encoder 將 Query 與 Doc 拼接後輸入 Transformer，進行全層的注意力交互。  
3. **優勢：** Cross-Encoder 能精確判斷 Query 與 Doc 之間的邏輯蘊含關係，有效剔除那些「關鍵詞匹配但語義無關」的文檔。

這需要在 data\_base/RAG\_QA\_service.py 的 rag\_answer\_question 函數 1 中，在 retriever.invoke(question) 之後插入重排序邏輯。

### **4.2 上下文感知檢索 (Contextual Retrieval)**

這是一種較新的技術，旨在解決獨立 Chunk 缺乏上下文的問題。

實作方法：  
在索引階段（Ingestion），利用 LLM 為每個 Chunk 生成一段簡短的「上下文解釋」，並將其添加到 Chunk 的開頭進行 Embedding。  
例如，原始 Chunk 為：「它在 2023 年增長了 15%。」  
生成的上下文感知 Chunk：「 它在 2023 年增長了 15%。」  
這樣，即使 Chunk 本身沒有提到主詞，向量檢索也能正確地將其與「Tesla 銷量」相關聯。這對於處理 data\_base/word\_chunk\_strategy.py 1 切分出的破碎段落尤為有效。

### **4.3 查詢擴展與 HyDE**

針對使用者簡短或模糊的查詢，應在 RAG\_QA\_service.py 1 中引入查詢轉換層：

1. **HyDE (Hypothetical Document Embeddings)：** 讓 LLM 先針對問題寫一個「假設性的答案」，然後用這個假設答案去檢索。這能更好地匹配文檔的陳述性語氣。  
2. **多視角查詢生成 (Multi-query Generation)：** 針對一個研究問題，讓 LLM 生成 3-5 個不同角度的子查詢，分別檢索後對結果進行去重融合（Reciprocal Rank Fusion, RRF）。

## ---

**5\. 多模態深度融合：視覺原生檢索**

為了突破 image\_summarizer.py 1 的限制，系統應朝向「視覺原生」方向演進。

### **5.1 ColPali 與多向量檢索 (Multi-Vector Retrieval)**

ColPali 是一種基於視覺語言模型（VLM）的檢索技術，它直接將文檔頁面的圖像視為由視覺 Token 組成的序列，並透過 ColBERT 風格的「後期交互（Late Interaction）」機制進行檢索。

架構升級建議：  
在 pdfserviceMD/PDF\_OCR\_services.py 1 中，除了進行 OCR 提取文本外，應並行生成頁面的 ColPali Visual Embeddings。這將允許系統直接通過文本查詢來「看」到頁面上的視覺特徵（如圖表的形狀、佈局的強調），而不僅僅是依賴 OCR 的文字結果。這對於檢索那些文字稀少但視覺資訊豐富的幻燈片或設計圖稿至關重要。

### **5.2 視覺思維鏈 (Visual Chain-of-Thought)**

在生成階段，目前的 Prompt 只是簡單地將圖片 Base64 插入 1。建議採用「視覺思維鏈」策略：

Python

\# 擬議的 Prompt 優化  
prompt\_text \= """  
Task: Step-by-step analysis of the provided context and images.  
1\. First, examine Image 1\. Describe its axes, data trends, and key anomalies.  
2\. Cross-reference this visual data with the provided text context.  
3\. Synthesize these findings to answer the user's question: {question}  
"""

這種強制模型「先觀察、再推理」的流程，已被證明能顯著減少多模態幻覺（Hallucination），提升回答的準確度。

## ---

**6\. 知識圖譜增強 (GraphRAG)：結構化推理**

向量檢索擅長尋找「相似性」，但拙於處理「關聯性」與「全域性」。GraphRAG 是提升研究深度的關鍵拼圖。

### **6.1 實體與關係的提取**

利用 structure\_analyzer.py 1 處理後的文本，引入一個後處理步驟：使用 LLM 提取實體（Entities）與關係（Relationships）。

* **實體：** 人物、公司、演算法、化學物質、法律條款。  
* **關係：** (Entity A) \--\[invents\]--\> (Entity B), (Company X) \--\[sues\]--\> (Company Y).

這些三元組應存儲於圖資料庫（如 Neo4j）中。由於 data\_base/vector\_store\_manager.py 1 目前僅管理 FAISS，這裡需要擴展以支持圖存儲。

### **6.2 基於圖譜的全局摘要 (Global Summarization)**

當使用者詢問「這份文檔的主要主題是什麼？」或「分析文檔中各個模組的依賴關係」時，單純的向量檢索會因為上下文窗口限制而失敗。

GraphRAG 透過 **Leiden 社群檢測算法**，將圖譜劃分為多個層級的語義社群。系統預先為每個社群生成摘要。當面對全局性問題時，系統直接利用這些社群摘要進行回答，從而實現對萬字長文的宏觀把握與深度綜合。

## ---

**7\. 代理人架構 (Agentic Workflow)：從回答到研究**

為了讓系統具備「研究員」的能力，必須將目前的線性 Router 1 轉型為代理人迴路。

### **7.1 自我修正與反思 (Self-Correction / Self-RAG)**

在 RAG\_QA\_service.py 1 中引入評估機制：

| 評估維度 | 描述 | 動作 |
| :---- | :---- | :---- |
| **檢索相關性** | 檢查檢索到的文檔是否包含回答問題所需的資訊。 | 若否，觸發查詢重寫 (Rewrite) 並重新檢索。 |
| **生成忠實度** | 檢查生成的答案是否嚴格基於檢索到的事實。 | 若否，標記並重新生成，強制引用來源。 |
| **回答有用性** | 檢查答案是否解決了使用者的核心疑問。 | 若否，嘗試分解問題或調用其他工具。 |

### **7.2 規劃與執行 (Plan-and-Solve)**

對於複雜的研究請求，引入一個 **Planner Agent**。它不直接回答問題，而是將問題分解為一系列子任務（Sub-tasks）。

範例流程：  
使用者：「比較 BERT 與 GPT-3 在架構上的差異及其對下游任務的影響。」

1. **Planner:** 分解為 Task 1: 檢索 BERT 架構, Task 2: 檢索 GPT-3 架構, Task 3: 檢索兩者在下游任務的表現, Task 4: 綜合對比。  
2. **Executor:** 依次或並行執行 Task 1-3，每次執行都是一次完整的 RAG 過程。  
3. **Synthesizer:** 匯總所有資訊，生成最終的深度比較報告。

這種架構利用了 core/llm\_factory.py 1 中不同用途的 LLM 配置（如使用高溫度的 LLM 進行規劃，低溫度的 LLM 進行資訊提取），實現了能力的專業化分工。

## ---

**8\. 結論**

本研究報告提出了一條清晰的技術演進路徑，旨在將現有的多模態 RAG 系統從一個基礎的資訊檢索工具，升級為一個具備深度認知能力的科研輔助平台。

透過引入 **語義分塊** 與 **Cross-Encoder 重排序**，我們解決了檢索精度的問題；透過 **ColPali** 與 **視覺思維鏈**，我們釋放了多模態數據的潛力；透過 **GraphRAG**，我們賦予了系統結構化推理的能力；最後，透過 **Agentic Workflow**，我們實現了從被動問答到主動研究的範式轉移。

這些技術的整合不僅能顯著提升系統回答複雜問題的能力，更能為使用者提供具備洞察力、邏輯嚴密且論證充分的研究報告，真正實現 AI 賦能的深度知識探索。實施這些改進將需要對現有的 data\_base 與 multimodal\_rag 模組進行大幅重構，但其帶來的分析能力躍升將是決定性的。

#### **引用的著作**

1. data\_base