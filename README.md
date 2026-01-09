# Multimodal Agentic RAG System 🧠📚

> **A Next-Generation Academic Research Assistant**  
> 基於代理人 (Agentic) 架構、具備自我修正與多模態理解能力的深度研究系統。

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![Status](https://img.shields.io/badge/Status-Research_Preview-orange)

## 🌟 專案願景 (Vision)

本系統旨在解決傳統 RAG (Retrieval-Augmented Generation) 在學術研究場景中的三大痛點：

1.  **碎片化 (Fragmentation)**：缺乏全域視角，難以處理跨文檔邏輯。
2.  **幻覺與和稀泥 (Averaging Hallucination)**：面對觀點衝突的文獻時傾向於取平均值，忽略反駁證據。
3.  **視覺盲區 (Visual Blindness)**：無法精確理解論文中的圖表數據。

透過引入 **Agentic Workflow** (Planner, Executor, Evaluator, Synthesizer) 與 **GraphRAG**，本系統能像人類研究員一樣進行「規劃 -> 執行 -> 評估 -> 修正」的深度研究循環。

---

## 🔥 核心功能 (Core Features)

### 1. 🔬 Deep Research (深度研究代理人)

- **Plan-and-Solve 架構**: 自動將複雜問題拆解為子任務。
- **Adaptive Loop (動態修正)**: 執行後自動調用 Evaluator 評分，若品質不佳 (Accuracy < 6.0) 則自動修正搜尋策略重試。
- **Conflict Arbitration (衝突仲裁)**: 當檢索到矛盾觀點（如 A 論文反駁 B 論文），系統能識別證據權重（Benchmark > Single Paper），避免和稀泥。

### 2. ⚖️ Academic Evaluation Engine (學術評估引擎)

- **1-10 分制多維度評分**:
  - **Accuracy (50%)**: 數據精確度與引用正確性。
  - **Completeness (30%)**: 觀點覆蓋率。
  - **Clarity (20%)**: 邏輯表達。
- **Pure LLM 對照模式**: 支援與無 RAG 的原生 LLM 進行 A/B Testing (`run_arena.py`)。

### 3. 🕸️ GraphRAG (知識圖譜增強)

- **全域視角**: 利用 NetworkX 構建實體關係圖，捕捉向量檢索遺漏的隱藏關聯。
- **Hybrid Search**: 預設結合 Vector Search + Graph Traversal，提升海量文檔下的抗噪能力。

### 4. 👁️ Multimodal Understanding (多模態)

- **Gemini Vision 整合**: 自動摘要 PDF 中的圖表。
- **Deep Image Verification**: (Opt-in) 針對特定圖表數據進行二次深度查證 (`re_examine_image`)。
- **圖文並茂報告**: 輸出的 Markdown 報告自動嵌入相關圖表引用。

### 5. 🌍 Advanced Translation (學術翻譯)

- **Layout-Aware**: 保持 PDF 原始排版 (Markdown-PDF 重建)。
- **Contextual Translation**: 專為學術術語優化的翻譯品質。

---

## 🛠️ 系統架構 (Architecture)

```mermaid
graph TD
    User[使用者] --> API[FastAPI Gateway]
    API --> Service[Deep Research Service]

    subgraph "Agentic Loop"
        Service --> Planner[Planner Agent]
        Planner --> Executor[Task Executor]
        Executor --> RAG["RAG / GraphRAG"]
        Executor --> Evaluator[Evaluator Agent]
        Evaluator --"低分重試"--> Planner
    end

    subgraph "Knowledge Base"
        PDF[PDF Files] --> OCR[Marker OCR]
        OCR --> VectorDB[FAISS (Vector)]
        OCR --> GraphDB[NetworkX (Graph)]
        OCR --> Vision[Gemini Vision]
    end

    RAG <--> VectorDB
    RAG <--> GraphDB

    Service --> Synthesizer[Synthesizer Agent]
    Synthesizer --> Report[Markdown Report]
```

---

## 🚀 快速開始 (Quick Start)

### 前置要求

- Python 3.10+
- CUDA (建議，用於 OCR 加速)
- Google Gemini API Key
- Supabase Project (用於 Auth 與 Logging)

### 安裝

```bash
git clone https://github.com/your-repo/multimodal-rag.git
cd multimodal-rag
pip install -r requirements.txt
```

### 設定環境變數

複製 `.env.example` 為 `.env` 並填入：

```env
GOOGLE_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
MARKER_USE_GPU=true
```

### 啟動服務

```bash
uvicorn main:app --reload
```

API 文件: `http://localhost:8000/docs`

---

## 🧪 實驗與評測 (Evaluation & Arena)

本專案包含一個自動化競技場腳本，用於比較 **Deep Research** 與 **Pure LLM** 的表現。

### 執行 Arena 測試

```bash
# 比較 3 個預設的黃金問題
python tests/run_arena.py --questions 3 --output results.json --user-id "your-uuid"

# 使用自定義問題集
python tests/run_arena.py --input tests/golden_set.json
```

---

## 📂 專案結構

```
├── main.py                 # FastAPI 入口
├── agents/                 # AI 代理人核心
│   ├── planner.py          # 任務規劃與修正
│   ├── evaluator.py        # 1-10分制評估引擎
│   └── synthesizer.py      # 報告合成與衝突仲裁
├── data_base/              # RAG 檢索邏輯
│   ├── deep_research_service.py # 深度研究主流程
│   ├── RAG_QA_service.py   # 基礎問答服務
│   └── vector_store_manager.py # FAISS 管理
├── graph_rag/              # 知識圖譜模組
├── pdfserviceMD/           # PDF OCR 與翻譯
├── multimodal_rag/         # 圖片理解模組
└── tests/                  # 單元測試與 Arena 腳本
```

---

## 📅 開發路線圖 (Roadmap)

- [x] **Phase 1-3**: 基礎 RAG 與 Agent 架構
- [x] **Phase 4**: 學術評估引擎 (1-10 分制)
- [x] **Phase 5**: 多文檔衝突仲裁系統
- [x] **Phase 6**: Deep Research 最終優化 (強制圖譜、信心校準)
- [x] **Phase 7**: PDF 生成引擎升級 (Pandoc + XeLaTeX + Datalab API)
- [x] **Phase 8**: 圖片管線整合 (OCR 圖片 → LLM 摘要 → RAG 索引)
- [x] **Phase 9**: Agentic 視覺查證 (Agent 主動看圖 + Re-Act 循環)
- [ ] **Phase 10**: ColPali 視覺向量嵌入 (Next Step)

---

## 📄 License

MIT License
