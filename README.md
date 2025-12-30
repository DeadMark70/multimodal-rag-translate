# 🔬 Multimodal RAG System

> 進階多模態 RAG 系統，支援 PDF/圖片處理、語義檢索、智能問答

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ 功能特色

### 📄 文件處理

- **PDF OCR**：Local Marker / Datalab API 雙模式
- **多語言翻譯**：Google Gemini AI 驅動
- **Markdown 輸出**：保留文件結構與格式

### 🔍 進階 RAG 檢索

- **語義分塊**：基於語意邊界的智能切分
- **上下文增強**：LLM 生成上下文前綴
- **Cross-Encoder 重排序**：ms-marco-MiniLM-L-12-v2
- **HyDE 查詢轉換**：假設文檔嵌入
- **多查詢融合**：Reciprocal Rank Fusion

### 🔗 GraphRAG 知識圖譜

- **實體關係抽取**：LLM 驅動的學術實體識別
- **社群檢測**：Leiden 演算法分群
- **Local Search**：實體擴展搜尋
- **Global Search**：社群 Map-Reduce 搜尋
- **圖譜管理 API**：重建、優化、狀態查詢

### 🤖 Agent 架構

- **Self-RAG 評估**：檢索相關性 + 答案忠實度
- **Plan-and-Solve**：複雜問題分解與綜合
- **Graph-aware Planning**：圖譜輔助任務規劃
- **深度研究端點**：`/rag/research`
- **🆕 Interactive Deep Research**：`/rag/plan` + `/rag/execute` 互動式研究
- **🆕 SSE 串流**：`/rag/execute/stream` 即時進度回報

### 💬 對話管理

- **🆕 對話歷史**：CRUD `/conversations` 端點
- **🆕 多輪對話**：`/rag/ask?conversation_id=xxx`
- **🆕 資料庫 Migration**：`migrations/` 目錄

### 🛡️ Multi-Doc Anti-Hallucination

- **🆕 文件分組 Context**：依來源檔案分組 chunks
- **🆕 來源標籤**：從資料庫查詢實際檔名
- **🆕 反幻覺指引**：強化 Prompt 警告

### 🖼️ 多模態支援

- **圖片內文字翻譯**：就地翻譯
- **視覺元素摘要**：圖表/表格智能描述
- **FAISS 向量索引**：GPU 加速檢索

---

## 🏗️ 技術架構

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
├────────────┬────────────┬────────────┬────────────┬─────────┤
│   /pdfmd   │    /rag    │   /graph   │ /imagemd   │/multimod│
│  PDF 翻譯  │  RAG 問答  │ 🆕知識圖譜 │  圖片翻譯  │ 多模態  │
├────────────┴────────────┴────────────┴────────────┴─────────┤
│                     Core Services                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Marker OCR  │  │ FAISS Index │  │   Google Gemini     │   │
│  │ (Local/API) │  │ (gemini)    │  │   (LLM)             │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  🆕 GraphRAG: NetworkX + Leiden Community Detection     │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Supabase (Auth + PostgreSQL)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速開始

### 前置需求

- Python 3.10+
- CUDA 11.8+ (GPU 加速)
- 8GB+ VRAM (推薦)

### 安裝

```bash
# 1. Clone 專案
git clone https://github.com/YOUR_USERNAME/multimodal-rag.git
cd multimodal-rag

# 2. 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安裝依賴
pip install -r requirements.txt
```

### 設定環境變數

建立 `config.env` 檔案：

```env
# Google Gemini API (必要)
GOOGLE_API_KEY=your-gemini-api-key

# Supabase (生產環境必要)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# HuggingFace (選用)
HF_TOKEN=your-hf-token

# OCR 設定
USE_LOCAL_MARKER=true          # true=本地 Marker, false=Datalab API
MARKER_USE_GPU=false           # GPU 加速 (需 CUDA)
DATALAB_API_KEY=your-api-key   # 僅 USE_LOCAL_MARKER=false 時需要

# 開發模式 (測試用)
DEV_MODE=false
```

### 啟動服務

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API 端點

### RAG 問答

```bash
# 基本問答
GET /rag/ask?question=什麼是機器學習

# 指定文件查詢
GET /rag/ask?question=摘要文件&doc_ids=uuid1,uuid2

# 🆕 關聯對話
GET /rag/ask?question=繼續上個問題&conversation_id=uuid

# 深度研究
POST /rag/research
{
  "question": "比較 Python 和 JavaScript 的優缺點",
  "max_subtasks": 3,
  "enable_reranking": true
}

# 🆕 互動式深度研究 - 規劃階段
POST /rag/plan
{"question": "比較 nnU-Net 和 SAM-Med3D", "max_subtasks": 5}

# 🆕 互動式深度研究 - 執行階段 (SSE 串流)
POST /rag/execute/stream
{"sub_tasks": [...], "enable_drilldown": true}
```

### GraphRAG 圖譜管理

```bash
# 取得圖譜狀態
GET /graph/status

# 取得視覺化資料 (react-force-graph 格式)
GET /graph/data

# 重建圖譜
POST /graph/rebuild
{"force": false}

# 優化圖譜 (實體融合 + 社群重建)
POST /graph/optimize
{"regenerate_communities": true}
```

### GraphRAG 增強問答

```bash
POST /rag/ask
{
  "question": "這些論文的主要趨勢是什麼？",
  "enable_graph_rag": true,
  "graph_search_mode": "auto"
}
```

### PDF 處理

```bash
# 上傳 PDF → OCR → 翻譯 → 返回翻譯 PDF
POST /pdfmd/ocr
Content-Type: multipart/form-data
file: [PDF 檔案]

# 取得翻譯後的 PDF
GET /pdfmd/file/{doc_id}
```

### 多模態處理

```bash
# 處理 PDF 中的文字與圖片
POST /multimodal/extract
Content-Type: multipart/form-data
file: [PDF 檔案]
```

---

## 📁 專案結構

.
├── main.py # FastAPI 入口
├── config.env # 環境變數 (不提交)
├── requirements.txt # Python 依賴
│
├── core/ # 核心模組
│ ├── auth.py # Supabase JWT 認證
│ ├── llm*factory.py # LLM 實例工廠
│ └── summary_service.py # 文件摘要生成
│
├── data_base/ # RAG 核心
│ ├── router.py # /rag 端點
│ ├── schemas.py # Pydantic 請求/回應模型
│ ├── RAG_QA_service.py # RAG 主服務
│ ├── vector_store_manager.py # FAISS 管理
│ ├── semantic_chunker.py # 語義分塊
│ ├── reranker.py # Cross-Encoder
│ └── query_transformer.py # HyDE/Multi-Query
│
├── graph_rag/ # 🆕 GraphRAG 模組
│ ├── schemas.py # Node, Edge, Community 定義
│ ├── store.py # NetworkX 圖譜存儲
│ ├── extractor.py # LLM 實體/關係抽取
│ ├── entity_resolver.py # 實體融合
│ ├── community_builder.py # Leiden 社群檢測
│ ├── local_search.py # 實體擴展搜尋
│ ├── global_search.py # 社群 Map-Reduce
│ └── router.py # /graph 端點
│
├── conversations/ # 🆕 對話歷史管理
│ ├── router.py # CRUD 端點 (/conversations/\*)
│ └── schemas.py # Pydantic 模型
│
├── migrations/ # 🆕 SQL Migrations
│ ├── 002_create_conversations.sql
│ └── 003_add_conversation_id_to_chat_logs.sql
│
├── agents/ # Agent 模組
│ ├── evaluator.py # Self-RAG 評估
│ ├── planner.py # 任務分解 (支援 GraphRAG)
│ └── synthesizer.py # 結果綜合
│
├── pdfserviceMD/ # PDF 處理
│ ├── router.py # /pdfmd 端點
│ ├── PDF_OCR_services.py # OCR 路由 (Local/API)
│ ├── local_marker_service.py # Local Marker OCR
│ └── translation_chunker.py # 頁面翻譯分塊
│
├── multimodal_rag/ # 多模態處理
│ ├── router.py # /multimodal 端點
│ └── image_summarizer.py # 圖片摘要
│
├── image_service/ # 圖片翻譯
│ ├── router.py # /imagemd 端點
│ └── ocr_service.py # DocTR OCR
│
├── agentlog/ # 專案文件
│ ├── codebase_overview.md # 架構總覽
│ ├── api_documentation.json # API 文件
│ └── frontend*\*.md # 前端整合指南
│
├── checklist/ # 程式碼審核文件
│
└── tests/ # 單元測試

````

---

## 🧪 測試

```bash
# 執行所有測試
pytest tests/ -v

# 執行特定模組測試
pytest tests/test_evaluator.py -v

# 測試覆蓋率
pytest tests/ --cov=. --cov-report=html
````

**目前測試狀態**：104 tests passing ✅

---

## 🔒 安全性

- ✅ Supabase JWT 認證
- ✅ 輸入驗證 (檔案類型/大小)
- ✅ 路徑遍歷防護
- ✅ 環境變數管理密鑰
- ✅ Per-user 資料隔離

---

## 📊 效能需求

| 服務              | VRAM 需求       |
| ----------------- | --------------- |
| Gemini Embedding  | API (無需 VRAM) |
| ms-marco Reranker | ~0.5 GB         |
| Marker OCR (GPU)  | ~3 GB           |
| DocTR (Image OCR) | ~1 GB           |
| **總計**          | **~4.5 GB**     |

---

## 🛣️ 開發路線

- [x] Phase 1: 語義分塊 + 上下文增強
- [x] Phase 2: Cross-Encoder + HyDE
- [x] Phase 3: Self-RAG + Plan-and-Solve
- [x] Phase 4: 上下文感知圖片摘要 (繁體中文 + 快取)
- [x] Phase 5: GraphRAG 知識圖譜 ✅
  - [x] 實體/關係抽取
  - [x] Leiden 社群檢測
  - [x] Local/Global Search
  - [x] 圖譜管理 API
  - [x] Planner 整合
- [x] Phase 5.4: 🆕 Interactive Deep Research ✅
- [x] Phase 5.5: 🆕 Conversation History ✅
- [x] Phase 5.6: 🆕 Multi-Doc Anti-Hallucination ✅
- [ ] Phase 6: ColPali (視覺嵌入)

---

## 📝 授權

MIT License - 詳見 [LICENSE](LICENSE)

---

## 🤝 貢獻

歡迎提交 Pull Request！請先閱讀貢獻指南。

1. Fork 專案
2. 建立功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request
