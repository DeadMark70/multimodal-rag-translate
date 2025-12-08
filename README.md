# 🔬 Multimodal RAG System

> 進階多模態 RAG 系統，支援 PDF/圖片處理、語義檢索、智能問答

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ 功能特色

### 📄 文件處理

- **PDF OCR**：使用 PaddleOCR 結構化識別
- **多語言翻譯**：Google Gemini AI 驅動
- **Markdown 輸出**：保留文件結構與格式

### 🔍 進階 RAG 檢索

- **語義分塊**：基於語意邊界的智能切分
- **上下文增強**：LLM 生成上下文前綴
- **Cross-Encoder 重排序**：BGE-Reranker-v2-M3
- **HyDE 查詢轉換**：假設文檔嵌入
- **多查詢融合**：Reciprocal Rank Fusion

### 🤖 Agent 架構

- **Self-RAG 評估**：檢索相關性 + 答案忠實度
- **Plan-and-Solve**：複雜問題分解與綜合
- **深度研究端點**：`/rag/research`

### 🖼️ 多模態支援

- **圖片內文字翻譯**：就地翻譯
- **視覺元素摘要**：圖表/表格智能描述
- **FAISS 向量索引**：GPU 加速檢索

---

## 🏗️ 技術架構

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI Server                      │
├─────────────┬─────────────┬──────────────┬──────────────┤
│   /pdfmd    │    /rag     │  /imagemd    │ /multimodal  │
│  PDF 翻譯   │  RAG 問答   │  圖片翻譯    │  多模態處理  │
├─────────────┴─────────────┴──────────────┴──────────────┤
│                     Core Services                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ PaddleOCR   │  │ FAISS Index │  │ Google Gemini   │  │
│  │ (Structure) │  │ (BGE-M3)    │  │ (LLM)           │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                     Supabase                             │
│              (Auth + PostgreSQL)                         │
└─────────────────────────────────────────────────────────┘
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

# 深度研究
POST /rag/research
{
  "question": "比較 Python 和 JavaScript 的優缺點",
  "max_subtasks": 3,
  "enable_reranking": true
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

```
.
├── main.py                 # FastAPI 入口
├── config.env              # 環境變數 (不提交)
├── requirements.txt        # Python 依賴
│
├── core/                   # 核心模組
│   ├── auth.py             # Supabase JWT 認證
│   └── llm_factory.py      # LLM 實例工廠
│
├── data_base/              # RAG 核心
│   ├── router.py           # /rag 端點
│   ├── RAG_QA_service.py   # RAG 主服務
│   ├── vector_store_manager.py  # FAISS 管理
│   ├── semantic_chunker.py # 語義分塊
│   ├── reranker.py         # Cross-Encoder
│   └── query_transformer.py # HyDE/Multi-Query
│
├── agents/                 # Agent 模組
│   ├── evaluator.py        # Self-RAG 評估
│   ├── planner.py          # 任務分解
│   └── synthesizer.py      # 結果綜合
│
├── pdfserviceMD/           # PDF 處理
│   ├── router.py           # /pdfmd 端點
│   └── PDF_OCR_services.py # PaddleOCR 服務
│
├── multimodal_rag/         # 多模態處理
│   ├── router.py           # /multimodal 端點
│   └── image_summarizer.py # 圖片摘要
│
├── image_service/          # 圖片翻譯
│   └── router.py           # /imagemd 端點
│
└── tests/                  # 單元測試
    ├── test_semantic_chunker.py
    ├── test_evaluator.py
    └── ...
```

---

## 🧪 測試

```bash
# 執行所有測試
pytest tests/ -v

# 執行特定模組測試
pytest tests/test_evaluator.py -v

# 測試覆蓋率
pytest tests/ --cov=. --cov-report=html
```

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

| 服務               | VRAM 需求 |
| ------------------ | --------- |
| BGE-M3 Embeddings  | ~1.5 GB   |
| BGE-Reranker-v2-M3 | ~1.5 GB   |
| PaddleOCR          | ~2 GB     |
| **總計**           | **~5 GB** |

---

## 🛣️ 開發路線

- [x] Phase 1: 語義分塊 + 上下文增強
- [x] Phase 2: Cross-Encoder + HyDE
- [x] Phase 3: Self-RAG + Plan-and-Solve
- [ ] Phase 4: ColPali 視覺嵌入 (需 8GB+ VRAM)
- [ ] Phase 5: GraphRAG 知識圖譜

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
