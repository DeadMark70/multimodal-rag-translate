# Codebase Overview & Architecture

## System Architecture

The Multimodal RAG System is a FastAPI-based application designed for:

1.  **PDF Processing**: OCR, translation, and recreation of PDFs.
2.  **RAG (Retrieval-Augmented Generation)**: Intelligent Q&A over document content using vector search and hybrid retrieval.
3.  **Agentic Workflows**: Complex query resolution using Planner, Executor (RAG), and Synthesizer agents.

### Core Components

- **FastAPI Backend (`main.py`)**: Entry point, middleware, and router registration.
- **PDF Service (`pdfserviceMD/`)**:
  - **OCR**: Hybrid approach using **Local Marker** (default) or **Datalab API**.
  - **Translation**: Uses `gemini-3.0-flash` for high-volume text translation.
  - **PDF Generation**: Rebuilds PDFs maintaining layout using `markdown-pdf`.
- **RAG Engine (`data_base/`)**:
  - **Vector Store**: FAISS for fast similarity search.
  - **Embeddings**: `models/gemini-embedding-001` (Google API).
  - **Reranker**: `jinaai/jina-reranker-v3` via local Hugging Face inference for high-precision re-ranking.
  - **Query Transformation**: Implements HyDE (Hypothetical Document Embeddings) and Multi-Query expansion.
- **GraphRAG (`graph_rag/`)**:
  - **Store**: NetworkX based graph storage.
  - **Extraction**: LLM-based entity and relation extraction.
  - **Community**: Leiden algorithm for community detection and summarization.
  - **Search**: Local and Global search strategies.
- **Agents (`agents/`)**:
  - **Planner**: Decomposes complex user queries into sub-tasks. 🆕 `refine_query_from_evaluation()` for smart retry. 🆕 `_is_similar_question()` uses character bigrams for CJK support.
  - **Evaluator**: Self-RAG implementation to score retrieval quality and generation hallucinations.
  - **Synthesizer**: Combines results from sub-tasks into a coherent final answer. 🆕 Academic report template.
- **LLM Factory (`core/llm_factory.py`)**: Centralized management of LLM instances with purpose-specific configurations (Temperature, Max Tokens).

## Data Flow

### 1. PDF Ingestion Pipeline

1.  **Upload**: User uploads PDF via `/pdfmd/upload_pdf_md`.
2.  **OCR**: System extracts text using Marker/Datalab.
3.  **Translation**: Text is translated and a new PDF is generated.
4.  **Return PDF**: Translated PDF is returned to user immediately.
5.  **Background Tasks** (non-blocking):
    - Text is chunked and stored in FAISS (RAG indexing).
    - 🆕 **GraphRAG extraction**: Entities/relations extracted and added to knowledge graph.
    - Executive summary is generated.

### 2. RAG Query Pipeline (`/rag/ask`)

1.  **Query Analysis**: Query is rewritten (HyDE/Multi-Query).
2.  **Retrieval**: Vectors are fetched from FAISS.
3.  **Reranking**: Top-K results are re-ordered by relevance.
    - **Generation**: LLM (`gemma-3-27b-it`) generates answer based on context.
4.  🆕 **Context Enricher**: Short chunks (<100 chars) expanded using parent chunks.

### 3. GraphRAG Pipeline (Integrated)

1.  **Ingestion**: 🆕 Automatically triggered after PDF upload via background task.
2.  **Extraction**: Entities and relations extracted using `gemini-2.5-flash`.
3.  **Storage**: NetworkX graph stored in `uploads/{user_id}/rag_index/graph.pkl`.
4.  **Optimization**: Performs entity resolution and Leiden community detection.
5.  **Search**: Supports Local (entity-centric) and Global (community-summary) search modes.

### 4. Research Pipeline (`/rag/research`)

1.  **Planning**: `Planner` breaks down the query.
2.  **Execution**: Each sub-task is executed via the RAG pipeline.
3.  **Evaluation**: `Evaluator` checks quality. 🆕 Triggers smart retry if score < 3.
4.  **Synthesis**: `Synthesizer` compiles the final report. 🆕 Academic format with 5 sections.

## Key Directories

| Directory         | Purpose           | Key Files                                                                  |
| :---------------- | :---------------- | :------------------------------------------------------------------------- |
| `core/`           | Infrastructure    | `llm_factory.py`, `auth.py`, `supabase_client.py`                          |
| `pdfserviceMD/`   | PDF Logic         | `PDF_OCR_services.py`, `ai_translate_md.py`                                |
| `data_base/`      | RAG Logic         | `vector_store_manager.py`, `RAG_QA_service.py`, `deep_research_service.py` |
| `graph_rag/`      | Knowledge Graph   | `store.py`, `extractor.py`, `community_builder.py`                         |
| `agents/`         | AI Agents         | `planner.py`, `evaluator.py`, `synthesizer.py`                             |
| `multimodal_rag/` | Vision RAG        | `image_summarizer.py`                                                      |
| `conversations/`  | 🆕 Chat History   | `router.py`, `schemas.py`                                                  |
| `migrations/`     | 🆕 SQL Migrations | `002_create_conversations.sql`, `003_add_conversation_id_to_chat_logs.sql` |

## Design Patterns

- **Factory Pattern**: `llm_factory.py` for creating configured LLMs.
- **Strategy Pattern**: Chunking strategies (`semantic`, `proposition`, `word`).
- **Router Pattern**: FastAPI `APIRouter` for modular API definition.
- **Asynchronous Processing**: Heavy tasks (OCR, Translation) run in threadpools or background tasks.
- **Graph-based Modeling**: `graph_rag/store.py` uses NetworkX for knowledge representation.
- **Anti-Hallucination**: 🆕 Document-grouped context with source labels in RAG prompts.
- **Evaluation-Driven Loop**: 🆕 Smart retry based on evaluator feedback in research pipeline.

## Model Configuration (Internal Defaults)

These are configured in `core/llm_factory.py` and are not currently exposed as env vars, but good to know:

- **Translation Model**: `gemini-3.0-flash`
- **Graph Extraction**: `gemini-3.0-flash`
- **Community Summary**: `gemini-3.0-flash`
- **General Model**: `gemma-3-27b-it`

## Roadmap Status

| Phase         | Feature                                 | Status      |
| :------------ | :-------------------------------------- | :---------- |
| **Phase 1-3** | Basic RAG + Agents                      | ✅ Complete |
| **Phase 4**   | Multimodal Features                     | ✅ Complete |
| **Phase 5**   | GraphRAG (Core Modules)                 | ✅ Complete |
| **Phase 5.3** | GraphRAG Integration                    | ✅ Complete |
| **Phase 5.4** | 🆕 Interactive Deep Research            | ✅ Complete |
| **Phase 5.5** | 🆕 Conversation History                 | ✅ Complete |
| **Phase 5.6** | 🆕 Multi-Doc Anti-Hallucination         | ✅ Complete |
| **Phase 5.7** | 🆕 Deep Research Upgrade (Phase 1+2)    | ✅ Complete |
| **Phase 5.8** | 🆕 Deep Image Analysis (Phase 3)        | ✅ Complete |
| **Phase 5.9** | 🆕 Academic Evaluation Engine           | ✅ Complete |
| **Phase 6**   | 🆕 Deep Research Final Optimization     | ✅ Complete |
| **Phase 7**   | 🆕 PDF Generation Engine Upgrade        | ✅ Complete |
| **Phase 8**   | 🆕 Image Pipeline Integration           | ✅ Complete |
| **Phase 9**   | 🆕 Agentic Visual Verification          | ✅ Complete |
| **Phase 10**  | ColPali (Visual Embeddings)             | 📝 Planned  |
| **Phase 11**  | 🆕 GraphRAG Batch Processing            | ✅ Complete |
| **Phase 12**  | 🆕 Translation & Embedding Optimization | ✅ Complete |
| **Phase 13**  | 🆕 Context Transparency (Deep Research) | ✅ Complete |
| **Phase 14**  | 🆕 Strict Relevance & Optimization      | ✅ Complete |
| **Phase 15**  | 🆕 Agentic RAG Precision & Conciseness  | ✅ Complete |

## Phase 13: Context Transparency (Deep Research)

提升 Deep Research 的可解釋性，讓使用者知道 AI 是依據哪些具體內容得出結論：

- **Context Bubble-up**: `DeepResearchService` 與 `RAG_QA_service` 改進，將檢索到的原始文本片段 (`contexts`) 向上傳遞。
- **Frontend Integration**: API 回應與 SSE 事件新增 `contexts` 欄位，前端可顯示每個子任務的參考依據。
- **Schema Update**: `SubTaskExecutionResult` 與 `TaskDoneData` 新增 `contexts: List[str]`。

三項效能與穩定性優化：

### 12.1 翻譯 Prompt 優化 (`translation_chunker.py`)

- 強化 `[[PAGE_X]]` marker 保留指令（置頂 ⛔ 警告）
- 增加保留項目清單（數學公式、HTML 標籤等）
- 結尾加入驗證提醒
- **效果**: 減少 marker 丟失導致的重試

### 12.2 圖片品質提升 (`image_summarizer.py`)

- `MAX_IMAGE_SIZE`: 1024 → 1500 像素
- `JPEG_QUALITY`: 85 → 95
- **效果**: 更高解析度圖片分析

### 12.3 Embedding Retry 機制 (`vector_store_manager.py`)

- 新增 `_add_documents_with_retry()` 函數
- 新增 `_create_faiss_with_retry()` 函數
- Exponential backoff: 30s → 60s → 120s
- 捕捉 `429 RESOURCE_EXHAUSTED` 自動重試
- **效果**: 免費 API 限制下的容錯

## Phase 11: GraphRAG Batch Processing

優化 GraphRAG 實體提取效能，使用批次並行處理：

- **router.py**: `_run_graph_extraction()` 函數改進
  - 新增 `batch_size` 參數（預設 3）
  - 使用 `asyncio.gather()` 並行處理批次
  - 預先過濾太短的 chunks
  - 增強批次進度 logging
- **效能提升**: 理論約 3 倍速度提升

## Phase 9: Agentic Visual Verification

賦予 Agent 主動「看圖」的能力，透過 Re-Act 循環：

- **visual_tools.py**: 安全的視覺查證工具（路徑驗證、擴展名白名單）
- **RAG_QA_service.py**: Re-Act 循環實作
  - `VISUAL_TOOL_INSTRUCTION`: 教導 LLM 使用 JSON 指令請求看圖
  - `_parse_visual_tool_request()`: 容錯 JSON 解析
  - `_execute_visual_verification_loop()`: 工具執行與合成
  - `enable_visual_verification` 參數
- **deep_research_service.py**: 傳遞參數到 RAG 調用

## Phase 8: Image Pipeline Integration

將 OCR 提取的圖片整合到 RAG 管線，使 Deep Research 能夠檢索圖片內容：

- **image_processor.py**: 新增圖片提取模組
  - `extract_images_from_markdown()`: 從 Markdown 提取圖片路徑與上下文
  - `create_visual_elements()`: 建立 VisualElement 物件
- **vector_store_manager.py**: 新增 `add_visual_summaries_to_knowledge_base()`
- **router.py**: 新增 `_process_document_images()` 整合到後處理流程
- **處理流程**: RAG 索引 → 圖片摘要 → GraphRAG → 總結生成

## Phase 7: PDF Generation Engine Upgrade

強化 Marker OCR → Pandoc → PDF 流程的穩健性：

- **markdown_cleaner.py**: 新增 Markdown 清洗模組
  - `fix_image_paths()`: 相對路徑轉絕對路徑，遺失圖片佔位符
  - `escape_latex_specials()`: 轉義 LaTeX 保留字 (%, #, &)
  - `enhance_wide_tables()`: 寬表格自動縮放 (adjustbox / scriptsize)
- **Pandoc 升級**: `--resource-path`, `--from=markdown+raw_tex`, `--listings`
- **容錯機制**: Debug .tex 保留, HTML fallback (weasyprint)

## Phase 6: Deep Research Final Optimization

提升 Deep Research 在大規模檢索與多文檔衝突場景的表現：

- **Phase 6.1A**: 預設開啟 GraphRAG (hybrid mode) 提升抗噪能力
- **Phase 6.1B**: 強制 Drill-down (iteration 0 不跳過) 確保邏輯深度
- **Phase 6.2**: 信心度校準 (衝突懲罰 ×0.8) 反映不確定性
- **Phase 6.3**: 對抗性查詢 (Counter-Query) 強制正反辯證

## Phase 5.9: Academic Evaluation Engine

新增 1-10 分制學術評估引擎：

- **評估維度**: Accuracy (50%), Completeness (30%), Clarity (20%)
- **Smart Retry**: 使用 `suggestion` 欄位驅動查詢精煉
- **Pure LLM 模式**: `evaluate_pure_llm()` 支援無文檔評估
- **Arena 腳本**: `tests/run_arena.py` RAG vs Pure LLM A/B 測試
- **閾值**: Accuracy < 6 觸發重試

## Phase 15: Agentic RAG Precision & Conciseness

大幅優化 Agentic RAG 的回答品質與精簡度，解決長篇大論與細節幻覺問題：

- **Synthesizer 優化**: 強制執行 **BLUF (Bottom Line Up Front)** 架構，字數限制 < 500 字，禁止廢話。
- **Technical Root Cause**: 要求在解釋差異時必須引用具體技術原因 (如 Inductive Bias, UpKern)，而非泛泛比較。
- **效能指標大幅提升**:
  - **Faithfulness**: 0.70 → **0.91** (超越 Naive RAG)
  - **Answer Correctness**: 0.48 → **0.63** (接近實用水平)
- **架構視覺化**: 建立 Mermaid 流程圖對比 Naive vs Agentic 決策路徑。
