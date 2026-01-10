# Tech Stack - Multimodal Agentic RAG System

## 1. 核心開發語言 (Core Language)
*   **Python 3.10+**: 作為整個後端、AI 模型串接與資料處理的主要開發語言。

## 2. Web 與 API 架構 (Backend & API)
*   **FastAPI**: 高效能的非同步 Web 框架，用於建立所有研究流程的 API。
*   **Uvicorn**: 作為 ASGI 伺服器運行 FastAPI。
*   **SSE-Starlette**: 支援 Server-Sent Events (SSE)，用於向前端即時串流 Agent 的思考進度。

## 3. 資料庫與儲存 (Database & Storage)
*   **Supabase**: 用於儲存對話紀錄、處理日誌與文件後設資料。

## 4. AI、LLM 與 RAG (AI & RAG)
*   **Google Generative AI (Gemini)**: 主要的 LLM 引擎。
*   **LangChain 0.3.0+**: 用於構建 Agentic Workflow 與鏈結各項 AI 組件。
*   **Sentence-Transformers**: 用於生成文本與圖片的向量嵌入。
*   **FAISS**: 本地向量資料庫，用於高效檢索。
*   **GraphRAG**: 利用 NetworkX, Leidenalg, python-igraph 構建學術知識圖譜與全域檢索能力。

## 5. OCR 與 文件處理 (OCR & Document Processing)
*   **DocTR**: 用於圖片與 PDF 頁面的視覺分析與 OCR 文字提取。
*   **marker-pdf**: 用於將 PDF 轉換為結構化的 Markdown 文件，保留排版邏輯。

## 6. 測試與品質控管 (Testing)
*   **Pytest**: 單元測試與整合測試框架。
*   **Pytest-asyncio**: 支援非同步測試。
