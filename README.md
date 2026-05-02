# Multimodal Agentic RAG Backend (`pdftopng`)

本專案是多模態 Agentic RAG 的後端服務，採 FastAPI 架構，提供 PDF/OCR、RAG 推理、GraphRAG、評估管線、對話持久化與統計 API。

本 README 以目前程式碼為準（`main.py`, `core/app_factory.py`, 各 router 與 `openapi.json`）。

---

## 1. 系統角色

後端負責：

1. 文件處理：上傳 PDF、OCR、翻譯、摘要與索引流程
2. 問答與研究：一般 RAG、Deep Research、Agentic Benchmark SSE
3. 知識圖譜：GraphRAG 抽取、重構、優化、node-vector 同步
4. 評估中心：題庫、模型設定、campaign 執行、metrics、trace
5. 對話持久化：conversation/message CRUD
6. 儀表板：統計資料供前端 Dashboard 顯示

---

## 2. 啟動與組裝方式

入口：`main.py`

```python
from core.app_factory import create_app
app = create_app()
```

`core/app_factory.py` 負責：

- 載入 `config.env`
- 設定 logging
- 註冊 CORS
- 註冊錯誤處理器與 request-id middleware
- 掛載所有 API routers
- 啟動期間執行 lifespan 初始化：
  - 建立必要資料夾
  - 初始化 Supabase client
  - 初始化 evaluation DB
  - 恢復 in-flight campaign
  - RAG warmup
  - PDF OCR warmup（非 fake/test mode）

---

## 3. 主要目錄結構

```text
pdftopng/
  main.py
  core/
    app_factory.py
    auth.py
    providers.py
    llm_factory.py
    errors.py
  pdfserviceMD/         # PDF/OCR/翻譯/摘要
  data_base/            # RAG ask/plan/execute/stream
  graph_rag/            # GraphRAG 狀態與維運
  evaluation/           # 測資/模型/campaign/metrics/trace
  conversations/        # 對話持久化 API
  stats/                # dashboard stats
  multimodal_rag/       # multimodal 抽取
  image_service/        # 圖片翻譯
  migrations/           # SQL migrations
  tests/
  openapi.json
  requirements.txt
  config.env.example
```

---

## 4. API 範圍（實際路由）

以下為目前 `openapi.json` 暴露的主路徑群組：

### 4.1 系統

- `GET /`

### 4.2 文件與 PDF（`/pdfmd`）

- `GET /pdfmd/list`
- `POST /pdfmd/ocr`
- `POST /pdfmd/upload_pdf_md`
- `GET /pdfmd/file/{doc_id}`
- `GET /pdfmd/file/{doc_id}/status`
- `POST /pdfmd/file/{doc_id}/retry-index`
- `POST /pdfmd/file/{doc_id}/translate`
- `DELETE /pdfmd/file/{doc_id}`
- `GET /pdfmd/file/{doc_id}/summary`
- `POST /pdfmd/file/{doc_id}/summary/regenerate`

### 4.3 RAG / Agentic（`/rag`）

- `POST /rag/ask`
- `POST /rag/ask/stream`
- `POST /rag/research`
- `POST /rag/plan`
- `POST /rag/execute`
- `POST /rag/execute/stream`
- `POST /rag/agentic/stream`

### 4.4 GraphRAG（`/graph`）

- `GET /graph/status`
- `GET /graph/documents`
- `GET /graph/data`
- `POST /graph/rebuild`
- `POST /graph/rebuild-full`
- `POST /graph/documents/{doc_id}/retry`
- `DELETE /graph/documents/{doc_id}`
- `POST /graph/optimize`
- `POST /graph/node-vector/sync`
- `GET /graph/node-vector/sync/status`

### 4.5 評估中心（`/api/evaluation`）

- test cases CRUD
- model configs CRUD
- available models list
- campaigns create/list/evaluate/cancel
- campaign results/metrics/traces
- campaign SSE stream

### 4.6 對話持久化（`/api/conversations`）

- list/create/get/update/delete conversation
- create message in conversation

### 4.7 其他

- `GET /stats/dashboard`
- `POST /multimodal/extract`
- `DELETE /multimodal/file/{doc_id}`
- `POST /imagemd/translate_image`

---

## 5. 認證與安全

### 5.1 JWT 驗證

`core/auth.py` 的 `get_current_user_id` 為所有受保護路由的共用 dependency：

- 從 `Authorization: Bearer <token>` 讀取 JWT
- 透過 `core.auth_repository.fetch_user_id_from_token` 驗證
- 缺 token 或失敗時回 401

### 5.2 CORS

`core/app_factory.py`：

- 預設允許本機開發來源（5173/4173/3000/80）
- 可用 `CORS_ORIGINS` 覆寫

### 5.3 錯誤處理

全域 handler 在 `core/errors.py` 與 `app_factory` 註冊：

- `AppError`
- `HTTPException`
- `RequestValidationError`
- 未處理例外

### 5.4 Request ID

HTTP middleware 會注入/回傳 `X-Request-Id`，便於 trace 與除錯。

---

## 6. Provider 與外部依賴策略

`core/providers.py` 封裝外部 provider，支援 real/fake 模式切換：

- LLM provider
- Datalab provider

切換規則：

- `TEST_MODE=true` 或 `USE_FAKE_PROVIDERS=true` -> fake providers

`core/llm_factory.py`：

- 統一管理 LLM purpose 與配置
- 以 cache 重用 model 實例
- 支援 request-scoped runtime override

---

## 7. 環境變數

請複製 `config.env.example` 為 `config.env`。

關鍵變數：

- `GOOGLE_API_KEY`（必要）
- `SUPABASE_URL`, `SUPABASE_KEY`
- `DATALAB_API_KEY`
- `HF_TOKEN`（選用）
- `TEST_MODE`, `USE_FAKE_PROVIDERS`
- `CI_BLOCK_EXTERNAL_NETWORK`
- `IMAGE_OCR_DEVICE`（`cpu|auto|cuda`）

---

## 8. 本機開發

### 8.1 建立虛擬環境

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 8.2 安裝依賴

```bash
pip install -r requirements.txt
```

### 8.3 啟動服務

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

文件：

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## 9. 測試與品質

`pytest.ini`：

- `testpaths = tests`
- `python_files = test_*.py`

常用命令：

```bash
pytest
pytest -q
ruff check .
```

測試範圍涵蓋：

- API contracts
- agentic/deep-research flow
- GraphRAG pipeline
- conversation persistence
- evaluation campaign pipeline
- provider/test-mode 安全邊界

---

## 10. 資料持久化與 Migration

### 10.1 SQL migrations（本 repo）

- `migrations/002_create_conversations.sql`
  - 建立 `conversations` 表
  - 啟用 RLS
  - 建立 user scoped policy
- `migrations/003_add_conversation_id_to_chat_logs.sql`
  - `chat_logs` 增加 `conversation_id`、`role`

### 10.2 其他 Supabase migration

前端 repo (`D:\flutterserver\Multimodal_RAG_System\supabase\migrations`) 另有 `messages` / `conversations.metadata` migration，請確保實際 DB schema 與兩側契約一致。

---

## 11. Docker 與整體部署

### 11.1 Backend image

`pdftopng/Dockerfile`：

- Base: `python:3.11-slim`
- 安裝系統依賴（poppler, pandoc, libgl1 ...）
- 安裝 torch/torchvision（CUDA index）與 Python requirements
- 以 uvicorn 啟動 port `8000`

### 11.2 Full stack compose

上層 `D:\flutterserver\docker-compose.yml`：

- `backend`: build from `./pdftopng`
- `frontend`: build from `./Multimodal_RAG_System`
- 兩者在同一 bridge network (`rag-net`)

---

## 12. 與前端整合重點

前端 (`Multimodal_RAG_System`) 預設呼叫：

- `VITE_API_BASE_URL=http://127.0.0.1:8000`

部署到 nginx reverse proxy 時，前端也可用相對 base URL（例如 `/`）轉發到 backend。

SSE 端點（例如 `/rag/ask/stream`, `/rag/execute/stream`, `/api/evaluation/campaigns/{id}/stream`）需要關閉 proxy buffering。

---

## 13. 常見問題

### Q1. 401 `Missing Authorization header`

前端未附 Bearer token 或使用未登入 session。請檢查 Supabase session 狀態與前端 `api.ts` interceptor。

### Q2. 啟動時外部 provider error

若本機只是跑測試/開發流程，可先設定：

```env
TEST_MODE=true
USE_FAKE_PROVIDERS=true
```

### Q3. Graph rebuild / node-vector sync 一直 skipped

通常是目前已有 active graph job，或圖譜節點數/可用文件數不足。請先查 `/graph/status`、`/graph/documents`。

### Q4. OCR/翻譯流程失敗

請檢查：

- `DATALAB_API_KEY` / `GOOGLE_API_KEY`
- 系統依賴（poppler, pandoc）
- `uploads/` 與 `output/` 目錄權限

---

## 14. 相關檔案

- App factory: `core/app_factory.py`
- Auth dependency: `core/auth.py`
- Provider registry: `core/providers.py`
- OpenAPI snapshot: `openapi.json`
- Frontend counterpart: `D:\flutterserver\Multimodal_RAG_System\README.md`
