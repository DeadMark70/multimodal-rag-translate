# Backend Tasks for Frontend Integration (Phase 2 & 3)

> **目標**: 支援前端所需的進階功能（GraphRAG 視覺化、參數控制、儀表板），並提供 Mock 資料以便前端並行開發。
> **建立日期**: 2025-12-21
> **狀態**: 待執行

---

## 1. 核心 Schema 更新 (Priority: High)
前端將會傳送新的控制參數，後端必須能接收並處理（即使目前邏輯尚未完全實作）。

### 📁 檔案: `data_base/schemas.py`
更新 `AskRequest` Pydantic 模型，新增以下欄位：

```python
class AskRequest(BaseModel):
    # ... 原有欄位 ...
    
    # GraphRAG 新增欄位
    enable_graph_rag: bool = False
    graph_search_mode: Literal["local", "global", "hybrid", "auto"] = "hybrid"
    enable_graph_planning: bool = False
```

---

## 2. GraphRAG API 端點實作 (Priority: High)
為了讓前端 `KnowledgeGraph` 元件有資料可渲染，需實作以下端點。**在 Phase 5 邏輯完成前，先回傳 Mock 資料。**

### 📁 新增檔案: `data_base/graph/router.py`

| HTTP Method | Endpoint | 說明 | Mock 回應範例 |
| :--- | :--- | :--- | :--- |
| `GET` | `/graph/status` | 圖譜統計資訊 | `{"node_count": 120, "edge_count": 350, "last_updated": "..."}` |
| `GET` | `/graph/data` | **視覺化資料 (Nodes & Links)** | 見下方 Mock 結構 |
| `POST` | `/graph/rebuild` | 強制重建圖譜 | `{"status": "started", "task_id": "..."}` |
| `POST` | `/graph/optimize` | 優化社群/摘要 | `{"status": "started", "task_id": "..."}` |

#### `/graph/data` Mock 資料結構
前端 `react-force-graph` 預期的格式：
```json
{
  "nodes": [
    {"id": "Transformer", "group": 1, "val": 10, "desc": "Deep Learning Architecture"},
    {"id": "BERT", "group": 1, "val": 8, "desc": "Pre-trained Model"},
    {"id": "Attention", "group": 2, "val": 5, "desc": "Mechanism"}
  ],
  "links": [
    {"source": "BERT", "target": "Transformer", "label": "based on"},
    {"source": "Transformer", "target": "Attention", "label": "uses"}
  ]
}
```

### 📁 修改檔案: `main.py`
- [ ] 註冊新的 router: `app.include_router(graph_router, prefix="/graph", tags=["Knowledge Graph"])`

---

## 3. RAG 服務邏輯調整 (Priority: Medium)
確保後端接收到新參數時不會報錯，並準備好接入點。

### 📁 修改檔案: `data_base/RAG_QA_service.py`
- [ ] 在 `rag_answer_question` 函式簽章中加入新參數 (`enable_graph_rag` 等)。
- [ ] 添加 Log 記錄：「Graph Mode enabled: {mode}」，以便確認參數傳遞正確。
- [ ] (暫時) 若 `enable_graph_rag=True`，在 Prompt 中加入一段文字：「(注意：使用者啟用了圖譜增強，目前處於模擬模式)」。

---

## 4. 深度研究進度 (Priority: Low)
針對 `POST /rag/research`，確認回傳結構是否支持前端顯示子任務狀態。

### 📁 檢查檔案: `agents/synthesizer.py`
- [ ] 確認 `ResearchReport` 模型中的 `sub_results` 包含 `task_id`, `question`, `status` (雖然目前是同步回傳，全都是 completed，但欄位要在)。

---

## 5. 執行清單

- [ ] 更新 `data_base/schemas.py`
- [ ] 建立 `data_base/graph/` 目錄 (若不存在)
- [ ] 建立 `data_base/graph/router.py` (含 Mock Data)
- [ ] 修改 `main.py` 註冊路由
- [ ] 修改 `RAG_QA_service.py` 接收參數
