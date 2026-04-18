# Native RAG 純淨化（Plain RAG）實施方案

- 文件路徑：`pdftopng/docs/exec-plans/plain_rag_downgrade.md`
- 狀態：Implemented（2026-04-18）
- 目的：將 Native RAG 還原為最基礎 Plain RAG（向量檢索 + LLM 生成），作為與 Deep Research / GraphRAG / Agentic 路徑的對照基線。

## 1. Objective

將 Native `/rag/ask`（含 stream）路徑降級為 Plain RAG，關閉或移除會提升檢索品質與抗幻覺能力的進階機制，確保 baseline 足夠「純」；同時保留 Deep Research / Agentic benchmark 舊有進階流程。

## 2. Feature Downgrade Matrix

| 功能項目 | 舊行為 | Plain RAG 行為 | 實作位置 |
|---|---|---|---|
| 適用範圍 | 所有 `rag_answer_question` 呼叫路徑共享同一行為 | 僅 Native `/ask` 啟用 plain；Deep Research / Agentic 強制 `plain_mode=False` | `data_base/router.py`、`data_base/deep_research_service.py`、`data_base/research_execution_core.py`、`evaluation/agentic_evaluation_service.py` |
| 混合檢索（Hybrid） | FAISS + BM25 + Ensemble | 僅 FAISS | `data_base/vector_store_manager.py` |
| 重排序（Reranking） | `enable_reranking=True`（多數流程） | 預設關閉 | `data_base/RAG_QA_service.py`、`data_base/schemas.py` |
| 查詢變換（HyDE/Multi-Query） | 可由參數啟用 | 預設關閉（保留相容參數） | `data_base/RAG_QA_service.py` |
| 上下文展開（Parent/Child Expansion） | 檢索後執行 `_expand_short_chunks` | 移除呼叫 | `data_base/RAG_QA_service.py` |
| 視覺查證 Prompt 注入 | 主 Prompt 可注入 `VISUAL_TOOL_INSTRUCTION` | 改為 `PLAIN_RAG_PROMPT_TEMPLATE`（不含工具指令） | `data_base/RAG_QA_service.py` |
| 索引預設配置 | production 預設 `semantic_contextual` | production 預設 `recursive_baseline` | `data_base/indexing_service.py` |

## 3. Technical Action Plan

### Phase 1: Retriever 純淨化（`vector_store_manager.py`）

- 新增 `plain_mode` 參數：
  - `get_user_retriever_async(..., plain_mode: bool = False)`
  - `get_user_retriever(..., plain_mode: bool = False)`
- `plain_mode=True` 時：直接回傳 `faiss_retriever`，跳過 BM25 與 `EnsembleRetriever`。

### Phase 2: QA 核心流程簡化（`RAG_QA_service.py`）

- `rag_answer_question` 調整：
  - `enable_reranking` 預設改為 `False`。
  - 新增 `plain_mode` 參數（預設 `True`），且檢索器呼叫改為 `get_user_retriever(..., plain_mode=plain_mode)`。
  - `_expand_short_chunks` 只在 `plain_mode=False` 時執行。
  - `plain_mode=True` 使用 `PLAIN_RAG_PROMPT_TEMPLATE`（不注入視覺查證工具提示）。
  - `plain_mode=False` 保留原本進階 prompt（含 anti-hallucination / conflict / visual 指示）。
- 保留既有相容參數（`enable_hyde` / `enable_multi_query` / `enable_crag` 等），以避免破壞既有呼叫端。

### Phase 3: 索引基線固定（`indexing_service.py`）

- `DEFAULT_PRODUCTION_INDEXING_PROFILE` 改為 `recursive_baseline`。
- 目的：避免 production ingestion 預設落在語意分塊 + context enrichment，讓 Native baseline 與 Plain RAG 更一致。

## 4. Verification Plan

- 單元測試更新：
  - `tests/test_rag_retrieval_logic.py`
    - 確認 `enable_reranking` 預設為 `False`
    - 確認檢索呼叫帶入 `plain_mode=True`
  - `tests/test_indexing_service.py`
    - 確認 production indexing profile 預設為 `recursive_baseline`
    - 保留 `semantic_contextual` 的顯式 profile 測試

- 建議最小回歸指令：
  - `python -m pytest tests/test_rag_retrieval_logic.py tests/test_indexing_service.py`

## 5. Impact Notes

- 這次降級會使 Native RAG 在精確術語匹配、跨 chunk 語境連續性、與查詢穩健性上明顯下降，屬於預期結果。
- Deep Research / GraphRAG / Agentic 路徑仍可作為高階對照組（是否啟用進階能力由其各自流程控制）。

## 6. Rollback

若需回復先前進階 Native 行為：

1. `indexing_service.py` 將 `DEFAULT_PRODUCTION_INDEXING_PROFILE` 改回 `semantic_contextual`
2. `RAG_QA_service.py`
   - `enable_reranking` 預設改回 `True`
   - `get_user_retriever(..., plain_mode=False)` 或移除 `plain_mode` 強制參數
   - 恢復 `_expand_short_chunks` 呼叫
   - 改回原先增強型 Prompt 模板
3. `vector_store_manager.py` 關閉 `plain_mode` 分支（或保留但不啟用）
