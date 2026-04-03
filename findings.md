# Findings & Decisions

## Requirements
- 實作使用者提供的「Agentic Eval 微優化」方案（evaluation flow only）。
- 必須保留 `Full Prompt Basis`：模型可見證據與 RAGAS contexts 一致。
- `hybrid_graph` 需改名為 `generic_graph`，並維持 generic core 行為。
- 強化 correctness（benchmark / figure_flow）但避免增加高成本推理輪次。

## Current Baseline Findings
- `evaluation/agentic_evaluation_service.py`
  - route profile 仍使用 `hybrid_graph`。
  - `figure_flow` 初始子任務上限為 2，且 tier_2 drilldown 仍允許 1 輪。
  - tier_3 drilldown 為 2 輪，且 followup cap 對 tier_3 為 2。
  - `visual_verify` 目前關閉 `enable_multi_query`。
- `data_base/RAG_QA_service.py`
  - graph context 已進 prompt，但 `return_docs=True` 時未把 graph evidence 加到 `documents`。
  - Prompt 具 anti-hallucination 規則，但缺少 benchmark/figure_flow 題型的輸出格式硬約束。
- `agents/synthesizer.py`
  - 已有 intent guidance，但 benchmark/figure_flow 的 correctness 導向約束還不夠嚴格。
- `evaluation/db.py`
  - trace 載入/寫入未對 `hybrid_graph` -> `generic_graph` 做正規化。

## Implementation Plan (Concrete)
1. `agentic_evaluation_service.py`
- route literal 改為 `generic_graph`。
- benchmark 初輪：數值子任務走 `generic_graph`，非數值走 `hybrid_compare`。
- figure_flow：初始子任務 1、drilldown 0。
- benchmark drilldown 上限收斂至 1（含 followup cap）。
- `visual_verify` 保留檢索增強：至少 `enable_multi_query=True`。

2. `RAG_QA_service.py`
- `_get_graph_context` 新增可選 evidence 回傳，不破壞舊字串回傳。
- `return_docs=True` + `enable_graph_rag=True` 時，graph evidence 轉成 `Document` 追加回傳。
- 增加題型導向 prompt 約束（benchmark/figure_flow）與嚴格 grounding 規則。

3. `agents/synthesizer.py`
- benchmark guidance：先列「指標-模型-數值-來源」，無數值標示「資料不足」。
- figure_flow guidance：先給有序流程 `A -> B -> C`，禁止新增未出現元件。
- 強化 source-tag/grounding 的輸出約束。

4. `evaluation/db.py`
- 實作 route profile alias 正規化，讀寫都能兼容舊 trace。

5. Tests
- 更新 `tests/test_agentic_evaluation_service.py`。
- 新增 `tests/test_rag_qa_graph_evidence.py`（或併入既有檔）驗證 graph evidence documents。
- 視需要補 `evaluation/db.py` route alias 測試。

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| `rg` 不可用 | 改用 `Select-String` |
| 指定的 `evaluation/metrics_aggregator.py` 不存在 | 直接在 `evaluation/db.py` 與 trace path 內實作相容層 |

## Resources
- `D:\flutterserver\pdftopng\evaluation\agentic_evaluation_service.py`
- `D:\flutterserver\pdftopng\data_base\RAG_QA_service.py`
- `D:\flutterserver\pdftopng\agents\synthesizer.py`
- `D:\flutterserver\pdftopng\evaluation\db.py`
- `D:\flutterserver\pdftopng\tests\test_agentic_evaluation_service.py`
- `D:\flutterserver\pdftopng\tests\test_rag_modes_agentic.py`
- `D:\flutterserver\pdftopng\tests\test_graphrag_integration.py`

## Implemented Outcomes
- `RouteProfile` 已改為 `generic_graph` 命名；`benchmark_data` 初輪僅數值題走 `generic_graph`，非數值走 `hybrid_compare`。
- `figure_flow` 已改為初始 1 task、drilldown 0；`benchmark_data` drilldown 上限收斂為 1。
- `visual_verify` 保留檢索增強（`enable_multi_query=True`）。
- `_get_graph_context` 新增可選 evidence 回傳；`rag_answer_question(return_docs=True, enable_graph_rag=True)` 會把 graph evidence 追加進 `RAGResult.documents`（`source=graph_evidence` + `evidence_type`）。
- RAG 與 synthesizer prompt guidance 已加入 benchmark/figure_flow 輸出約束與 strict grounding（無證據則標示「資料不足」）。
- `evaluation/db.py` 已加入 route profile alias 正規化，讀寫 trace 都會把 `hybrid_graph` 轉為 `generic_graph`。
