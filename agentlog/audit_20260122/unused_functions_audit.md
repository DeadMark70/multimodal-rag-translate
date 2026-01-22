# Unused Functions Audit Report (2026-01-22)

This report lists identified dead code and potential false positives from the Python backend.

## Summary
- **Tool used**: Vulture (min-confidence 60%)
- **Total items flagged**: ~150+
- **Primary categories**: FastAPI Entry Points, Schema Fields, Agent Skills, Internal Utilities.

## 1. Likely Dead Code (High Confidence)
These functions or classes appear to be internal and are not called within the project or registered as API endpoints.

| File Path | Symbol | Line | Recommendation |
|-----------|--------|------|----------------|
| `agents/evaluator.py` | `evaluate_rag_detailed` | 721 | Check if deprecated by `RAGEvaluator` class. |
| `agents/evaluator.py` | `compare_rag_vs_pure_llm` | 751 | Check if used in Arena mode tests. |
| `agents/planner.py` | `needs_planning` | 309 | **Keep**. Intended for future dynamic routing. |
| `agents/planner.py` | `needs_graph_analysis` | 339 | **Keep**. Intended for future dynamic routing. |
| `data_base/RAG_QA_service.py` | `rag_answer_question_simple` | 866 | Likely redundant. |
| `data_base/semantic_chunker.py` | `create_semantic_chunker` | 327 | Check if used in initialization. |
| `graph_rag/community_builder.py` | `rebuild_communities` | 224 | Internal utility, check usage. |
| `multimodal_rag/utils.py` | `ensure_directory` | 5 | Redundant if `os.makedirs` used directly. |

## 2. False Positives: FastAPI Entry Points
These are decorated with `@router.get/post` and are called by the API, not internally. **Do not remove.**

| File Path | Symbol | Reason |
|-----------|--------|--------|
| `conversations/router.py` | `list_conversations`, `create_conversation`, etc. | API Endpoints |
| `data_base/router.py` | `ask_question`, `research_question`, etc. | API Endpoints |
| `graph_rag/router.py` | `get_graph_status`, `rebuild_graph`, etc. | API Endpoints |
| `pdfserviceMD/router.py` | `list_documents`, `upload_pdf_md`, etc. | API Endpoints |
| `main.py` | `startup_event`, `read_root` | App Lifecycle |

## 3. False Positives: Pydantic Schema Fields
Flagged because they are accessed via dynamic keys or JSON serialization. **Do not remove.**

| File Path | Symbol | Reason |
|-----------|--------|--------|
| `conversations/schemas.py` | `created_at`, `updated_at` | Model Fields |
| `data_base/schemas.py` | `assistant`, `created_at`, `confidence_score` | Model Fields |
| `graph_rag/schemas.py` | `USES`, `CITES`, `EXTENDS` | Constants/Enum-like fields |

## 4. Agent Skills & Scripts
Located in `.agent/skills/` or `experiments/`. These are often standalone scripts or templates.

| File Path | Symbol | Reason |
|-----------|--------|--------|
| `.agent/skills/...` | Various | Templates/Examples |
| `experiments/evaluation_pipeline.py` | `mock_rag_answer` | Testing Utility |

## Conclusion
Most "unused" items in routers are API entry points. The real optimization targets are in `agents/`, `data_base/`, and `multimodal_rag/utils.py`.
