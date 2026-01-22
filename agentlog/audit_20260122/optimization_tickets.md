# Optimization Implementation Tickets

This file contains granular tasks for future optimization tracks.

## Track A: Security & Lint Cleanup
- [ ] Refactor `pdfserviceMD/router.py`: Change `doc_id` to `UUID` in `get_processing_status`, `get_pdf_file`, `delete_pdf_file`, `get_document_summary_endpoint`, `regenerate_summary_endpoint`.
- [ ] Refactor `multimodal_rag/router.py`: Change `doc_id` to `UUID` in `delete_multimodal_document`.
- [ ] Run `ruff check . --fix` to remove unused imports and variables.
- [ ] Run `ruff format .` to standardize indentation and spacing.
- [ ] Manually remove extraneous `f` prefixes from strings flagged by Ruff (F541).

## Track B: Dead Code Removal
- [ ] Remove `rag_answer_question_simple` from `data_base/RAG_QA_service.py`.
- [ ] Remove `ensure_directory` from `multimodal_rag/utils.py` (replace with `os.makedirs(..., exist_ok=True)`).
- [ ] Remove `evaluate_rag_detailed` and `compare_rag_vs_pure_llm` from `agents/evaluator.py`.
- [ ] Archive standalone scripts in `experiments/` that are no longer needed for development.

## Track C: Refactoring & Technical Debt
- [ ] Rename ambiguous variables (e.g., `l`) in `pdfserviceMD/markdown_cleaner.py`.
- [ ] Split multiple statements on one line in `pdfserviceMD/markdown_cleaner.py`.
- [ ] Refactor `main.py` to move middleware and router registration into helper functions.
- [ ] Add `# noqa: E402` to necessary early imports in `main.py`.
