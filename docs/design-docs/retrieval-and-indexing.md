# Retrieval And Indexing

## Purpose

Describe the boundaries between ask/research execution, indexing, and metadata ownership.

## Core Paths

- Ask and research entrypoints: `data_base/router.py`
- Legacy answer API and public re-exports: `data_base/RAG_QA_service.py`
- Answer orchestration stages: `data_base/rag_pipeline.py`
- Retrieval, filtering, corrective retrieval, and generation:
  `data_base/rag_retrieval.py`, `data_base/rag_filtering.py`,
  `data_base/rag_crag.py`, and `data_base/rag_generation.py`
- Graph decisions, evidence, and observability: `data_base/rag_graph_runtime.py`
- Shared execution core: `data_base/research_execution_core.py`
- Index orchestration: `data_base/indexing_service.py`
- Metadata helpers: `data_base/document_metadata.py`

## Design Rules

- New writes use canonical `doc_id`.
- Legacy `original_doc_uid` remains compatibility-only for reads/deletes.
- Keep `RAG_QA_service.py` as a compatibility facade. New pipeline behavior belongs
  to the stage owner instead of the facade.
- Patch private helpers in their owning module; only the documented public facade
  names are compatibility contracts.
- Background indexing and graph maintenance should surface explicit states instead of silently masking partial failure.
