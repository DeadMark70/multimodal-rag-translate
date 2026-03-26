# Retrieval And Indexing

## Purpose

Describe the boundaries between ask/research execution, indexing, and metadata ownership.

## Core Paths

- Ask and research entrypoints: `data_base/router.py`
- Retrieval and answer generation: `data_base/RAG_QA_service.py`
- Shared execution core: `data_base/research_execution_core.py`
- Index orchestration: `data_base/indexing_service.py`
- Metadata helpers: `data_base/document_metadata.py`

## Design Rules

- New writes use canonical `doc_id`.
- Legacy `original_doc_uid` remains compatibility-only for reads/deletes.
- Background indexing and graph maintenance should surface explicit states instead of silently masking partial failure.
