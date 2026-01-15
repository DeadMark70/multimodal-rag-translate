# PDF Service (pdfserviceMD) Technical Documentation

## 1. Technical Implementation Details

### Core Logic
The `pdfserviceMD` module manages the end-to-end processing of PDF documents, transforming them into translated, indexable content. The pipeline consists of the following stages:

1.  **Ingestion & Validation**: Accepts PDF uploads, validates file type, and saves to a user-specific directory.
2.  **OCR Processing (Hybrid Strategy)**:
    -   **Strategy**: Uses `ocr_service_sync` which selects between Local Marker (on-device, free) or Datalab API (cloud, paid) based on the `USE_LOCAL_MARKER` environment variable.
    -   **Output**: Generates Markdown content with `[[PAGE_N]]` markers to preserve pagination context.
    -   **Image Extraction**: Extracts images from the PDF for separate multimodal processing.
3.  **Translation (Chunk-Aware)**:
    -   **Logic**: Uses `ai_translate_md.py` which delegates to `translation_chunker.py`.
    -   **Mechanism**: Splits Markdown by `[[PAGE_N]]` markers to respect LLM context windows, translates chunks to Traditional Chinese using Gemini, and reassembles them.
4.  **PDF Generation**:
    -   **Engine**: Converts the translated Markdown back into a PDF using Pandoc (`markdown_to_pdf.py` / `Pandoc_md_to_pdf.py`), preserving layout and embedding translated images.
5.  **Post-Processing (Async)**:
    -   **RAG Indexing**: Indexes text content into the Vector Store (`data_base`).
    -   **Image Summarization**: Summarizes extracted images and indexes them (`multimodal_rag`).
    -   **GraphRAG**: Extracts entities and builds a knowledge graph (`graph_rag`).
    -   **Summary**: Generates an executive summary of the document.

### Algorithms
-   **Page-Based Chunking**: Ensures translation context is maintained per page while fitting within API limits.
-   **Visual Placeholder Replacement**: Identifies image locations in OCR output and re-inserts them into the translated document.

## 2. Codebase Map

| File Path | Responsibility |
| :--- | :--- |
| `pdfserviceMD/router.py` | Main API entry point. Handles uploads, orchestrates the processing pipeline, and manages document status/retrieval. |
| `pdfserviceMD/PDF_OCR_services.py` | Wrapper service for OCR. Switches between Local Marker and Datalab API. |
| `pdfserviceMD/local_marker_service.py` | Implementation of local OCR using the `marker-pdf` library. |
| `pdfserviceMD/ai_translate_md.py` | Entry point for translation services. |
| `pdfserviceMD/translation_chunker.py` | Logic for splitting Markdown into manageable chunks for LLM translation. |
| `pdfserviceMD/markdown_to_pdf.py` | Converts translated Markdown to PDF (Pandoc wrapper). |
| `pdfserviceMD/image_processor.py` | Utilities for extracting and handling images from Markdown/PDFs. |
| `pdfserviceMD/markdown_process.py` | Helpers for cleaning and preprocessing Markdown text. |

## 3. Usage Guide

**⚠️ IMPORTANT: All commands must be executed within the project's virtual environment (`.venv`).**

### API Usage
This module is primarily accessed via HTTP API.

**Upload & Process PDF:**
`POST /pdfmd/upload_pdf_md`
-   **Form Data**: `file=@/path/to/doc.pdf`
-   **Auth**: Requires JWT token.

**Get Status:**
`GET /pdfmd/file/{doc_id}/status`

**Get Translated PDF:**
`GET /pdfmd/file/{doc_id}`

### Standalone Testing (via Script)
To test OCR or translation logic directly:

```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Example: Run a test script (create one if needed)
python debug_import.py
```

## 4. Dependencies

### Internal Modules
-   `core`: Authentication, LLM factory.
-   `data_base`: RAG storage and vector management.
-   `multimodal_rag`: Image summarization logic.
-   `graph_rag`: Knowledge graph extraction.
-   `supabase_client`: Database interactions.

### External Libraries
-   `fastapi`: API framework.
-   `marker-pdf`: Local OCR engine.
-   `httpx`: Async HTTP client (for Datalab API).
-   `pypandoc` / `pandoc`: PDF generation.
-   `google-generativeai`: Translation model.
