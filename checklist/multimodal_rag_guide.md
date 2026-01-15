# Multimodal RAG (multimodal_rag) Technical Documentation

## 1. Technical Implementation Details

### Core Logic
The `multimodal_rag` module handles the extraction, analysis, and indexing of non-textual information (figures, images, charts) from PDF documents, enabling the system to "see" and "reason" about visual data.

1.  **Extraction (`structure_analyzer.py`)**:
    -   **Engine**: Relies on **Datalab Layout API** (formerly Marker API) to analyze document structure.
    -   **Visual Cropping**: Converts PDF pages to images using `pdf2image`, then uses the bounding boxes returned by the API to physically crop and save visual elements (`visuals/` directory).
    -   **Text Correlation**: Analyzes the surrounding Markdown text to extract context for each image (e.g., "Figure 1", captions, nearby paragraphs).

2.  **Summarization (`image_summarizer.py`)**:
    -   **Engine**: Google Gemini Vision API.
    -   **Prompting**: Generates context-aware prompts. It feeds the image *and* its textual context (from step 1) to the VLM (Vision-Language Model).
    -   **Caching**: Implements MD5 hash-based LRU caching to prevent re-processing identical images (saving costs and time).
    -   **Optimization**: Pre-processes images (resizing/compression) before sending to the API.

3.  **Visual Verification (Advanced)**:
    -   Supports a "Re-Examine" workflow where an Agent can ask specific questions about an already indexed image (e.g., "What is the value of the red bar in Figure 2?").

4.  **Indexing (`router.py`)**:
    -   Orchestrates the pipeline: Extraction -> Summarization -> Indexing (via `data_base.vector_store_manager`).
    -   Visual summaries are indexed as text vectors but tagged with metadata (`source="image"`) so they can be retrieved alongside regular text.

## 2. Codebase Map

| File Path | Responsibility |
| :--- | :--- |
| `multimodal_rag/router.py` | Main API endpoint (`/extract`). Coordinates the extraction and summarization process. |
| `multimodal_rag/structure_analyzer.py` | Interfaces with Datalab API to understand page layout and crop images. |
| `multimodal_rag/image_summarizer.py` | Manages VLM interactions for describing images. Includes caching and prompting logic. |
| `multimodal_rag/schemas.py` | Defines data structures (`ExtractedDocument`, `VisualElement`) shared across modules. |
| `multimodal_rag/utils.py` | Helper functions. |

## 3. Usage Guide

**⚠️ IMPORTANT: All commands must be executed within the project's virtual environment (`.venv`).**

### API Usage
**Extract & Process:**
`POST /multimodal/extract`
-   **Form Data**: `file=@/path/to/doc.pdf`
-   **Process**: Uploads -> Datalab API -> Crop -> Gemini Vision -> Index.

### Standalone Testing
To test image summarization or structure analysis:

```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# No dedicated script provided by default.
# Integration is typically tested via the main PDF upload flow in `pdfserviceMD`.
```

## 4. Dependencies

### Internal Modules
-   `core`: LLM factory (for Vision model).
-   `data_base`: Indexing logic (`index_extracted_document`).
-   `supabase_client`: Database access.

### External Libraries
-   `httpx`: Async HTTP client for Datalab API.
-   `pdf2image`: Converts PDF pages to images (requires Poppler installed on system).
-   `opencv-python` (`cv2`): Image processing and cropping.
-   `Pillow`: Image manipulation.
-   `numpy`: Array handling.
