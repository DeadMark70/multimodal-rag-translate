# PDF Translation & RAG API Documentation

## 1. API Overview
- **Version**: 3.1.0
- **Base URL**: `http://localhost:8000` (Example)
- **Supported Clients**: Flutter (Mobile/Web), React, REST Clients.
- **Description**: Comprehensive suite for PDF OCR, translation, Multi-modal RAG, and GraphRAG services.

## 2. Authentication
- **Type**: Bearer Token (Supabase JWT)
- **Header**: `Authorization: Bearer {supabase_access_token}`
- **Note**: Development mode (`DEV_MODE=true`) allows bypassing authentication for testing.

## 3. Core Modules & Endpoints

### 3.1 PDF & OCR Service (`/pdfmd`)
Handles document ingestion, OCR (Local/Marker), and translation.
- `POST /pdfmd/upload_pdf_md`: Upload PDF for OCR + Translation + RAG indexing.
- `GET /pdfmd/list`: List user documents (max 50).
- `GET /pdfmd/file/{doc_id}/status`: Poll processing status (ocr, translating, indexing, etc.).
- `GET /pdfmd/file/{doc_id}/summary`: Get AI-generated executive briefing.
- `doc_id` path parameter for `/pdfmd/file/{doc_id}*` must be a valid UUID. Invalid format returns `422 Unprocessable Entity`.

### 3.2 RAG Question Answering (`/rag`)
Intelligent retrieval and research across user documents.
- `POST /rag/ask`: Context-aware Q&A. Supports `enable_evaluation`, `enable_graph_rag`, and `enable_visual_verification`.
- `POST /rag/research`: Deep Research mode using Plan-and-Solve workflow.
- `POST /rag/plan` & `POST /rag/execute/stream`: Human-in-the-loop research planning and SSE-streamed execution.

### 3.3 GraphRAG Service (`/graph`)
Global relationship analysis and knowledge graph visualization.
- `GET /graph/status`: Check graph health (node/edge counts).
- `GET /graph/data`: Get `react-force-graph` compatible JSON for UI visualization.
- `POST /graph/optimize`: Trigger entity resolution and community re-summarization.

### 3.4 Conversations (`/api/conversations`)
Persistence for chat history.
- `GET /api/conversations`: List user chat sessions.
- `POST /api/conversations`: Create new session.
- `GET /api/conversations/{id}`: Fetch session with full message history.

### 3.5 Image Translation (`/imagemd`)
- `POST /imagemd/translate_image`: In-place translation of text within images.

### 3.6 Analytics (`/stats`)
- `GET /stats/dashboard`: Performance metrics (Accuracy rate, Grounded vs Hallucinated counts).

## 4. Key Data Schemas

### Evaluation Metrics (Academic Grade)
```json
{
  "accuracy": 8.5,      // 50% weight (1-10)
  "completeness": 7.0,  // 30% weight (1-10)
  "clarity": 9.0,       // 20% weight (1-10)
  "weighted_score": 8.15,
  "is_passing": true,
  "faithfulness": "grounded"
}
```

### Research Response
```json
{
  "question": "...",
  "summary": "Short briefing...",
  "detailed_answer": "Full markdown report...",
  "sub_tasks": [...],
  "all_sources": ["uuid-1", "uuid-2"],
  "confidence": 0.85
}
```

## 5. Implementation Guides
For technical details on individual modules, refer to:
- [Agents Implementation](agents_implementation_guide.md)
- [Core Services](core_service_guide.md)
- [RAG Database](database_guide.md) (Checklist)
- [Graph RAG](graph_rag_guide.md) (Checklist)
- [Multi-modal RAG](multimodal_rag_guide.md) (Checklist)
- [PDF Service](pdfservice_md_guide.md) (Checklist)

## 6. Refactor Notes (2026-02-09)
- Phase 3 and Phase 4 refactors were internal-only (dead code cleanup and maintainability extraction).
- No additional external API path/schema contract changes were introduced in these phases.
- UUID validation for `doc_id` remains enforced on `/pdfmd/file/{doc_id}*` and `/multimodal/file/{doc_id}`.
