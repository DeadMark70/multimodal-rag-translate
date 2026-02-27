"""Contract tests for v3 normalization changes."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from pdfserviceMD.schemas import UploadPdfResponse
from pdfserviceMD.service import UploadPipelineContext

TEST_USER_ID = "test-user-123"


@contextmanager
def _build_client(with_auth: bool = True):
    """Builds test client with startup stubs."""
    from core.auth import get_current_user_id

    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
    ):
        if with_auth:
            app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
        with TestClient(app) as client:
            yield client
    app.dependency_overrides = {}


def test_get_rag_ask_is_removed() -> None:
    """GET /rag/ask should not be available after route unification."""
    with _build_client(with_auth=True) as client:
        response = client.get("/rag/ask?question=hello")
        assert response.status_code == 405
        payload = response.json()
        assert "error" in payload
        assert payload["error"]["code"] == "BAD_REQUEST"
        assert isinstance(payload["error"]["message"], str)
        assert payload["error"]["request_id"]


def test_upload_pdf_returns_json_contract() -> None:
    """Upload endpoint must return JSON response envelope, not binary blob."""
    context = UploadPipelineContext(
        doc_id="doc-123",
        markdown_text="content",
        book_title="demo",
        user_folder="uploads/test-user-123/doc-123",
        response=UploadPdfResponse(
            doc_id="doc-123",
            status="completed",
            message="Upload accepted. Background indexing started.",
            pdf_available=True,
            pdf_download_url="/pdfmd/file/doc-123",
            pdf_error=None,
            rag_status="processing_background",
        ),
    )

    with _build_client(with_auth=True) as client, patch(
        "pdfserviceMD.router.run_upload_pipeline", new=AsyncMock(return_value=context)
    ), patch("pdfserviceMD.router.run_post_processing_tasks", new=AsyncMock()):
        response = client.post(
            "/pdfmd/upload_pdf_md",
            files={"file": ("demo.pdf", b"dummy-pdf", "application/pdf")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["doc_id"] == "doc-123"
        assert payload["pdf_available"] is True
        assert payload["pdf_download_url"] == "/pdfmd/file/doc-123"


def test_openapi_contains_http_bearer_security_scheme() -> None:
    """OpenAPI should expose HTTP bearer auth scheme for protected routes."""
    with _build_client(with_auth=False) as client:
        openapi = client.get("/openapi.json").json()
        security_schemes = openapi["components"]["securitySchemes"]
        assert "HTTPBearer" in security_schemes
        assert security_schemes["HTTPBearer"]["type"] == "http"
        assert security_schemes["HTTPBearer"]["scheme"] == "bearer"


def test_validation_error_uses_error_envelope() -> None:
    """422 responses should use standardized error envelope."""
    with _build_client(with_auth=True) as client:
        response = client.get("/pdfmd/file/not-a-uuid/status")
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "VALIDATION_ERROR"
        assert payload["error"]["message"] == "Request validation failed"
        assert payload["error"]["request_id"]
        assert "details" in payload["error"]
