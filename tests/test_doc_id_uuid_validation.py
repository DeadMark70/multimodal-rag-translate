"""UUID path parameter validation tests for document APIs."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from main import app

TEST_USER_ID = "test-user-123"
INVALID_DOC_ID = "not-a-uuid"


@pytest.fixture
def client():
    """Test client with auth override and startup stubs."""
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
    ):
        app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
        with TestClient(app) as test_client:
            yield test_client
        app.dependency_overrides = {}


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", f"/pdfmd/file/{INVALID_DOC_ID}/status"),
        ("get", f"/pdfmd/file/{INVALID_DOC_ID}"),
        ("delete", f"/pdfmd/file/{INVALID_DOC_ID}"),
        ("get", f"/pdfmd/file/{INVALID_DOC_ID}/summary"),
        ("post", f"/pdfmd/file/{INVALID_DOC_ID}/summary/regenerate"),
        ("delete", f"/multimodal/file/{INVALID_DOC_ID}"),
    ],
)
def test_invalid_doc_id_returns_422(client: TestClient, method: str, path: str) -> None:
    """Invalid UUID path params should be rejected by FastAPI validation."""
    response = getattr(client, method)(path)
    assert response.status_code == 422


def test_valid_doc_id_status_endpoint_not_422(client: TestClient) -> None:
    """Valid UUID should reach endpoint logic instead of path-validation failure."""
    valid_doc_id = str(uuid4())
    mock_result = MagicMock()
    mock_result.data = None

    with patch("pdfserviceMD.router.supabase") as mock_supabase:
        (
            mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.single.return_value.execute.return_value
        ) = mock_result
        response = client.get(f"/pdfmd/file/{valid_doc_id}/status")

    assert response.status_code == 404


def test_valid_doc_id_multimodal_delete_not_422(client: TestClient) -> None:
    """Valid UUID delete should execute handler and not fail validation."""
    valid_doc_id = str(uuid4())
    with patch(
        "multimodal_rag.router.delete_document_from_knowledge_base", return_value=None
    ):
        response = client.delete(f"/multimodal/file/{valid_doc_id}")
    assert response.status_code == 200
