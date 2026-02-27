"""Regression test: research metadata is preserved in conversation detail API."""

from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from main import app

TEST_USER_ID = "test-user-123"
TEST_CONV_ID = str(uuid4())


@pytest.fixture
def client():
    """Test client with auth override and startup stubs."""
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
    ):
        app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
        with TestClient(app) as c:
            yield c
        app.dependency_overrides = {}


def test_get_research_conversation_returns_full_metadata(client: TestClient) -> None:
    research_metadata = {
        "summary": "This is a summary",
        "detailed_answer": "This is a detailed answer",
        "sub_tasks": [{"id": 1, "question": "Q1", "answer": "A1", "sources": ["doc1"]}],
        "confidence": 0.95,
        "all_sources": ["doc1"],
        "total_iterations": 1,
        "question": "Original Question",
    }
    returned = {
        "id": TEST_CONV_ID,
        "user_id": TEST_USER_ID,
        "title": "Research: Original Question",
        "type": "research",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "metadata": research_metadata,
        "messages": [],
    }

    with patch(
        "conversations.router.get_user_conversation_detail",
        new=AsyncMock(return_value=returned),
    ):
        response = client.get(f"/api/conversations/{TEST_CONV_ID}")

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "research"
    assert data["metadata"] == research_metadata
    assert data["metadata"]["summary"] == "This is a summary"
    assert data["metadata"]["sub_tasks"][0]["answer"] == "A1"
