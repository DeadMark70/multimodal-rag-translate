"""API tests for conversations router."""

from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from main import app

TEST_USER_ID = "test-user-123"
TEST_CONV_ID = str(uuid4())
TEST_MSG_ID = str(uuid4())


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


def test_list_conversations(client: TestClient) -> None:
    expected = [
        {
            "id": TEST_CONV_ID,
            "title": "Test Chat",
            "type": "chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {"tags": ["test"]},
        }
    ]
    with patch(
        "conversations.router.list_user_conversations",
        new=AsyncMock(return_value=expected),
    ) as mock_service:
        response = client.get("/api/conversations")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["title"] == "Test Chat"
    assert payload[0]["metadata"] == {"tags": ["test"]}
    mock_service.assert_awaited_once_with(user_id=TEST_USER_ID)


def test_create_conversation(client: TestClient) -> None:
    returned = {
        "id": TEST_CONV_ID,
        "title": "New Chat",
        "type": "chat",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "metadata": {"foo": "bar"},
    }
    with patch(
        "conversations.router.create_user_conversation",
        new=AsyncMock(return_value=returned),
    ) as mock_service:
        payload = {"title": "New Chat", "type": "chat", "metadata": {"foo": "bar"}}
        response = client.post("/api/conversations", json=payload)

    assert response.status_code == 201
    assert response.json()["title"] == "New Chat"
    assert response.json()["metadata"] == {"foo": "bar"}
    assert mock_service.await_count == 1


def test_get_conversation_details_with_messages(client: TestClient) -> None:
    returned = {
        "id": TEST_CONV_ID,
        "title": "Existing Chat",
        "type": "chat",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "metadata": {},
        "messages": [
            {
                "id": TEST_MSG_ID,
                "role": "user",
                "content": "Hello",
                "metadata": {"tokens": 10},
                "created_at": datetime.now().isoformat(),
            },
            {
                "id": str(uuid4()),
                "role": "assistant",
                "content": "Hi there",
                "metadata": {"tokens": 15},
                "created_at": datetime.now().isoformat(),
            },
        ],
    }
    with patch(
        "conversations.router.get_user_conversation_detail",
        new=AsyncMock(return_value=returned),
    ) as mock_service:
        response = client.get(f"/api/conversations/{TEST_CONV_ID}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == TEST_CONV_ID
    assert len(data["messages"]) == 2
    assert data["messages"][0]["content"] == "Hello"
    assert data["messages"][1]["role"] == "assistant"
    assert mock_service.await_count == 1


def test_create_message(client: TestClient) -> None:
    returned = {
        "id": str(uuid4()),
        "role": "user",
        "content": "New Message",
        "metadata": {"context": "test"},
        "created_at": datetime.now().isoformat(),
    }
    with patch(
        "conversations.router.create_conversation_message",
        new=AsyncMock(return_value=returned),
    ) as mock_service:
        payload = {
            "role": "user",
            "content": "New Message",
            "metadata": {"context": "test"},
        }
        response = client.post(f"/api/conversations/{TEST_CONV_ID}/messages", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert data["content"] == "New Message"
    assert data["role"] == "user"
    assert data["metadata"] == {"context": "test"}
    assert mock_service.await_count == 1
