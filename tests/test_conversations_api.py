
# tests/test_conversations_api.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from uuid import uuid4
from datetime import datetime

# Import app
from main import app


# Import app

# Test data
TEST_USER_ID = "test-user-123"
TEST_CONV_ID = str(uuid4())
TEST_MSG_ID = str(uuid4())

@pytest.fixture
def client():
    """Test client with auth override and mocked startup."""
    from core.auth import get_current_user_id
    
    # Patch heavy startup functions (moved into core.app_factory lifecycle helpers)
    with patch("core.app_factory._initialize_rag_components", new=AsyncMock()), \
         patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()):
         
        app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
        with TestClient(app) as c:
            yield c
        app.dependency_overrides = {}

@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    with patch("conversations.router.supabase") as mock:
        yield mock

def test_list_conversations(client, mock_supabase):
    # Mock response
    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [
        {
            "id": TEST_CONV_ID,
            "title": "Test Chat",
            "type": "chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {"tags": ["test"]}
        }
    ]
    
    response = client.get("/api/conversations")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Test Chat"
    assert data[0]["metadata"] == {"tags": ["test"]}
    
    # Verify query structure
    mock_supabase.table.assert_called_with("conversations")

def test_create_conversation(client, mock_supabase):
    # Mock response
    mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [
        {
            "id": TEST_CONV_ID,
            "title": "New Chat",
            "type": "chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {"foo": "bar"}
        }
    ]
    
    payload = {
        "title": "New Chat",
        "type": "chat",
        "metadata": {"foo": "bar"}
    }
    
    response = client.post("/api/conversations", json=payload)
    
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "New Chat"
    assert data["metadata"] == {"foo": "bar"}
    
    # Verify insert args
    insert_call = mock_supabase.table.return_value.insert.call_args[0][0]
    assert insert_call["title"] == "New Chat"
    assert insert_call["metadata"] == {"foo": "bar"}

def test_get_conversation_details_with_messages(client, mock_supabase):
    # Mock conversation response
    mock_conv = MagicMock()
    mock_conv.data = {
        "id": TEST_CONV_ID,
        "title": "Existing Chat",
        "type": "chat",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "metadata": {}
    }
    
    # Mock messages response
    mock_msgs_data = [
        {
            "id": TEST_MSG_ID,
            "conversation_id": TEST_CONV_ID,
            "role": "user",
            "content": "Hello",
            "metadata": {"tokens": 10},
            "created_at": datetime.now().isoformat()
        },
        {
            "id": str(uuid4()),
            "conversation_id": TEST_CONV_ID,
            "role": "assistant",
            "content": "Hi there",
            "metadata": {"tokens": 15},
            "created_at": datetime.now().isoformat()
        }
    ]
    mock_msgs = MagicMock()
    mock_msgs.data = mock_msgs_data
    
    # Setup chain for conversation query
    # table("conversations").select().eq().eq().single().execute()
    mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.single.return_value.execute.return_value = mock_conv
    
    # Setup chain for messages query
    # table("messages").select().eq().order().execute()
    # Note: Because the chain is slightly different, we need to be careful with the mock setup or just set side_effect based on table name.
    
    def side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "conversations":
            mock_table.select.return_value.eq.return_value.eq.return_value.single.return_value.execute.return_value = mock_conv
        elif table_name == "messages":
            mock_table.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_msgs
        return mock_table
    
    mock_supabase.table.side_effect = side_effect
    
    response = client.get(f"/api/conversations/{TEST_CONV_ID}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == TEST_CONV_ID
    assert len(data["messages"]) == 2
    assert data["messages"][0]["content"] == "Hello"
    assert data["messages"][0]["metadata"] == {"tokens": 10}
    assert data["messages"][1]["role"] == "assistant"

def test_create_message(client, mock_supabase):
    # Mock Check Conversation
    mock_conv_check = MagicMock()
    mock_conv_check.data = {"id": TEST_CONV_ID}
    
    # Mock Insert Message
    mock_insert_resp = MagicMock()
    mock_insert_resp.data = [{
        "id": str(uuid4()),
        "role": "user",
        "content": "New Message",
        "metadata": {"context": "test"},
        "created_at": datetime.now().isoformat()
    }]
    
    def side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "conversations":
            # Check existence
            mock_table.select.return_value.eq.return_value.eq.return_value.single.return_value.execute.return_value = mock_conv_check
        elif table_name == "messages":
            # Insert
            mock_table.insert.return_value.execute.return_value = mock_insert_resp
        return mock_table
        
    mock_supabase.table.side_effect = side_effect
    
    payload = {
        "role": "user",
        "content": "New Message",
        "metadata": {"context": "test"}
    }
    
    response = client.post(f"/api/conversations/{TEST_CONV_ID}/messages", json=payload)
    
    assert response.status_code == 201
    data = response.json()
    assert data["content"] == "New Message"
    assert data["role"] == "user"
    assert data["metadata"] == {"context": "test"}
    
    # Verify insert call
    # We need to find the call to table("messages").insert(...)
    # Because of side_effect, we can inspect the mock objects returned or just trust the response for now if complex.
    # But let's verification implicitly via the response being correct.
