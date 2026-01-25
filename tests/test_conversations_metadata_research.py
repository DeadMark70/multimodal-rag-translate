import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from uuid import uuid4
from datetime import datetime
import json

# Import app
from main import app
from core.auth import get_current_user_id

# Test data
TEST_USER_ID = "test-user-123"
TEST_CONV_ID = str(uuid4())

@pytest.fixture
def client():
    """Test client with auth override."""
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}

@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    with patch("conversations.router.supabase") as mock:
        yield mock

def test_get_research_conversation_returns_full_metadata(client, mock_supabase):
    """
    Test that GET /api/conversations/{id} returns the full research metadata.
    """
    # Mock research metadata matching ExecutePlanResponse structure
    research_metadata = {
        "summary": "This is a summary",
        "detailed_answer": "This is a detailed answer",
        "sub_tasks": [
            {"id": 1, "question": "Q1", "answer": "A1", "sources": ["doc1"]}
        ],
        "confidence": 0.95,
        "all_sources": ["doc1"],
        "total_iterations": 1,
        "question": "Original Question"
    }
    
    # Mock conversation response
    mock_conv = MagicMock()
    mock_conv.data = {
        "id": TEST_CONV_ID,
        "user_id": TEST_USER_ID,
        "title": "Research: Original Question",
        "type": "research",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "metadata": research_metadata
    }
    
    # Mock messages response (empty for this test)
    mock_msgs = MagicMock()
    mock_msgs.data = []
    
    def side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "conversations":
            mock_table.select.return_value.eq.return_value.eq.return_value.single.return_value.execute.return_value = mock_conv
        elif table_name == "messages":
            mock_table.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_msgs
        return mock_table
    
    mock_supabase.table.side_effect = side_effect
    
    # API prefix from main.py is /api/conversations
    response = client.get(f"/api/conversations/{TEST_CONV_ID}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "research"
    assert data["metadata"] == research_metadata
    assert data["metadata"]["summary"] == "This is a summary"
    assert len(data["metadata"]["sub_tasks"]) == 1
    assert data["metadata"]["sub_tasks"][0]["answer"] == "A1"

if __name__ == "__main__":
    import sys
    pytest.main([__file__])
