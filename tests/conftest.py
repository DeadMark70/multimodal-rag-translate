"""
Pytest Configuration and Shared Fixtures

Provides common fixtures for Phase 1 unit and integration tests.
"""

# Standard library
import os
import sys
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party
import pytest
from langchain_core.documents import Document

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_embeddings():
    """Creates a mock HuggingFaceEmbeddings instance."""
    mock = MagicMock()
    
    # Mock embed_documents to return fake embeddings
    def fake_embed(texts: List[str]) -> List[List[float]]:
        """Generate fake embeddings that simulate semantic similarity."""
        embeddings = []
        for i, text in enumerate(texts):
            # Simple fake embedding: different values for different sentences
            base_val = hash(text) % 100 / 100.0
            embedding = [base_val + j * 0.01 for j in range(384)]  # 384-dim like bge-m3
            embeddings.append(embedding)
        return embeddings
    
    mock.embed_documents = fake_embed
    return mock


@pytest.fixture
def mock_llm():
    """Creates a mock LLM instance for testing."""
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content="Mock LLM response"))
    return mock


@pytest.fixture
def sample_documents():
    """Creates sample documents for testing."""
    return [
        Document(
            page_content="這是第一段測試文字。這段文字討論人工智慧的發展。",
            metadata={"page_number": 1, "original_doc_uid": "test-doc-1"}
        ),
        Document(
            page_content="Tesla 在 2023 年營收增長 20%，同時其在歐洲的市場份額有所下降。",
            metadata={"page_number": 2, "original_doc_uid": "test-doc-1"}
        ),
        Document(
            page_content="機器學習是人工智慧的一個分支。深度學習又是機器學習的一個分支。",
            metadata={"page_number": 3, "original_doc_uid": "test-doc-1"}
        ),
    ]


@pytest.fixture
def sample_markdown():
    """Sample markdown text for chunking tests."""
    return """[[PAGE_1]]
# 人工智慧導論

人工智慧（AI）是計算機科學的一個分支。它致力於創建能夠模擬人類智能的系統。

## 機器學習

機器學習是 AI 的核心技術之一。它讓電腦能從數據中學習。

[[PAGE_2]]
## 深度學習

深度學習使用多層神經網路。這種方法在圖像識別和自然語言處理中表現優異。

### 應用場景

1. 圖像識別
2. 語音識別
3. 自然語言處理
"""


@pytest.fixture
def temp_user_dir(tmp_path):
    """Creates a temporary user directory for testing."""
    user_dir = tmp_path / "uploads" / "test-user" / "rag_index"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


@pytest.fixture
def integration_test_user_id():
    """Returns the user ID for integration testing."""
    return "c1bae279-c099-4c45-ba19-2bb393ca4e4b"


@pytest.fixture
def target_documents():
    """Returns the list of target documents for testing."""
    return [
        "17f74b87-a50b-472d-a551-5b73035e58b5/SwinUNETR.pdf",
        "27f40556-b6a9-4744-a742-6e4815c14e42/nnU-Net Revisited.pdf"
    ]


@pytest.fixture
def conflict_resolution_question():
    """Returns the standard conflict resolution question for this track."""
    return "SwinUNETR 與 nnU-Net 在醫學影像分割任務上，誰的表現更好？請根據文獻中的實驗數據進行比較。"


# ============================================================================
# Async Helpers
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Patch Helpers
# ============================================================================

@pytest.fixture
def patch_llm_factory(mock_llm):
    """Patches the LLM factory to return mock LLM."""
    with patch("core.llm_factory.get_llm", return_value=mock_llm):
        yield mock_llm
