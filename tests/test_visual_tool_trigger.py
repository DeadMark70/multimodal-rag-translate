import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from data_base.RAG_QA_service import rag_answer_question, _parse_visual_tool_request

def test_visual_tool_parsing():
    """Verify that the JSON instruction for VERIFY_IMAGE is correctly parsed."""
    # Test standard format
    response = 'Need more details. {"action": "VERIFY_IMAGE", "path": "uploads/img1.png", "question": "What is the x-axis?"}'
    parsed = _parse_visual_tool_request(response)
    assert parsed is not None
    assert parsed["action"] == "VERIFY_IMAGE"
    assert parsed["path"] == "uploads/img1.png"
    
    # Test markdown code block format
    response_md = '```json\n{"action": "VERIFY_IMAGE", "path": "uploads/img2.png", "question": "Value in 2024?"}\n```'
    parsed_md = _parse_visual_tool_request(response_md)
    assert parsed_md is not None
    assert parsed_md["path"] == "uploads/img2.png"

@pytest.mark.asyncio
async def test_visual_verification_trigger_logic():
    """Verify that rag_answer_question triggers the visual verification loop."""
    user_id = "c1bae279-c099-4c45-ba19-2bb393ca4e4b"
    question = "圖表中的具體數據是多少？"
    
    # Mock LLM to return a VERIFY_IMAGE request initially
    with patch("data_base.RAG_QA_service.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_get_llm.return_value = mock_llm
        
        # 1. First call returns the tool request
        # 2. Second call (synthesis) returns the final answer
        mock_llm.ainvoke.side_effect = [
            MagicMock(content='{"action": "VERIFY_IMAGE", "path": "test.png", "question": "Data?"}'),
            MagicMock(content='根據視覺分析，數據是 42。')
        ]
        
        # Mock retriever to return some docs with images
        with patch("data_base.RAG_QA_service.get_user_retriever") as mock_get_retriever:
            from langchain_core.documents import Document
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [
                Document(page_content="Img summary", metadata={"source": "image", "image_path": "test.png", "doc_id": "d1"})
            ]
            mock_get_retriever.return_value = mock_retriever
            
            # Mock the actual tool execution to avoid file system checks
            with patch("data_base.visual_tools.verify_image_details", new_callable=AsyncMock) as mock_tool:
                mock_tool.return_value = {"success": True, "result": "The value is 42"}
                
                # We need to mock os.path.exists for the image
                with patch("os.path.exists", return_value=True):
                    # Mock _encode_image to avoid reading real files
                    with patch("data_base.RAG_QA_service._encode_image", return_value="fake_base64"):
                        
                        answer, _ = await rag_answer_question(
                            question=question,
                            user_id=user_id,
                            enable_visual_verification=True
                        )
                        
                        assert "42" in answer
                        assert mock_tool.called
                        assert mock_llm.ainvoke.call_count == 2