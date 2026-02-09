import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from data_base.query_transformer import QueryTransformer, transform_query_with_hyde, transform_query_multi

@pytest.mark.asyncio
async def test_hyde_generation():
    """Verify HyDE document generation."""
    transformer = QueryTransformer()
    question = "SwinUNETR 的核心優勢是什麼？"
    
    with patch("data_base.query_transformer.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "SwinUNETR 結合了 Swin Transformer 的層次化特徵提取能力與 UNet 的對稱結構，這使其在 3D 醫學影像分割中表現優異..."
        mock_llm.ainvoke.return_value = mock_response
        
        hyde_doc = await transformer.generate_hyde_document(question)
        
        assert "SwinUNETR" in hyde_doc
        assert len(hyde_doc) > len(question)
        # Verify get_llm was called with correct purpose
        mock_get_llm.assert_called_with("query_rewrite")

@pytest.mark.asyncio
async def test_multi_query_generation():
    """Verify multi-query generation and parsing."""
    transformer = QueryTransformer()
    question = "比較 SwinUNETR 與 nnU-Net"
    
    with patch("data_base.query_transformer.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock LLM response with numbered list
        mock_response = MagicMock()
        mock_response.content = "1. SwinUNETR 的架構細節\n2. nnU-Net 的自動化配置特點\n3. 兩者在各類數據集上的表現對比"
        mock_llm.ainvoke.return_value = mock_response
        
        queries = await transformer.generate_multi_queries(question)
        
        # Original + 3 generated = 4
        assert len(queries) == 4
        assert queries[0] == question
        assert "SwinUNETR" in queries[1]
        assert "nnU-Net" in queries[2]

@pytest.mark.asyncio
async def test_transformer_fallbacks():
    """Verify fallback behavior on LLM failure."""
    QueryTransformer()
    question = "Test question"
    
    with patch("data_base.query_transformer.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.ainvoke.side_effect = Exception("LLM Error")
        
        # HyDE should fallback to original question
        hyde_doc = await transform_query_with_hyde(question)
        assert hyde_doc == question
        
        # Multi-query should fallback to [question]
        queries = await transform_query_multi(question)
        assert queries == [question]