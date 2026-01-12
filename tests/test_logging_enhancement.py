import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from experiments.evaluation_pipeline import EvaluationPipeline

@pytest.mark.asyncio
async def test_run_tier_logging_naive():
    """Test that Naive RAG returns thought_process and tool_calls"""
    pipeline = EvaluationPipeline()
    
    with patch("experiments.evaluation_pipeline.rag_answer_question") as mock_rag:
        # Mock result
        mock_result = MagicMock()
        mock_result.answer = "naive answer"
        mock_result.source_doc_ids = ["doc1"]
        mock_result.documents = [MagicMock(page_content="context1")]
        mock_result.usage = {"total_tokens": 10}
        mock_result.thought_process = "naive thought"
        mock_result.tool_calls = [{"action": "search"}]
        
        mock_rag.return_value = mock_result
        
        result = await pipeline.run_tier("Naive RAG", "q", "model")
        
        assert result["thought_process"] == "naive thought"
        assert result["tool_calls"] == [{"action": "search"}]
        assert result["retrieved_contexts"][0]["text"] == "context1"


@pytest.mark.asyncio
async def test_run_tier_logging_agentic():
    """Test that Full Agentic RAG aggregates diagnostics"""
    pipeline = EvaluationPipeline()
    
    with patch("experiments.evaluation_pipeline.get_deep_research_service") as mock_get_service, \
         patch("experiments.evaluation_pipeline.asyncio.sleep") as mock_sleep:
        
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_service.generate_plan = AsyncMock(return_value=MagicMock(sub_tasks=[]))
        
        # Mock subtasks with diagnostics
        mock_sub1 = MagicMock()
        mock_sub1.id = 1
        mock_sub1.contexts = []
        mock_sub1.usage = {}
        mock_sub1.thought_process = "thought 1"
        mock_sub1.tool_calls = [{"t": 1}]
        
        mock_sub2 = MagicMock()
        mock_sub2.id = 2
        mock_sub2.contexts = []
        mock_sub2.usage = {}
        mock_sub2.thought_process = "thought 2"
        mock_sub2.tool_calls = [{"t": 2}]
        
        mock_exec_res = MagicMock()
        mock_exec_res.sub_tasks = [mock_sub1, mock_sub2]
        mock_service.execute_plan = AsyncMock(return_value=mock_exec_res)
        
        result = await pipeline.run_tier("Full Agentic RAG", "q", "model")
        
        assert "thought 1" in result["thought_process"]
        assert "thought 2" in result["thought_process"]
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0] == {"t": 1}

def test_save_results_csv_diagnostics(tmp_path):
    """Test that save_results_csv includes new diagnostic fields"""
    pipeline = EvaluationPipeline()
    csv_path = tmp_path / "test.csv"
    
    results = {
        "model1": {
            "q1": {
                "Naive RAG": {
                    "answer": "ans",
                    "scores": {"faithfulness": 0.1},
                    "usage": {"total_tokens": 100},
                    "thought_process": "thought text",
                    "tool_calls": [{"a": 1}],
                    "behavior_pass": True
                }
            }
        }
    }
    
    pipeline.save_results_csv(results, str(csv_path))
    
    import csv
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["Thought_Process"] == "thought text"
        assert "a" in rows[0]["Tool_Calls"]
