"""
Unit Tests for EvaluationPipeline Module

Tests the initialization and core structure of the EvaluationPipeline.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

class TestEvaluationPipelineStructure:
    """Tests for EvaluationPipeline structure."""

    def test_initialization(self):
        """Tests that EvaluationPipeline can be initialized."""
        try:
            from experiments.evaluation_pipeline import EvaluationPipeline
        except ImportError:
            pytest.fail("Could not import EvaluationPipeline from experiments.evaluation_pipeline")
        
        pipeline = EvaluationPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, "models")
        assert hasattr(pipeline, "tiers")
        assert len(pipeline.models) == 5
        assert len(pipeline.tiers) == 5

    def test_models_list(self):
        """Tests that the pipeline supports the required models."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline()
        expected_models = [
            "gemma-3-27b",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-flash"
        ]
        
        for model in expected_models:
            assert model in pipeline.models

    def test_tiers_list(self):
        """Tests that the pipeline supports the required tiers."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline()
        expected_tiers = [
            "Naive RAG",
            "Advanced RAG",
            "Graph RAG",
            "Long Context Mode",
            "Full Agentic RAG"
        ]
        
        for tier in expected_tiers:
            assert tier in pipeline.tiers

class TestTokenMonitoring:
    """Tests for token monitoring logic."""

    def test_extract_token_usage_success(self):
        """Tests that token usage is correctly extracted from a response object."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from langchain_core.messages import AIMessage
        
        pipeline = EvaluationPipeline()
        
        # Mock response with usage_metadata
        mock_response = AIMessage(
            content="Test response",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30
            }
        )
        
        usage = pipeline.extract_token_usage(mock_response)
        
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_extract_token_usage_missing(self):
        """Tests that missing usage_metadata returns zeros."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from langchain_core.messages import AIMessage
        
        pipeline = EvaluationPipeline()
        
        mock_response = AIMessage(content="No usage info")
        
        usage = pipeline.extract_token_usage(mock_response)
        
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == 0

class TestBenchmarkQuestions:
    """Tests for benchmark questions loading."""

    def test_benchmark_file_exists(self):
        """Tests that experiments/benchmark_questions.json exists and is valid JSON."""
        import json
        import os
        
        file_path = "experiments/benchmark_questions.json"
        assert os.path.exists(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
            
            # Check structure of first item
            item = data[0]
            assert "question" in item
            assert "ground_truth" in item
            assert "type" in item

class TestMockRAG:
    """Tests for mock RAG functionality."""

    @pytest.mark.asyncio
    async def test_mock_rag_answer(self):
        """Tests that mock_rag_answer returns consistent mocked results."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        pipeline = EvaluationPipeline()
        
        question = "Test question"
        answer, contexts = await pipeline.mock_rag_answer(question)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(contexts, list)
        assert len(contexts) > 0
        assert "Mocked context" in contexts[0]

class TestTierExecution:
    """Tests for individual tier execution."""

    @pytest.mark.asyncio
    async def test_run_tier_naive_rag(self):
        """Tests execution of Naive RAG tier."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from data_base.RAG_QA_service import RAGResult
        from langchain_core.documents import Document
        
        pipeline = EvaluationPipeline()
        mock_result = RAGResult(
            answer="Naive answer",
            source_doc_ids=["doc1"],
            documents=[Document(page_content="Naive context")]
        )
        
        with patch("experiments.evaluation_pipeline.rag_answer_question", return_value=mock_result) as mock_rag:
            res = await pipeline.run_tier("Naive RAG", "Question", "gemini-2.0-flash")
            
            assert res["answer"] == "Naive answer"
            assert res["contexts"] == ["Naive context"]
            mock_rag.assert_called_once()
            # Check params
            args, kwargs = mock_rag.call_args
            assert kwargs["enable_reranking"] is False
            assert kwargs["enable_hyde"] is False

    @pytest.mark.asyncio
    async def test_run_tier_advanced_rag(self):
        """Tests execution of Advanced RAG tier."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from data_base.RAG_QA_service import RAGResult
        from langchain_core.documents import Document
        
        pipeline = EvaluationPipeline()
        mock_result = RAGResult(
            answer="Advanced answer",
            source_doc_ids=["doc1"],
            documents=[Document(page_content="Advanced context")]
        )
        
        with patch("experiments.evaluation_pipeline.rag_answer_question", return_value=mock_result) as mock_rag:
            res = await pipeline.run_tier("Advanced RAG", "Question", "gemini-2.0-flash")
            
            assert res["answer"] == "Advanced answer"
            mock_rag.assert_called_once()
            args, kwargs = mock_rag.call_args
            assert kwargs["enable_reranking"] is True
            assert kwargs["enable_hyde"] is True

    @pytest.mark.asyncio
    async def test_run_tier_graph_rag(self):
        """Tests execution of Graph RAG tier."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from data_base.RAG_QA_service import RAGResult
        from langchain_core.documents import Document
        
        pipeline = EvaluationPipeline()
        mock_result = RAGResult(
            answer="Graph answer",
            source_doc_ids=["doc1"],
            documents=[Document(page_content="Graph context")]
        )
        
        with patch("experiments.evaluation_pipeline.rag_answer_question", return_value=mock_result) as mock_rag:
            res = await pipeline.run_tier("Graph RAG", "Question", "gemini-2.0-flash")
            
            assert res["answer"] == "Graph answer"
            mock_rag.assert_called_once()
            args, kwargs = mock_rag.call_args
            assert kwargs["enable_graph_rag"] is True

    @pytest.mark.asyncio
    async def test_run_tier_long_context(self):
        """Tests execution of Long Context Mode tier."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        from langchain_core.messages import AIMessage
        
        pipeline = EvaluationPipeline()
        
        # Mocking getting full text and LLM call
        mock_full_text = "This is the full text of all PDFs."
        mock_response = AIMessage(content="Long context answer")
        
        with patch("experiments.evaluation_pipeline.EvaluationPipeline.get_user_full_text", return_value=mock_full_text):
            with patch("experiments.evaluation_pipeline.get_llm") as mock_get_llm:
                mock_llm = AsyncMock()
                mock_llm.ainvoke.return_value = mock_response
                mock_get_llm.return_value = mock_llm
                
                res = await pipeline.run_tier("Long Context Mode", "Question", "gemini-2.0-flash")
                
                assert res["answer"] == "Long context answer"
                assert res["contexts"] == [mock_full_text]
                mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_tier_full_agentic(self):
        """Tests execution of Full Agentic RAG tier."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline()
        
        mock_service = MagicMock()
        mock_service.generate_plan = AsyncMock()
        mock_service.execute_plan = AsyncMock()
        
        mock_plan = MagicMock()
        mock_plan.sub_tasks = []
        mock_service.generate_plan.return_value = mock_plan
        
        mock_result = MagicMock()
        mock_result.detailed_answer = "Agentic answer"
        mock_result.summary = "Summary"
        mock_result.all_sources = ["source1"]
        mock_service.execute_plan.return_value = mock_result
        
        with patch("experiments.evaluation_pipeline.get_deep_research_service", return_value=mock_service):
            with patch("asyncio.sleep", return_value=None) as mock_sleep:
                res = await pipeline.run_tier("Full Agentic RAG", "Question", "gemini-2.0-flash")
                
                assert res["answer"] == "Agentic answer"
                assert res["source_doc_ids"] == ["source1"]
                mock_service.generate_plan.assert_called_once()
                mock_service.execute_plan.assert_called_once()
                mock_sleep.assert_called_with(60) # Verify 1-minute pause

class TestReporting:
    """Tests for report generation."""

    def test_save_results_json(self):
        """Tests saving results to JSON."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        import json
        import os
        
        pipeline = EvaluationPipeline()
        results = {
            "test_run": {
                "answer": "Test Answer",
                "scores": {"faithfulness": 1.0}
            }
        }
        
        output_path = "tests/test_results.json"
        pipeline.save_results_json(results, output_path)
        
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["test_run"]["answer"] == "Test Answer"
        
        # Cleanup
        os.remove(output_path)

    def test_save_results_csv(self):
        """Tests saving results to CSV."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        import csv
        import os
        
        pipeline = EvaluationPipeline()
        
        # Nested structure mimicking the real report
        results = {
            "model_1": {
                "question_1": {
                    "Naive RAG": {
                        "answer": "Ans",
                        "scores": {"faithfulness": 0.9, "answer_correctness": 0.8},
                        "usage": {"total_tokens": 100},
                        "behavior_pass": True
                    }
                }
            }
        }
        
        
        output_path = "tests/test_results.csv"
        pipeline.save_results_csv(results, output_path)
        
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["Model"] == "model_1"
            assert rows[0]["Tier"] == "Naive RAG"
            assert rows[0]["Behavior_Pass"] == "True"
            assert float(rows[0]["Faithfulness"]) == 0.9
        
        # Cleanup
        os.remove(output_path)

class TestBehavioralCheck:

    """Tests for behavioral validation."""



    def test_check_visual_verification_pass(self):

        """Tests that behavior check passes when tool usage is detected."""

        from experiments.evaluation_pipeline import EvaluationPipeline

        

        pipeline = EvaluationPipeline()

        

        # Simulating a result that indicates tool usage

        result = {

            "answer": "Answer based on visual tool.",

            "tool_usage": ["visual_verification"] 

        }

        

        passed = pipeline.check_visual_verification(result)

        assert passed is True



    def test_check_visual_verification_fail(self):

        """Tests that behavior check fails when tool usage is missing."""

        from experiments.evaluation_pipeline import EvaluationPipeline

        

        pipeline = EvaluationPipeline()

        

        result = {

            "answer": "Standard answer.",

            "tool_usage": []

        }

        

        passed = pipeline.check_visual_verification(result)

        assert passed is False



    @pytest.mark.asyncio

    async def test_run_full_evaluation(self):

        """Tests the full evaluation loop with mocks."""

        from experiments.evaluation_pipeline import EvaluationPipeline

        import os

        import json

        

        pipeline = EvaluationPipeline()

        # Limit models and tiers for faster test

        pipeline.models = ["m1"]

        pipeline.tiers = ["Naive RAG"]

        

        questions = [{"question": "q1", "ground_truth": "gt1", "type": "standard"}]

        q_path = "tests/test_questions.json"

        with open(q_path, "w", encoding="utf-8") as f:

            json.dump(questions, f)

            

        mock_tier_res = {"answer": "ans", "contexts": ["ctx"], "usage": {"total_tokens": 10}}

        mock_scores = {"faithfulness": 1.0, "answer_correctness": 1.0}

        

        with patch("experiments.evaluation_pipeline.EvaluationPipeline.run_tier", return_value=mock_tier_res):

            with patch("experiments.evaluation_pipeline.EvaluationPipeline.calculate_ragas_metrics", return_value=mock_scores):

                with patch("experiments.evaluation_pipeline.on_startup_rag_init", return_value=None):

                    output_prefix = "tests/full_eval"

                    res = await pipeline.run_full_evaluation(q_path, output_prefix)

                    

                    assert "m1" in res

                    assert "q1" in res["m1"]

                    assert res["m1"]["q1"]["Naive RAG"]["scores"]["faithfulness"] == 1.0

                    

                    # Verify files created

                    files = os.listdir("tests")

                    assert any(f.startswith("full_eval") and f.endswith(".json") for f in files)

                    assert any(f.startswith("full_eval") and f.endswith(".csv") for f in files)

                    

                    # Cleanup

                    for f in files:

                        if f.startswith("full_eval"):

                            os.remove(os.path.join("tests", f))

        

        os.remove(q_path)

    @pytest.mark.asyncio
    async def test_full_loop_behavior_pass(self):
        """Tests that Behavior_Pass is correctly recorded in CSV after full evaluation."""
        from experiments.evaluation_pipeline import EvaluationPipeline
        import os
        import json
        import csv
        
        pipeline = EvaluationPipeline()
        pipeline.models = ["m1"]
        pipeline.tiers = ["Full Agentic RAG"]
        
        # A visual verification question
        questions = [{"question": "q_visual", "ground_truth": "gt", "type": "visual_verification"}]
        q_path = "tests/test_questions_visual.json"
        with open(q_path, "w", encoding="utf-8") as f:
            json.dump(questions, f)
            
        # Mock result that SHOULD pass behavioral check (contains keyword)
        mock_tier_res = {
            "answer": "根據視覺查證, Figure 1 shows...", 
            "contexts": ["ctx"], 
            "usage": {"total_tokens": 10}
        }
        mock_scores = {"faithfulness": 1.0, "answer_correctness": 1.0}
        
        with patch("experiments.evaluation_pipeline.EvaluationPipeline.run_tier", return_value=mock_tier_res):
            with patch("experiments.evaluation_pipeline.EvaluationPipeline.calculate_ragas_metrics", return_value=mock_scores):
                with patch("experiments.evaluation_pipeline.on_startup_rag_init", return_value=None):
                    with patch("asyncio.sleep", return_value=None):
                        output_prefix = "tests/behavior_eval"
                        await pipeline.run_full_evaluation(q_path, output_prefix)
                        
                        # Find the CSV file
                        files = os.listdir("tests")
                        csv_file = next(f for f in files if f.startswith("behavior_eval") and f.endswith(".csv"))
                        
                        with open(os.path.join("tests", csv_file), "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            assert len(rows) == 1
                            assert rows[0]["Behavior_Pass"] == "True" # Keyword detection worked
                        
                        # Cleanup
                        for f in files:
                            if f.startswith("behavior_eval"):
                                os.remove(os.path.join("tests", f))
        
        os.remove(q_path)


            
            
