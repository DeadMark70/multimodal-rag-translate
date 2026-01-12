"""
Evaluation Pipeline Module

This module defines the EvaluationPipeline class, which orchestrates the
comparative evaluation of RAG models and configurations using Ragas metrics
and tiered benchmarking.
"""

import logging
import asyncio
from typing import List, Dict, Any

from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_correctness
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset

from core.llm_factory import get_llm, set_session_model_override
from data_base.RAG_QA_service import rag_answer_question, RAGResult
from data_base.deep_research_service import get_deep_research_service
from data_base.schemas_deep_research import ExecutePlanRequest

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """
    Orchestrates the evaluation process for Multimodal Agentic RAG.
    
    Supports ablation studies across multiple models and tiers, integrating
    Ragas for objective metric calculation.
    """
    
    def __init__(self, user_id: str = "c1bae279-c099-4c45-ba19-2bb393ca4e4b"):
        self.user_id = user_id
        self.models: List[str] = [
            "gemma-3-27b",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-flash"
        ]
        self.tiers: List[str] = [
            "Naive RAG",
            "Advanced RAG",
            "Graph RAG",
            "Long Context Mode",
            "Full Agentic RAG"
        ]
        # Evaluator model for Ragas
        self.evaluator_model = "gemini-3-pro-preview" 

    async def run_tier(self, tier: str, question: str, model_name: str) -> Dict[str, Any]:
        """
        Executes a single evaluation tier for a given question and model.
        
        Args:
            tier: The name of the tier to run.
            question: The question to ask.
            model_name: The LLM model to use.
            
        Returns:
            A dictionary containing the answer, contexts, and usage metadata.
        """
        logger.info(f"Running tier '{tier}' for model '{model_name}'...")
        
        # Set session model override
        set_session_model_override(model_name)
        
        try:
            if tier == "Naive RAG":
                # Tier 1: Naive RAG (Weak Baseline)
                result = await rag_answer_question(
                    question=question,
                    user_id=self.user_id,
                    enable_reranking=False,
                    enable_hyde=False,
                    enable_multi_query=False,
                    enable_graph_rag=False,
                    enable_visual_verification=False,
                    return_docs=True
                )
                
                return {
                    "answer": result.answer,
                    "contexts": [d.page_content for d in result.documents],
                    "source_doc_ids": result.source_doc_ids,
                    "usage": {"total_tokens": 0} # Placeholder
                }

            elif tier == "Advanced RAG":
                # Tier 2: Advanced RAG (Strong Baseline)
                result = await rag_answer_question(
                    question=question,
                    user_id=self.user_id,
                    enable_reranking=True,
                    enable_hyde=True,
                    enable_multi_query=True,
                    enable_graph_rag=False,
                    enable_visual_verification=False,
                    return_docs=True
                )
                return {
                    "answer": result.answer,
                    "contexts": [d.page_content for d in result.documents],
                    "source_doc_ids": result.source_doc_ids,
                    "usage": {"total_tokens": 0}
                }

            elif tier == "Graph RAG":
                # Tier 3: Graph RAG (Structured Enhanced)
                # Same as Advanced RAG + enable_graph_rag=True
                result = await rag_answer_question(
                    question=question,
                    user_id=self.user_id,
                    enable_reranking=True,
                    enable_hyde=True,
                    enable_multi_query=True,
                    enable_graph_rag=True,
                    enable_visual_verification=False,
                    return_docs=True
                )
                return {
                    "answer": result.answer,
                    "contexts": [d.page_content for d in result.documents],
                    "source_doc_ids": result.source_doc_ids,
                    "usage": {"total_tokens": 0}
                }
            
            return {"error": f"Tier {tier} not implemented"}
        finally:
            # Clear override after run
            set_session_model_override(None)

    def extract_token_usage(self, response) -> dict:
        """
        Extracts token usage metadata from a LangChain response object.
        
        Args:
            response: The response object from an LLM call.
            
        Returns:
            A dictionary containing input_tokens, output_tokens, and total_tokens.
        """
        usage = getattr(response, "usage_metadata", {})
        if not usage:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
        
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }

    async def calculate_ragas_metrics(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str], 
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Calculates Faithfulness and Answer Correctness using Ragas.
        
        Args:
            question: The input question.
            answer: The generated answer.
            contexts: List of retrieved context strings.
            ground_truth: The reference answer.
            
        Returns:
            A dictionary with metric names and their scores.
        """
        logger.info(f"Calculating Ragas metrics for question: {question[:50]}...")
        
        # Prepare data for Ragas
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth]
        }
        dataset = Dataset.from_dict(data)
        
        # Wrap the evaluator LLM
        # Note: Using gemini-3-pro-preview as requested in spec if possible
        # For now using get_llm with the configured evaluator model
        evaluator_llm = get_llm("evaluator", model_name="gemini-3-pro-preview")
        ragas_llm = LangchainLLMWrapper(evaluator_llm)
        
        try:
            # Run evaluation
            # Use a thread pool or run_in_executor if evaluate is blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: evaluate(
                    dataset=dataset,
                    metrics=[faithfulness, answer_correctness],
                    llm=ragas_llm
                )
            )
            
            # Extract scores (ragas result is a Result object that acts like a dict)
            return {
                "faithfulness": float(result.get("faithfulness", 0.0)),
                "answer_correctness": float(result.get("answer_correctness", 0.0))
            }
        except Exception as e:
            logger.error(f"Error calculating Ragas metrics: {e}")
            return {
                "faithfulness": 0.0,
                "answer_correctness": 0.0,
                "error": str(e)
            }

    async def mock_rag_answer(self, question: str) -> tuple[str, List[str]]:
        """
        Simulates a RAG response for testing purposes.
        
        Args:
            question: The input question.
            
        Returns:
            A tuple of (mocked_answer, mocked_contexts).
        """
        # Simple rule-based mock for testing
        mocked_answer = f"This is a mocked answer to: {question}. It contains relevant information."
        mocked_contexts = [
            f"Mocked context 1 for {question}: Medical imaging is a field of medicine.",
            f"Mocked context 2 for {question}: nnU-Net is a popular framework."
        ]
        return mocked_answer, mocked_contexts
