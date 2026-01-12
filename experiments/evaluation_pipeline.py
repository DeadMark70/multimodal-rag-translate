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
from ragas.metrics import faithfulness, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from core.llm_factory import get_llm, set_session_model_override
from data_base.RAG_QA_service import rag_answer_question, RAGResult
from data_base.deep_research_service import get_deep_research_service
from data_base.schemas_deep_research import ExecutePlanRequest
from data_base.router import on_startup_rag_init
from data_base.vector_store_manager import get_embeddings

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
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash-lite"
        ]
        self.tiers: List[str] = [
            "Naive RAG",
            "Advanced RAG",
            "Graph RAG",
            "Long Context Mode",
            "Full Agentic RAG"
        ]
        # Evaluator model for Ragas
        self.evaluator_model = "gemini-3-flash-preview" 

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
                    "retrieved_contexts": [
                        {"text": d.page_content, "metadata": d.metadata} 
                        for d in result.documents
                    ],
                    "source_doc_ids": result.source_doc_ids,
                    "usage": result.usage or {"total_tokens": 0},
                    "thought_process": getattr(result, "thought_process", None),
                    "tool_calls": getattr(result, "tool_calls", [])
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
                    "retrieved_contexts": [
                        {"text": d.page_content, "metadata": d.metadata} 
                        for d in result.documents
                    ],
                    "source_doc_ids": result.source_doc_ids,
                    "usage": result.usage or {"total_tokens": 0},
                    "thought_process": getattr(result, "thought_process", None),
                    "tool_calls": getattr(result, "tool_calls", [])
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
                    "retrieved_contexts": [
                        {"text": d.page_content, "metadata": d.metadata} 
                        for d in result.documents
                    ],
                    "source_doc_ids": result.source_doc_ids,
                    "usage": result.usage or {"total_tokens": 0},
                    "thought_process": getattr(result, "thought_process", None),
                    "tool_calls": getattr(result, "tool_calls", [])
                }


            elif tier == "Long Context Mode":
                # Tier 4: Long Context Mode (Context Stuffing)
                # Read all PDFs and feed directly to LLM
                full_text = self.get_user_full_text()
                if not full_text:
                    return {"error": "No documents found for user"}
                
                llm = get_llm("rag_qa", model_name=model_name)
                prompt = f"""以下是所有相關文獻的完整內容：

{full_text}

請根據以上資料回答問題：
{question}

請以繁體中文回答。"""
                
                from langchain_core.messages import HumanMessage
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                
                return {
                    "answer": response.content,
                    "contexts": [full_text],
                    "retrieved_contexts": [{"text": full_text, "metadata": {"source": "all_docs"}}],
                    "source_doc_ids": [], # We don't have specific IDs here as we sent everything
                    "usage": self.extract_token_usage(response),
                    "thought_process": None,
                    "tool_calls": []
                }

            elif tier == "Full Agentic RAG":
                # Tier 5: Full Agentic RAG (Ours - Ultimate)
                # Rate limit management: 1-minute pause as requested
                logger.info("Tier 5: Applying 1-minute rate-limit pause before execution...")
                await asyncio.sleep(60)
                
                service = get_deep_research_service()
                
                # Step 1: Generate Plan
                plan_res = await service.generate_plan(
                    question, 
                    self.user_id, 
                    enable_graph_planning=True
                )
                
                # Step 2: Execute Plan
                exec_request = ExecutePlanRequest(
                    original_question=question,
                    sub_tasks=plan_res.sub_tasks,
                    enable_drilldown=True,
                    max_iterations=1,
                    enable_reranking=True,
                    enable_deep_image_analysis=True
                )
                
                exec_res = await service.execute_plan(exec_request, self.user_id)
                
                # Aggregate contexts, token usage, and diagnostics from all subtasks
                all_contexts = []
                all_retrieved_contexts = []
                aggregated_usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
                all_thoughts = []
                all_tool_calls = []
                
                for subtask in exec_res.sub_tasks:
                    all_contexts.extend(subtask.contexts)
                    # Note: subtask.contexts are just strings, we need to pass documents through if possible.
                    # Wait, subtask has only 'contexts' (List[str])...
                    # Let me check if SubTaskExecutionResult has raw docs or if I should add it.
                    for ctx in subtask.contexts:
                        all_retrieved_contexts.append({"text": ctx, "metadata": {"subtask_id": subtask.id}})
                    
                    if subtask.usage:
                        aggregated_usage["input_tokens"] += subtask.usage.get("input_tokens", 0)
                        aggregated_usage["output_tokens"] += subtask.usage.get("output_tokens", 0)
                        aggregated_usage["total_tokens"] += subtask.usage.get("total_tokens", 0)
                    
                    if subtask.thought_process:
                        all_thoughts.append(f"Subtask {subtask.id}: {subtask.thought_process}")
                    
                    if subtask.tool_calls:
                        all_tool_calls.extend(subtask.tool_calls)
                
                return {
                    "answer": exec_res.detailed_answer,
                    "summary": exec_res.summary,
                    "contexts": all_contexts,
                    "retrieved_contexts": all_retrieved_contexts,
                    "source_doc_ids": exec_res.all_sources,
                    "usage": aggregated_usage,
                    "thought_process": "\n\n".join(all_thoughts) if all_thoughts else None,
                    "tool_calls": all_tool_calls
                }



            
            return {"error": f"Tier {tier} not implemented"}
        finally:
            # Clear override after run
            set_session_model_override(None)

    def get_user_full_text(self) -> str:
        """
        Retrieves all text content from the user's vector store.
        
        Returns:
            Concatenated text of all indexed documents.
        """
        from data_base.vector_store_manager import get_user_vector_store_path, get_embeddings
        import os
        from langchain_community.vectorstores import FAISS
        
        user_index_path = get_user_vector_store_path(self.user_id)
        embeddings = get_embeddings()
        
        if not os.path.exists(os.path.join(user_index_path, "index.faiss")):
            logger.warning(f"Vector store not found for user {self.user_id}")
            return ""
            
        try:
            vector_db = FAISS.load_local(
                user_index_path,
                embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            
            all_docs = list(vector_db.docstore._dict.values())
            # Filter for text only and avoid duplicate chunks if they were expanded
            # Actually, just take all unique chunks
            seen_content = set()
            unique_texts = []
            for d in all_docs:
                if d.metadata.get("source") != "image" and d.page_content not in seen_content:
                    unique_texts.append(d.page_content)
                    seen_content.add(d.page_content)
            
            return "\n\n".join(unique_texts)
        except Exception as e:
            logger.error(f"Error loading full text from vector store: {e}")
            return ""

    def extract_token_usage(self, response) -> dict:
        """
        Extracts token usage metadata from a LangChain response object.
        
        Args:
            response: The response object from an LLM call.
            
        Returns:
            A dictionary containing input_tokens, output_tokens, and total_tokens.
        """
        # Try modern LangChain usage_metadata first (LangChain 0.2+)
        usage = getattr(response, "usage_metadata", {})
        if usage:
            return {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
            
        # Fallback to legacy llm_output (LangChain < 0.2 or specific models)
        llm_output = getattr(response, "llm_output", {})
        if llm_output and "token_usage" in llm_output:
            token_usage = llm_output["token_usage"]
            return {
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0)
            }
        
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
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
        logger.info(f"Skipping Ragas metrics for question: {question[:50]} (DISABLED for Debugging)")
        return {
            "faithfulness": 0.0,
            "answer_correctness": 0.0
        }
        
        # Disabled for debugging phase
        # logger.info(f"Calculating Ragas metrics for question: {question[:50]}...")
        # 
        # # Prepare data for Ragas
        # data = {
        #     "question": [question],
        #     "answer": [answer],
        #     "contexts": [contexts],
        #     "ground_truth": [ground_truth]
        # }
        dataset = Dataset.from_dict(data)
        
        # Wrap the evaluator LLM
        # Note: Using gemini-2.5-pro as requested
        evaluator_llm = get_llm("evaluator", model_name=self.evaluator_model)
        ragas_llm = LangchainLLMWrapper(evaluator_llm)
        
        # Get and wrap embeddings
        embeddings_model = get_embeddings()
        if not embeddings_model:
             logger.error("Embeddings model not initialized for Ragas")
             return {"faithfulness": 0.0, "answer_correctness": 0.0, "error": "Embeddings not initialized"}
             
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_model)
        
        # Configure metrics with the evaluator LLM and Embeddings
        # Note: In 0.4.x, older metrics still use .llm attribute for LangChain wrappers
        faithfulness.llm = ragas_llm
        answer_correctness.llm = ragas_llm
        answer_correctness.embeddings = ragas_embeddings
        
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
            
            # Extract scores
            try:
                # Ragas 0.4.x Result object supports __getitem__ but might not have .get()
                f_score = result["faithfulness"]
                ac_score = result["answer_correctness"]
                
                # Handle single-item list return (common in Ragas 0.4.x for single-row datasets)
                if isinstance(f_score, (list, tuple)):
                    f_score = f_score[0] if f_score else 0.0
                if isinstance(ac_score, (list, tuple)):
                    ac_score = ac_score[0] if ac_score else 0.0
                
                return {
                    "faithfulness": float(f_score),
                    "answer_correctness": float(ac_score)
                }
            except (KeyError, TypeError) as e:
                # Fallback or detailed error logging
                logger.error(f"Error extracting scores from result: {e}. Result type: {type(result)}")
                return {
                    "faithfulness": 0.0,
                    "answer_correctness": 0.0,
                    "error": f"Score extraction failed: {e}"
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

    def save_results_json(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Saves the results dictionary to a JSON file.
        
        Args:
            results: The results dictionary.
            output_path: Path to save the JSON file.
        """
        import json
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving JSON results: {e}")

    def save_results_csv(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Saves the results to a flattened CSV file.
        
        Args:
            results: The nested results dictionary.
            output_path: Path to save the CSV file.
        """
        import csv
        import os
        import json
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        flattened_data = []
        
        # Iterate through the nested structure
        # Structure: Model -> Question -> Tier -> Result Dict
        for model_name, questions in results.items():
            for question, tiers in questions.items():
                for tier_name, res in tiers.items():
                    if "error" in res:
                        continue
                        
                    scores = res.get("scores", {})
                    usage = res.get("usage", {})
                    
                    row = {
                        "Model": model_name,
                        "Question": question,
                        "Tier": tier_name,
                        "Answer": res.get("answer", ""),
                        "Faithfulness": scores.get("faithfulness", ""),
                        "Answer_Correctness": scores.get("answer_correctness", ""),
                        "Total_Tokens": usage.get("total_tokens", ""),
                        "Input_Tokens": usage.get("input_tokens", ""),
                        "Output_Tokens": usage.get("output_tokens", ""),
                        "Behavior_Pass": res.get("behavior_pass", ""),
                        "Thought_Process": res.get("thought_process", ""),
                        "Tool_Calls": json.dumps(res.get("tool_calls", []), ensure_ascii=False)
                    }
                    flattened_data.append(row)
        
        if not flattened_data:
            logger.warning("No data to save to CSV")
            return

        try:
            fieldnames = [
                "Model", "Question", "Tier", "Answer", 
                "Faithfulness", "Answer_Correctness", 
                "Total_Tokens", "Input_Tokens", "Output_Tokens",
                "Behavior_Pass", "Thought_Process", "Tool_Calls"
            ]
            
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
            logger.info(f"CSV results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving CSV results: {e}")

    def check_visual_verification(self, result: Dict[str, Any]) -> bool:
        """
        Checks if the visual verification tool was used during execution.
        
        Args:
            result: The result dictionary from run_tier.
            
        Returns:
            True if visual verification usage was detected, False otherwise.
        """
        # Check explicit tool usage field
        tool_usage = result.get("tool_usage", [])
        if "visual_verification" in tool_usage:
            return True
            
        # Fallback: Check for key phrases in the answer that indicate tool usage
        # (e.g. "根據視覺查證", "Using visual verification")
        answer = result.get("answer", "")
        if "視覺查證" in answer or "visual verification" in answer.lower():
            return True
            
        return False

    async def run_full_evaluation(self, questions_path: str, output_prefix: str = "experiments/results/evaluation") -> Dict[str, Any]:
        """
        Runs the full evaluation loop across all models and tiers.
        
        Args:
            questions_path: Path to the benchmark_questions.json file.
            output_prefix: Prefix for the output JSON and CSV files.
            
        Returns:
            The full results dictionary.
        """
        import json
        import os
        from datetime import datetime
        
        # Load questions
        with open(questions_path, "r", encoding="utf-8") as f:
            benchmark_questions = json.load(f)
        
        results = {}
        
        # Initialize RAG components (Embedding, etc.)
        try:
            await on_startup_rag_init()
        except Exception as e:
            logger.warning(f"RAG init failed (might be ok if already init): {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name in self.models:
            results[model_name] = {}
            for q_item in benchmark_questions:
                question = q_item["question"]
                ground_truth = q_item["ground_truth"]
                q_type = q_item.get("type", "standard")
                
                results[model_name][question] = {}
                
                for tier in self.tiers:
                    try:
                        # 1. Run the tier
                        res = await self.run_tier(tier, question, model_name)
                        
                        if "error" in res:
                            logger.error(f"Error in {model_name} | {tier} | {question[:30]}: {res['error']}")
                            results[model_name][question][tier] = res
                            continue
                        
                        # 2. Calculate Ragas metrics
                        # Note: Metrics calculation uses the evaluator model (gemini-3-pro-preview)
                        # We don't want to override the model for Ragas itself
                        scores = await self.calculate_ragas_metrics(
                            question=question,
                            answer=res["answer"],
                            contexts=res["contexts"],
                            ground_truth=ground_truth
                        )
                        
                        # 3. Behavioral check
                        behavior_pass = False
                        if q_type == "visual_verification":
                            behavior_pass = self.check_visual_verification(res)
                        
                        # 4. Store results
                        res["scores"] = scores
                        res["behavior_pass"] = behavior_pass
                        res["type"] = q_type
                        
                        results[model_name][question][tier] = res
                        
                        # Partial save to prevent data loss
                        self.save_results_json(results, f"{output_prefix}_{timestamp}.json")
                        
                    except Exception as e:
                        logger.error(f"Unexpected error in evaluation loop: {e}", exc_info=True)
                        results[model_name][question][tier] = {"error": str(e)}

        # Final save
        self.save_results_json(results, f"{output_prefix}_{timestamp}.json")
        self.save_results_csv(results, f"{output_prefix}_{timestamp}.csv")
        
        return results
