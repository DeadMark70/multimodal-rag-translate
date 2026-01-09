#!/usr/bin/env python3
"""
Arena Experiment Script for RAG vs Pure LLM Evaluation.

This script runs A/B testing to compare RAG/Deep Research answers against
Pure LLM answers using the Phase 4 evaluation engine.

Usage:
    python tests/run_arena.py --questions 3 --output results.json
    python tests/run_arena.py --input golden_set.json --output results.csv
"""

# Standard library
import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# Local application
from agents.evaluator import (
    RAGEvaluator,
    DetailedEvaluationResult,
    compare_rag_vs_pure_llm,
)
from core.llm_factory import get_llm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Sample questions for testing (if no input file provided)
SAMPLE_QUESTIONS = [
    "什麼是 Transformer 架構？請說明其核心組件。",
    "RAG (Retrieval-Augmented Generation) 的原理是什麼？",
    "解釋 Self-Attention 機制如何運作。",
]


async def get_pure_llm_answer(question: str) -> str:
    """
    Gets an answer from Pure LLM without any document context.
    
    Args:
        question: The question to answer.
        
    Returns:
        Pure LLM answer string.
    """
    llm = get_llm("chat")
    prompt = f"""你是一個專業的學術助理。請回答以下問題：

問題：{question}

請提供準確、完整且有條理的回答。如果你不確定答案，請誠實表明。"""
    
    message = HumanMessage(content=prompt)
    response = await llm.ainvoke([message])
    return response.content


async def get_mock_rag_answer(question: str) -> tuple[str, list[Document]]:
    """
    Gets a mock RAG answer with simulated document context.
    
    In production, this would call the actual RAG pipeline.
    For testing, we simulate with a simple LLM call.
    
    Args:
        question: The question to answer.
        
    Returns:
        Tuple of (answer, documents).
    """
    llm = get_llm("chat")
    
    # Simulate RAG with explicit source attribution
    prompt = f"""你是一個 RAG 系統。請回答以下問題，並在回答中明確引用來源。

問題：{question}

請按照以下格式回答：
1. 先提供答案
2. 在回答中使用 [來源 1]、[來源 2] 等標記引用資訊
3. 確保答案準確且有依據"""
    
    message = HumanMessage(content=prompt)
    response = await llm.ainvoke([message])
    
    # Create mock documents
    documents = [
        Document(
            page_content=f"這是關於 {question[:20]} 的參考資料片段 1。",
            metadata={"source": "mock_doc_1.pdf", "page": 1}
        ),
        Document(
            page_content=f"這是關於 {question[:20]} 的參考資料片段 2。",
            metadata={"source": "mock_doc_2.pdf", "page": 5}
        ),
    ]
    
    return response.content, documents


async def run_single_experiment(
    question: str,
    experiment_id: int,
) -> dict[str, Any]:
    """
    Runs a single A/B experiment comparing RAG vs Pure LLM.
    
    Args:
        question: The research question.
        experiment_id: Unique ID for this experiment run.
        
    Returns:
        Dict with experiment results.
    """
    logger.info(f"[Exp {experiment_id}] Question: {question[:50]}...")
    
    # Get answers from both systems
    logger.info(f"[Exp {experiment_id}] Getting Pure LLM answer...")
    pure_llm_answer = await get_pure_llm_answer(question)
    
    logger.info(f"[Exp {experiment_id}] Getting RAG answer...")
    rag_answer, documents = await get_mock_rag_answer(question)
    
    # Run comparison evaluation
    logger.info(f"[Exp {experiment_id}] Evaluating both answers...")
    comparison = await compare_rag_vs_pure_llm(
        question=question,
        rag_answer=rag_answer,
        pure_llm_answer=pure_llm_answer,
        documents=documents,
    )
    
    result = {
        "experiment_id": experiment_id,
        "question": question,
        "timestamp": datetime.now().isoformat(),
        "rag_answer_preview": rag_answer[:200] + "...",
        "pure_llm_answer_preview": pure_llm_answer[:200] + "...",
        "rag_score": comparison["rag_score"],
        "pure_llm_score": comparison["pure_llm_score"],
        "score_difference": comparison["score_difference"],
        "winner": comparison["winner"],
        "rag_accuracy": comparison["rag_eval"]["accuracy"],
        "rag_completeness": comparison["rag_eval"]["completeness"],
        "rag_clarity": comparison["rag_eval"]["clarity"],
        "pure_llm_accuracy": comparison["pure_llm_eval"]["accuracy"],
        "pure_llm_completeness": comparison["pure_llm_eval"]["completeness"],
        "pure_llm_clarity": comparison["pure_llm_eval"]["clarity"],
    }
    
    logger.info(
        f"[Exp {experiment_id}] Result: RAG={comparison['rag_score']:.1f} vs "
        f"Pure LLM={comparison['pure_llm_score']:.1f} → {comparison['winner']}"
    )
    
    return result


async def run_arena(
    questions: list[str],
    output_file: str = "arena_results.json",
) -> list[dict[str, Any]]:
    """
    Runs the full arena experiment.
    
    Args:
        questions: List of questions to test.
        output_file: Path to save results.
        
    Returns:
        List of experiment results.
    """
    results = []
    
    for i, question in enumerate(questions, 1):
        result = await run_single_experiment(question, i)
        results.append(result)
    
    # Calculate summary statistics
    rag_wins = sum(1 for r in results if r["winner"] == "rag")
    pure_llm_wins = sum(1 for r in results if r["winner"] == "pure_llm")
    ties = sum(1 for r in results if r["winner"] == "tie")
    
    avg_rag_score = sum(r["rag_score"] for r in results) / len(results)
    avg_pure_llm_score = sum(r["pure_llm_score"] for r in results) / len(results)
    
    summary = {
        "total_experiments": len(results),
        "rag_wins": rag_wins,
        "pure_llm_wins": pure_llm_wins,
        "ties": ties,
        "avg_rag_score": round(avg_rag_score, 2),
        "avg_pure_llm_score": round(avg_pure_llm_score, 2),
        "rag_advantage": round(avg_rag_score - avg_pure_llm_score, 2),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results
    output_data = {
        "summary": summary,
        "experiments": results,
    }
    
    # Save as JSON
    if output_file.endswith(".json"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    # Save as CSV
    elif output_file.endswith(".csv"):
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    
    logger.info(f"\n{'='*50}")
    logger.info("ARENA RESULTS SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total Experiments: {summary['total_experiments']}")
    logger.info(f"RAG Wins: {rag_wins} ({100*rag_wins/len(results):.1f}%)")
    logger.info(f"Pure LLM Wins: {pure_llm_wins} ({100*pure_llm_wins/len(results):.1f}%)")
    logger.info(f"Ties: {ties} ({100*ties/len(results):.1f}%)")
    logger.info(f"Average RAG Score: {avg_rag_score:.2f}/10")
    logger.info(f"Average Pure LLM Score: {avg_pure_llm_score:.2f}/10")
    logger.info(f"RAG Advantage: +{summary['rag_advantage']:.2f}")
    logger.info(f"Results saved to: {output_file}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Arena: RAG vs Pure LLM A/B Testing"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=3,
        help="Number of sample questions to test (default: 3)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to JSON file with custom questions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="arena_results.json",
        help="Output file path (supports .json or .csv)"
    )
    
    args = parser.parse_args()
    
    # Load questions
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
            questions = data.get("questions", data) if isinstance(data, dict) else data
    else:
        questions = SAMPLE_QUESTIONS[:args.questions]
    
    logger.info(f"Running Arena with {len(questions)} questions...")
    
    # Run arena
    asyncio.run(run_arena(questions, args.output))


if __name__ == "__main__":
    main()
