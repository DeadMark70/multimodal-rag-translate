import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.llm_factory import get_llm
from data_base.RAG_QA_service import rag_answer_question
from data_base.deep_research_service import get_deep_research_service
from data_base.schemas_deep_research import ExecutePlanRequest, EditableSubTask
from agents.evaluator import RAGEvaluator

USER_ID = "c1bae279-c099-4c45-ba19-2bb393ca4e4b"
QUESTION = "SwinUNETR 與 nnU-Net 在醫學影像分割任務上，誰的表現更好？請根據文獻中的實驗數據進行比較。"

async def run_vanilla_llm():
    print("\n[Baseline 1] Running Vanilla LLM...")
    llm = get_llm("rag_qa")
    from langchain_core.messages import HumanMessage
    response = await llm.ainvoke([HumanMessage(content=QUESTION)])
    return response.content

async def run_naive_rag():
    print("\n[Baseline 2] Running Naive RAG...")
    # Naive RAG: No reranking, no graph, no transformation
    answer, sources = await rag_answer_question(
        question=QUESTION,
        user_id=USER_ID,
        enable_reranking=False,
        enable_hyde=False,
        enable_multi_query=False,
        enable_graph_rag=False
    )
    return answer, sources

async def run_full_agentic_rag():
    print("\n[Ours] Running Full Agentic RAG (Deep Research)...")
    service = get_deep_research_service()
    
    # Step 1: Generate Plan
    plan_res = await service.generate_plan(QUESTION, USER_ID, enable_graph_planning=True)
    
    # Step 2: Execute Plan
    exec_request = ExecutePlanRequest(
        original_question=QUESTION,
        sub_tasks=plan_res.sub_tasks,
        enable_drilldown=True,
        max_iterations=1,
        enable_reranking=True,
        enable_deep_image_analysis=True
    )
    
    exec_res = await service.execute_plan(exec_request, USER_ID)
    return exec_res

async def main():
    # --- RAG Initialization ---
    print("Initializing RAG components...")
    from data_base.router import on_startup_rag_init
    await on_startup_rag_init()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "question": QUESTION,
        "experiments": {}
    }
    
    # 1. Vanilla LLM
    try:
        results["experiments"]["vanilla_llm"] = await run_vanilla_llm()
    except Exception as e:
        results["experiments"]["vanilla_llm"] = f"Error: {e}"

    # 2. Naive RAG
    try:
        ans, src = await run_naive_rag()
        results["experiments"]["naive_rag"] = {"answer": ans, "sources": src}
    except Exception as e:
        results["experiments"]["naive_rag"] = f"Error: {e}"

    # 3. Full Agentic RAG
    try:
        res = await run_full_agentic_rag()
        results["experiments"]["agentic_rag"] = {
            "summary": res.summary,
            "detailed": res.detailed_answer,
            "confidence": res.confidence,
            "sources": res.all_sources,
            "iterations": res.total_iterations
        }
    except Exception as e:
        results["experiments"]["agentic_rag"] = f"Error: {e}"

    # Save Results
    os.makedirs("experiments/results", exist_ok=True)
    output_path = "experiments/results/baseline_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nExperiment complete. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
