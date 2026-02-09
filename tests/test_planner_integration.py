import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from agents.planner import TaskPlanner  # noqa: E402
from data_base.router import on_startup_rag_init  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)

async def run_planner_test():
    print("Initializing RAG components...")
    await on_startup_rag_init()
    
    planner = TaskPlanner()
    question = "In the nnU-Net Revisited paper, where exactly is the BraTS dataset located in Figure 1?"
    
    print(f"\nGenerating plan for question: {question}")
    plan = await planner.plan(question)
    
    print("\n--- GENERATED SUB-TASKS ---")
    visual_task_found = False
    visual_keywords = ["視覺", "查證", "圖", "Figure", "image", "verify"]
    
    for task in plan.sub_tasks:
        print(f"{task.id}. [{task.task_type}] {task.question}")
        if any(kw.lower() in task.question.lower() for kw in visual_keywords):
            visual_task_found = True
            
    if visual_task_found:
        print("\n✅ SUCCESS: Visual verification sub-task detected!")
    else:
        print("\n❌ FAILURE: No visual verification sub-task found in the plan.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_planner_test())
