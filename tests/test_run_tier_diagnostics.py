import asyncio
import logging
import sys
from pathlib import Path
import json

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from experiments.evaluation_pipeline import EvaluationPipeline
from data_base.router import on_startup_rag_init

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_single_run():
    print("Initializing RAG components...")
    await on_startup_rag_init()
    
    pipeline = EvaluationPipeline()
    print(f"Active Models: {pipeline.models}")
    
    # We only expect gemini-2.0-flash-lite
    assert pipeline.models == ["gemini-2.0-flash-lite"]
    
    question = "What is the primary contribution of the SwinUNETR paper?"
    model = "gemini-2.0-flash-lite"
    tier = "Naive RAG"
    
    print(f"Running test: {tier} | {model}...")
    result = await pipeline.run_tier(tier, question, model)
    
    print("\n--- TEST RESULT ---")
    print(f"Answer snippet: {result.get('answer', '')[:100]}...")
    print(f"Total Tokens: {result.get('usage', {}).get('total_tokens')}")
    print(f"Has Thought Process: {result.get('thought_process') is not None or 'None' in str(result.get('thought_process'))}")
    print(f"Tool Calls Count: {len(result.get('tool_calls', []))}")
    print(f"Retrieved Contexts Count: {len(result.get('retrieved_contexts', []))}")
    
    if result.get('retrieved_contexts'):
        print(f"First context metadata sample: {result.get('retrieved_contexts')[0].get('metadata')}")

    # Final assertion to ensure basic structure is there
    assert "total_tokens" in result["usage"]
    assert "retrieved_contexts" in result
    print("\n✅ Basic diagnostic structure verified!")

if __name__ == "__main__":
    asyncio.run(test_single_run())
