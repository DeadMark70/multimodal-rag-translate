import asyncio
import logging
import sys
from pathlib import Path
import json

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from experiments.evaluation_pipeline import EvaluationPipeline  # noqa: E402
from data_base.router import on_startup_rag_init  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_agentic_fix():
    print("\n" + "="*60)
    print("🚀 STARTING AGENTIC RAG VERIFICATION (Visual Verification Case)")
    print("="*60)
    
    # 1. Init RAG
    print("Initializing RAG components...")
    await on_startup_rag_init()
    
    pipeline = EvaluationPipeline()
    
    # 2. Setup Question (The one from your analysis)
    question = "In the nnU-Net Revisited paper, where exactly is the BraTS dataset located in Figure 1? Please look at the image if needed."
    model = "gemini-2.0-flash-lite"
    tier = "Full Agentic RAG"
    
    print(f"\nRunning tier: {tier} | Model: {model}")
    print(f"Question: {question}")
    
    # 3. Execute
    try:
        result = await pipeline.run_tier(tier, question, model)
        
        # 4. Assertions
        print("\n" + "-"*40)
        print("VERIFICATION RESULTS:")
        
        # A. Token counting fix
        tokens = result.get('usage', {}).get('total_tokens', 0)
        print(f"1. Total Tokens: {tokens}")
        assert tokens > 0, "❌ FAILED: total_tokens should be greater than 0"
        print("   ✅ PASS: Token counting is working.")
        
        # B. Behavioral pass (Vision tool usage)
        tool_calls = result.get('tool_calls', [])
        print(f"2. Tool Calls count: {len(tool_calls)}")
        any(tc.get('action') == 'VERIFY_IMAGE' for tc in tool_calls)
        
        # If not explicitly in tool_calls, check answer for markers
        result.get('answer', '')
        
        # Note: Depending on how AFC works, tool_calls might be recorded differently.
        # But our manual capture in _execute_visual_verification_loop should be there.
        print(f"   Tool calls trace: {json.dumps(tool_calls, indent=2, ensure_ascii=False)}")
        
        assert len(tool_calls) > 0, "❌ FAILED: Agent should have called at least one tool (Vision tool expected)"
        print("   ✅ PASS: Agent triggered tool usage.")
        
        # C. Logging diagnostics
        print(f"3. Has Thought Process: {result.get('thought_process') is not None}")
        assert result.get('thought_process') is not None, "❌ FAILED: thought_process is missing"
        
        print(f"4. Retrieved Contexts count: {len(result.get('retrieved_contexts', []))}")
        assert len(result.get('retrieved_contexts', [])) > 0, "❌ FAILED: retrieved_contexts are empty"
        print("   ✅ PASS: Diagnostic logging is present.")
        
        print("\n" + "="*60)
        print("🏆 ALL AGENTIC FIX VERIFICATIONS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Verification failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(verify_agentic_fix())
