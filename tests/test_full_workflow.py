from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from core import providers as provider_module
from core.providers import configure_providers
from data_base.RAG_QA_service import RAGResult
from data_base.deep_research_service import DeepResearchService
from data_base.schemas_deep_research import ExecutePlanRequest, EditableSubTask


class _FakeLLMResponse:
    """Minimal fake LLM response object used by the workflow test."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {"total_tokens": 0}


class _PlannerLLM:
    """Planner LLM stub that returns parseable sub-task output."""

    async def ainvoke(self, _messages):
        return _FakeLLMResponse(
            "1. [RAG] What are the core architectural components of SwinUNETR?\n"
            "2. [GRAPH] How does SwinUNETR differ from traditional UNet in design and advantages?"
        )


class _SynthesizerLLM:
    """Synthesizer LLM stub that returns a stable final report."""

    async def ainvoke(self, _messages):
        return _FakeLLMResponse(
            "SwinUNETR 相較於傳統 UNet 的主要優勢在於它能更好地建模全域上下文，"
            "同時保留分層視覺表徵。\n\n"
            "- 它使用分層 Transformer 編碼器來補強長距離關聯。\n"
            "- 與傳統 UNet 相比，這讓它在複雜 3D 醫學影像場景更具優勢。"
        )


@pytest.fixture(autouse=True)
def fake_provider_registry():
    """Force fake providers so this workflow test never touches real external APIs."""
    original_registry = provider_module._registry
    configure_providers(use_fake=True)
    yield
    provider_module._registry = original_registry


async def _fake_rag_answer_question(question: str, **_kwargs) -> RAGResult:
    """Return deterministic retrieval results for workflow execution tests."""
    if "core architectural components" in question:
        answer = "SwinUNETR uses a hierarchical Swin Transformer encoder connected to a UNet-style decoder."
        source_id = "doc-arch"
        context = "Hierarchical Swin Transformer encoder with skip connections to a decoder."
    else:
        answer = "Compared with traditional UNet, SwinUNETR has stronger global-context modeling and richer multiscale features."
        source_id = "doc-compare"
        context = "Global-context modeling and multiscale features improve complex 3D segmentation."

    return RAGResult(
        answer=answer,
        source_doc_ids=[source_id],
        documents=[Document(page_content=context, metadata={"doc_id": source_id})],
        usage={"total_tokens": 0},
    )

@pytest.mark.asyncio
async def test_deep_research_full_workflow():
    """
    End-to-End Test: Planning -> Execution -> Synthesis.
    Using a real DeepResearchService instance.
    """
    service = DeepResearchService()
    user_id = "test-user-id-001"
    question = "請分析 SwinUNETR 的核心架構，並說明其相較於傳統 UNet 的優勢。"

    with patch("agents.planner.get_llm", return_value=_PlannerLLM()), \
         patch("agents.synthesizer.get_llm", return_value=_SynthesizerLLM()), \
         patch("data_base.deep_research_service.rag_answer_question", new=AsyncMock(side_effect=_fake_rag_answer_question)), \
         patch.object(service, "_drill_down_loop", new=AsyncMock(return_value=0)), \
         patch("data_base.deep_research_service.persist_research_conversation", new=AsyncMock()):
        # 1. Test Planning Phase
        plan_res = await service.generate_plan(question, user_id, enable_graph_planning=True)

        assert plan_res.status == "waiting_confirmation"
        assert len(plan_res.sub_tasks) >= 2
        assert plan_res.sub_tasks[0].task_type == "rag"
        assert plan_res.sub_tasks[1].task_type == "graph_analysis"
        print(f"\n[Step 1] Generated {len(plan_res.sub_tasks)} sub-tasks.")

        # 2. Test Execution Phase (Simulated confirmation)
        exec_request = ExecutePlanRequest(
            original_question=question,
            sub_tasks=plan_res.sub_tasks,
            enable_drilldown=True,
            max_iterations=1,
            enable_reranking=False,
        )

        print("[Step 2] Executing plan...")
        exec_res = await service.execute_plan(exec_request, user_id)

    # 3. Verify Results
    assert exec_res.question == question
    assert exec_res.summary != ""
    assert isinstance(exec_res.sub_tasks, list)
    assert len(exec_res.sub_tasks) == 2
    assert exec_res.total_iterations == 0
    assert "SwinUNETR" in exec_res.summary
    assert sorted(exec_res.all_sources) == ["doc-arch", "doc-compare"]

    print(f"[Step 3] Execution finished. Total iterations: {exec_res.total_iterations}")
    print(f"Summary: {exec_res.summary[:100]}...")

    assert exec_res.confidence >= 0.0

@pytest.mark.asyncio
async def test_deep_research_no_tasks():
    """Verify behavior when no tasks are enabled."""
    service = DeepResearchService()
    user_id = "test-user"
    request = ExecutePlanRequest(
        original_question="Test",
        sub_tasks=[EditableSubTask(id=1, question="Q1", task_type="rag", enabled=False)],
        enable_drilldown=False
    )
    
    res = await service.execute_plan(request, user_id)
    assert "沒有啟用的子任務" in res.summary
