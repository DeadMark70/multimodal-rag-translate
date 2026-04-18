from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agents.planner import ResearchPlan, SubTask, classify_question_intent
from data_base.schemas_deep_research import (
    AtomicFact,
    EditableSubTask,
    ExecutePlanResponse,
    ResearchPlanResponse,
    SubTaskExecutionResult,
)
from evaluation.agentic_evaluation_service import (
    AGENTIC_EVAL_PROFILE,
    AGENTIC_IMAGE_ANALYSIS_ENABLED,
    AgenticEvaluationService,
    _drilldown_iterations_for_strategy,
    _is_numeric_benchmark_subtask,
    _route_profile_for_task,
)


@pytest.mark.asyncio
async def test_generate_agentic_plan_caps_initial_plan_to_strategy_limit() -> None:
    service = AgenticEvaluationService()
    generated_plan = ResearchPlan(
        original_question="What changed?",
        sub_tasks=[
            SubTask(id=1, question="Task 1", task_type="rag"),
            SubTask(id=2, question="Task 2", task_type="graph_analysis"),
            SubTask(id=3, question="Task 3", task_type="rag"),
            SubTask(id=4, question="Task 4", task_type="rag"),
        ],
        estimated_complexity="complex",
    )

    with patch("evaluation.agentic_evaluation_service.TaskPlanner") as mock_planner_cls:
        mock_planner = mock_planner_cls.return_value
        mock_planner.plan = AsyncMock(return_value=generated_plan)

        response = await service.generate_agentic_plan(
            question="Compare model A and model B differences",
            user_id="user-1",
        )

    mock_planner_cls.assert_called_once_with(max_subtasks=2, enable_graph_planning=True)
    assert response.status == "waiting_confirmation"
    assert len(response.sub_tasks) == 2
    assert [task.id for task in response.sub_tasks] == [1, 2]
    assert response.estimated_complexity == "complex"


@pytest.mark.asyncio
async def test_generate_agentic_plan_figure_flow_uses_single_initial_subtask() -> None:
    service = AgenticEvaluationService()
    generated_plan = ResearchPlan(
        original_question="請重建 MICCSS 模組流程",
        sub_tasks=[
            SubTask(id=1, question="Task 1", task_type="rag"),
            SubTask(id=2, question="Task 2", task_type="rag"),
            SubTask(id=3, question="Task 3", task_type="graph_analysis"),
        ],
        estimated_complexity="complex",
    )

    with patch("evaluation.agentic_evaluation_service.TaskPlanner") as mock_planner_cls:
        mock_planner = mock_planner_cls.return_value
        mock_planner.plan = AsyncMock(return_value=generated_plan)

        response = await service.generate_agentic_plan(
            question="請重建 MICCSS 模組流程",
            user_id="user-1",
        )

    mock_planner_cls.assert_called_once_with(max_subtasks=2, enable_graph_planning=True)
    assert len(response.sub_tasks) == 1
    assert response.sub_tasks[0].id == 1
    assert response.sub_tasks[0].question == "請重建 MICCSS 模組流程"


@pytest.mark.asyncio
async def test_generate_agentic_plan_figure_flow_anchors_original_question_and_filters_broad_aux() -> None:
    service = AgenticEvaluationService()
    generated_plan = ResearchPlan(
        original_question="請重建 MICCSS CSS 流程並說明翻轉分支與 SiamSSM 累加機制",
        sub_tasks=[
            SubTask(id=1, question="What is the overall architecture of nnMamba?", task_type="rag"),
            SubTask(
                id=2,
                question="Reconstruct MICCSS CSS branch order and SiamSSM accumulation flow",
                task_type="rag",
            ),
        ],
        estimated_complexity="medium",
    )

    with patch("evaluation.agentic_evaluation_service.TaskPlanner") as mock_planner_cls:
        mock_planner = mock_planner_cls.return_value
        mock_planner.plan = AsyncMock(return_value=generated_plan)

        response = await service.generate_agentic_plan(
            question="請重建 MICCSS CSS 流程並說明翻轉分支與 SiamSSM 累加機制",
            user_id="user-1",
        )

    assert len(response.sub_tasks) == 2
    assert response.sub_tasks[0].question == "請重建 MICCSS CSS 流程並說明翻轉分支與 SiamSSM 累加機制"
    assert response.sub_tasks[1].question == "Reconstruct MICCSS CSS branch order and SiamSSM accumulation flow"


def test_route_profile_benchmark_initial_round_prefers_numeric_only_generic_graph() -> None:
    numeric_graph_route = _route_profile_for_task(
        strategy_tier="tier_3_multi_hop_analysis",
        question_intent="benchmark_data",
        task_type="rag",
        task_question="Across papers, compare FLOPs and Params (141.14G vs 224.35G) and arbitrate efficiency",
        iteration=0,
    )
    numeric_non_graph_route = _route_profile_for_task(
        strategy_tier="tier_3_multi_hop_analysis",
        question_intent="benchmark_data",
        task_type="rag",
        task_question="Provide Dice score details (0.88 vs 0.82) for each model",
        iteration=0,
    )
    non_numeric_route = _route_profile_for_task(
        strategy_tier="tier_3_multi_hop_analysis",
        question_intent="benchmark_data",
        task_type="rag",
        task_question="Explain Dice supervision mechanism in 3D training",
        iteration=0,
    )

    assert numeric_graph_route == "generic_graph"
    assert numeric_non_graph_route == "hybrid_compare"
    assert non_numeric_route == "hybrid_compare"


def test_drilldown_iterations_match_intent_constraints() -> None:
    assert (
        _drilldown_iterations_for_strategy(
            strategy_tier="tier_2_structured_compare",
            question_intent="figure_flow",
        )
        == 0
    )
    assert (
        _drilldown_iterations_for_strategy(
            strategy_tier="tier_3_multi_hop_analysis",
            question_intent="benchmark_data",
        )
        == 1
    )


def test_numeric_benchmark_subtask_requires_metric_and_numeric_context() -> None:
    assert (
        _is_numeric_benchmark_subtask(
            task_type="rag",
            task_question="Compare FLOPs 141.14G vs 224.35G for each model",
        )
        is True
    )
    assert (
        _is_numeric_benchmark_subtask(
            task_type="rag",
            task_question="Explain Dice supervision mechanism in 3D segmentation",
        )
        is False
    )
    assert (
        _is_numeric_benchmark_subtask(
            task_type="rag",
            task_question="Analyze SAM-Med3D architecture details",
        )
        is False
    )


def test_classify_question_intent_keeps_methodology_compare_out_of_benchmark() -> None:
    q6_like = (
        "根據論文方法描述，比較 U-Mamba 與 Weak-Mamba-UNet 在模型角色與監督機制上的核心差異，"
        "並說明 three-view cross-supervision 與 pseudo-label Dice supervision。"
    )
    q4_like = (
        "請結合文獻中的 Params 與 FLOPs 報告，裁決 Mamba 在 3D 醫療分割是否具最高計算效率。"
    )
    assert classify_question_intent(q6_like) == "comparison_disambiguation"
    assert classify_question_intent(q4_like) == "benchmark_data"


def test_retrieval_quality_gate_blocks_contextless_answers() -> None:
    service = AgenticEvaluationService()
    service._required_coverage = []
    results = [
        SubTaskExecutionResult(
            id=1,
            question="Task 1",
            answer="This is a long enough answer with no obvious failure marker." * 8,
            contexts=[],
            sources=[],
        )
    ]
    assert service._should_skip_drilldown(results) is False


def test_gap_targeted_followup_requires_coverage_keyword_overlap() -> None:
    service = AgenticEvaluationService()
    assert service._is_gap_targeted_followup(
        question="What are the limitations and caveats of method A?",
        coverage_gaps=["confusion_or_limitation"],
    )
    assert not service._is_gap_targeted_followup(
        question="Provide broad historical background of the field",
        coverage_gaps=["confusion_or_limitation"],
    )


@pytest.mark.asyncio
async def test_generate_agentic_plan_bypasses_planner_for_tier1_detail_lookup() -> None:
    service = AgenticEvaluationService()
    with patch("evaluation.agentic_evaluation_service.TaskPlanner") as mock_planner_cls:
        response = await service.generate_agentic_plan(
            question="What is SONO-MultiKAN?",
            user_id="user-1",
        )

    mock_planner_cls.assert_not_called()
    assert len(response.sub_tasks) == 1
    assert response.sub_tasks[0].question == "What is SONO-MultiKAN?"
    assert response.estimated_complexity == "simple"


@pytest.mark.asyncio
async def test_run_case_uses_dedicated_agentic_execution_constraints() -> None:
    service = AgenticEvaluationService(max_concurrent_tasks=3)
    plan_response = ResearchPlanResponse(
        original_question="What changed?",
        sub_tasks=[
            EditableSubTask(id=1, question="Task 1", task_type="rag"),
            EditableSubTask(id=2, question="Task 2", task_type="graph_analysis"),
            EditableSubTask(id=3, question="Task 3", task_type="rag"),
        ],
        estimated_complexity="medium",
        doc_ids=None,
    )
    execute_response = ExecutePlanResponse(
        question="What changed?",
        summary="summary",
        detailed_answer="final answer",
        sub_tasks=[],
        all_sources=[],
        confidence=1.0,
        total_iterations=1,
    )

    service.generate_agentic_plan = AsyncMock(return_value=plan_response)
    service.run_execute_plan = AsyncMock(return_value=execute_response)

    async def passthrough(func, **kwargs):
        return await func(**kwargs)

    with patch("evaluation.agentic_evaluation_service.run_with_retry", new=AsyncMock(side_effect=passthrough)):
        result = await service.run_case(
            question_id="Q1",
            question="What changed?",
            user_id="user-1",
            run_number=1,
        )

    request = service.run_execute_plan.await_args.kwargs["request"]
    assert request.enable_drilldown is False
    assert request.max_iterations == 1
    assert request.enable_deep_image_analysis is AGENTIC_IMAGE_ANALYSIS_ENABLED
    assert len(request.sub_tasks) == 3
    assert result.answer == "final answer"
    assert result.agent_trace["execution_profile"] == AGENTIC_EVAL_PROFILE
    assert result.agent_trace["strategy_tier"] == "tier_1_detail_lookup"


@pytest.mark.asyncio
async def test_synthesize_execution_results_forces_single_task_synthesis_lite() -> None:
    service = AgenticEvaluationService()
    service._active_question_intent = "figure_flow"
    service._active_strategy_tier = "tier_2_structured_compare"
    one_result = SubTaskExecutionResult(
        id=1,
        question="請重建 CSS 流程",
        answer="raw answer",
        contexts=["ctx"],
        sources=["doc-1"],
    )

    with patch(
        "evaluation.agentic_evaluation_service.synthesize_results",
        new=AsyncMock(
            return_value=SimpleNamespace(
                summary="normalized summary",
                detailed_answer="A -> B -> C",
                confidence=0.9,
            )
        ),
    ) as mock_synthesize:
        response = await service._synthesize_execution_results(
            original_question="請重建 CSS 流程",
            all_results=[one_result],
            total_iterations=0,
        )

    kwargs = mock_synthesize.await_args.kwargs
    assert kwargs["enabled"] is True
    assert kwargs["force_llm_for_single"] is True
    assert kwargs["enable_conflict_arbitration"] is True
    assert response.detailed_answer == "A -> B -> C"


def test_route_kwargs_always_enable_crag_for_agentic_execution() -> None:
    service = AgenticEvaluationService()
    kwargs = service._route_kwargs(
        route_profile="hybrid_compare",
        enable_reranking=True,
        enable_visual_verification=False,
        task_type="rag",
        stage_hint="exploration",
    )
    assert kwargs["enable_crag"] is True
    assert kwargs["plain_mode"] is False


@pytest.mark.asyncio
async def test_agentic_drilldown_uses_structured_fact_state_for_followup_context() -> None:
    service = AgenticEvaluationService()
    service._active_question_intent = "benchmark_data"
    service._active_strategy_tier = "tier_2_structured_compare"
    service._required_coverage = ["confusion_or_limitation"]

    current_results = [
        SubTaskExecutionResult(
            id=1,
            question="Compare model metrics",
            answer="Model A reports Dice 0.90 and Model B reports Dice 0.87.",
            sources=["doc-1"],
            contexts=["ctx"],
            tool_calls=[
                {
                    "action": "VERIFY_IMAGE",
                    "question": "What does XYZ in Figure 3 stand for?",
                    "success": True,
                    "result": "Figure 3 introduces XYZ as Cross-Year Zonal Yield.",
                }
            ],
            visual_verification_meta={
                "visual_verification_attempted": True,
                "visual_tool_call_count": 1,
                "visual_force_fallback_used": False,
            },
        )
    ]

    with patch("evaluation.agentic_evaluation_service.TaskPlanner") as mock_planner_cls, patch.object(
        service,
        "_extract_atomic_facts",
        new=AsyncMock(
            return_value=[
                AtomicFact(
                    claim="Model A reports Dice 0.90 and Model B reports Dice 0.87.",
                    source_doc_ids=["doc-1"],
                )
            ]
        ),
    ):
        mock_planner = mock_planner_cls.return_value
        mock_planner.create_followup_tasks = AsyncMock(return_value=[])

        iterations = await service._drill_down_loop(
            original_question="Compare model metrics and limitations",
            current_results=current_results,
            user_id="user-1",
            doc_ids=None,
            enable_reranking=True,
            max_iterations=1,
        )

    assert iterations == 0
    current_findings = mock_planner.create_followup_tasks.await_args.kwargs["current_findings"]
    assert "Structured Fact State" in current_findings
    assert "Model A reports Dice 0.90 and Model B reports Dice 0.87." in current_findings
    assert "Visual Verification Findings" in current_findings
    assert "potential_terms=XYZ" in current_findings
