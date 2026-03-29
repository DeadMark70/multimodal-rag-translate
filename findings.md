# Findings & Decisions

## Requirements
- Inspect the current backend production evaluation path for agentic RAG in `D:\flutterserver\pdftopng`.
- Focus on `evaluation/rag_modes.py`, `evaluation/agentic_evaluation_service.py`, `data_base/research_execution_core.py`, `agents/planner.py`, and related schemas.
- Report:
- 1. How agentic evaluation currently executes.
- 2. Whether it is actually using Deep Research semantics.
- 3. Current limits on drilldown, subtasks, and images.
- 4. Likely mismatch versus a dedicated agentic evaluation baseline.
- Do not edit product files.

## Research Findings
- Session catchup reported no recoverable prior Codex session state.
- `evaluation/rag_modes.py` routes `mode == "agentic"` to `AgenticEvaluationService.run_case()` instead of the normal `rag_answer_question()` path.
- `AgenticEvaluationService` inherits `ResearchExecutionCore`, so evaluation reuses shared Deep Research-style planning and execution primitives rather than a dedicated evaluation executor.
- The evaluation wrapper hard-codes `enable_drilldown=True`, `max_iterations=2`, and `enable_deep_image_analysis=True` in `ExecutePlanRequest`.
- Initial planning in `ResearchExecutionCore.generate_plan()` uses `TaskPlanner(max_subtasks=5, enable_graph_planning=...)`.
- Drill-down planning in `ResearchExecutionCore._drill_down_loop()` uses `TaskPlanner(max_subtasks=3, enable_graph_planning=False)`.
- `TaskPlanner.create_followup_tasks()` also caps follow-up tasks to `[:3]` even after parsing.
- Initial execution path always calls `rag_answer_question(... enable_graph_rag=True, graph_search_mode=\"generic\", enable_visual_verification=enable_deep_image_analysis)`.
- Drill-down execution uses graph mode only for `graph_analysis` tasks; otherwise it uses `graph_search_mode=\"auto\"`.
- `ResearchExecutionCore._should_skip_drilldown()` explicitly forces at least one drill-down opportunity at iteration 0.
- `ExecutePlanRequest` allows `max_iterations` from 1 to 5 at the schema level, but evaluation pins it to 2.
- Image handling limits live in `RAG_QA_service`, not in the evaluation wrapper: `MAX_IMAGES = 3` and `MAX_VISUAL_ITERATIONS = 2`.
- Visual verification only runs when `enable_visual_verification` is true and retrieved docs expose `image_paths`; otherwise “deep image analysis” is effectively inactive for that subtask.
- Usage from the visual verification loop is not aggregated back into `usage_metadata`, so evaluation token accounting can undercount image-heavy runs.
- User-facing `DeepResearchService` adds interactive confirmation, optional streaming, and optional conversation persistence on top of the shared core; evaluation bypasses those layers and calls `run_execute_plan()` directly.
- Repository guidance in `agent.md` already records that user-facing Deep Research and evaluation agentic may share primitives but should not share an unversioned wrapper once behavior diverges.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use source inspection plus light verification commands | Best way to answer execution semantics and hard-coded/runtime limits without changing code |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| None so far | N/A |

## Resources
- `D:\flutterserver\pdftopng\evaluation\rag_modes.py`
- `D:\flutterserver\pdftopng\evaluation\agentic_evaluation_service.py`
- `D:\flutterserver\pdftopng\data_base\research_execution_core.py`
- `D:\flutterserver\pdftopng\agents\planner.py`
- `D:\flutterserver\pdftopng\data_base\schemas_deep_research.py`
- `D:\flutterserver\pdftopng\evaluation\trace_schemas.py`
- `D:\flutterserver\pdftopng\evaluation\campaign_engine.py`
- `D:\flutterserver\pdftopng\data_base\deep_research_service.py`
- `D:\flutterserver\pdftopng\data_base\RAG_QA_service.py`
- `D:\flutterserver\pdftopng\agent.md`

## Visual/Browser Findings
- None.

---
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*
