# Task Plan: Inspect Agentic RAG Evaluation Path

## Goal
Trace the current backend production evaluation path for agentic RAG in `D:\flutterserver\pdftopng` and report how it executes, whether it uses Deep Research semantics, what limits it applies, and where it likely diverges from a dedicated agentic evaluation baseline.

## Current Phase
Phase 4

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify constraints and requirements
- [x] Document findings in findings.md
- **Status:** complete

### Phase 2: Execution Path Trace
- [x] Read target modules and related schemas
- [x] Map call flow from evaluation entrypoint to planner/research core
- [x] Document decisions with rationale
- **Status:** complete

### Phase 3: Limits & Semantics Analysis
- [x] Extract drilldown/subtask/image limits
- [x] Determine whether Deep Research semantics are used
- [x] Compare against a dedicated agentic evaluation baseline
- **Status:** complete

### Phase 4: Verification
- [x] Re-check traced flow against source references
- [x] Confirm report answers all requested points
- [ ] Log verification in progress.md
- **Status:** in_progress

### Phase 5: Delivery
- [ ] Review notes and findings
- [ ] Deliver concise report to user
- [ ] Avoid any file edits outside planning files
- **Status:** pending

## Key Questions
1. What is the exact runtime path for agentic evaluation requests today?
2. Does the path invoke Deep Research-style planning/execution semantics or only reuse parts of the stack?
3. Where are the concrete limits for drilldown depth, subtasks, and images enforced?
4. What differences likely matter when comparing this path to a dedicated agentic evaluation baseline?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use planning files in `pdftopng` | Required by `AGENTS.md` for complex multi-step inspection tasks |
| Do not edit product code | User explicitly asked for inspection/report only |
| Compare evaluation path against `DeepResearchService` rather than only core methods | Needed to distinguish shared execution primitives from full user-facing Deep Research semantics |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| None so far | 1 | N/A |

## Notes
- Focus files: `evaluation/rag_modes.py`, `evaluation/agentic_evaluation_service.py`, `data_base/research_execution_core.py`, `agents/planner.py`, and related schemas.
- Planning files are the only files to be created/updated.
