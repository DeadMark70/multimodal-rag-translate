# v9 Preflight Call Budget Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align v9 route call caps and the frontend preflight envelope with their legitimate provider phase sets.

**Architecture:** The backend route planner remains the source of each execution contract. Its caps will cover evidence, visual, final, and graph phases where applicable. The frontend only supplies a five-call admission envelope that covers the planner-inclusive worst case.

**Tech Stack:** Python 3.11, pytest, React, TypeScript, Vitest.

## Global Constraints

- Keep the 50,000 runtime token envelope unchanged.
- Keep all runtime feasibility and per-phase call limits fail-closed.
- Do not skip required visual, graph, evidence, or final-answer phases to admit a request.

---

### Task 1: Route contract capacity

**Files:**
- Modify: `data_base/agentic_v9/route_planner.py`
- Modify: `tests/test_agentic_v9_route_planner.py`
- Modify: `tests/test_agentic_v9_budget_feasibility.py`

- [x] Add failing cases for visual exact/multi-document contracts requiring three calls and graph-visual contracts requiring four calls.
- [x] Increase route capacity only to the required worst-case phase count.
- [x] Run the focused pytest files and commit backend changes.

### Task 2: Preflight envelope alignment

**Files:**
- Modify: `Multimodal_RAG_System/src/components/evaluation/CampaignRunner.tsx`
- Modify: `Multimodal_RAG_System/src/components/evaluation/CampaignRunner.test.tsx`

- [x] Change the frontend preflight call envelope from three to five and update its contract test.
- [x] Run the focused frontend test and lint/type validation.
- [x] Commit the frontend changes with the backend contract alignment.
