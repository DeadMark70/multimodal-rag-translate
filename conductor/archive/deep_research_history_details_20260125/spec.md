# Specification: Deep Research History Details

## Overview
Currently, the Deep Research history only displays basic information. Users cannot access the detailed results (Research Report, Sub-task execution details, Sources, etc.) of past sessions. This track aims to persist the full execution results in the database and provide a detailed UI (Expandable List + Modal) to view them.

## Context
- **Backend:** `DeepResearchService` generates `ExecutePlanResponse` (Plan + Execution Results). This is currently returned but not fully persisted for retrieval.
- **Frontend:** The history list needs to support opening these details.
- **Database:** `conversations` table has a `metadata` JSONB column (type='research') that can store this data.

## Functional Requirements

### Backend
1.  **Persistence:**
    - Update `DeepResearchService.execute_plan` (or the router handler) to save the full `ExecutePlanResponse` JSON into the `conversations.metadata` column upon successful execution.
    - Ensure `metadata` contains:
        - `summary`
        - `detailed_answer`
        - `sub_tasks` (List of QA pairs, sources, thoughts)
        - `confidence`
        - `all_sources`
    - Update `title` based on `original_question` if generic.
2.  **Retrieval:**
    - Ensure `GET /api/conversations/{id}` (or the list endpoint) returns the `metadata` field.

### Frontend
1.  **History List:**
    - Identify "Research" type items.
    - Provide a "View Details" button or click action.
2.  **Detail Modal:**
    - Create a reusable `ResearchDetailModal` component.
    - **Header:** Display Original Question, Date, and Confidence Score.
    - **Body:**
        - **Summary:** Display the Executive Summary.
        - **Full Report:** Display the Detailed Answer (Markdown support).
        - **Process/Steps:** An interactive **Accordion** list showing each Sub-task:
            - Question
            - Answer
            - Sources (Links/Citations)
            - Thought Process (if available)

## Non-Functional Requirements
- **Performance:** Ensure loading history remains snappy.
- **UX:** The Modal should be responsive and easy to close.

## Acceptance Criteria
- [ ] Completing a new Deep Research session saves the full result to DB.
- [ ] Users can open a past Research session from the history sidebar.
- [ ] The Detail Modal accurately displays the Summary, Detailed Report, and Sub-tasks.
- [ ] Sub-tasks are collapsible/expandable.
