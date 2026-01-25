# Implementation Plan - Track: deep_research_history_details

## Phases

### Phase 1: Database & Backend API Updates [x] [checkpoint: ee128c8]
- [x] Task: Create database migration for `metadata` column (25bb835)
    - [x] Create `supabase/migrations/20260125_add_metadata_to_conversations.sql`
    - [x] Add `metadata` JSONB column to `conversations` table if not exists (ensure JSONB type)
- [x] Task: Update Backend Schemas and Models (78464ed)
    - [x] Update `ExecutePlanRequest` (Pydantic model) to accept optional `conversation_id`
    - [x] Update `ConversationResponse` schema to include `metadata` field
- [x] Task: Implement Persistence Logic (d5f750b)
    - [x] Modify `DeepResearchService.execute_plan` or the router handler
    - [x] Logic: After successful execution, update `conversations` table:
        - Set `metadata` = `ExecutePlanResponse`
        - Update `title` based on `original_question` (if generic or default)
- [x] Task: Verify API Response (45ebcba)
    - [x] Test `GET /api/conversations/{id}` returns populated `metadata`
- [x] Task: Conductor - User Manual Verification 'Database & Backend API Updates' (Protocol in workflow.md)

### Phase 2: Frontend API & Types Integration [x]
- [x] Task: Update Frontend Types
    - [x] Update `src/types/conversation.ts` to include `metadata` in `Conversation` interface
    - [x] Define `ExecutePlanResponse` interface matching backend
- [x] Task: Update API Services
    - [x] Update `conversationApi.getConversation` (if needed) to ensure metadata is parsed
    - [x] Update `useDeepResearch` hook to pass `conversation_id` when executing plan
- [x] Task: Conductor - User Manual Verification 'Frontend API & Types Integration' (Protocol in workflow.md)

### Phase 3: Frontend UI Components [x]
- [x] Task: Create `ResearchDetailModal` Component
    - [x] Create `src/components/rag/ResearchDetailModal.tsx`
    - [x] Implement Header (Question, Date, Confidence)
    - [x] Implement Tab/Section switching (Summary vs. Details)
    - [x] Implement Markdown Renderer for `detailed_answer`
- [x] Task: Create `ResearchStepsAccordion` Component
    - [x] Create `src/components/rag/ResearchStepsAccordion.tsx`
    - [x] Implement collapsible list for `sub_tasks`
    - [x] Display Question, Answer, Sources for each step
- [x] Task: Integrate with History List
    - [x] Update `ConversationsSidebar` or `HistoryItem`
    - [x] Add "View Details" action for `type === 'research'`
    - [x] Connect action to open `ResearchDetailModal`
- [x] Task: Conductor - User Manual Verification 'Frontend UI Components' (Protocol in workflow.md)

### Phase 4: Integration & Polish [x]
- [x] Task: End-to-End Testing
    - [x] Run full Deep Research flow
    - [x] Verify persistence
    - [x] Verify UI rendering of history
- [x] Task: Error Handling & Refinement
    - [x] Handle missing/malformed metadata gracefully
    - [x] Optimize large Markdown rendering
- [x] Task: Conductor - User Manual Verification 'Integration & Polish' (Protocol in workflow.md)