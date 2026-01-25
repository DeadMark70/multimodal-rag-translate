# Implementation Plan - Track: deep_research_history_details

## Phases

### Phase 1: Database & Backend API Updates
- [x] Task: Create database migration for `metadata` column (25bb835)
    - [x] Create `supabase/migrations/20260125_add_metadata_to_conversations.sql`
    - [x] Add `metadata` JSONB column to `conversations` table if not exists (ensure JSONB type)
- [ ] Task: Update Backend Schemas and Models
    - [ ] Update `ExecutePlanRequest` (Pydantic model) to accept optional `conversation_id`
    - [ ] Update `ConversationResponse` schema to include `metadata` field
- [ ] Task: Implement Persistence Logic
    - [ ] Modify `DeepResearchService.execute_plan` or the router handler
    - [ ] Logic: After successful execution, update `conversations` table:
        - Set `metadata` = `ExecutePlanResponse`
        - Update `title` based on `original_question` (if generic or default)
- [ ] Task: Verify API Response
    - [ ] Test `GET /api/conversations/{id}` returns populated `metadata`
- [ ] Task: Conductor - User Manual Verification 'Database & Backend API Updates' (Protocol in workflow.md)

### Phase 2: Frontend API & Types Integration
- [ ] Task: Update Frontend Types
    - [ ] Update `src/types/conversation.ts` to include `metadata` in `Conversation` interface
    - [ ] Define `ExecutePlanResponse` interface matching backend
- [ ] Task: Update API Services
    - [ ] Update `conversationApi.getConversation` (if needed) to ensure metadata is parsed
    - [ ] Update `useDeepResearch` hook to pass `conversation_id` when executing plan
- [ ] Task: Conductor - User Manual Verification 'Frontend API & Types Integration' (Protocol in workflow.md)

### Phase 3: Frontend UI Components
- [ ] Task: Create `ResearchDetailModal` Component
    - [ ] Create `src/components/rag/ResearchDetailModal.tsx`
    - [ ] Implement Header (Question, Date, Confidence)
    - [ ] Implement Tab/Section switching (Summary vs. Details)
    - [ ] Implement Markdown Renderer for `detailed_answer`
- [ ] Task: Create `ResearchStepsAccordion` Component
    - [ ] Create `src/components/rag/ResearchStepsAccordion.tsx`
    - [ ] Implement collapsible list for `sub_tasks`
    - [ ] Display Question, Answer, Sources for each step
- [ ] Task: Integrate with History List
    - [ ] Update `ConversationsSidebar` or `HistoryItem`
    - [ ] Add "View Details" action for `type === 'research'`
    - [ ] Connect action to open `ResearchDetailModal`
- [ ] Task: Conductor - User Manual Verification 'Frontend UI Components' (Protocol in workflow.md)

### Phase 4: Integration & Polish
- [ ] Task: End-to-End Testing
    - [ ] Run full Deep Research flow
    - [ ] Verify persistence
    - [ ] Verify UI rendering of history
- [ ] Task: Error Handling & Refinement
    - [ ] Handle missing/malformed metadata gracefully
    - [ ] Optimize large Markdown rendering
- [ ] Task: Conductor - User Manual Verification 'Integration & Polish' (Protocol in workflow.md)
