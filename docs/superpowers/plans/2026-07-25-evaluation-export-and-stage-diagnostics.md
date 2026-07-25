# Evaluation Export and Stage Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Separate v9 capability gaps from execution failures, persist safe failed-run diagnostics, and make the Evaluation Center export control download JSON.

**Architecture:** Keep the existing errors and export APIs, adding one typed stage-warning projection and endpoint. Store safe failure diagnostics in each run's existing derived_metrics JSON. The Ablation tab loads warnings lazily and uses the existing export response as its post-download preview.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, SQLite, pytest; React, TypeScript, Chakra UI, Vitest, Testing Library, Axios.

## Global Constraints

- Preserve POST /api/evaluation/campaigns/{campaign_id}/export and its current answer/excerpt defaults.
- Never persist raw prompts, provider responses, stack traces, API keys, or secrets as failure diagnostics.
- partial is never a Sanitized Error and never changes failed-run counts.
- A warning is emitted only for a required v9 stage that is partial or required_but_not_satisfied; not_triggered and not_requested are silent.
- Use {campaignId}-redacted.json unless full prompts or raw trace payloads are enabled; those options use {campaignId}-custom.json.
- Backend repository: D:/flutterserver/pdftopng. Frontend repository: D:/flutterserver/Multimodal_RAG_System.
- Do not stage existing untracked data, .pytest-tmp, documents, or unrelated changes.

---

## File Map

| Area | Files | Responsibility |
| --- | --- | --- |
| Backend contract | evaluation/campaign_schemas.py, evaluation/router.py | Typed Stage Warnings response and owned endpoint. |
| Backend analytics | evaluation/analytics.py | Mutually-exclusive errors/warnings and safe reason mapping. |
| Backend runtime | evaluation/campaign_engine.py | Safe failure diagnostics before result persistence. |
| Backend tests | tests/test_evaluation_research_analytics.py, tests/test_evaluation_analytics_api.py, tests/test_campaign_engine.py, tests/test_evaluation_export_redaction.py | Projection, auth, persistence, redaction. |
| Frontend contract | src/types/evaluation.ts, src/services/evaluationApi.ts, src/services/evaluationApi.test.ts | Stage-warning types and client. |
| Frontend integration | src/pages/EvaluationCenter.tsx, src/pages/EvaluationCenter.ui.test.tsx | Lazy warning loading and export error handoff. |
| Frontend UI | src/components/evaluation/AblationDashboardTab.tsx, src/components/evaluation/AblationDashboardTab.test.tsx | Warnings, download, preview, and Sanitized Errors. |

---

### Task 1: Add mutually-exclusive Stage Warnings analytics

**Files:**
- Modify: pdftopng/evaluation/campaign_schemas.py:698-714
- Modify: pdftopng/evaluation/analytics.py:182-193,801-848
- Modify: pdftopng/evaluation/router.py:612-630
- Test: pdftopng/tests/test_evaluation_research_analytics.py
- Test: pdftopng/tests/test_evaluation_analytics_api.py

**Interfaces:**

~~~
class StageWarningRow(BaseModel):
    run_id: str
    campaign_id: str
    question_id: str
    mode: CampaignMode
    stage_name: str
    status: Literal["partial", "required_but_not_satisfied"]
    failure_reason: str
    created_at: datetime

class CampaignStageWarningsResponse(BaseModel):
    campaign_id: str
    rows: list[StageWarningRow] = Field(default_factory=list)

async def campaign_stage_warnings(
    self, *, user_id: str, campaign_id: str
) -> CampaignStageWarningsResponse: ...
~~~

Route: GET /api/evaluation/campaigns/{campaign_id}/stage-warnings.

- [ ] **Step 1: Write failing analytics tests**

Seed a completed result with three trace events:

~~~
partial_graph = EvaluationTraceEvent(
    stage_name="agentic_v9_graph_locator",
    status="partial",
    payload={"execution_state": "required_but_not_satisfied"},
    error={"reason": "no_eligible_graph_source_evidence"},
    ...,
)
not_triggered = EvaluationTraceEvent(
    stage_name="agentic_v9_graph_locator",
    status="success",
    payload={"execution_state": "not_triggered"},
    error={},
    ...,
)
failed_generation = EvaluationTraceEvent(
    stage_name="answer_generation",
    status="failed",
    error={"code": "PROVIDER_TIMEOUT", "message": "request timed out"},
    ...,
)
~~~

Assert campaign_errors contains only answer_generation/PROVIDER_TIMEOUT.
Assert campaign_stage_warnings contains only the partial graph row and its exact
failure_reason. Add a second case using only error.reason =
page_assets_unavailable; it must not produce Error details unavailable.

- [ ] **Step 2: Write failing route/auth tests**

Call the endpoint as the owner and assert one row with run_id, Q14, agentic,
agentic_v9_graph_locator, partial, and no_eligible_graph_source_evidence.
Call it as another user and assert 404.

- [ ] **Step 3: Run tests to verify failure**

Run: .\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_analytics.py tests/test_evaluation_analytics_api.py -q

Expected: FAIL because schemas, method, and route do not exist; partial trace
events are still admitted by _build_campaign_errors.

- [ ] **Step 4: Implement minimal backend contract**

1. Add both Pydantic models next to SanitizedErrorRow.
2. Add the authenticated route next to /errors.
3. Add _sanitize_failure_reason(raw), reusing secret, multiline, traceback,
   and length protections already used by _sanitize_error_message.
4. Change _build_campaign_errors: trace/LLM rows are admitted only when
   status == "failed", never merely because error is a nonempty dict.
5. Add _build_campaign_stage_warnings. Admit only agentic_v9_ stages that are
   partial or whose payload.execution_state is required_but_not_satisfied.
   Reject not_triggered and not_requested. Resolve reason in this exact order:

~~~
reason = (
    item.error.get("reason")
    or item.payload.get("failure_reason")
    or item.error.get("message")
    or "capability_gap_reason_not_recorded"
)
~~~

6. Set row.status to required_but_not_satisfied when that is the payload
   execution_state; otherwise preserve partial. Join warning rows to their
   owning result for question and mode and sort by
   (created_at, run_id, stage_name).
7. campaign_stage_warnings loads the same owned campaign context and trace
   events used by campaign_errors.

- [ ] **Step 5: Verify and commit**

Run: .\.venv\Scripts\python.exe -m pytest tests/test_evaluation_research_analytics.py tests/test_evaluation_analytics_api.py tests/test_evaluation_export_redaction.py -q

Expected: PASS; redaction and ownership tests remain unchanged.

~~~
git add evaluation/campaign_schemas.py evaluation/analytics.py evaluation/router.py tests/test_evaluation_research_analytics.py tests/test_evaluation_analytics_api.py
git commit -m "fix(evaluation): separate stage warnings from failures"
~~~

### Task 2: Persist safe diagnostics for every future failed run

**Files:**
- Modify: pdftopng/evaluation/campaign_engine.py:194-215,1880-2055
- Test: pdftopng/tests/test_campaign_engine.py

**Interfaces:**

Every failed result receives derived_metrics.failure_diagnostics:

~~~
{
  "error_code": str,
  "safe_error_message": str,
  "last_completed_stage": str | None,
  "provider_status": str | None,
  "retry_count": int,
  "timeout_state": str | None,
  "budget_state": str | None,
}
~~~

- [ ] **Step 1: Write failing tests for both failure paths**

Add a runner that raises a RuntimeError with a fake API key and traceback-like
newline text. Add a second runner that returns BenchmarkExecutionResult with
error_message and an agent trace containing scalar last_completed_stage,
provider_status, retry_count, timeout_state, and budget_state.

For the exception case assert:

~~~
diagnostics = result.derived_metrics["failure_diagnostics"]
assert diagnostics["error_code"] == "RuntimeError"
assert diagnostics["safe_error_message"] != ""
assert "sk-secret" not in diagnostics["safe_error_message"]
assert "traceback" not in diagnostics["safe_error_message"].lower()
assert diagnostics["last_completed_stage"] == "campaign_unit_execution"
~~~

For the result-error case assert the trace scalar fields survive and that an
empty error message becomes failure_reason_not_recorded.

- [ ] **Step 2: Run tests to verify failure**

Run: .\.venv\Scripts\python.exe -m pytest tests/test_campaign_engine.py -k failure_diagnostics -q

Expected: FAIL because no failure_diagnostics exists and raw error strings are
stored.

- [ ] **Step 3: Implement one safe diagnostic builder**

Add near _build_derived_metrics:

~~~
def _safe_failure_message(raw: Any) -> str: ...
def _failure_diagnostics(
    *, unit: CampaignUnit, payload: BenchmarkExecutionResult | Exception
) -> dict[str, Any]: ...
~~~

Rules:
1. Redact secrets and collapse traceback/multiline text to Provider error
   details were redacted.
2. error_code is Exception.__class__.__name__ or an explicit scalar
   agent_trace.error_code.
3. last_completed_stage is a scalar trace field; for bare exceptions use
   campaign_unit_execution, otherwise None.
4. Accept provider_status, timeout_state, and budget_state only as strings;
   accept retry_count only as a nonnegative integer.
5. Never leave safe_error_message blank.

- [ ] **Step 4: Apply it at the persistence source**

Have _build_derived_metrics add failure_diagnostics for Exception and
BenchmarkExecutionResult.error_message cases. Before each failed
CampaignResultRepository.create call, use safe_error_message for
error_message. Give root-span and LLM-usage failure events the same safe
message so every displayed surface agrees.

- [ ] **Step 5: Verify and commit**

Run: .\.venv\Scripts\python.exe -m pytest tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py tests/test_evaluation_research_analytics.py -q

Expected: PASS; every newly persisted failed run has safe nonempty diagnostics.

~~~
git add evaluation/campaign_engine.py tests/test_campaign_engine.py
git commit -m "fix(evaluation): persist safe failed run diagnostics"
~~~

### Task 3: Add the frontend Stage Warnings contract and lazy load

**Files:**
- Modify: Multimodal_RAG_System/src/types/evaluation.ts:946-965
- Modify: Multimodal_RAG_System/src/services/evaluationApi.ts:357-387
- Modify: Multimodal_RAG_System/src/services/evaluationApi.test.ts
- Modify: Multimodal_RAG_System/src/pages/EvaluationCenter.tsx:30-58,208-241,408-419
- Test: Multimodal_RAG_System/src/pages/EvaluationCenter.ui.test.tsx

**Interfaces:**

~~~
export interface StageWarningRow {
  run_id: string;
  campaign_id: string;
  question_id: string;
  mode: CampaignMode;
  stage_name: string;
  status: 'partial' | 'required_but_not_satisfied';
  failure_reason: string;
  created_at: string;
}
export interface CampaignStageWarningsResponse {
  campaign_id: string;
  rows: StageWarningRow[];
}
~~~

- [ ] **Step 1: Write failing API-client and tab-load tests**

In evaluationApi.test.ts, mock api.get and assert
getCampaignStageWarnings('cmp-1') requests
/api/evaluation/campaigns/cmp-1/stage-warnings.

In EvaluationCenter.ui.test.tsx, mock getCampaignStageWarnings, select the
Ablation tab, and assert it is called with cmp-1. Also assert
exportCampaignAnalysis remains uncalled: no export preview endpoint is added.

- [ ] **Step 2: Run tests to verify failure**

Run: npm test -- --run src/services/evaluationApi.test.ts src/pages/EvaluationCenter.ui.test.tsx

Expected: FAIL because the type, API method, and tab data are absent.

- [ ] **Step 3: Implement types, client, and data composition**

1. Add the two interfaces next to CampaignErrorsResponse.
2. Add:

~~~
export async function getCampaignStageWarnings(
  campaignId: string
): Promise<CampaignStageWarningsResponse> {
  const response = await api.get<CampaignStageWarningsResponse>(
    `/api/evaluation/campaigns/${campaignId}/stage-warnings`
  );
  return response.data;
}
~~~

3. In Ablation's Promise.all, fetch stage warnings with ablation, human
   evaluation, and errors.
4. Pass stageWarnings and selectedCampaignId to AblationDashboardTab.
5. Pass onExportError={(message) => setDashboardError(message)}; do not add a
   parallel notification subsystem.

- [ ] **Step 4: Verify and commit**

Run: npm test -- --run src/services/evaluationApi.test.ts src/pages/EvaluationCenter.ui.test.tsx

Expected: PASS; warnings load only with Ablation and export is never eagerly
called.

~~~
git add src/types/evaluation.ts src/services/evaluationApi.ts src/services/evaluationApi.test.ts src/pages/EvaluationCenter.tsx src/pages/EvaluationCenter.ui.test.tsx
git commit -m "feat(evaluation-ui): load stage warning diagnostics"
~~~

### Task 4: Render capability gaps and make Export JSON download

**Files:**
- Modify: Multimodal_RAG_System/src/components/evaluation/AblationDashboardTab.tsx:21-34,92-101,233-310
- Test: Multimodal_RAG_System/src/components/evaluation/AblationDashboardTab.test.tsx

**Interfaces:**

~~~
interface AblationDashboardData {
  // existing fields
  stageWarnings?: CampaignStageWarningsResponse;
}
interface AblationDashboardTabProps {
  data?: AblationDashboardData;
  campaignId: string;
  onExportError?: (message: string) => void;
}
~~~

- [ ] **Step 1: Write failing component tests**

Mock exportCampaignAnalysis and browser download seams:

~~~
Object.defineProperty(URL, "createObjectURL", { value: vi.fn(() => "blob:campaign-export") });
Object.defineProperty(URL, "revokeObjectURL", { value: vi.fn() });
const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
~~~

Click Export JSON with defaults. Assert the exact six-option request payload,
the temporary anchor click, download attribute cmp-1-redacted.json, URL
revocation, and Preview: 2 runs, 1 LLM calls after the mocked response.

Enable Full prompts and assert cmp-1-custom.json. Add a Graph partial and
Visual required warning, plus a not_triggered fixture. Assert only the first
two reasons render under Stage warnings / capability gaps. Reject export and
assert onExportError is called while Preview: not generated stays unchanged.

- [ ] **Step 2: Run test to verify failure**

Run: npm test -- --run src/components/evaluation/AblationDashboardTab.test.tsx

Expected: FAIL because the button has no handler and warnings do not render.

- [ ] **Step 3: Implement local export state and download helper**

1. Import useState and exportCampaignAnalysis.
2. Keep exportPreview in component state, initially undefined. Before export
   show Preview: not generated; do not use an empty page-level preview.
3. Add:

~~~
function exportFilename(campaignId: string, options: Required<ExportCampaignRequest>): string {
  const rawContent = options.include_full_prompts || options.include_raw_trace_payloads;
  return `${campaignId}-${rawContent ? "custom" : "redacted"}.json`;
}
~~~

4. The async handler disables Export JSON while awaiting the API. It builds
   JSON.stringify(exported, null, 2), creates an application/json Blob, clicks
   a temporary anchor, removes it, revokes the object URL in finally, and then
   stores the returned export as the preview.
5. On rejection, retain the previous preview, call onExportError with the safe
   request error text, and clear the loading state.
6. Preserve all checkbox options exactly when calling the API. Keep the full
   prompts badge.

- [ ] **Step 4: Render warnings and preserve failure table semantics**

Render Stage warnings / capability gaps immediately before Sanitized Errors
with Question, Mode, Stage, Status, Reason columns. Render
No capability gaps recorded. for an empty list. Use API-provided rows directly
and never infer warnings from errors in the browser. Keep Sanitized Errors for
failed rows only; no not_triggered row is displayed.

- [ ] **Step 5: Verify and commit**

Run: npm test -- --run src/components/evaluation/AblationDashboardTab.test.tsx src/services/evaluationApi.test.ts src/pages/EvaluationCenter.ui.test.tsx

Expected: PASS; download, filename, preview update, warning/error separation,
and lazy data loading all pass.

~~~
git add src/components/evaluation/AblationDashboardTab.tsx src/components/evaluation/AblationDashboardTab.test.tsx
git commit -m "fix(evaluation-ui): export JSON and show capability gaps"
~~~

### Task 5: Cross-stack verification

**Files:** Modify no additional file unless verification exposes a direct contract mismatch in the files named above.

- [ ] **Step 1: Run focused backend suite**

Run: .\.venv\Scripts\python.exe -m pytest tests/test_campaign_engine.py tests/test_evaluation_research_analytics.py tests/test_evaluation_analytics_api.py tests/test_evaluation_export_redaction.py -q

Expected: PASS.

- [ ] **Step 2: Run focused frontend suite**

Run: npm test -- --run src/components/evaluation/AblationDashboardTab.test.tsx src/services/evaluationApi.test.ts src/pages/EvaluationCenter.ui.test.tsx

Expected: PASS.

- [ ] **Step 3: Run required static checks**

Run in pdftopng: .\.venv\Scripts\python.exe -m compileall evaluation

Run in Multimodal_RAG_System:
~~~
npm run lint
npm run build
~~~

Expected: exit code 0. If a command has a pre-existing failure, record its exact
output separately and do not attribute it to this change.

- [ ] **Step 4: Verify commit scope**

Run git status --short and git log --oneline -4 in each repository. Confirm only
the task commits are added; do not stage .pytest-tmp, data, or previous
untracked plans/specifications.

---

## Plan Self-Review

- **Coverage:** Task 1 separates partial warnings from errors and exposes reasons. Task 2 makes future failed runs diagnosable without retaining secrets. Task 3 connects the typed API lazily. Task 4 makes the existing export contract usable and fixes false Preview 0 counts. Task 5 verifies both stacks.
- **Scope:** The design adds no preview endpoint, no telemetry storage migration, and no Graph/Visual policy change.
- **Consistency:** StageWarningRow exists in backend, frontend types, API client, page loading, and rendering. Filename classification uses only raw-prompt/raw-trace options, matching the approved spec.
