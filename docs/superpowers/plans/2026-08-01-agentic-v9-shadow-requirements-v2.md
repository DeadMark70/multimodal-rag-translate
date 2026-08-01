# Agentic v9 Shadow Requirements v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the behavior-neutral Agentic v9 shadow analyzer so it conservatively decomposes atomic answer obligations, separates response constraints, preserves mixed representation needs, and exports auditable candidate-only diagnostics without adding an LLM call or changing runtime behavior.

**Architecture:** Add a focused deterministic question-decomposition module and keep `requirement_shadow.py` responsible for public v2 schemas, representation analysis, and candidate mapping. The existing runtime remains a fail-soft post-execution observer; export continues to copy the durable trace generically.

**Tech Stack:** Python 3.11+, dataclasses, regular expressions, Pydantic v2, LangChain `Document`, pytest, pytest-asyncio, Ruff.

## Global Constraints

- The authoritative design is `docs/superpowers/specs/2026-08-01-agentic-v9-shadow-requirements-v2-design.md`.
- Perform zero new LLM, embedding, retrieval, reranker, graph, or visual calls.
- Do not read QID, ground truth, expected sources, expected evidence, paper-specific answers, or evaluation-only metadata in production code.
- Do not change routing, retrieval, reranking, graph/visual execution, sufficiency, context packing, synthesis, response status, token accounting, or latency semantics.
- Emit at most eight requirements and eight response constraints.
- Candidate evidence never becomes `supported`; `supported_count` must remain zero.
- Any analyzer exception must fail soft and must not fail or downgrade a run.
- Use `apply_patch` for edits and stage only files named in the active task.
- Do not run Ruff formatter repository-wide; existing large files are not uniformly Ruff-formatted.

---

## File Structure

- Create `data_base/agentic_v9/requirement_decomposition.py`: deterministic span protection, top-level block parsing, obligation extraction, constraint classification, entity-distributive expansion, bounds, and fallback.
- Modify `data_base/agentic_v9/requirement_shadow.py`: public v2 Pydantic contract, mixed representation classification, stable candidate identity, and conversion from decomposition drafts.
- Modify `evaluation/agentic_v9_campaign_runtime.py`: update only the fail-soft unavailable payload version.
- Create `tests/test_agentic_v9_requirement_decomposition.py`: syntax and semantic decomposition unit tests.
- Modify `tests/test_agentic_v9_requirement_shadow.py`: v2 schema, mixed representation, candidate identity, and 16-question offline smoke tests.
- Modify `tests/test_agentic_v9_campaign_runtime.py`: runtime neutrality and fail-soft v2 assertions.
- Modify `tests/test_evaluation_export_redaction.py`: durable redacted v2 requirements/constraints export assertion.

---

### Task 1: Safe Structural Decomposition Foundation

**Files:**
- Create: `data_base/agentic_v9/requirement_decomposition.py`
- Create: `tests/test_agentic_v9_requirement_decomposition.py`

**Interfaces:**
- Produces: `split_top_level_blocks(question: str) -> tuple[QuestionBlock, ...]`
- Produces: immutable `QuestionBlock(text: str, method: DecompositionMethod, confidence: DecompositionConfidence)`
- Produces for later tasks: `DecompositionMethod = Literal["numbered", "coordinated", "entity_distributive", "fallback"]`
- Produces for later tasks: `DecompositionConfidence = Literal["high", "medium", "low"]`

- [ ] **Step 1: Write failing tests for protected numeric spans and genuine numbered blocks**

Add tests containing a full Q16-like question and an ambiguous unnumbered question:

```python
from data_base.agentic_v9.requirement_decomposition import split_top_level_blocks


def test_numbered_blocks_do_not_split_parenthetical_ids_or_decimals() -> None:
    question = (
        "請回答以下三個子問題：1. GEPAR3D 中 tooth 1) 誤判為 tooth 32) "
        "的懲罰與原因為何？ 2. ODES 的 P(x,y) 公式與 |A^c(x,y)| 定義為何？ "
        "3. noise 0.4 時兩個 Dice 與 Theorem 1 的 m 範圍為何？"
    )
    blocks = split_top_level_blocks(question)
    assert len(blocks) == 3
    assert "tooth 1)" in blocks[0].text
    assert "tooth 32)" in blocks[0].text
    assert "0.4" in blocks[2].text
    assert all(block.method == "numbered" for block in blocks)
    assert all(block.confidence == "high" for block in blocks)


def test_unsequenced_numeric_text_falls_back_without_splitting() -> None:
    blocks = split_top_level_blocks("比較 class 1) 與 class 2) 在 v3.1 的結果。")
    assert len(blocks) == 1
    assert blocks[0].method == "fallback"
    assert blocks[0].confidence == "low"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_requirement_decomposition.py -q -p no:cacheprovider
```

Expected: collection fails because `requirement_decomposition` does not exist.

- [ ] **Step 3: Implement immutable structural types and a boundary-safe scanner**

Create the module with these public types:

```python
from dataclasses import dataclass
from typing import Literal

DecompositionMethod = Literal[
    "numbered", "coordinated", "entity_distributive", "fallback"
]
DecompositionConfidence = Literal["high", "medium", "low"]


@dataclass(frozen=True, slots=True)
class QuestionBlock:
    text: str
    method: DecompositionMethod
    confidence: DecompositionConfidence
```

Implement `split_top_level_blocks` using these exact safeguards:

1. Normalize repeated whitespace while preserving punctuation.
2. Find numeric markers only at the start or after `：:；;。！？!?`.
3. Accept numeric forms `1.` and `1、`; accept Chinese forms `（一）` and `一、`.
4. Reject candidates inside decimal/version/percentage/range/parenthetical identifier spans.
5. Accept numbered mode only when at least two markers form a monotonic sequence beginning at one.
6. Otherwise return the complete normalized question as one `fallback/low` block.

Keep protected spans as offsets rather than replacing user text, so emitted text is byte-for-byte readable.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all structural tests pass.

- [ ] **Step 5: Add boundary variants and rerun**

Add tests for `1、2、`, `（一）（二）`, percentages, Figure/Table identifiers, and an empty question. Define empty input to return an empty tuple; the public shadow builder will handle its own bounded fallback.

- [ ] **Step 6: Commit Task 1**

```powershell
git add data_base/agentic_v9/requirement_decomposition.py tests/test_agentic_v9_requirement_decomposition.py
git commit -m feat(agentic-v9):add-safe-shadow-decomposition
```

---

### Task 2: Obligations, Constraints, and Conservative Entity Distribution

**Files:**
- Modify: `data_base/agentic_v9/requirement_decomposition.py`
- Modify: `tests/test_agentic_v9_requirement_decomposition.py`

**Interfaces:**
- Produces: `decompose_question(question: str, *, max_requirements: int = 8, max_constraints: int = 8) -> QuestionDecomposition`
- Produces: `DecomposedRequirement(text, method, confidence)`
- Produces: `ResponseConstraintDraft(text, kind)` where `kind` is `conditional_scope | output_format | prohibition | allowed_labels`
- Produces: `QuestionDecomposition(requirements, response_constraints, truncated_requirement_count, truncated_constraint_count)`

- [ ] **Step 1: Write failing tests for constraints instead of false missing requirements**

```python
def test_conditional_scope_is_a_constraint_not_a_requirement() -> None:
    result = decompose_question(
        "Weak-Mamba-UNet 與 Semi-Mamba-UNet 都有 first claim。"
        "請判斷能否唯一決定誰最先；若不能，必須按 claim scope 分開回答。"
    )
    assert len(result.requirements) == 1
    assert [item.kind for item in result.response_constraints] == [
        "conditional_scope"
    ]
    assert "claim scope" in result.response_constraints[0].text


def test_classification_labels_are_constraints() -> None:
    result = decompose_question(
        "僅根據 nnFormer 與 U-Mamba，請分別分類："
        "A. 有量化 ensemble 證據；B. 只有公平比較。"
    )
    assert len(result.requirements) == 2
    assert {item.kind for item in result.response_constraints} == {"allowed_labels"}
    assert all("A." not in item.text and "B." not in item.text for item in result.requirements)
```

- [ ] **Step 2: Write failing tests for coordinated and distributive obligations**

Add exact regression shapes without QID branching:

```python
def test_css_question_yields_four_generic_obligations() -> None:
    result = decompose_question(
        "根據架構描述，請重建 CSS 的特徵融合流程，並說明三個翻轉分支"
        "與 SiamSSM 的運算/累加機制。"
    )
    assert len(result.requirements) == 4
    joined = "\n".join(item.text for item in result.requirements)
    for anchor in ("融合流程", "翻轉分支", "SiamSSM", "累加機制"):
        assert anchor in joined


def test_explicit_other_entities_use_bounded_distribution() -> None:
    result = decompose_question(
        "在 SAMed、MedSAM、SAM-Med2D、SAM-Med3D 中，哪一個符合條件？"
        "請指出唯一符合者與關鍵技術，並簡述另外三者為何不符合。"
    )
    assert len(result.requirements) <= 5
    joined = "\n".join(item.text for item in result.requirements)
    for entity in ("SAMed", "MedSAM", "SAM-Med2D", "SAM-Med3D"):
        assert entity in joined
```

- [ ] **Step 3: Write failing tests for Q15/Q16-like atomicity and unseen text**

The Q15-like assertion must yield four obligations: resolve the referenced Figure alternative, report a Table value, calculate a delta, and report batch size. The Q16-like assertion must yield six obligations after three safe top-level blocks are split by explicit continuation cues.

Add an unseen bilingual case:

```python
def test_unseen_bilingual_obligations_generalize_without_templates() -> None:
    result = decompose_question(
        "Compare Model-A and Model-B：請分別回報 latency 與 memory，"
        "並解釋 trade-off；不要宣稱為通用排名。"
    )
    assert 3 <= len(result.requirements) <= 5
    assert any(item.kind == "prohibition" for item in result.response_constraints)
    assert all(item.confidence in {"high", "medium", "low"} for item in result.requirements)
```

- [ ] **Step 4: Run the focused suite and verify RED**

Run the Task 1 command. Expected: new obligation/constraint assertions fail because only structural block splitting exists.

- [ ] **Step 5: Implement the deterministic classification pipeline**

Add frozen dataclasses for the produced interfaces. Implement in this order:

1. `_extract_constraints`: strip matched conditional, format, prohibition, and allowed-label clauses from obligation input while preserving them as drafts.
2. `_split_obligation_clauses`: split at top-level question marks, semicolons, and explicit continuation cues only when the resulting clause has an obligation cue.
3. `_split_obligation_objects`: propagate an explicit answer verb over a bounded object list introduced by `與`, `以及`, `/`, or `and`; require meaningful noun anchors on both sides.
4. `_extract_referenced_prerequisite`: when a Figure introduces named alternatives and a later obligation selects one alternative, emit one medium-confidence prerequisite requirement without inventing its answer.
5. `_extract_entity_list` and `_expand_distributive`: expand only with `每個`, `各自`, `分別`, or `另外三者`; cap expansion before the global eight-item cap.
6. `_bounded_result`: preserve the first eight requirements/constraints and record both overflow counts.
7. If no valid obligation remains, emit one `fallback/low` requirement containing the non-constraint question text.

Every rule must depend only on generic syntax and lexical cues. Do not import or inspect evaluation golden data.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the Task 1 command. Expected: all decomposition tests pass.

- [ ] **Step 7: Add anti-over-splitting tests**

Cover these cases:

- `Channel-Spatial Siamese` stays one phrase.
- `A^c(x,y)` and `P(x,y)` stay intact.
- A bare `與` between two model names does not automatically create requirements.
- More than eight valid obligations sets `truncated_requirement_count` and returns exactly eight.
- More than eight constraints sets `truncated_constraint_count` and returns exactly eight.

- [ ] **Step 8: Commit Task 2**

```powershell
git add data_base/agentic_v9/requirement_decomposition.py tests/test_agentic_v9_requirement_decomposition.py
git commit -m feat(agentic-v9):classify-shadow-obligations
```

---

### Task 3: Public v2 Shadow Contract and Candidate Mapping

**Files:**
- Modify: `data_base/agentic_v9/requirement_shadow.py`
- Modify: `tests/test_agentic_v9_requirement_shadow.py`

**Interfaces:**
- Consumes: `decompose_question(...) -> QuestionDecomposition`
- Preserves: `build_requirement_shadow(*, question: str, documents: Sequence[Document]) -> RequirementShadowAnalysis`
- Produces schema: `shadow_requirements_v2`
- Produces: `ShadowResponseConstraint`, `information_needs`, `decomposition_method`, `decomposition_confidence`, and bounded truncation counters

- [ ] **Step 1: Update tests first to require the v2 contract**

Change the existing assertions and add:

```python
def test_v2_never_promotes_candidate_to_supported() -> None:
    analysis = build_requirement_shadow(
        question="請回報模型分數。",
        documents=[Document(page_content="模型分數為 0.91。", metadata={})],
    )
    assert analysis.schema_version == "shadow_requirements_v2"
    assert analysis.support_assessment == "candidate_only"
    assert analysis.summary.supported_count == 0
    assert all(item.coverage_status != "supported" for item in analysis.requirements)


def test_mixed_figure_and_markdown_table_needs_are_preserved() -> None:
    analysis = build_requirement_shadow(
        question=(
            "根據 Figure 1 辨識策略 (b)，再從 Table 1 回報 mIoU。"
        ),
        documents=[
            Document(
                page_content=(
                    "Figure 1 summary: strategy (b) fine-tunes all components.\n"
                    "| Strategy | mIoU |\n|---|---:|\n| b | 0.877 |"
                ),
                metadata={"source": "image", "type": "figure"},
            )
        ],
    )
    needs = {need for item in analysis.requirements for need in item.information_needs}
    assert {"visual_pattern", "markdown_table"} <= needs
    assert analysis.summary.supported_count == 0
```

- [ ] **Step 2: Add failing tests for stable fallback evidence identity**

Use two documents without canonical IDs but with different content. Assert refs begin with `content:` and remain stable if document order changes. Assert duplicate content collapses to one candidate ref.

- [ ] **Step 3: Run requirement-shadow tests and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_requirement_shadow.py -q -p no:cacheprovider
```

Expected: schema-version and missing v2-field failures.

- [ ] **Step 4: Implement v2 Pydantic models with compatibility projection**

Update `ShadowRequirement`:

```python
information_need: InformationNeed
information_needs: list[InformationNeed] = Field(min_length=1, max_length=4)
decomposition_method: DecompositionMethod
decomposition_confidence: DecompositionConfidence
```

Add:

```python
class ShadowResponseConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    constraint_id: str = Field(pattern=r"^C[1-8]$")
    kind: Literal[
        "conditional_scope", "output_format", "prohibition", "allowed_labels"
    ]
    text: str = Field(min_length=1, max_length=512)
```

Update `RequirementShadowAnalysis` to v2 with `response_constraints` and `truncated`. Add `constraint_count`, `low_confidence_count`, `truncated_requirement_count`, and `truncated_constraint_count` to the summary. Keep `supported_count` defaulted and validated at zero.

- [ ] **Step 5: Replace singular precedence with ordered mixed-needs analysis**

Implement `_information_needs(text) -> list[InformationNeed]` by recording all matching representations in first-mention order and deduplicating. Use the first item as the compatibility `information_need`. Add `plain_text` only when a separate prose obligation exists; do not add it mechanically to every structured question.

Visual decision rules remain diagnostic:

- Markdown Table alone: `not_requested`.
- Figure provenance plus a candidate image summary: `optional`.
- Exact spatial/pattern request with no candidate summary: `required`.
- Mixed Figure/Table: retain both needs; do not let Table erase Figure.

- [ ] **Step 6: Implement stable evidence identity and candidate mapping**

Use `get_document_id` from `data_base.document_metadata`. Resolve refs in this order:

1. canonical doc ID plus canonical chunk ID;
2. canonical doc ID plus `content-<sha256-prefix>`;
3. `content:<sha256>` when no canonical doc ID exists.

Deduplicate by the complete evidence ref. Continue limiting each requirement to eight candidate refs. Candidate mapping may use representation compatibility and generic anchors but never evaluation metadata.

- [ ] **Step 7: Run requirement-shadow tests and verify GREEN**

Run the Task 3 command. Expected: all tests pass.

- [ ] **Step 8: Add a 16-question offline schema smoke test**

Load only the `question` field from `evaluation/golden/agentic_v9_questions_v2.json`; do not read ground truth, source docs, expected route, atomic facts, or expected evidence. For every question, assert:

```python
assert analysis.schema_version == "shadow_requirements_v2"
assert 1 <= len(analysis.requirements) <= 8
assert len(analysis.response_constraints) <= 8
assert analysis.summary.supported_count == 0
```

Keep the detailed Q5/Q7/Q11/Q13/Q15/Q16 expectations in dedicated generic regression tests rather than branching production logic by QID.

- [ ] **Step 9: Commit Task 3**

```powershell
git add data_base/agentic_v9/requirement_shadow.py tests/test_agentic_v9_requirement_shadow.py
git commit -m feat(agentic-v9):emit-shadow-requirements-v2
```

---

### Task 4: Runtime Fail-soft and Redacted Export Integration

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_evaluation_export_redaction.py`

**Interfaces:**
- Consumes: unchanged `build_requirement_shadow(...)`
- Persists: `agentic_v9.requirement_shadow.schema_version == "shadow_requirements_v2"`
- Exports: existing top-level `requirement_summary` containing v2 requirements and constraints

- [ ] **Step 1: Update runtime tests first**

Change the success test to assert v2 fields and exact provider neutrality:

```python
assert shadow["schema_version"] == "shadow_requirements_v2"
assert shadow["behavior_influence"] is False
assert shadow["support_assessment"] == "candidate_only"
assert provider.ainvoke.await_count == 1
```

Change the failure test to require the unavailable payload to report
`shadow_requirements_v2` while preserving `response_status == "complete"` and
the retrieved documents.

- [ ] **Step 2: Update export test first**

Seed one v2 durable materialization containing one requirement and one
`conditional_scope` constraint. Assert the redacted response returns both under
`requirement_summary`, does not expose prompt/answer secrets, and leaves the
existing run/LLM-call counts unchanged.

- [ ] **Step 3: Run integration tests and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_campaign_runtime.py -q -p no:cacheprovider -k requirement_shadow
.venv\Scripts\python.exe -m pytest tests\test_evaluation_export_redaction.py::test_export_defaults_redact_full_prompts_and_errors_are_sanitized -q -p no:cacheprovider
```

Expected: v1 schema assertions fail before runtime fallback and fixture updates.

- [ ] **Step 4: Make the minimal runtime version change**

Keep the analyzer invocation position and try/except boundary unchanged. Update
only the unavailable diagnostic payload from `shadow_requirements_v1` to
`shadow_requirements_v2`. Do not add runtime branching based on requirements,
constraints, confidence, coverage, or visual decisions.

No analytics implementation change is expected because export already copies
the durable requirement payload through `redact_sensitive_value`.

- [ ] **Step 5: Run integration tests and verify GREEN**

Run both Task 4 commands. Expected: all selected tests pass.

- [ ] **Step 6: Prove token/call neutrality in the runtime regression**

For the existing simple non-comparison runtime fixture, assert the provider has
exactly one final-answer invocation and the returned usage remains the fixture's
measured input/output total. Do not mock a separate Shadow provider because no
such provider may exist.

- [ ] **Step 7: Commit Task 4**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_export_redaction.py
git commit -m test(agentic-v9):verify-shadow-v2-observability
```

---

### Task 5: Full Verification and Handoff

**Files:**
- Verify only; no production edits expected.

**Interfaces:**
- Verifies every interface and constraint produced by Tasks 1-4.

- [ ] **Step 1: Run all focused and related tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_requirement_decomposition.py tests\test_agentic_v9_requirement_shadow.py tests\test_agentic_v9_campaign_runtime.py tests\test_evaluation_v9_attempt_persistence.py tests\test_campaign_schemas.py -q --disable-warnings -p no:cacheprovider
.venv\Scripts\python.exe -m pytest tests\test_evaluation_export_redaction.py::test_export_defaults_redact_full_prompts_and_errors_are_sanitized -q --disable-warnings -p no:cacheprovider
```

Expected: zero failures.

- [ ] **Step 2: Run lint without repository-wide formatting**

```powershell
.venv\Scripts\python.exe -m ruff check data_base\agentic_v9\requirement_decomposition.py data_base\agentic_v9\requirement_shadow.py evaluation\agentic_v9_campaign_runtime.py tests\test_agentic_v9_requirement_decomposition.py tests\test_agentic_v9_requirement_shadow.py tests\test_agentic_v9_campaign_runtime.py tests\test_evaluation_export_redaction.py
.venv\Scripts\python.exe -m ruff format --check data_base\agentic_v9\requirement_decomposition.py tests\test_agentic_v9_requirement_decomposition.py
git diff --check
```

Expected: Ruff check passes, the two new files are formatted, and Git reports no whitespace errors.

- [ ] **Step 3: Review behavioral isolation**

Inspect the final diff and confirm:

- no new invocation of `get_llm`, provider factory, embedding, retrieval, reranker, graph, or visual services;
- runtime does not branch on Shadow v2 output;
- no QID, ground-truth, expected-source, or paper-specific production rule exists;
- no unrelated workspace files are staged.

- [ ] **Step 4: Record the smoke handoff**

After deployment, run one Agentic v9 repeat over the fixed 16 questions and
export redacted JSON with raw trace payload. Review only:

- requirement counts and confidence;
- constraints separated from requirements;
- Q16 protected numeric spans;
- Q15 mixed Figure/Table needs;
- candidate refs without `unknown:chunk-N`;
- absence of a Shadow LLM phase.

Do not enable corrective retrieval or alter Visual routing based on this smoke.

- [ ] **Step 5: Commit any test-only verification adjustment if required**

If verification required no changes, do not create an empty commit. If a test
fixture needed correction, stage only that fixture/test and use:

```powershell
git commit -m test(agentic-v9):close-shadow-v2-regressions
```

## Completion Definition

The implementation is complete only when all Task 5 commands pass, the diff
proves behavioral isolation, and a new redacted 16-question export contains
`shadow_requirements_v2` for every completed run. Promotion from Shadow
diagnostics to behavioral control remains explicitly out of scope.
