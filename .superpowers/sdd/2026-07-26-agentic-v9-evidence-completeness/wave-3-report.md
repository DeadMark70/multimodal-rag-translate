# Wave 3 Report — Slot-Bound Retrieval and Missing-Slot Repair

Date: 2026-07-26
Worktree: `D:\flutterserver\pdftopng\.worktrees\agentic-v9-evidence-completeness`
Wave base: `461387c2b91abd00eac9bd69ccb0f738d8433d65`
Branch: `feature/agentic-v9-evidence-completeness`

## Status

Tasks 8–9 were implemented sequentially with test-first RED/GREEN cycles and
one required implementation commit per task. Retrieval and repair remain
answer-free, source-authorized, locator-aware, bounded by the setup/runtime
reserve, and fail closed at sufficiency.

## Task 8 — Compile source- and locator-aware retrieval tasks

Commit:
`cc17eb01a1ded862e26dbc3d47bf46f3f360a687 feat(agentic-v9): bind retrieval to atomic slots`

Files:

- `data_base/agentic_v9/retrieval_tasks.py`
- `evaluation/agentic_v9_campaign_runtime.py`
- `tests/test_agentic_v9_retrieval_tasks.py`
- `tests/test_agentic_v9_campaign_runtime.py`

Decisions:

- QueryContract v2 slots are grouped only when canonical authorized document
  IDs, locator hints, and visual policy are compatible.
- Every group retains the ordered individual `target_slot_ids`.
- Group queries use only source-name hints, entity IDs, answer-free slot
  descriptions, and locator hints. The original question is excluded.
- Source-name hints are intersected with the authoritative
  `source_name_to_doc_ids` mapping and the run-wide authorized scope.
- Evidence packet construction filters each claimed slot against that slot's
  `authorized_source_doc_ids`; globally authorized but slot-unauthorized
  documents are discarded.
- Legacy v1 contracts keep their existing route-specific task behavior.

RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_retrieval_tasks.py::test_q16_compiles_atomic_source_and_locator_groups_without_answer_text tests/test_agentic_v9_campaign_runtime.py::test_text_evidence_outside_atomic_slot_authorized_ids_cannot_support_it -q
```

Result: `2 failed`. Q16 had no atomic ODES/Table 3/Theorem 1 task groups, and a
globally authorized `doc-b` packet incorrectly supported a `doc-a`-only slot.

GREEN:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py -q
```

Result: `25 passed, 24 warnings`.

Focused Ruff:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base/agentic_v9/retrieval_tasks.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py
```

Result: `All checks passed!`

## Task 9 — Group and persist missing-slot repair

Commit:
`b072b3fb14739d07357410462827fe3acd2eead3 feat(agentic-v9): repair unresolved atomic slots`

Files:

- `data_base/agentic_v9/repair.py`
- `data_base/agentic_v9/execution_core.py`
- `evaluation/agentic_v9_campaign_runtime.py`
- `evaluation/trace_schemas.py`
- `tests/test_agentic_v9_repair.py`
- `tests/test_agentic_v9_execution_core.py`
- `tests/test_agentic_v9_campaign_runtime.py`

Decisions:

- Repair eligibility is the intersection of required contract slots,
  `repairable_slot_ids`, and persisted slot resolutions whose status is exactly
  `not_found`.
- `supported`, `conflicted`, and `explicitly_unavailable` resolutions are never
  repaired, even if a malformed caller includes them in `repairable_slot_ids`.
- The grouping key is the ordered canonical authorized source group, normalized
  locator hints (identifier/type), and compatible entity terms.
- Each repair task narrows the source scope; repair cannot introduce a
  canonical document ID or source-name mapping absent from the contract.
- Repairs are hard-capped at two tasks per round and two rounds total, in both
  the planner and execution core.
- Sufficiency is recomputed immediately after every executed repair round.
- Serialized repair records contain the round, target slot IDs, task query,
  source constraints, locator constraints, resulting evidence IDs, and a
  post-sufficiency stop reason (`evidence_complete`, `continue_repair`,
  `no_repairable_slots`, or `repair_round_cap_reached`).
- The typed trace schema rejects more than two rounds, unordered/duplicate
  rounds, or executed repair records without a persisted stop reason.

RED:

1. Grouping/status RED:

   ```powershell
   D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_repair.py::test_atomic_repair_groups_only_required_not_found_slots_by_constraints tests/test_agentic_v9_repair.py::test_repair_has_an_absolute_two_round_cap -q
   ```

   Result: `1 failed, 1 passed`; the old planner selected supported/conflicted
   slots first and emitted one task per slot.

2. Core round-cap RED:

   ```powershell
   D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_execution_core.py::test_core_recomputes_sufficiency_after_each_of_at_most_two_repairs -q
   ```

   Result: `1 failed`; the core executed repair rounds `[1, 2, 3, 4, 5]`.

3. Q16 persisted-trace RED:

   ```powershell
   D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py::test_q16_repair_trace_persists_constraints_evidence_and_stop_reason -q
   ```

   Result: `1 failed`; repair tasks and constraints were present, but
   `resulting_evidence_ids` remained empty.

4. Typed persistence RED:

   ```powershell
   D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_repair.py::test_persisted_trace_rejects_executed_repair_without_stop_reason -q
   ```

   Result: `1 failed`; an executed repair without a stop reason was accepted.

Focused GREEN:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_repair.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py -q
```

Result: `33 passed, 24 warnings` before the final trace-schema regression was
added; that regression separately passed after GREEN.

Focused Ruff:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base/agentic_v9/repair.py data_base/agentic_v9/execution_core.py evaluation/agentic_v9_campaign_runtime.py evaluation/trace_schemas.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py
```

Result: `All checks passed!`

## Wave 3 gate — Deterministic Q16 repair

The fixture starts with:

- ODES `Equation 2` slot `S3`: `not_found`
- U-KAN `Table 3` slot `S5`: supported
- U-KAN `Theorem 1` slot `S7`: `not_found`

Round 1 emits exactly two repairs:

- `S3`, authorized IDs `["odes"]`, locator `["Equation 2"]`
- `S7`, authorized IDs `["ukan"]`, locator `["Theorem 1"]`

The resulting trace records both repair evidence IDs and
`stop_reason="evidence_complete"`. The original question contains
`SECRET-ODES` and `SECRET-THEOREM`; neither string appears in a repair query.

Final verification:

```powershell
$env:EVALUATION_TEST_TMPDIR='C:\Users\user\AppData\Local\Temp\agentic-v9-wave3'
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_repair.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_schemas.py tests/test_evaluation_v9_attempt_persistence.py -q -o cache_dir='C:\Users\user\AppData\Local\Temp\agentic-v9-wave3-cache'
```

Result: `78 passed, 24 warnings`.

## Self-review

- `git diff --check` was clean before each commit.
- Touched-file Ruff was clean with `--no-cache`.
- Task 8 and Task 9 each have the exact requested commit message and file set.
- No expected answer, benchmark gold field, or prior generated answer enters
  initial or repair task compilation.
- Per-slot canonical ID checks apply when evidence packets are constructed, not
  only when retrieval is requested.
- Source narrowing preserves authoritative name-to-ID pairs and never adds IDs.
- Setup-owned deadline/final-reserve checks remain authoritative; repair exits
  when final reserve is unavailable.
- Structured final answer and deterministic sufficiency behavior were not
  relaxed.

## Concerns

1. The isolated worktree has no local `.venv`; verification used the main
   checkout venv at `D:\flutterserver\pdftopng\.venv`.
2. The worktree's existing `.pytest-tmp` and `.ruff_cache` directories reject
   new temp files under the sandbox ACL. Ruff was run with `--no-cache`, and the
   persistence suite was rerun successfully with an OS temp root.
3. The 24 warnings are existing `storage3` Pydantic v2 deprecations plus a
   pytest cache write warning; Wave 3 introduced no new warning category.
4. Repository-wide `pytest -q` cannot collect in this worktree because six
   legacy experiment/RAGAS modules import an absent `experiments` package.
   Retrying with only those six modules excluded reached `1083 passed,
   1 skipped`, but also produced `148 failed, 56 errors`; the failures are
   dominated by isolated-worktree environment/data ACL gaps (`config.env`
   absent, SQLite/database files unable to open, and missing benchmark/data
   artifacts). The focused Wave 3 plus adjacent schema/persistence verification
   remains the authoritative green result: `78 passed`.

## Fix round 1/5 — Constraint retention, canonical repair grouping, terminal repair state

Commit:
`0a780b399efeea7fb7f50ff82066aa7306aeaaad fix(agentic-v9): enforce atomic retrieval repair constraints`

Root causes:

- Runtime state retained only `task_id -> slot_ids`, discarding each
  `RetrievalTask`'s narrowed source scope and locator contract. Packet
  construction consequently checked the run-wide scope and stamped every
  grouped slot.
- Repair grouping used raw ordered locator/entity tuples, so spelling, case,
  whitespace, and entity order consumed separate tasks under the two-task cap.
- The core's loop condition silently exited on reserve/deadline and called
  `plan_repair` once more before noticing terminal sufficiency. The adapter's
  provisional `continue_repair` could therefore remain durable, and a
  successful round could append an empty next-round decision.

Changes:

- Added shared `slot_constraints.py` for authoritative direct-ID/source-name
  resolution, canonical locator type+identifier sets, normalized unordered term
  sets, and structured chunk-locator matching.
- Retrieval compilation, repair grouping, and packet binding use the same
  source resolver.
- Runtime state retains full retrieval tasks by ID. Packet construction
  enforces task scope, task locator constraints, per-slot canonical source
  scope, and per-slot locator compatibility independently.
- Chunk projections and persisted `SourceLocator` values retain figure, table,
  formula, section, printed-page, bbox, and PDF-page metadata.
- Equivalent `Table 3` and compatible entity variants group before the
  two-task cap, preventing later distinct locator groups from starvation.
- The core checks complete/no-repairable sufficiency before requesting another
  repair decision and reports terminal loop reasons to the adapter.
- Terminal `final_budget_protected`, `deadline_exhausted`, completion, and cap
  reasons replace provisional `continue_repair` and cannot be overwritten by
  the final sufficiency projection.

RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py::test_source_name_only_slot_rejects_a_different_globally_authorized_doc tests/test_agentic_v9_campaign_runtime.py::test_same_document_chunk_with_wrong_locator_cannot_support_slot tests/test_agentic_v9_campaign_runtime.py::test_grouped_task_chunk_is_bound_only_to_its_matching_atomic_slot tests/test_agentic_v9_repair.py::test_equivalent_locator_and_term_variants_group_before_two_task_cap tests/test_agentic_v9_execution_core.py::test_core_does_not_request_repair_when_initial_sufficiency_is_terminal tests/test_agentic_v9_execution_core.py::test_core_records_terminal_reserve_reason_after_executed_repair -q
```

Result: `6 failed`. Failures showed source-name-only leakage, wrong-locator
support, loss of full task constraints, grouping starvation, one unnecessary
repair decision on complete evidence, and no terminal callback.

Persisted reserve/empty-round RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py::test_runtime_persists_terminal_reserve_reason_after_repair tests/test_agentic_v9_campaign_runtime.py::test_q16_repair_trace_persists_constraints_evidence_and_stop_reason -q
```

Result: `2 failed`. The reserve-stopped repair persisted
`continue_repair`, and Q16 appended a second repair record because locator-bound
fixtures lacked structured locator metadata.

GREEN:

```powershell
$env:EVALUATION_TEST_TMPDIR='C:\Users\user\AppData\Local\Temp\agentic-v9-wave3-fix1-final'
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_sufficiency_gate.py -q -o cache_dir='C:\Users\user\AppData\Local\Temp\agentic-v9-wave3-fix1-final-cache'
```

Result: `55 passed, 24 warnings`.

Focused Ruff:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base/agentic_v9/slot_constraints.py data_base/agentic_v9/retrieval_tasks.py data_base/agentic_v9/repair.py data_base/agentic_v9/execution_core.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_sufficiency_gate.py
```

Result: `All checks passed!`

Self-review:

- A source-name-only slot mapped to `doc-a` rejects a globally authorized
  `doc-b`.
- A `Table 4` chunk cannot support a `Table 3` slot in the same document.
- One grouped task chunk can support only the independently matching slot.
- Equivalent locator/term variants group without expanding source scope.
- Successful Q16 repair persists exactly one executed round.
- Reserve exhaustion after an executed repair persists the terminal
  `final_budget_protected` reason.
- Repair and retrieval queries remain answer-free; no gold fields were added.

## Fix round 2/5 — Co-located slot disambiguation, terminal precedence, v1 compatibility

Commit:
`7b57194 fix(agentic-v9): disambiguate grouped slot evidence`

Root causes:

- Packet binding treated source and locator compatibility as sufficient for
  every slot in a grouped v2 task. Two atomic slots sharing a document and
  `Table 3` therefore both received a chunk even when its text described only
  one requested fact.
- Python's `while ... else` recorded `repair_round_cap_reached` immediately
  after the final allowed retrieval, before inspecting that round's successful
  sufficiency result.
- Fix round 1 applied new structured locator enforcement to both contract
  versions, rejecting ordinary v1 chunks that previously bound through their
  retrieval task.

Changes:

- Added deterministic, answer-free content disambiguation for ambiguous v2
  peers that share a canonical source and locator. It derives distinguishing
  terms only from slot descriptions and entity IDs, with expected-answer-type
  shape checks when types distinguish peers; it never reads expected values,
  benchmark gold, or generated answers.
- Kept strict task source, task locator, per-slot source, per-slot locator, and
  content binding behind `contract_version == "2"`. V1 retains its prior
  task-target binding behavior while still honoring the retrieval task's
  authorized source scope.
- Replaced the repair-loop `else` with explicit post-round terminal precedence:
  evidence complete/no repairable slots, then deadline/final reserve, then
  repair cap.

RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py::test_same_source_and_locator_chunk_supports_only_matching_slot_fact tests/test_agentic_v9_campaign_runtime.py::test_v1_locator_hint_accepts_ordinary_retrieved_chunk_without_metadata tests/test_agentic_v9_execution_core.py::test_successful_last_allowed_repair_keeps_completion_terminal_reason -q
```

Result: `4 failed` (the parametrized terminal test contributes two cases).
The grouped chunk stamped both S5 and S6, the v1 ordinary chunk produced no
packet, and successful one-round/two-round repair routes both persisted the cap
reason instead of completion.

Focused GREEN:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_campaign_runtime.py::test_same_source_and_locator_chunk_supports_only_matching_slot_fact tests/test_agentic_v9_campaign_runtime.py::test_v1_locator_hint_accepts_ordinary_retrieved_chunk_without_metadata tests/test_agentic_v9_execution_core.py::test_successful_last_allowed_repair_keeps_completion_terminal_reason -q
```

Result: `4 passed, 24 warnings`.

Requested suite GREEN:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_execution_core.py -q
```

Result: `53 passed, 24 warnings`.

Focused Ruff:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base/agentic_v9/slot_constraints.py evaluation/agentic_v9_campaign_runtime.py data_base/agentic_v9/execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_execution_core.py
```

Result: `All checks passed!`

Self-review:

- The same Table 3 chunk mentioning only the U-KAN metric binds S5 and not the
  co-located proposed-method S6 slot.
- Content matching uses contract descriptors, entity identifiers, and answer
  type shapes only; the tests and implementation contain no expected answer
  values beyond a synthetic chunk fact used to demonstrate number-shaped
  evidence.
- V1 locator-hinted tasks accept ordinary chunks without structured locator
  metadata, preserving the earlier compatibility contract.
- Successful repair on both a one-round route and the second/final round keeps
  `evidence_complete`; cap cannot overwrite completion, deadline, or protected
  final reserve.
- `git diff --check` and focused Ruff were clean before commit.

## Fix round 3/5 — Fail-closed co-located slot association

Commit:
`77f1de8 fix(agentic-v9): fail closed on ambiguous slot evidence`

Root cause:

- `slot_content_matches_chunk` returned supported when co-located peers had no
  unique descriptor terms.
- Expected-answer-type shape checks ran only when peer types differed, so two
  numeric peers could accept text containing no answer value.
- Descriptor presence and answer shape were independent whole-chunk checks.
  A chunk containing both peer names and one value could therefore support both
  slots even when the value was associated with only one name.

Changes:

- Co-located slots with no unique description/entity discriminator now fail
  closed.
- Every co-located slot must have a signal matching its expected answer shape.
  Numeric, equation, and definition signals receive local association checks;
  comparison slots require comparison language.
- Structured locator references are blanked before numeric signal detection,
  so `Table 3` cannot masquerade as a numeric answer.
- Structured signals bind only to the uniquely closest slot discriminator.
  Missing or tied associations fail closed, and multi-word discriminators use
  mean distance so they are not penalized for having more terms.
- The helper remains invoked only by the existing contract-v2 co-located
  runtime path; v1 behavior is unchanged.

RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py -q
```

Result: `3 failed, 1 warning`. The failures independently demonstrated
fail-open identical descriptors, global name/value matching, and acceptance of
numeric text without an answer value.

Focused GREEN:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_campaign_runtime.py::test_same_source_and_locator_chunk_supports_only_matching_slot_fact tests/test_agentic_v9_campaign_runtime.py::test_v1_locator_hint_accepts_ordinary_retrieved_chunk_without_metadata -q
```

Result: `5 passed, 24 warnings`.

Requested suite GREEN:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py -q
```

Result: `35 passed, 24 warnings`.

Focused Ruff:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base/agentic_v9/slot_constraints.py tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py
```

Result: `All checks passed!`

Self-review:

- Identical co-located numeric slots cannot be deterministically distinguished
  and both reject the chunk.
- When both names appear but only U-KAN is locally associated with `0.8`, only
  U-KAN is supported.
- `Table 3` plus a U-KAN descriptor but no numeric answer is unsupported.
- The preserved distinct U-KAN/proposed-method runtime regression passes.
- The preserved v1 ordinary-chunk compatibility regression passes.
- No expected answer, benchmark gold, or prior generated answer enters the
  matching algorithm.
- `git diff --check` and focused Ruff were clean before commit.

## Fix round 4/5 ??Slot-condition numbers and unavailable numeric results

Commit:
`5b2a2c9492e00892b0d9961fc3e8e83eb26f9308 fix(agentic-v9): exclude slot condition numbers`

Root cause:

- `_answer_signal_spans` received only the expected answer type and chunk text,
  so every non-locator number was eligible for local association.
- Numeric query conditions from slot descriptions and identifiers ??including
  the Q16 noise level `0.4`, years, model versions, and other condition counts ??
  could therefore masquerade as a requested numeric result.
- The signal scan did not recognize explicit negative/unavailable result
  statements, so a nearby number could remain eligible even when the requested
  result was stated as not reported.

Changes:

- Numeric signal extraction now receives the complete slot and canonicalizes
  numbers from its description, entity IDs, locator hints, source-name hints,
  and authorized document IDs before local association.
- Slot-derived condition values are removed from candidate answer spans,
  including decimal spelling variants such as `.4`, `0.4`, and `0.40`.
- Structured locator masking remains in place for identifiers such as
  `Table 3`.
- Sentence-local `no result`, `not reported`, and `unavailable` language
  rejects numeric candidates, including semicolon-separated denials.
- A number equal to a condition value is retained only when it is independently
  linked to a non-condition slot descriptor, preserving true evidence such as
  `noise level 0.4, U-KAN Dice was 0.4`.
- The helper is still reached only through the existing v2 co-located binding
  path; v1 behavior is unchanged.

Initial RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py -q
```

Result: `3 failed, 8 passed, 1 warning`. The exact Q16-style condition-only
statement, a negated numeric result, and the year-condition variant were
accepted.

Semicolon-negation RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py::test_negated_numeric_result_is_not_answer_evidence -q
```

Result: `1 failed, 1 warning`. `U-KAN Dice was 0.81; ... result was not
reported` still supported the slot until unavailable detection was widened
from a clause to its containing sentence.

Focused GREEN:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py -q
```

Result: `44 passed, 24 warnings`.

Focused Ruff:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base/agentic_v9/slot_constraints.py tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py
```

Result: `All checks passed!`

Format verification:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff format --no-cache --check data_base/agentic_v9/slot_constraints.py tests/test_agentic_v9_slot_constraints.py
```

Result: `2 files already formatted`.

Self-review:

- The exact statement `U-KAN was evaluated at noise level 0.4; no Dice result
  was reported.` cannot support the Q16-style U-KAN Dice slot.
- Positive results remain supported when a separate Dice value exists, even
  when that value happens to equal the condition value.
- Regressions cover noise levels, decimal normalization, years, model/version
  identifiers in both descriptions and entity IDs, fold counts, and locators.
- Existing fail-closed co-located matching, retrieval compilation, campaign
  runtime binding, and v1 compatibility tests remain green.
- The 24 warnings are the existing pytest cache-option and `storage3` Pydantic
  deprecation warnings.

## Fix round 5/5 ??Universal numeric association and peer-local unavailability

Commit:
`276c0be74d98fa329ceb989fbd2556904d219ec5 fix(agentic-v9): require semantic numeric evidence`

Root causes:

- Fix round 4 invoked semantic result association only when a candidate number
  equaled a slot-derived condition. Any other nearby number, such as the `10`
  in `evaluated on 10 cases`, bypassed that gate and could support Dice through
  proximity alone.
- Unavailability was checked across the whole sentence. A valid U-KAN value
  was therefore discarded when the same sentence said that the co-located
  proposed-method result was not reported.

Changes:

- Every numeric candidate now requires an explicit result link (`is`, `was`,
  `scored`, `as`, `of`, `:`, `=`, and the existing result verbs) and the
  current slot's unique discriminator before proximity comparison.
- Slot-derived condition-number filtering remains an additional early
  exclusion. A condition-equal value is still allowed only when the text
  independently states it as the requested result.
- Unavailability is scoped from the current slot's associated discriminator to
  the next peer discriminator, bounded by the containing sentence. This keeps
  a peer's `not reported` clause from negating a valid current-slot value while
  still rejecting same-result continuations such as `; however, the result was
  not reported`.
- Existing valid result forms including `U-KAN metric as 0.8`, `Dice: 0.81`,
  and `U-KAN scored 0.81` remain supported.

Breaker RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py::test_unrelated_case_count_is_not_a_numeric_dice_result tests/test_agentic_v9_slot_constraints.py::test_unavailable_peer_does_not_negate_ukan_result tests/test_agentic_v9_slot_constraints.py::test_numeric_result_requires_explicit_ukan_association -q
```

Result: `3 failed, 2 passed, 1 warning`. The unrelated `10` supported U-KAN,
and sentence-wide peer unavailability rejected U-KAN in both mixed-peer
variants. The two positive result forms already passed.

Runtime compatibility RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py -q
```

Result: `1 failed, 48 passed, 24 warnings`. Universal association initially
rejected the existing valid `Table 3 reports the U-KAN metric as 0.8` runtime
fixture; recognizing `as/of` as explicit result links restored it.

Focused GREEN:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py -q
```

Result: `49 passed, 24 warnings`.

Focused Ruff:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base/agentic_v9/slot_constraints.py tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py
```

Result: `All checks passed!`

Format verification:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff format --no-cache --check data_base/agentic_v9/slot_constraints.py tests/test_agentic_v9_slot_constraints.py
```

Result: `2 files already formatted`.

Self-review:

- `U-KAN was evaluated on 10 cases at noise level 0.4; the Dice analysis is
  pending.` cannot support the U-KAN Dice slot.
- `U-KAN Dice was 0.81, while the proposed-method Dice was not reported.`
  supports U-KAN only.
- A negated proposed-method value in the same sentence also leaves U-KAN
  supported while rejecting the proposed-method slot.
- Positive `was`, `as`, label, and `scored` result forms remain supported,
  including the condition-equal `noise 0.4, Dice was 0.4` case.
- Locator exclusion, condition-number variants, identical-slot fail-closed
  behavior, v2 runtime binding, and v1 compatibility remain covered.
- The 24 warnings remain the existing pytest cache-option and `storage3`
  Pydantic deprecation warnings.

## Design Amendment B — Pre-edit structured result contract risk/test matrix

Date: 2026-07-27

| Area | Current propagation/behavior | Amendment risk | Required test/mitigation |
| --- | --- | --- | --- |
| Schema compatibility | `RequiredSlot` is embedded in `QueryContract`; v1 and v2 payloads deserialize through Pydantic defaults. `ExpectedAnswerType` currently covers number, equation, definition, comparison, explanation, and text. | Making new fields required, or inferring facts while loading old payloads, would break persisted v1/older-v2 traces. Extending answer types can also change validation consumers. | Add strict `SlotCondition`; add optional `requested_measure`/`expected_result_unit` and default-empty `conditions`. Prove v1 and older v2 payloads load without invented result roles. |
| Deterministic planner | `_decompose`, known bundles, `_slot_for_named_source`, and `_slot` construct all deterministic slots. Q16 currently encodes noise `0.4` inside prose and calls the Theorem range `text`. | Conditions may remain mixed into prose; result unit may be inferred from an open-ended blacklist; ambiguous slots may be marked complete. | Q16 must produce `requested_measure="Dice"`, `expected_result_unit="dimensionless"`, and `conditions=[noise_level = 0.4]`; Theorem 1 must be a range slot. Unresolved deterministic role separation must degrade the v2 plan. |
| LLM ambiguity planner and prompt | `_PlannerSlot`, strict `_PlannerDecision`, planner prompt JSON, `_parse_decision`, `_validate_planner_scope`, `_validate_answer_free`, and the `RequiredSlot` projection form the LLM path. | New fields could be silently dropped, accepted from malformed JSON, contain unauthorized/gold-like values, or bypass numeric/dependency validation. | Strict JSON must require/expose the structured fields; safe fallback on missing/unknown result role, unauthorized scope, invalid dependencies, or answer-like values not present in the question. Condition values are admitted only when present in the original question. |
| Persistence/materialization | Runtime writes `contract.model_dump(mode="json")`; worker retains `query_contract` inside trace payload; storage sanitizes source-name fields and JSON-materializes the rest; analytics revalidates with `QueryContract`. | New fields could be dropped or rewritten during trace sanitation/materialization, or old stored payloads could stop loading. | Runtime and durable persistence tests must round-trip requested measure, result unit, and conditions while preserving source/locator binding; old payload reads remain additive. |
| Answer-leakage boundary | Planning receives only question and authorized source metadata. Numeric leakage validation currently scans route reason, descriptions, and locators. Retrieval compilation consumes planner-owned slot metadata. | A benchmark key point, reference/gold/expected answer, retrieved answer, or a planner-authored condition value could become a requested result. | Tests use sentinel benchmark/gold/reference/expected fields and prove none enter planning; validate structured fields against question-only numeric/text authority; retrieval queries contain role/condition metadata but no answer key. |
| Numeric matcher | Current matcher extracts numbers, infers requested units from description prose, filters description-derived numbers, and uses a bounded following-unit blacklist. | A condition can independently satisfy a slot; unfamiliar units remain fail-open; entity/measure/value/unit binding is implicit. | Structured numeric role is authoritative: bind current entity/discriminator + requested measure + explicit result link + compatible value/unit. Dimensionless Dice rejects `10 cases`, `patient count 10`, and `year 2024`; `Dice 0.81 at noise 0.4` supports. Requested patient count/year controls support. |
| Equation/definition/range | Equation and definition have dedicated signals; range currently falls through text. | Generic non-empty text can support a structured result without the requested shape. | Add equation, definition, and range fixtures with positive and fail-closed negative controls. |
| Categorical/boolean | Both currently fall through text because the schema has no distinct types. | Any descriptor-bearing sentence can be stamped supported without a category/boolean result. | Add categorical and boolean answer types with explicit result association and positive/negative fixtures. |
| Comparison/explanation/list | Comparison has a keyword signal; explanation/text accept non-empty text; list has no type. | Condition-only prose or a mere descriptor mention may satisfy the slot. | Add type-appropriate comparison, explanation, and list positive/negative fixtures; no numeric-unit inference for nonnumeric types. |
| Co-located binding | `_slot_ids_supported_by_chunk` independently checks source, locator, peers, then `slot_content_matches_chunk`. | A shared Table 3 chunk can stamp two numeric slots from one associated value. | Preserve independent same-table U-KAN/proposed-method slot support and fail closed on missing/tied result association. |
| Legacy v1 | Runtime bypasses v2 locator/content constraints for the established v1 packet path. | Applying structured-role enforcement globally would regress ordinary v1 chunks. | Preserve the v1 ordinary-chunk compatibility regression unchanged. |
| Older v2 missing fields | Missing additive fields currently cannot express which prose numbers are conditions versus results. | Guessing from description recreates the blacklist design and can produce false support. | Older v2 remains readable, but ambiguous numeric role matching fails closed/degrades; unambiguous legacy behavior remains readable and nonnumeric compatibility is retained where safe. |

Exact RED tests to add before production edits:

1. `test_required_slot_structured_result_fields_round_trip_and_old_payloads_remain_readable`
2. `test_q16_deterministic_contract_separates_dice_result_from_noise_condition`
3. `test_ambiguity_planner_accepts_strict_structured_result_json`
4. `test_ambiguity_planner_rejects_unasked_condition_or_result_values`
5. `test_ambiguity_planner_degrades_when_result_role_is_unknown`
6. `test_dimensionless_dice_rejects_condition_counts_patients_and_years`
7. `test_dice_result_binds_measure_value_unit_and_noise_condition`
8. `test_requested_patient_count_and_year_are_supported`
9. `test_same_table_numeric_slots_bind_independently_with_structured_roles`
10. `test_nonnumeric_result_types_use_type_appropriate_checks`
11. `test_runtime_persists_structured_slot_contract_with_source_locator_binding`
12. `test_materialization_round_trips_structured_slot_contract`
13. `test_v1_packet_binding_and_existing_v2_payload_compatibility`

## Breaker exception A ??Pre-edit unit-aware numeric evidence risk/design note

Date: 2026-07-27

- Numeric candidate extraction currently masks structured locators and scans
  number/decimal/percent-shaped spans. It preserves source offsets but does not
  classify the token following a candidate.
- Preceding semantic association requires an explicit numeric-result link and
  the current slot's unique discriminator. This blocks unrelated numbers such
  as `evaluated on 10 cases`, but accepts syntactically result-like text such as
  `Dice result of 10 cases` or `Dice was 10 cases`.
- Following unit/context is therefore the remaining load-bearing risk. The
  localized design is to classify only a bounded set of count/condition units
  immediately following the candidate (cases, folds, subjects, samples,
  images, patients, epochs, iterations, runs, seeds, years, pages, and
  table/figure variants), plus `%`/percent wording.
- Unit rejection must be slot-aware rather than blanket. `expected_answer_type`
  establishes that the slot is numeric, while explicit slot wording determines
  whether percent, count, year, or another bounded unit is itself the requested
  result shape. Dimensionless metrics such as Dice reject those following
  units; explicitly requested percentages/counts/years retain them.
- Slot-derived condition-number exclusion remains an independent earlier
  filter, including decimal normalization and locator/model/year conditions.
  A condition-equal value still requires a genuine requested-result
  association.
- Peer-local negation remains bounded from the current discriminator to the
  next peer discriminator. Unit classification will run before that unchanged
  check so one peer's unavailable result cannot suppress another peer's valid
  numeric evidence.
- Main false-negative risk: inferring a requested unit from loose prose. The
  implementation will use explicit bounded unit words in the slot description,
  avoid a schema change, and keep v1 untouched because this matcher remains on
  the existing co-located v2 path only.

## Breaker exception A ??Unit-aware numeric evidence implementation

Commit:
`eccc4063a305195f42321d8c7550906972c2dc8c fix(agentic-v9): validate numeric evidence units`

Root cause:

- Numeric extraction, condition exclusion, preceding semantic association, and
  peer-local unavailability all ran without inspecting the candidate's
  following token.
- Consequently, syntactically linked but semantically incompatible values such
  as `Dice result of 10 cases` and `Dice was 10 cases` passed as Dice evidence.

Changes:

- Added a bounded following-unit classifier for singular/plural cases, folds,
  subjects, samples, images, patients, epochs, iterations, runs, seeds, years,
  pages, tables, figures, and percent/percentage forms. Hyphenated and
  parenthesized immediate variants share the same classification path.
- Numeric candidates carrying one of those units are rejected unless the slot
  description explicitly requests that same result shape.
- Requested-unit inference remains local to `RequiredSlot.description` and
  `expected_answer_type="number"`:
  count/number/size/total/how-many phrasing authorizes its adjacent unit;
  explicit percentage/result phrasing authorizes percent; and bounded
  `in years/epochs/iterations` plus publication-year wording authorizes those
  measurement shapes.
- Locator wording is not unit authorization: `in Table 3` cannot turn
  `10 tables` into Dice evidence. Explicit `table count` or `number of tables`
  remains representable.
- Existing condition exclusion, prefix association, peer-local negation,
  structured locator masking, and the v2-only caller boundary were unchanged.

Primary RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py::test_dice_rejects_semantically_linked_case_count tests/test_agentic_v9_slot_constraints.py::test_dimensionless_dice_rejects_count_and_percent_units tests/test_agentic_v9_slot_constraints.py::test_numeric_result_allows_explicitly_requested_unit -q
```

Result: `17 failed, 5 passed, 1 warning`. Both exact case-count statements
supported Dice, as did every non-requested count/condition/percent variant.
The dimensionless, requested-percent, requested-count, and requested-year
controls already passed.

Locator-authorization RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py::test_locator_wording_does_not_request_a_table_count -q
```

Result: `1 failed, 1 warning`. The first unit-inference draft treated
`in Table 3` as requesting table units; inference was narrowed to explicit
count shape or bounded measurement units.

Focused GREEN:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py -q
```

Result: `72 passed, 24 warnings`.

Focused Ruff:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base/agentic_v9/slot_constraints.py tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py
```

Result: `All checks passed!`

Format verification:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff format --no-cache --check data_base/agentic_v9/slot_constraints.py tests/test_agentic_v9_slot_constraints.py
```

Result: `2 files already formatted`.

Self-review:

- Both exact `10 cases` regressions are unsupported for a Dice slot.
- All listed non-requested count/condition units and percent representations
  are rejected for dimensionless Dice.
- `Dice 0.81`, requested `accuracy 91%`/`91 percent`, requested patient count,
  and requested duration in years remain supported.
- The existing condition-equal real result, locator exclusions, peer-local
  unavailability, co-located fail-closed behavior, runtime binding, and v1
  compatibility regressions remain green.
- The requested-unit vocabulary is intentionally bounded. A new domain unit
  requires adding an explicit alias and a positive requested-shape regression,
  rather than silently widening inference.
- The 24 warnings remain the existing pytest cache-option and `storage3`
  Pydantic deprecation warnings.

## Design Amendment B — Structured result contract implementation

Date: 2026-07-27

Files:

- `data_base/agentic_v9/schemas.py`
- `data_base/agentic_v9/contract_planner.py`
- `data_base/agentic_v9/slot_constraints.py`
- `data_base/agentic_v9/retrieval_tasks.py`
- `data_base/agentic_v9/repair.py`
- `prompts/agentic_v9_contract_planner.json`
- `tests/test_agentic_v9_schemas.py`
- `tests/test_agentic_v9_contract_planner.py`
- `tests/test_agentic_v9_route_planner.py`
- `tests/test_agentic_v9_retrieval_tasks.py`
- `tests/test_agentic_v9_slot_constraints.py`
- `tests/test_agentic_v9_campaign_runtime.py`
- `tests/test_evaluation_v9_attempt_persistence.py`

Changes:

- Added strict answer-free `SlotCondition` plus additive `requested_measure`,
  `expected_result_unit`, and `conditions` fields on `RequiredSlot`.
- Extended result types with range, categorical, boolean, and list while
  preserving existing values.
- Deterministic Q16 planning now separates U-KAN/proposed-method Dice from the
  `noise_level = 0.4` condition and marks Theorem 1 as a range.
- The ambiguity prompt and strict response model expose the new fields.
  Planner-authored measures, condition fields, values, and condition units
  must be grounded in the original question; unknown numeric roles and units
  degrade through the existing safe fallback.
- Retrieval and repair queries propagate structured role and condition
  metadata without accepting the original question's benchmark/gold wrapper.
- Numeric matching now requires requested measure, entity when declared,
  explicit result linkage, a numeric value, and compatible result unit.
  Dimensionless matching rejects any detected unit generically; it no longer
  depends on the former count/year/epoch/etc. unit alias blacklist.
- Conditions are the only source of condition-number exclusion. A condition
  value never satisfies a slot unless it is separately linked as the requested
  result.
- Singleton and co-located v2 slots use the same fail-closed result matcher.
  Equation, definition, range, categorical, boolean, comparison, explanation,
  list, and compatibility text paths use type-specific checks.
- Runtime trace and durable materialization retain the structured contract
  unchanged apart from the existing source-name sanitizer.

Initial schema collection RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_schemas.py::test_required_slot_structured_result_fields_round_trip_and_old_payloads_remain_readable tests/test_agentic_v9_contract_planner.py::test_q16_has_seven_ordered_slots_without_expected_numeric_answers tests/test_agentic_v9_contract_planner.py::test_ambiguity_planner_accepts_strict_structured_result_json tests/test_agentic_v9_contract_planner.py::test_ambiguity_planner_rejects_unasked_condition_or_result_values tests/test_agentic_v9_contract_planner.py::test_ambiguity_planner_degrades_when_result_role_is_unknown tests/test_agentic_v9_slot_constraints.py::test_dimensionless_dice_rejects_condition_counts_patients_and_years tests/test_agentic_v9_slot_constraints.py::test_dice_result_binds_measure_value_unit_and_noise_condition tests/test_agentic_v9_slot_constraints.py::test_requested_patient_count_and_year_are_supported tests/test_agentic_v9_slot_constraints.py::test_nonnumeric_result_types_use_type_appropriate_checks tests/test_agentic_v9_slot_constraints.py::test_existing_v2_numeric_payload_without_result_role_fails_closed -q
```

Result: collection failed because `SlotCondition` and the additive slot fields
did not exist.

Behavioral RED after the additive schema shell:

The same command produced `11 failed, 6 passed, 1 warning`. Failures showed
Q16 roles were absent, ambiguity output dropped structured fields, singleton
numeric and nonnumeric evidence passed without result-shape checks, and older
ambiguous v2 numeric slots failed open.

Additional nonnumeric leakage RED:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_contract_planner.py::test_ambiguity_planner_rejects_unasked_condition_or_result_values -q
```

Result: `1 failed, 2 passed, 1 warning`; the planner accepted an unasked
`requested_measure="gold accuracy"` until question-token grounding was added.

Focused amendment GREEN:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agentic_v9_schemas.py::test_required_slot_structured_result_fields_round_trip_and_old_payloads_remain_readable tests/test_agentic_v9_contract_planner.py::test_q16_has_seven_ordered_slots_without_expected_numeric_answers tests/test_agentic_v9_contract_planner.py::test_ambiguity_planner_accepts_strict_structured_result_json tests/test_agentic_v9_contract_planner.py::test_ambiguity_planner_rejects_unasked_condition_or_result_values tests/test_agentic_v9_contract_planner.py::test_ambiguity_planner_degrades_when_result_role_is_unknown tests/test_agentic_v9_slot_constraints.py::test_dimensionless_dice_rejects_condition_counts_patients_and_years tests/test_agentic_v9_slot_constraints.py::test_dice_result_binds_measure_value_unit_and_noise_condition tests/test_agentic_v9_slot_constraints.py::test_requested_patient_count_and_year_are_supported tests/test_agentic_v9_slot_constraints.py::test_nonnumeric_result_types_use_type_appropriate_checks tests/test_agentic_v9_slot_constraints.py::test_existing_v2_numeric_payload_without_result_role_fails_closed -q
```

Result: `17 passed, 1 warning` before the categorical/boolean and nonnumeric
leakage cases were added.

Final combined GREEN:

```powershell
$env:EVALUATION_TEST_TMPDIR='C:\Users\user\AppData\Local\Temp\agentic-v9-wave3b-final'
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_prompt_loader.py tests/test_agentic_v9_schemas.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_route_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_v9_attempt_persistence.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_sufficiency_gate.py -q
```

Result: `201 passed, 24 warnings`.

Focused Ruff and format:

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check --no-cache data_base/agentic_v9/schemas.py data_base/agentic_v9/contract_planner.py data_base/agentic_v9/slot_constraints.py data_base/agentic_v9/retrieval_tasks.py data_base/agentic_v9/repair.py tests/test_agentic_v9_schemas.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_route_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_v9_attempt_persistence.py
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff format --no-cache --check data_base/agentic_v9/schemas.py data_base/agentic_v9/contract_planner.py data_base/agentic_v9/slot_constraints.py data_base/agentic_v9/retrieval_tasks.py data_base/agentic_v9/repair.py tests/test_agentic_v9_schemas.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_route_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_slot_constraints.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_v9_attempt_persistence.py
git diff --check
```

Results: `All checks passed!`, `12 files already formatted`, and clean diff
check (Git emitted only the repository's existing LF-to-CRLF notices).

Compatibility:

- V1 continues to use its established ordinary packet-binding path.
- Persisted v1 and older v2 slots deserialize with `None`/empty additive
  defaults; no requested measure, unit, or condition is invented.
- Older ambiguous numeric v2 slots are readable but become degraded and cannot
  produce false supported evidence without an explicit result role.
- Older text v2 payloads retain their nonempty-text compatibility path.
- Source/locator intersection, grouped repair caps, terminal stop precedence,
  budgets, and actual-route trace persistence remain covered by the combined
  suite.

Concerns:

1. Deterministic role extraction intentionally recognizes only explicit
   question phrasing. A novel numeric request that cannot separate measure and
   unit degrades instead of guessing.
2. The persistence suite requires an OS temp root because the isolated
   worktree's `.pytest-tmp` ACL rejects writes.
3. The 24 warnings remain the existing pytest `cache_dir` option warning and
   `storage3` Pydantic deprecations; no new warning category was introduced.
