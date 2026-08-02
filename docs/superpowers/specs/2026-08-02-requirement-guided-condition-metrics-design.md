# Requirement-Guided Condition Metrics Design

## Goal

Make a campaign with multiple `ablation_conditions` directly interpretable in
the Evaluation Center. In particular, `v9-baseline` and `v9-guided` must not be
collapsed into a single `agentic` mode row.

## Scope

1. Extend the existing campaign `/ablation` analytics response with a typed
   condition comparison section.
2. Aggregate RAGAS correctness, faithfulness, and relevancy by condition.
3. Build paired deltas using the identity `(question_id, repeat_number)` and
   condition ids. A pair is included only when both arms have a completed run
   and the metric is finite on both runs.
4. Surface condition metrics and paired deltas in the existing Ablation tab.
5. Include the same condition comparison projection and per-run RAGAS metric
   map in redacted campaign exports.

## Non-goals

- Do not change Agentic V9 retrieval, requirement guidance, prompts, or the
  process-wide `AGENTIC_V9_REQUIREMENT_GUIDED_RUNTIME` default.
- Do not alter Mode Comparison; it intentionally remains a mode-level view.
- Do not infer missing metrics as zero or include failed/unpaired runs in a
  paired delta.

## Backend Contract

`AblationResponse.summaries` gains `condition_comparison` only when the
campaign has two or more recorded condition ids. It contains:

- `conditions`: one row per condition with label, flags, execution counts,
  metric validity counts, means, mean tokens, and mean latency.
- `paired`: one comparison row for the configured baseline/guided ordering.
  It records the condition ids, completed-pair count, metric-specific pair
  counts, mean deltas (`guided - baseline`), and excluded-pair reasons.
- `availability`: whether RAGAS rows were found and an explicit warning if
  scores are unavailable.

The implementation reads `ragas_scores` through the existing owned campaign
query and uses `condition_id` captured in `derived_metrics`. Condition labels
and flags come from the run snapshot/derived metadata, never from current
environment variables.

The export payload reuses this server-side projection at
`metrics.condition_comparison`, and adds a finite-only per-run
`ragas_metrics` map. Existing export redaction controls continue to govern
answers, prompts, and excerpts; numeric metrics and condition metadata are
not sensitive prompt content.

## Frontend Behavior

The Ablation tab renders a `Condition Metrics` section ahead of the existing
condition-count table:

- condition label/id, completed/failed runs;
- correctness, faithfulness, relevancy, tokens, and latency;
- an explicit `N/A` for missing metrics, not zero.

For a compatible two-arm campaign it also renders `Paired Delta (guided -
baseline)`, including completed pair count and exclusions. Campaigns without
condition comparison data keep the current generic ablation display unchanged.

## Testing

- Backend analytics tests cover complete pairs, failed/unpaired exclusion,
  missing metric handling, and environment-independent condition metadata.
- Export tests assert per-run RAGAS metrics and the same condition comparison
  projection are included without exposing restricted prompt content.
- Frontend component tests cover rendered condition metrics, delta display,
  `N/A` handling, and legacy/no-condition empty compatibility.

## Acceptance Criteria

For a `v9-baseline` / `v9-guided` campaign, the user can see each arm's RAGAS
means, tokens, latency, successful pair count, and guided-minus-baseline
deltas in the Ablation tab and the exported JSON. A provider failure such as
the baseline Q4 failure is visibly excluded rather than silently biasing a
paired result.
