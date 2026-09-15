# Evaluation partial reruns — 2026-09-16

- Numeric audit references such as `1` normalize to `[Ref 1]` and still require a real matching source. An invalid source mapping cannot cancel an otherwise usable drill-down request: retain initial evidence, run the search, and synthesize without trusting the invalid matrix. Such fallback answers remain `qualified_partial`.
- `POST /api/evaluation/campaigns/{campaign_id}/reruns` accepts optional `modes` and `scope: "missing_only"` (RAGAS only). Question and mode filters apply together; existing scores, including zero, are retained by missing-only runs. `selected` still explicitly recomputes the chosen metrics. Execution-and-RAGAS reruns use the original model snapshot and restrict downstream scoring to the selected questions and modes.
- Batch compatibility depends on evaluator settings and metric policy, rather than each answer/context/reference value. Individual score signatures and persistence remain separate. Concurrency and retry tuning are deferred.
- Terminal scoring checks overall execution and metric coverage. Latest work items supersede historical failures, while missing results or scores keep the campaign `completed_with_errors`. Changes apply when state is next derived; historical exports are not rewritten.

Example: repair only Q30's Naive faithfulness score:

```json
{"scope":"missing_only","stages":"ragas","question_ids":["Q30"],"modes":["naive"],"metric_names":["faithfulness"]}
```

Example: regenerate Q13 with its original Agentic v10 configuration and then score it:

```json
{"scope":"selected","stages":"execution_and_ragas","question_ids":["Q13"],"modes":["agentic-v10"],"metric_names":[]}
```
