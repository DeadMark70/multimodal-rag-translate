# v9 Preflight Call Budget Alignment Design

## Goal

Allow every supported v9 route in the 16-case evaluation set to pass preflight when its required provider phases fit the configured runtime envelope, without weakening runtime admission checks.

## Root cause

Route contracts reserve two calls for several routes, but table/figure questions add `visual_extract`; graph questions add `graph_route`. Their legitimate phase sets therefore require three or four calls. The frontend preflight ceiling is also fixed at three calls, while an ambiguity-planner plus graph-and-visual route can require five.

## Decision

Raise non-graph route contract caps to three, graph route caps to four, and the frontend preflight envelope to five. The dynamic planner still consumes an additional contract call. Existing token budgets, per-phase limits, final-answer reservation, and runtime feasibility checks remain unchanged.

## Safety

This does not make every request admissible: the post-contract validator still rejects a contract whose actual phase set exceeds its cap, token envelope, remaining call budget, or per-phase limit.
