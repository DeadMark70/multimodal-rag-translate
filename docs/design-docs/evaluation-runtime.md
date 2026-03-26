# Evaluation Runtime

## Purpose

Describe evaluation as a persisted runtime subsystem.

## Ownership

- Router: `evaluation/router.py`
- Storage: `evaluation/storage.py` and `evaluation/db.py`
- Engine: `evaluation/campaign_engine.py`
- Traces and metrics: `evaluation/trace_schemas.py`, campaign result accessors

## Runtime Rules

- Campaigns persist status and results in SQLite.
- SQLite runs in WAL mode for concurrent campaign work.
- Results, traces, metrics, manual evaluate, cancel, and SSE stream are separate API concerns.
- Model discovery is control-plane behavior; runtime generation stays behind provider/factory seams.
