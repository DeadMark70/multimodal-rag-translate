# PLANS

## Planning System

- Active work: `docs/exec-plans/active/`
- Completed milestones: `docs/exec-plans/completed/`
- Debt ledger: `docs/exec-plans/tech-debt-tracker.md`

## Naming Policy

- New plan files should use `YYYY-MM-short-title.md`.
- Existing historical filenames may remain as-is unless there is a strong reason to rename them.

## Required Sections

1. Objective
2. Scope
3. Work items
4. Risks
5. Success metrics
6. Exit criteria

## Change Policy

1. Update current-state docs before moving a plan to completed.
2. Keep completed plans immutable except for factual corrections or broken links.
3. If a change affects router prefixes, endpoint families, or runtime contracts, update `docs/generated/api-surface.md` in the same change set.
