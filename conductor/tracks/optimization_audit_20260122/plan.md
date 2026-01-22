# Implementation Plan: Project Optimization & Security Audit

This plan outlines the steps to perform a comprehensive audit of the Python backend to improve maintainability, normativity, and security.

## Phase 1: Preparation & Tooling Setup [checkpoint: f2dedcf]
- [x] Task: Research and configure static analysis tools for dead code detection (e.g., `vulture`, `pylint`). 6d966ff
- [x] Task: Set up a dedicated audit workspace in `agentlog/` for storing temporary logs. 6d966ff
- [x] Task: Conductor - User Manual Verification 'Preparation & Tooling Setup' (Protocol in workflow.md) f2dedcf

## Phase 2: Dead Code (Unused Functions) Audit [checkpoint: 17027bd]
- [x] Task: Write a custom script or use `vulture` to scan the codebase for unused functions and classes. a11348f
- [x] Task: Manually verify false positives in core modules (`agents`, `data_base`, `pdfserviceMD`). a11348f
- [x] Task: Generate the audit report at `agentlog/unused_functions_audit.md`. a11348f
- [x] Task: Conductor - User Manual Verification 'Dead Code Audit' (Protocol in workflow.md) 17027bd

## Phase 3: Security & Auth Audit [checkpoint: 82fa15f]
- [x] Task: Audit all FastAPI routers to ensure consistent use of `get_current_user_id` dependency. cad46a2
- [x] Task: Scan the repository for hardcoded secrets, API keys, or insecure `.env` patterns. cad46a2
- [x] Task: Review file handling logic in `pdfserviceMD/router.py` and `image_service/router.py` for path traversal or injection risks. cad46a2
- [x] Task: Conductor - User Manual Verification 'Security & Auth Audit' (Protocol in workflow.md) 82fa15f

## Phase 4: PEP 8 & Code Style Audit
- [ ] Task: Run `flake8` or `ruff` to identify PEP 8 violations across the backend.
- [ ] Task: Identify "High-Impact" files with the most violations for prioritized refactoring.
- [ ] Task: Document major architectural inconsistencies found during the style audit.
- [ ] Task: Conductor - User Manual Verification 'PEP 8 & Code Style Audit' (Protocol in workflow.md)

## Phase 5: Final Proposal & Synthesis
- [ ] Task: Consolidate all findings into a final "Optimization Proposal" (added to `agentlog/optimization_audit.md`).
- [ ] Task: Create specific implementation tasks (tickets) for the identified fixes in future tracks.
- [ ] Task: Conductor - User Manual Verification 'Final Proposal & Synthesis' (Protocol in workflow.md)
