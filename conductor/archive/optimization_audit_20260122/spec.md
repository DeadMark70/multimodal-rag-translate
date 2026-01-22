# Specification: Project Optimization & Security Audit

## Overview
This track focuses on improving the codebase's normativity, maintainability, and security. It involves a comprehensive audit to identify dead code (unused functions), enforce PEP 8 standards, and address critical security vulnerabilities (Auth, Secrets, and Input Validation). The final output includes a detailed audit report and a set of prioritized optimization tasks.

## Functional Requirements
1.  **Dead Code Audit**:
    - Scan the entire Python backend for unused functions, classes, and variables.
    - Generate a report in `agentlog/unused_functions_audit.md` listing:
        - File Path
        - Function/Symbol Name
        - Line Number
2.  **Security Audit**:
    - **API & Auth**: Verify all routers use `get_current_user_id` and handle permissions correctly.
    - **Secret Management**: Scan for hardcoded secrets or insecure environment variable usage.
    - **Input Validation**: Audit file processing modules (`pdfserviceMD`, `image_service`) for unsafe operations.
3.  **Code Style (PEP 8)**:
    - Identify violations of PEP 8 standards across the codebase.
    - Prioritize fixes for high-impact modules (routers and core services).

## Non-Functional Requirements
- **Maintainability**: Improve code readability and structure through standardization.
- **Security**: Minimize the attack surface by removing unused code and fixing auth gaps.
- **Transparency**: Provide a clear audit trail in the `agentlog/` directory.

## Acceptance Criteria
- [ ] A comprehensive `agentlog/unused_functions_audit.md` file exists and is populated with identified dead code.
- [ ] A report on security findings (Auth, Secrets, Inputs) is provided.
- [ ] A prioritized list of refactoring tasks based on PEP 8 and architectural standards is generated.
- [ ] No existing functionality is broken during the analysis phase.

## Out of Scope
- Automated removal of code (this track is for **Audit and Proposal**; deletion occurs in subsequent implementation tracks).
- Major architectural changes (e.g., switching databases).
- Frontend (React) code audit.
