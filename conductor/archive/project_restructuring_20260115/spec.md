# Specification: project_restructuring_20260115

## 1. Overview
This track focuses on restructuring the `checklist` directory to provide structured, module-specific technical documentation. This will allow AI agents to understand the system's architecture and logic by reading concise summaries instead of the entire codebase. Additionally, the project's root `README.md` will be updated to reflect the current project state and architecture.

## 2. Functional Requirements
### 2.1 Technical Documentation (checklist/)
Create 7 comprehensive `.md` files in the `checklist/` directory, one for each primary router/module. Each file must follow a consistent structure:
1.  **Technical Implementation Details**: Deep dive into the core logic and algorithms of the module.
2.  **Codebase Map**: A list of key files and their responsibilities within the module.
3.  **Usage Guide**: Instructions on how to use the module's API and key functions. **Must explicitly state that execution requires the project's virtual environment (`.venv`).**
4.  **Dependencies**: List of internal module dependencies and external libraries.

The 7 modules are:
- `pdfserviceMD` (PDF OCR & Translation)
- `data_base` (RAG Database)
- `image_service` (Image Service)
- `multimodal_rag` (Multimodal RAG)
- `stats` (Statistics)
- `graph_rag` (Graph RAG)
- `conversations` (Conversations)

### 2.2 README Update
Update the root `README.md` to include:
- A refreshed project overview.
- Updated installation steps (**emphasizing `.venv` usage**).
- A high-level description of the system architecture.
- Current project progress and status.

## 3. Non-Functional Requirements
- **Consistency**: All 7 module documents must share the same headers and formatting style.
- **Clarity**: Documentation must be precise enough for an AI agent to perform tasks within that module with minimal codebase scanning.
- **Environment Context**: All commands and script executions documented must assume and explicitly instruct the use of the `.venv` virtual environment.

## 4. Acceptance Criteria
- [ ] 7 markdown files exist in `checklist/` corresponding to the identified modules.
- [ ] Each file contains the 4 required sections (Technical Details, Map, Guide, Dependencies).
- [ ] The `README.md` is updated and reflects the current state of the project.
- [ ] All previous/outdated files in `checklist/` that are no longer relevant are moved to an `archive/` or `outdate/` folder.

## 5. Out of Scope
- Modifying the actual implementation code of the routers (unless required to clarify documentation).
- Adding new features to the existing modules.
