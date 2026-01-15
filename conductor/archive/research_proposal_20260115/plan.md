# Plan: research_proposal_writing

## Phase 1: Data Gathering and Analysis [checkpoint: a25a063]
- [x] Task: Analyze existing proposal and experimental data
    - [x] Use `skill:docx` to read `experiments/doc/研究計畫.docx` and extract section headers and rules
    - [x] Read `experiments/doc/research_done.txt` to extract current QA process and preliminary findings
    - [x] Use `skill:csv-data-summarizer` to analyze CSV files in `experiments/doc/` for token usage and performance metrics
    - [x] (Optional) Use `skill:pdf` to sample source PDFs in `experiments/doc/` for answer verification
- [x] Task: Conductor - User Manual Verification 'Phase 1: Data Gathering and Analysis' (Protocol in workflow.md)

## Phase 2: Content Drafting (Markdown)
- [x] Task: Draft "Preliminary Results" section
    - [x] Convert CSV stats into Markdown tables
    - [x] Write narrative summary of current performance and process
- [x] Task: Draft "Future Methodology" section
    - [x] Detail Ragas implementation plan
    - [x] Outline scale-up plan (50 PDFs, 10+ questions)
    - [x] Describe statistical validation (10+ iterations, T-tests/ANOVA)
    - [x] Incorporate Ablation study design
- [x] Task: Draft "Budget and Timeline" section
    - [x] Populate budget justification based on scale-up
    - [x] Define project timeline for expanded experiments
- [x] Task: Conductor - User Manual Verification 'Phase 2: Content Drafting (Markdown)' (Protocol in workflow.md)

## Phase 3: Final Document Generation [checkpoint: e8d2492]
- [x] Task: Synchronize and Update DOCX
    - [x] Use `skill:docx` to insert/update sections in `experiments/doc/研究計畫.docx`
    - [x] Fill in the budget table in the DOCX file (NTD)
    - [x] Ensure formatting strictly matches original document rules
- [x] Task: Final review of generated documents
- [x] Task: Conductor - User Manual Verification 'Phase 3: Final Document Generation' (Protocol in workflow.md)
