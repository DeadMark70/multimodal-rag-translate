# Specification: research_proposal_writing

## 1. Overview
This track focuses on updating and expanding the existing research proposal (`experiments/doc/研究計畫.docx`). The goal is to incorporate preliminary experimental results (`research_done.txt`, CSV data) and detail a comprehensive methodology for future large-scale experiments (using Ragas, more data) to justify budget requests.

## 2. Functional Requirements
### 2.1 File Analysis & Content Extraction
- **Analyze Existing Proposal**: Read `experiments/doc/研究計畫.docx` to understand the current structure and specific formatting rules.
- **Analyze Experimental Data**:
    -   Read `experiments/doc/research_done.txt` to extract experimental process descriptions and QA examples.
    -   Analyze CSV files in `experiments/doc/` to calculate statistics (token usage, context length, etc.) and identify trends.
    -   (Optional) Use `skill:pdf` to verify answers against source PDFs in `experiments/doc/` if necessary for accuracy.

### 2.2 Content Generation (Markdown Intermediate)
-   **Create `experiments/doc/research_proposal_content.md`** as the working draft.
-   **Preliminary Results Section**:
    -   Generate Markdown tables from CSV data.
    -   Write a narrative summary of the current experimental performance.
    -   Describe the experimental process based on `research_done.txt`.
    -   Present any comparative metrics available.
-   **Methodology Section (Future Work)**:
    -   **Automated Evaluation (Ragas)**: Detail the plan to introduce Ragas for quantifying Faithfulness and Answer Relevance.
    -   **Benchmarking Scale-up**: Propose expanding the dataset from 13 to 40-50 PDFs and increasing test questions from 4 to 10+ per document to ensure statistical significance.
    -   **Statistical Validation**: Plan for at least 10 iterations of testing for each scenario and apply appropriate statistical methods (e.g., T-test, ANOVA) to validate the results.
    -   **Ablation Studies**: Describe plans to compare GraphRAG vs. Vector RAG (building on current prototypes).
-   **Budget & Resources**:
    -   Justify the need for increased API tokens and computational resources based on the scaled-up experimental plan.
    -   Fill in the budget table (in NTD) found in the docx template.
    -   Include a timeline for the expanded experiments.

### 2.3 Document Synchronization
-   **Update DOCX**: Use `skill:docx` to write the content from `research_proposal_content.md` into `experiments/doc/研究計畫.docx`, strictly adhering to the existing formatting and rules.

## 3. Non-Functional Requirements
-   **Format Compliance**: strict adherence to the styles and rules defined in the original `研究計畫.docx`.
-   **Academic Tone**: The writing must be formal, objective, and suitable for a research grant proposal.
-   **Sync**: Keep the Markdown draft and the Docx file synchronized.

## 4. Acceptance Criteria
- [ ] `research_proposal_content.md` is created with all sections populated.
- [ ] `experiments/doc/研究計畫.docx` is updated with the new content.
- [ ] The "Preliminary Results" section accurately reflects the data from `research_done.txt` and CSVs.
- [ ] The "Methodology" section clearly outlines the Ragas implementation, scale-up plan (40-50 PDFs, 10+ questions), and statistical validation (10+ iterations).
- [ ] The budget table is filled with justifiable estimates in NTD.

## 5. Out of Scope
-   Running the actual new experiments (Ragas, 50 PDFs). This track is about *writing the plan* to do them.
