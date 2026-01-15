# Specification: RAG System Audit & Experimental Baseline Verification

## 1. Overview
This track comprehensively audits and verifies advanced RAG settings (HyDE, Multi-Query, Reranker, GraphRAG, Deep Research, and Visual Verification). The focus is on user `c1bae279-c099-4c45-ba19-2bb393ca4e4b` who provides the test documents (SwinUNETR vs nnU-Net).

## 2. Target Components
*   **Retrieval**: HyDE, Multi-Query.
*   **Ranking**: Cross-Encoder Reranker.
*   **GraphRAG**: Hybrid search & Planning.
*   **Deep Research**: Sub-task enforcement & Feedback loops.
*   **Visual Verification**: Multi-modal detail analysis.
*   **Evaluation**: Integration of RAGAS metrics (Faithfulness, Relevance).

## 3. Test Scenarios
*   **Conflict Resolution**: Correctly handling contradictory claims in SwinUNETR vs nnU-Net papers.
*   **Multi-hop**: Connecting facts across multiple papers.
*   **Noise**: Ranking relevant chunks above irrelevant ones from the user's library.

## 4. Baselines
*   **Vanilla LLM**: No RAG.
*   **Naive RAG**: Vector only.
*   **Ours**: Full stack (GraphRAG + Deep Research).
