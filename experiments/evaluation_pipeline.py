"""
Evaluation Pipeline Module

This module defines the EvaluationPipeline class, which orchestrates the
comparative evaluation of RAG models and configurations using Ragas metrics
and tiered benchmarking.
"""

from typing import List

class EvaluationPipeline:
    """
    Orchestrates the evaluation process for Multimodal Agentic RAG.
    
    Supports ablation studies across multiple models and tiers, integrating
    Ragas for objective metric calculation.
    """
    
    def __init__(self):
        self.models: List[str] = [
            "gemma-3-27b",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-flash"
        ]
        self.tiers: List[str] = [
            "Naive RAG",
            "Advanced RAG",
            "Graph RAG",
            "Long Context Mode",
            "Full Agentic RAG"
        ]
