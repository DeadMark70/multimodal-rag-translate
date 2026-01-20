"""
LLM Factory Module

Provides cached LLM instances with purpose-specific configurations.
Each purpose has its own optimized settings to avoid configuration conflicts.
"""

# Standard library
import logging
from functools import lru_cache
from typing import Literal, Optional

# Third-party
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logger = logging.getLogger(__name__)

# Purpose type for type safety
LLMPurpose = Literal[
    "rag_qa", "translation", "image_caption", "visual_verification",
    "context_generation", "proposition_extraction", "query_rewrite",
    "evaluator", "planner", "synthesizer", "summary",
    "graph_extraction", "community_summary"  # GraphRAG purposes
]

# Model mapping - only translation uses different model
_MODEL_BY_PURPOSE: dict[str, str] = {
    "translation": "gemini-2.5-flash-lite",
    "graph_extraction": "gemini-2.5-flash-lite",   # Fast extraction for GraphRAG
    "community_summary": "gemini-2.5-flash-lite",   # Fast summarization for communities
}

# Default model for all other purposes
_DEFAULT_MODEL = "gemma-3-27b-it"

# Session-wide override for testing
_session_model_override: Optional[str] = None

def set_session_model_override(model_name: Optional[str]) -> None:
    """
    Sets a session-wide model override and clears the LLM cache.
    
    Args:
        model_name: The model name to use for all purposes (except translation/graph extraction 
                   if they have explicit mapping, unless also overridden).
    """
    global _session_model_override
    _session_model_override = model_name
    clear_llm_cache()
    if model_name:
        logger.info(f"Session model override set to: {model_name}")
    else:
        logger.info("Session model override cleared")

# Configuration for each purpose
_LLM_CONFIGS: dict[str, dict] = {
    "rag_qa": {
        "temperature": 0.3,
        "max_output_tokens": 4096,
    },
    "translation": {
        "temperature": 0.1,
        "max_output_tokens": 65000,  # Increased for long document translation
    },
    "image_caption": {
        "temperature": 0.2,
        "max_output_tokens": 768,
    },
    "visual_verification": {
        "temperature": 0.2,
        "max_output_tokens": 2048,  # Increased for detailed deep research analysis
    },
    "context_generation": {
        "temperature": 0.1,
        "max_output_tokens": 256,
    },
    "proposition_extraction": {
        "temperature": 0.1,
        "max_output_tokens": 1024,
    },
    "query_rewrite": {
        "temperature": 0.3,
        "max_output_tokens": 512,
    },
    "evaluator": {
        "temperature": 0.1,
        "max_output_tokens": 2048,
        # Custom instruction to be lenient on extra info/citations
        "convert_system_message_to_human": True, 
    },
    "planner": {
        "temperature": 0.2,
        "max_output_tokens": 512,
    },
    "synthesizer": {
        "temperature": 0.3,
        "max_output_tokens": 2048,
    },
    "summary": {
        "temperature": 0.3,
        "max_output_tokens": 1024,
    },
    # GraphRAG purposes
    "graph_extraction": {
        "temperature": 0.1,  # Low temperature for consistent extraction
        "max_output_tokens": 2048,
    },
    "community_summary": {
        "temperature": 0.2,
        "max_output_tokens": 1024,
    },
}

@lru_cache(maxsize=30)  # Increased to accommodate different model instances and overrides
def get_llm(purpose: LLMPurpose, model_name: str = None) -> ChatGoogleGenerativeAI:
    """
    Returns a cached LLM instance for a specific purpose.

    Each purpose has its own optimized configuration (temperature, max_tokens).
    Instances are cached using lru_cache to avoid repeated initialization.

    Args:
        purpose: The intended use case for the LLM.
        model_name: Optional model name to override the default for this purpose.

    Returns:
        Configured ChatGoogleGenerativeAI instance.
    """
    config = _LLM_CONFIGS.get(purpose, _LLM_CONFIGS["rag_qa"])
    
    # Priority: 1. explicit model_name, 2. session override, 3. purpose-specific, 4. default
    if model_name:
        model = model_name
    elif _session_model_override:
        model = _session_model_override
    else:
        model = _MODEL_BY_PURPOSE.get(purpose, _DEFAULT_MODEL)

    logger.info(f"Initializing LLM for purpose: {purpose} (model: {model}, config: {config})")

    return ChatGoogleGenerativeAI(
        model=model,
        **config
    )


def clear_llm_cache() -> None:
    """
    Clears the LLM instance cache.

    Useful for testing or when configuration needs to be reloaded.
    """
    get_llm.cache_clear()
    logger.info("LLM cache cleared")
