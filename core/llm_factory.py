"""
LLM Factory Module

Provides cached LLM instances with purpose-specific configurations.
Each purpose has its own optimized settings to avoid configuration conflicts.
"""

# Standard library
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import lru_cache
from typing import Any, Literal, Optional

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
_DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Session-wide override for testing
_session_model_override: Optional[str] = None
_runtime_llm_overrides: ContextVar[dict[str, Any]] = ContextVar(
    "runtime_llm_overrides",
    default={},
)

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


@contextmanager
def llm_runtime_override(**overrides: Any):
    """
    Apply request-scoped runtime overrides for nested LLM calls.

    ContextVar keeps overrides task-local so concurrent requests do not leak
    model parameters into each other.
    """
    current = dict(_runtime_llm_overrides.get())
    merged = {**current, **{key: value for key, value in overrides.items() if value is not None}}
    token: Token[dict[str, Any]] = _runtime_llm_overrides.set(merged)
    try:
        yield
    finally:
        _runtime_llm_overrides.reset(token)

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

@lru_cache(maxsize=64)
def _get_llm_cached(
    purpose: LLMPurpose,
    model_name: Optional[str],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_output_tokens: Optional[int],
    thinking_budget: Optional[int],
    include_thoughts: Optional[bool],
    thinking_level: Optional[str],
) -> ChatGoogleGenerativeAI:
    """
    Returns a cached LLM instance for a specific purpose.

    Each purpose has its own optimized configuration (temperature, max_tokens).
    Instances are cached using lru_cache to avoid repeated initialization.
    """
    config = dict(_LLM_CONFIGS.get(purpose, _LLM_CONFIGS["rag_qa"]))

    if temperature is not None:
        config["temperature"] = temperature
    if top_p is not None:
        config["top_p"] = top_p
    if top_k is not None:
        config["top_k"] = top_k
    if max_output_tokens is not None:
        config["max_output_tokens"] = max_output_tokens
    if thinking_budget is not None:
        config["thinking_budget"] = thinking_budget
    if include_thoughts is not None:
        config["include_thoughts"] = include_thoughts
    if thinking_level is not None:
        config["thinking_level"] = thinking_level

    if model_name:
        model = model_name
    elif _session_model_override:
        model = _session_model_override
    else:
        model = _MODEL_BY_PURPOSE.get(purpose, _DEFAULT_MODEL)

    logger.info(
        "Initializing LLM for purpose: %s (model: %s, config: %s)",
        purpose,
        model,
        config,
    )

    return ChatGoogleGenerativeAI(model=model, **config)


def get_llm(purpose: LLMPurpose, model_name: str = None) -> ChatGoogleGenerativeAI:
    """
    Returns a cached LLM instance for a specific purpose.

    Runtime overrides come from task-local context so nested async flows can use
    request-scoped model settings safely.
    """
    overrides = _runtime_llm_overrides.get()
    effective_model = model_name or overrides.get("model_name")
    return _get_llm_cached(
        purpose=purpose,
        model_name=effective_model,
        temperature=overrides.get("temperature"),
        top_p=overrides.get("top_p"),
        top_k=overrides.get("top_k"),
        max_output_tokens=overrides.get("max_output_tokens"),
        thinking_budget=overrides.get("thinking_budget"),
        include_thoughts=overrides.get("include_thoughts"),
        thinking_level=overrides.get("thinking_level"),
    )


def clear_llm_cache() -> None:
    """
    Clears the LLM instance cache.

    Useful for testing or when configuration needs to be reloaded.
    """
    _get_llm_cached.cache_clear()
    logger.info("LLM cache cleared")
