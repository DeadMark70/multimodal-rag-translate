"""LLM Factory Module.

Provides cached LLM instances with purpose-specific configurations.
Each purpose has its own optimized settings to avoid configuration conflicts.

Architecture note:
- this is the only runtime module that instantiates `ChatGoogleGenerativeAI`
- downstream business logic should import `get_llm(...)` from `core.providers`
- direct `google-genai` usage is reserved for control-plane helpers/services
"""

# Standard library
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import lru_cache
from typing import Any, Literal, Optional

# Third-party
from langchain_google_genai import ChatGoogleGenerativeAI

from core.llm_usage_callback import EvaluationUsageCallback
from evaluation.model_capabilities import get_thinking_capability

# Configure logging
logger = logging.getLogger(__name__)

# Purpose type for type safety
LLMPurpose = Literal[
    "rag_qa",
    "translation",
    "image_caption",
    "visual_verification",
    "context_generation",
    "proposition_extraction",
    "query_rewrite",
    "evaluator",
    "planner",
    "synthesizer",
    "summary",
    "graph_extraction",
    "community_summary",  # GraphRAG purposes
]
GraphRAGPurpose = Literal["graph_extraction", "community_summary"]
ExtractionProfile = Literal["standard", "high_precision"]

_GRAPH_RAG_MODEL = "gemini-3.1-flash-lite"

# Model mapping - only translation uses different model
_MODEL_BY_PURPOSE: dict[str, str] = {
    "translation": "gemini-2.5-flash-lite",
    "graph_extraction": _GRAPH_RAG_MODEL,
    "community_summary": _GRAPH_RAG_MODEL,
}

# Default model for all other purposes
_DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Session-wide override for testing
_session_model_override: Optional[str] = None
_runtime_llm_overrides: ContextVar[dict[str, Any]] = ContextVar(
    "runtime_llm_overrides",
    default={},
)
_GRAPH_RAG_THINKING_BUDGETS: dict[GraphRAGPurpose, int] = {
    "graph_extraction": 2048,
    "community_summary": 1024,
}
_GRAPH_RAG_THINKING_LEVELS: dict[GraphRAGPurpose, dict[ExtractionProfile, str]] = {
    "graph_extraction": {
        "standard": "medium",
        "high_precision": "high",
    },
    "community_summary": {
        "standard": "low",
        "high_precision": "low",
    },
}


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
def llm_runtime_override(*, clear: tuple[str, ...] = (), **overrides: Any):
    """
    Apply request-scoped runtime overrides for nested LLM calls.

    ContextVar keeps overrides task-local so concurrent requests do not leak
    model parameters into each other.
    """
    current = dict(_runtime_llm_overrides.get())
    for key in clear:
        current.pop(key, None)
    supplied = {key: value for key, value in overrides.items() if value is not None}
    if "setup_max_output_tokens" not in current and "max_output_tokens" in supplied:
        supplied.setdefault("setup_max_output_tokens", supplied["max_output_tokens"])
    if "setup_max_input_tokens" not in current and "max_input_tokens" in supplied:
        supplied.setdefault("setup_max_input_tokens", supplied["max_input_tokens"])
    merged = {**current, **supplied}
    token: Token[dict[str, Any]] = _runtime_llm_overrides.set(merged)
    try:
        yield
    finally:
        _runtime_llm_overrides.reset(token)


def current_llm_runtime_overrides() -> dict[str, Any]:
    """Return a copy of the request-scoped LLM runtime metadata."""
    return dict(_runtime_llm_overrides.get())


def _resolve_model_name(purpose: LLMPurpose, model_name: Optional[str] = None) -> str:
    """Resolve the effective model name for a purpose before SDK initialization."""
    if model_name:
        return model_name
    if _session_model_override:
        return _session_model_override
    return _MODEL_BY_PURPOSE.get(purpose, _DEFAULT_MODEL)


def get_graph_rag_model_name(
    purpose: GraphRAGPurpose,
    *,
    model_name: Optional[str] = None,
) -> str:
    """Return the effective GraphRAG model name for persisted run metadata."""
    return _resolve_model_name(purpose, model_name=model_name)


def get_graph_rag_runtime_overrides(
    purpose: GraphRAGPurpose,
    *,
    model_name: Optional[str] = None,
    extraction_profile: ExtractionProfile = "standard",
) -> dict[str, Any]:
    """Return model-family-aware thinking overrides for GraphRAG calls."""
    model = _resolve_model_name(purpose, model_name=model_name)
    capability = get_thinking_capability(model)
    overrides: dict[str, Any] = {"include_thoughts": False}

    if capability.control_type == "level":
        overrides["thinking_level"] = _GRAPH_RAG_THINKING_LEVELS[purpose][
            extraction_profile
        ]
    else:
        overrides["thinking_budget"] = _GRAPH_RAG_THINKING_BUDGETS[purpose]

    return overrides


@contextmanager
def graph_rag_llm_runtime_override(
    purpose: GraphRAGPurpose,
    *,
    model_name: Optional[str] = None,
    extraction_profile: ExtractionProfile = "standard",
):
    """Apply Setup-compatible thinking config for GraphRAG calls."""
    context = _runtime_llm_overrides.get()
    setup_thinking = context.get("thinking_enabled")
    if setup_thinking is not None:
        # An active Evaluation Setup is authoritative, even if a nested caller
        # supplies a stale or conflicting model_name argument.
        active_model = context.get("model_name") or model_name
        effective_model = active_model or _resolve_model_name(
            purpose, model_name=model_name
        )
        capability = get_thinking_capability(str(effective_model))
        overrides: dict[str, Any] = {
            "include_thoughts": bool(context.get("include_thoughts", False)),
        }
        if setup_thinking:
            if (
                capability.control_type == "budget"
                and context.get("thinking_budget") is not None
            ):
                overrides["thinking_budget"] = context["thinking_budget"]
            elif (
                capability.control_type == "level"
                and context.get("thinking_level") is not None
            ):
                overrides["thinking_level"] = context["thinking_level"]
        with llm_runtime_override(
            clear=("thinking_budget", "thinking_level"),
            **overrides,
        ):
            yield
        return

    active_model = model_name or context.get("model_name")
    with llm_runtime_override(
        clear=("thinking_budget", "thinking_level"),
        **get_graph_rag_runtime_overrides(
            purpose,
            model_name=active_model,
            extraction_profile=extraction_profile,
        ),
    ):
        yield


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

    model = _resolve_model_name(purpose, model_name=model_name)

    logger.info(
        "Initializing LLM for purpose: %s (model: %s, config: %s)",
        purpose,
        model,
        config,
    )

    return ChatGoogleGenerativeAI(
        model=model,
        callbacks=[
            EvaluationUsageCallback(
                purpose=purpose,
                provider="google",
                model_name=model,
            )
        ],
        **config,
    )


def get_llm(purpose: LLMPurpose, model_name: str = None) -> ChatGoogleGenerativeAI:
    """
    Returns a cached LLM instance for a specific purpose.

    Runtime overrides come from task-local context so nested async flows can use
    request-scoped model settings safely.
    """
    overrides = _runtime_llm_overrides.get()
    setup_model = (
        overrides.get("model_name") if "thinking_enabled" in overrides else None
    )
    effective_model = setup_model or model_name or overrides.get("model_name")
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


def get_llm_usage_metrics(response: Any) -> dict[str, int]:
    """
    Extract normalized token usage metrics from a LangChain/Google response object.

    Returns zeroes when usage metadata is unavailable.
    """
    usage = get_flat_llm_usage(response)
    if not usage:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }

    return usage


def get_flat_llm_usage(response: Any) -> dict[str, int]:
    """Extract only canonical flat token fields from a provider response."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata") or response.get("usage")
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage = usage.dict()
    if not isinstance(usage, dict) or not usage:
        return {}

    output_details = usage.get("output_token_details")
    details = output_details if isinstance(output_details, dict) else {}
    reasoning = usage.get(
        "reasoning_tokens",
        usage.get("thoughts_token_count", details.get("reasoning", 0)),
    )
    input_tokens = _usage_token_int(
        usage.get(
            "input_tokens",
            usage.get("prompt_tokens", usage.get("prompt_token_count", 0)),
        )
    )
    output_tokens = _usage_token_int(
        usage.get(
            "output_tokens",
            usage.get("completion_tokens", usage.get("candidates_token_count", 0)),
        )
    )
    reasoning_tokens = _usage_token_int(reasoning)
    reported_total = usage.get("total_tokens", usage.get("total_token_count"))
    total_tokens = (
        _usage_token_int(reported_total)
        if reported_total is not None
        else input_tokens + output_tokens + reasoning_tokens
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def _usage_token_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0
