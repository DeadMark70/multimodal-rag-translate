from evaluation.model_capabilities import (
    get_thinking_capability,
    normalize_model_config_for_runtime,
)


def test_gemini_25_flash_uses_budget_control() -> None:
    capability = get_thinking_capability("gemini-2.5-flash")

    assert capability.supported is True
    assert capability.control_type == "budget"
    assert capability.supports_disable is True
    assert capability.supports_dynamic is True
    assert capability.budget_min == 0
    assert capability.budget_max == 24576


def test_gemini_3_uses_level_control() -> None:
    capability = get_thinking_capability("gemini-3.0-flash")

    assert capability.supported is True
    assert capability.control_type == "level"
    assert capability.levels == ["minimal", "low", "medium", "high"]
    assert capability.default_level == "medium"


def test_unknown_model_disables_thinking_controls() -> None:
    capability = get_thinking_capability("unknown-model")

    assert capability.supported is False
    assert capability.control_type == "none"


def test_runtime_normalization_budget_model_drops_level() -> None:
    runtime = normalize_model_config_for_runtime(
        {
            "model_name": "gemini-2.5-flash",
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "thinking_mode": True,
            "thinking_budget": 8192,
            "thinking_level": "high",
        }
    )

    assert runtime["thinking_budget"] == 8192
    assert runtime["thinking_enabled"] is True
    assert "thinking_level" not in runtime


def test_runtime_normalization_level_model_drops_budget() -> None:
    runtime = normalize_model_config_for_runtime(
        {
            "model_name": "gemini-3.0-flash",
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "thinking_mode": True,
            "thinking_budget": 8192,
            "thinking_level": "high",
        }
    )

    assert runtime["thinking_level"] == "high"
    assert runtime["thinking_enabled"] is True
    assert "thinking_budget" not in runtime


def test_runtime_normalization_no_thinking_mode_removes_thinking_fields() -> None:
    runtime = normalize_model_config_for_runtime(
        {
            "model_name": "gemini-2.5-flash",
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "thinking_mode": False,
            "thinking_budget": 8192,
            "thinking_level": "high",
        }
    )

    assert "thinking_budget" not in runtime
    assert "thinking_level" not in runtime


def test_runtime_normalization_exposes_explicit_thinking_disabled_state() -> None:
    runtime = normalize_model_config_for_runtime(
        {
            "model_name": "gemini-2.5-flash-lite",
            "thinking_mode": False,
            "thinking_budget": 8192,
            "thinking_level": "high",
        }
    )

    assert runtime["thinking_enabled"] is False
    assert "thinking_budget" not in runtime
    assert "thinking_level" not in runtime
