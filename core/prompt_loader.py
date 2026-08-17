from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Formatter
from typing import Any


class PromptConfigError(ValueError):
    """Raised when prompt configuration is invalid or cannot be formatted."""


@dataclass(frozen=True)
class PromptDefinition:
    version: int
    description: str
    required_variables: tuple[str, ...]
    template: str


class PromptRegistry:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._prompts = self._load()

    def get(self, key: str) -> PromptDefinition:
        try:
            return self._prompts[key]
        except KeyError as exc:
            raise PromptConfigError(f"Unknown prompt key: {key}") from exc

    def format(self, key: str, **variables: Any) -> str:
        prompt = self.get(key)
        return _format_definition(prompt, **variables)

    def _load(self) -> dict[str, PromptDefinition]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PromptConfigError(
                f"Could not read prompt config: {self.path}"
            ) from exc

        if not isinstance(raw, dict) or not isinstance(raw.get("prompts"), dict):
            raise PromptConfigError("Prompt config must contain a 'prompts' object")

        prompts: dict[str, PromptDefinition] = {}
        for key, value in raw["prompts"].items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise PromptConfigError(
                    "Prompt entries must be objects keyed by string"
                )
            prompts[key] = _parse_prompt(key, value)
        return prompts


def _format_definition(prompt: PromptDefinition, **variables: Any) -> str:
    missing = [name for name in prompt.required_variables if name not in variables]
    if missing:
        raise PromptConfigError(
            f"Missing required variables for prompt: {', '.join(missing)}"
        )

    declared = set(prompt.required_variables)
    used = _template_variables(prompt.template)
    undeclared = sorted(used - declared)
    if undeclared:
        raise PromptConfigError(
            f"Template uses undeclared variables: {', '.join(undeclared)}"
        )

    if not used:
        return prompt.template

    try:
        return prompt.template.format(**variables)
    except (KeyError, IndexError, ValueError) as exc:
        raise PromptConfigError("Could not format prompt template") from exc


_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _prompt_registry(filename: str) -> PromptRegistry:
    return PromptRegistry(_PROMPTS_DIR / filename)


@lru_cache(maxsize=1)
def get_rag_qa_prompt_registry() -> PromptRegistry:
    return _prompt_registry("rag_qa_prompts.json")


@lru_cache(maxsize=1)
def get_agentic_rag_prompt_registry() -> PromptRegistry:
    return _prompt_registry("agentic_rag_prompts.json")


@lru_cache(maxsize=1)
def get_graph_rag_prompt_registry() -> PromptRegistry:
    return _prompt_registry("graph_rag_prompts.json")


@lru_cache(maxsize=1)
def get_rag_pipeline_prompt_registry() -> PromptRegistry:
    return _prompt_registry("rag_pipeline_prompts.json")


@lru_cache(maxsize=1)
def get_default_prompt_registry() -> PromptRegistry:
    return get_rag_qa_prompt_registry()


def format_prompt(key: str, **variables: Any) -> str:
    """Format a prompt from the default production registry."""
    return get_default_prompt_registry().format(key, **variables)


def format_agentic_rag_prompt(key: str, **variables: Any) -> str:
    return get_agentic_rag_prompt_registry().format(key, **variables)


def format_graph_rag_prompt(key: str, **variables: Any) -> str:
    return get_graph_rag_prompt_registry().format(key, **variables)


def format_rag_pipeline_prompt(key: str, **variables: Any) -> str:
    return get_rag_pipeline_prompt_registry().format(key, **variables)


def _parse_prompt(key: str, value: dict[str, Any]) -> PromptDefinition:
    version = value.get("version")
    description = value.get("description")
    required_variables = value.get("required_variables")
    template = value.get("template")

    if not isinstance(version, int):
        raise PromptConfigError(f"Prompt '{key}' version must be an integer")
    if not isinstance(description, str):
        raise PromptConfigError(f"Prompt '{key}' description must be a string")
    if not isinstance(required_variables, list) or not all(
        isinstance(item, str) for item in required_variables
    ):
        raise PromptConfigError(
            f"Prompt '{key}' required_variables must be a list of strings"
        )
    if not isinstance(template, str):
        raise PromptConfigError(f"Prompt '{key}' template must be a string")

    prompt = PromptDefinition(
        version=version,
        description=description,
        required_variables=tuple(required_variables),
        template=template,
    )

    undeclared = sorted(_template_variables(template) - set(prompt.required_variables))
    if undeclared:
        raise PromptConfigError(
            f"Prompt '{key}' template uses undeclared variables: {', '.join(undeclared)}"
        )

    return prompt


def _template_variables(template: str) -> set[str]:
    names: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            name = field_name.split(".", 1)[0].split("[", 1)[0]
            if name.isidentifier():
                names.add(name)
    return names
