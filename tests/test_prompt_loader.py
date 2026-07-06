from pathlib import Path

import pytest

from core.prompt_loader import (
    PromptConfigError,
    PromptRegistry,
    format_agentic_rag_prompt,
    format_graph_rag_prompt,
    format_prompt,
    format_rag_pipeline_prompt,
    get_agentic_rag_prompt_registry,
    get_graph_rag_prompt_registry,
    get_rag_pipeline_prompt_registry,
    get_rag_qa_prompt_registry,
)


def test_prompt_registry_formats_prompt_from_json(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        """
{
  "prompts": {
    "plain_rag_answer": {
      "version": 1,
      "description": "Plain RAG answer prompt.",
      "required_variables": ["question", "context_text"],
      "template": "Q: {question}\\nC: {context_text}"
    }
  }
}
""",
        encoding="utf-8",
    )

    registry = PromptRegistry(path)

    assert (
        registry.format(
            "plain_rag_answer",
            question="What is RAG?",
            context_text="Retrieved chunk",
        )
        == "Q: What is RAG?\nC: Retrieved chunk"
    )


def test_prompt_registry_rejects_missing_required_variable(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        """
{
  "prompts": {
    "plain_rag_answer": {
      "version": 1,
      "description": "Plain RAG answer prompt.",
      "required_variables": ["question", "context_text"],
      "template": "Q: {question}\\nC: {context_text}"
    }
  }
}
""",
        encoding="utf-8",
    )

    registry = PromptRegistry(path)

    with pytest.raises(PromptConfigError, match="context_text"):
        registry.format("plain_rag_answer", question="What is RAG?")


def test_prompt_registry_rejects_template_variables_not_declared(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        """
{
  "prompts": {
    "broken": {
      "version": 1,
      "description": "Broken prompt.",
      "required_variables": ["question"],
      "template": "Q: {question}\\nC: {context_text}"
    }
  }
}
""",
        encoding="utf-8",
    )

    with pytest.raises(PromptConfigError, match="context_text"):
        PromptRegistry(path)


def test_prompt_registry_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        """
{
  "prompts": {
    "plain_rag_answer": {
      "version": 1,
      "description": "Plain RAG answer prompt.",
      "required_variables": ["question"],
      "template": "Q: {question}"
    }
  }
}
""",
        encoding="utf-8",
    )

    registry = PromptRegistry(path)

    with pytest.raises(PromptConfigError, match="missing"):
        registry.format("missing", question="What is RAG?")


def test_default_format_prompt_uses_production_registry() -> None:
    formatted = format_prompt(
        "plain_rag_answer",
        context_text="context",
        graph_section="",
        history_section="",
        question="question",
        intent_constraints="constraints",
    )

    assert "context" in formatted
    assert "question" in formatted
    assert "constraints" in formatted


def test_domain_registry_accessors_load_expected_files() -> None:
    assert get_rag_qa_prompt_registry().path.name == "rag_qa_prompts.json"
    assert get_agentic_rag_prompt_registry().path.name == "agentic_rag_prompts.json"
    assert get_graph_rag_prompt_registry().path.name == "graph_rag_prompts.json"
    assert get_rag_pipeline_prompt_registry().path.name == "rag_pipeline_prompts.json"


def test_domain_formatters_delegate_to_domain_registries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agentic_path = tmp_path / "agentic.json"
    agentic_path.write_text(
        """
{
  "prompts": {
    "sample": {
      "version": 1,
      "description": "Agentic sample prompt.",
      "required_variables": ["name"],
      "template": "agentic {name}"
    }
  }
}
""",
        encoding="utf-8",
    )

    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        """
{
  "prompts": {
    "sample": {
      "version": 1,
      "description": "Graph sample prompt.",
      "required_variables": ["name"],
      "template": "graph {name}"
    }
  }
}
""",
        encoding="utf-8",
    )

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        """
{
  "prompts": {
    "sample": {
      "version": 1,
      "description": "Pipeline sample prompt.",
      "required_variables": ["name"],
      "template": "pipeline {name}"
    }
  }
}
""",
        encoding="utf-8",
    )

    agentic_registry = PromptRegistry(agentic_path)
    graph_registry = PromptRegistry(graph_path)
    pipeline_registry = PromptRegistry(pipeline_path)

    from core import prompt_loader

    monkeypatch.setattr(
        prompt_loader,
        "get_agentic_rag_prompt_registry",
        lambda: agentic_registry,
    )
    monkeypatch.setattr(
        prompt_loader,
        "get_graph_rag_prompt_registry",
        lambda: graph_registry,
    )
    monkeypatch.setattr(
        prompt_loader,
        "get_rag_pipeline_prompt_registry",
        lambda: pipeline_registry,
    )

    assert format_agentic_rag_prompt("sample", name="Ada") == "agentic Ada"
    assert format_graph_rag_prompt("sample", name="Ada") == "graph Ada"
    assert format_rag_pipeline_prompt("sample", name="Ada") == "pipeline Ada"
