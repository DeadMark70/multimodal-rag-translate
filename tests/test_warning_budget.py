from pathlib import Path

import yaml

from scripts.run_pytest_with_warning_budget import (
    parse_warning_count,
    warning_budget_exit_code,
)


def test_parse_warning_count_defaults_to_zero_without_summary():
    assert parse_warning_count("42 passed in 1.23s") == 0


def test_parse_warning_count_accepts_singular_summary():
    assert parse_warning_count("1 passed, 1 warning in 0.10s") == 1


def test_parse_warning_count_accepts_plural_summary():
    assert parse_warning_count("10 passed, 56 warnings in 2.00s") == 56


def test_parse_warning_count_ignores_unrelated_numbers():
    assert parse_warning_count("collected 900 tests\n56 passed in 12.34s") == 0


def test_warning_budget_allows_lower_and_equal_counts():
    assert warning_budget_exit_code(0, 55, 56) == 0
    assert warning_budget_exit_code(0, 56, 56) == 0


def test_warning_budget_rejects_only_a_higher_count():
    assert warning_budget_exit_code(0, 57, 56) == 1


def test_warning_budget_preserves_pytest_failure_exit_code():
    assert warning_budget_exit_code(5, 57, 56) == 5


def _workflow() -> dict:
    root = Path(__file__).resolve().parent.parent
    return yaml.safe_load(
        (root / ".github" / "workflows" / "no-external-api-test.yml").read_text(
            encoding="utf-8"
        )
    )


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_workflow_has_read_only_timed_python_jobs_with_dependency_caching():
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"deployment-compile", "quality-and-tests"}

    deployment = workflow["jobs"]["deployment-compile"]
    quality = workflow["jobs"]["quality-and-tests"]
    assert deployment["timeout-minutes"] == 10
    assert quality["timeout-minutes"] == 45

    deployment_setup = _step(deployment, "Setup deployment Python")["with"]
    quality_setup = _step(quality, "Setup quality Python")["with"]
    assert deployment_setup == {
        "python-version": "3.11",
        "cache": "pip",
        "cache-dependency-path": "requirements.txt",
    }
    assert quality_setup == {
        "python-version": "3.13",
        "cache": "pip",
        "cache-dependency-path": "requirements.txt",
    }


def test_deployment_compile_covers_all_production_python_entrypoints():
    deployment = _workflow()["jobs"]["deployment-compile"]
    command = str(_step(deployment, "Compile deployment source")["run"]).split()

    assert command[:4] == ["python", "-m", "compileall", "-q"]
    assert set(command[4:]) == {
        "agents",
        "conversations",
        "core",
        "data_base",
        "evaluation",
        "graph_rag",
        "image_service",
        "main.py",
        "multimodal_rag",
        "pdfserviceMD",
        "stats",
        "supabase_client.py",
    }


def test_quality_workflow_runs_complete_fake_provider_gate_contract():
    workflow = _workflow()
    job = workflow["jobs"]["quality-and-tests"]
    assert job["env"] == {
        "TEST_MODE": "true",
        "USE_FAKE_PROVIDERS": "true",
        "CI_BLOCK_EXTERNAL_NETWORK": "true",
    }
    commands = "\n".join(
        str(step.get("run", "")) for step in job["steps"]
    )
    assert "python -m pip install -r requirements.txt" in commands
    assert "python -m ruff check . --select E9,F63,F7,F82" in commands
    assert "python scripts/check_complexity_ratchet.py --check" in commands
    assert (
        "python scripts/run_pytest_with_warning_budget.py --max-warnings 56 -- -q"
        in commands
    )
    assert "python scripts/sync_openapi_artifacts.py --check" in commands
    assert "python scripts/check_markdown_links.py" in commands
