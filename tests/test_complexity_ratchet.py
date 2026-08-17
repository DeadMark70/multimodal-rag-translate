from pathlib import Path

from scripts.check_complexity_ratchet import (
    PRODUCTION_ROOTS,
    compare_complexity,
    parse_ruff_findings,
)


def _finding(filename: str, function: str, score: int) -> dict:
    return {
        "code": "C901",
        "filename": filename,
        "location": {"row": 10, "column": 1},
        "message": f"`{function}` is too complex ({score} > 10)",
    }


def test_production_roots_include_agent_runtime():
    assert "agents" in PRODUCTION_ROOTS


def test_parse_ruff_findings_normalizes_paths_and_function_names(tmp_path: Path):
    payload = [
        _finding(str(tmp_path / "core" / "service.py"), "answer", 14),
        _finding("evaluation\\runner.py", "run_case", 11),
    ]

    assert parse_ruff_findings(payload, tmp_path) == {
        "core/service.py::answer": 14,
        "evaluation/runner.py::run_case": 11,
    }


def test_parse_ruff_findings_rejects_malformed_json_shape(tmp_path: Path):
    try:
        parse_ruff_findings({"not": "a list"}, tmp_path)
    except ValueError as exc:
        assert "list" in str(exc)
    else:
        raise AssertionError("malformed Ruff payload must fail closed")


def test_compare_allows_unchanged_decreased_and_removed_entries():
    baseline = {"core/a.py::same": 12, "core/a.py::lower": 15, "core/a.py::gone": 20}
    current = {"core/a.py::same": 12, "core/a.py::lower": 13}

    assert compare_complexity(baseline, current) == []


def test_compare_rejects_existing_score_increase():
    assert compare_complexity({"core/a.py::work": 12}, {"core/a.py::work": 13}) == [
        "core/a.py::work increased from 12 to 13"
    ]


def test_compare_allows_new_score_at_threshold():
    assert compare_complexity({}, {"core/new.py::work": 10}, threshold=10) == []


def test_compare_rejects_new_score_above_threshold_and_sorts_diagnostics():
    assert compare_complexity(
        {},
        {"core/z.py::work": 11, "core/a.py::work": 12},
        threshold=10,
    ) == [
        "core/a.py::work is new with complexity 12 (threshold 10)",
        "core/z.py::work is new with complexity 11 (threshold 10)",
    ]


def test_parse_ruff_findings_uses_highest_duplicate_score(tmp_path: Path):
    payload = [
        _finding("core/service.py", "answer", 11),
        _finding("core/service.py", "answer", 13),
    ]

    assert parse_ruff_findings(payload, tmp_path) == {"core/service.py::answer": 13}
