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
