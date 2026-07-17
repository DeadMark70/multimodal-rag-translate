"""Strict research-summary contract regression tests."""

from evaluation.research_analytics import nearest_rank


def test_nearest_rank_percentiles_are_observed_values() -> None:
    assert nearest_rank([100, 200, 300, 400, 500], 0.50) == 300
    assert nearest_rank([100, 200, 300, 400, 500], 0.95) == 500
