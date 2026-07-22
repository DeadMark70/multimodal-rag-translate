from __future__ import annotations

from dataclasses import replace

from evaluation.benchmark_release import (
    BenchmarkRun,
    build_manifest,
    build_smoke_manifest,
    clustered_paired_bootstrap,
    ratio_of_sums,
    validate_benchmark_runs,
)


def _run(question_id: str, repeat: int, *, mode: str, version: str, shadow: bool = False, score: float = 0.5, tokens: int = 10) -> BenchmarkRun:
    return BenchmarkRun(
        run_id=f"{question_id}-{repeat}-{mode}-{version}{'-shadow' if shadow else ''}",
        campaign_id="campaign",
        question_id=question_id,
        repeat_number=repeat,
        mode=mode,
        condition_id=f"{mode}-{version}{'-shadow' if shadow else ''}",
        execution_profile=f"{mode}-{version}",
        agentic_execution_version=version,
        shadow_evaluation_policy="research" if shadow else None,
        completed=True,
        timed_out=False,
        accounting_complete=True,
        snapshot_fingerprint="snapshot",
        quality_score=score,
        runtime_tokens=tokens,
        latency_ms=100.0,
        category="cat",
    )


def test_manifest_rejects_shadow_as_official_comparator() -> None:
    result = validate_benchmark_runs(
        [
            _run("Q1", 1, mode="naive", version="v8"),
            _run("Q1", 1, mode="agentic", version="v8"),
            _run("Q1", 1, mode="agentic", version="v9", shadow=True),
        ]
    )

    assert result.comparable is False
    assert "missing_official_v9_arm" in result.reasons
    assert "shadow_arm_excluded" in result.reasons


def test_clustered_bootstrap_aggregates_repeats_before_question_resampling() -> None:
    runs = [
        _run("Q1", 1, mode="naive", version="v8", score=0.2),
        _run("Q1", 1, mode="agentic", version="v9", score=0.6),
        _run("Q1", 2, mode="naive", version="v8", score=0.4),
        _run("Q1", 2, mode="agentic", version="v9", score=0.8),
        _run("Q2", 1, mode="naive", version="v8", score=0.1),
        _run("Q2", 1, mode="agentic", version="v9", score=0.5),
        _run("Q2", 2, mode="naive", version="v8", score=0.3),
        _run("Q2", 2, mode="agentic", version="v9", score=0.7),
    ]
    report = clustered_paired_bootstrap(runs, seed=7, resamples=200)

    assert report.mean_delta == 0.4
    assert report.cluster_count == 2
    assert report.repeat_aggregation == "mean_per_question_before_bootstrap"
    assert report == clustered_paired_bootstrap(runs, seed=7, resamples=200)


def test_smoke_manifest_has_exactly_nine_official_arms_and_no_shadow() -> None:
    manifest = build_smoke_manifest(benchmark_id="smoke-1")

    assert manifest.kind == "smoke"
    assert len(manifest.ordered_blocks) == 9
    assert {item.question_id for item in manifest.ordered_blocks} == {"Q9", "Q15", "Q16"}
    assert all(item.shadow_evaluation_policy is None for item in manifest.ordered_blocks)


def test_formal_manifest_excludes_non_blocking_ablation_and_requires_full_matrix() -> None:
    runs = [
        _run(question_id, repeat, mode=mode, version=version, tokens=20 if version == "v9" else 10)
        for question_id in [f"Q{index}" for index in range(1, 17)]
        for repeat in range(1, 9)
        for mode, version in (("naive", "v8"), ("agentic", "v8"), ("agentic", "v9"))
    ]
    ablation = replace(
        _run("Q1", 1, mode="agentic", version="v8"),
        condition_id="ablation:v8-a-phase-policy",
    )
    manifest = build_manifest(benchmark_id="formal-1", runs=[*runs, ablation])

    assert manifest.kind == "formal"
    assert len(manifest.ordered_blocks) == 384
    assert manifest.non_blocking_ablations == ("ablation:v8-a-phase-policy",)


def test_ratio_of_sums_is_not_mean_of_per_run_ratios() -> None:
    v9 = [
        _run("Q1", 1, mode="agentic", version="v9", tokens=100),
        _run("Q2", 1, mode="agentic", version="v9", tokens=10),
    ]
    naive = [
        _run("Q1", 1, mode="naive", version="v8", tokens=10),
        _run("Q2", 1, mode="naive", version="v8", tokens=10),
    ]

    assert ratio_of_sums(v9, naive) == 5.5
