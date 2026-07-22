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


def _run(
    question_id: str,
    repeat: int,
    *,
    mode: str,
    version: str,
    shadow: bool = False,
    score: float = 0.5,
    tokens: int = 10,
    golden_question_fingerprint: str | None = None,
    environment_fingerprint: str = "environment-v1",
) -> BenchmarkRun:
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
        golden_question_fingerprint=golden_question_fingerprint or f"golden-{question_id}",
        environment_fingerprint=environment_fingerprint,
        evaluator_fingerprint="evaluator-v1",
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


def _complete_two_question_matrix() -> list[BenchmarkRun]:
    return [
        _run(question_id, 1, mode=mode, version=version)
        for question_id in ("Q1", "Q2")
        for mode, version in (("naive", "v8"), ("agentic", "v8"), ("agentic", "v9"))
    ]


def test_validation_rejects_per_pair_golden_mismatch() -> None:
    runs = _complete_two_question_matrix()
    runs = [
        replace(run, golden_question_fingerprint="golden-Q1-other")
        if run.question_id == "Q1" and run.run_id.endswith("agentic-v9")
        else run
        for run in runs
    ]

    result = validate_benchmark_runs(runs)

    assert result.comparable is False
    assert "incompatible_golden_question" in result.reasons
    assert "incompatible_benchmark_environment" not in result.reasons


def test_validation_rejects_benchmark_wide_environment_mismatch_despite_distinct_goldens() -> None:
    runs = _complete_two_question_matrix()
    runs = [
        replace(run, environment_fingerprint="environment-v2") if run.question_id == "Q2" else run
        for run in runs
    ]

    result = validate_benchmark_runs(runs)

    assert result.comparable is False
    assert "incompatible_benchmark_environment" in result.reasons
    assert "incompatible_golden_question" not in result.reasons


def test_validation_rejects_benchmark_wide_evaluator_mismatch_despite_valid_pairs() -> None:
    runs = _complete_two_question_matrix()
    runs = [
        replace(run, evaluator_fingerprint="evaluator-v2") if run.question_id == "Q2" else run
        for run in runs
    ]

    result = validate_benchmark_runs(runs)

    assert result.comparable is False
    assert "incompatible_benchmark_evaluator_metadata" in result.reasons
    assert "incompatible_evaluator_metadata" not in result.reasons


def test_validation_accepts_distinct_question_goldens_with_one_environment_and_evaluator() -> None:
    result = validate_benchmark_runs(_complete_two_question_matrix())

    assert result.comparable is True
    assert result.reasons == ()
