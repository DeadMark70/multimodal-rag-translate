"""Deterministic benchmark identity and question-cluster statistics.

The helpers in this module deliberately operate on immutable run projections.
They do not inspect provider configuration or mutate campaign state, which makes
the release report repeatable and prevents a UI from reconstructing a more
permissive comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil
from random import Random
from statistics import mean
from typing import Literal


OFFICIAL_ARMS = ("naive", "agentic-v8", "agentic-v9")


@dataclass(frozen=True, slots=True)
class ArmIdentity:
    mode: str
    condition_id: str
    execution_profile: str
    agentic_execution_version: str
    shadow_evaluation_policy: str | None = None

    @property
    def is_shadow(self) -> bool:
        return self.shadow_evaluation_policy is not None or self.mode == "agentic-v9-shadow"

    @property
    def is_non_blocking_ablation(self) -> bool:
        """A-type policy ablations are recorded, never official comparators."""
        return self.condition_id.startswith("ablation:")

    @property
    def official_label(self) -> str | None:
        if self.is_shadow or self.is_non_blocking_ablation:
            return None
        if self.mode == "naive":
            return "naive"
        if self.agentic_execution_version == "v8":
            return "agentic-v8"
        if self.agentic_execution_version == "v9":
            return "agentic-v9"
        return None


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    run_id: str
    campaign_id: str
    question_id: str
    repeat_number: int
    mode: str
    condition_id: str
    execution_profile: str
    agentic_execution_version: str
    shadow_evaluation_policy: str | None
    completed: bool
    timed_out: bool
    accounting_complete: bool
    golden_question_fingerprint: str | None
    environment_fingerprint: str | None
    evaluator_fingerprint: str | None
    quality_score: float | None
    runtime_tokens: int | None
    latency_ms: float | None
    category: str | None

    @property
    def identity(self) -> ArmIdentity:
        return ArmIdentity(
            mode=self.mode,
            condition_id=self.condition_id,
            execution_profile=self.execution_profile,
            agentic_execution_version=self.agentic_execution_version,
            shadow_evaluation_policy=self.shadow_evaluation_policy,
        )

    @property
    def pair_key(self) -> tuple[str, int]:
        return (self.question_id, self.repeat_number)


@dataclass(frozen=True, slots=True)
class ManifestBlock:
    question_id: str
    repeat_number: int
    mode: str
    condition_id: str
    execution_profile: str
    agentic_execution_version: str
    shadow_evaluation_policy: str | None
    golden_question_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    benchmark_id: str
    kind: Literal["smoke", "formal", "insufficient"]
    ordered_blocks: tuple[ManifestBlock, ...]
    arm_order_seed: int
    evaluator_blinding: dict[str, object]
    environment_fingerprint: str | None = None
    evaluator_fingerprint: str | None = None
    non_blocking_ablations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationResult:
    comparable: bool
    reasons: tuple[str, ...]
    official_runs: tuple[BenchmarkRun, ...]
    shadow_runs: tuple[BenchmarkRun, ...]
    ablation_runs: tuple[BenchmarkRun, ...] = ()


@dataclass(frozen=True, slots=True)
class ClusteredBootstrapResult:
    mean_delta: float | None
    ci_lower: float | None
    ci_upper: float | None
    cluster_count: int
    resamples: int
    seed: int
    repeat_aggregation: str = "mean_per_question_before_bootstrap"
    method: str = "paired_question_cluster_bootstrap"


def build_smoke_manifest(*, benchmark_id: str, seed: int = 20260722) -> BenchmarkManifest:
    """Return the immutable nine-unit smoke schedule without executing it."""
    blocks = [
        ManifestBlock(
            question_id=question_id,
            repeat_number=1,
            mode=mode,
            condition_id=condition,
            execution_profile=profile,
            agentic_execution_version=version,
            shadow_evaluation_policy=None,
        )
        for question_id in ("Q9", "Q15", "Q16")
        for mode, condition, profile, version in (
            ("naive", "naive-official", "naive_eval", "v8"),
            ("agentic", "agentic-v8-official", "agentic_eval_v8", "v8"),
            ("agentic", "agentic-v9-official", "agentic_eval_v9", "v9"),
        )
    ]
    return BenchmarkManifest(
        benchmark_id=benchmark_id,
        kind="smoke",
        ordered_blocks=tuple(_ordered_blocks(blocks, seed)),
        arm_order_seed=seed,
        evaluator_blinding={
            "enabled": True,
            "shown_mode_label": False,
            "method": "deterministic_per_question_repeat_arm_order",
        },
    )


def build_manifest(*, benchmark_id: str, runs: list[BenchmarkRun], seed: int = 20260722) -> BenchmarkManifest:
    """Freeze observed arm identity before release calculations."""
    validation = validate_benchmark_runs(runs)
    official = validation.official_runs
    question_count = len({run.question_id for run in official})
    repeat_numbers = {run.repeat_number for run in official}
    expected_formal = (
        validation.comparable
        and question_count == 16
        and repeat_numbers == set(range(1, 9))
        and len(official) == 384
    )
    kind: Literal["smoke", "formal", "insufficient"]
    if len(official) == 9 and question_count == 3 and repeat_numbers == {1}:
        kind = "smoke"
    elif expected_formal:
        kind = "formal"
    else:
        kind = "insufficient"
    environment_fingerprints = {
        run.environment_fingerprint for run in official if run.environment_fingerprint
    }
    evaluator_fingerprints = {
        run.evaluator_fingerprint for run in official if run.evaluator_fingerprint
    }
    blocks = [
        ManifestBlock(
            question_id=run.question_id,
            repeat_number=run.repeat_number,
            mode=run.mode,
            condition_id=run.condition_id,
            execution_profile=run.execution_profile,
            agentic_execution_version=run.agentic_execution_version,
            shadow_evaluation_policy=run.shadow_evaluation_policy,
            golden_question_fingerprint=run.golden_question_fingerprint,
        )
        for run in official
    ]
    return BenchmarkManifest(
        benchmark_id=benchmark_id,
        kind=kind,
        ordered_blocks=tuple(_ordered_blocks(blocks, seed)),
        arm_order_seed=seed,
        evaluator_blinding={
            "enabled": True,
            "shown_mode_label": False,
            "method": "deterministic_per_question_repeat_arm_order",
        },
        environment_fingerprint=(
            next(iter(environment_fingerprints))
            if len(environment_fingerprints) == 1
            else None
        ),
        evaluator_fingerprint=(
            next(iter(evaluator_fingerprints))
            if len(evaluator_fingerprints) == 1
            else None
        ),
        non_blocking_ablations=tuple(
            sorted({run.condition_id for run in validation.ablation_runs})
        ),
    )


def validate_benchmark_runs(runs: list[BenchmarkRun]) -> ValidationResult:
    """Reject invalid official comparisons instead of selecting a subset."""
    reasons: set[str] = set()
    shadow = tuple(run for run in runs if run.identity.is_shadow)
    ablations = tuple(
        run for run in runs if not run.identity.is_shadow and run.identity.is_non_blocking_ablation
    )
    official = tuple(
        run
        for run in runs
        if not run.identity.is_shadow and not run.identity.is_non_blocking_ablation
    )
    if shadow:
        reasons.add("shadow_arm_excluded")
    if not official:
        reasons.add("no_official_runs")
    by_pair: dict[tuple[str, int], dict[str, list[BenchmarkRun]]] = {}
    for run in official:
        if not run.completed:
            reasons.add("official_run_not_completed")
        if run.timed_out:
            reasons.add("official_run_timed_out")
        if not run.accounting_complete:
            reasons.add("partial_accounting")
        if not run.golden_question_fingerprint:
            reasons.add("missing_golden_question_fingerprint")
        if not run.environment_fingerprint:
            reasons.add("missing_environment_fingerprint")
        if not run.evaluator_fingerprint:
            reasons.add("missing_evaluator_metadata")
        arm = run.identity.official_label
        if arm is None:
            reasons.add("unknown_official_arm")
            continue
        by_pair.setdefault(run.pair_key, {}).setdefault(arm, []).append(run)
    for arms in by_pair.values():
        for arm in OFFICIAL_ARMS:
            count = len(arms.get(arm, []))
            if count == 0:
                reasons.add(f"missing_official_{arm.replace('agentic-', '')}_arm")
            elif count > 1:
                reasons.add("duplicate_official_arm")
        paired = [row for rows in arms.values() for row in rows]
        golden_fingerprints = {row.golden_question_fingerprint for row in paired}
        if len(golden_fingerprints) != 1 or None in golden_fingerprints:
            reasons.add("incompatible_golden_question")
        evaluator_fingerprints = {row.evaluator_fingerprint for row in paired}
        if len(evaluator_fingerprints) != 1 or None in evaluator_fingerprints:
            reasons.add("incompatible_evaluator_metadata")
    # A golden fingerprint is intentionally pair-local: distinct questions have
    # distinct ground truth and expected evidence.  Only the immutable runtime
    # environment and evaluator must agree benchmark-wide before question-level
    # deltas can share one clustered CI.
    benchmark_environments = {run.environment_fingerprint for run in official}
    if len(benchmark_environments) != 1 or None in benchmark_environments:
        reasons.add("incompatible_benchmark_environment")
    benchmark_evaluators = {run.evaluator_fingerprint for run in official}
    if len(benchmark_evaluators) != 1 or None in benchmark_evaluators:
        reasons.add("incompatible_benchmark_evaluator_metadata")
    return ValidationResult(
        comparable=not reasons,
        reasons=tuple(sorted(reasons)),
        official_runs=official,
        shadow_runs=shadow,
        ablation_runs=ablations,
    )


def clustered_paired_bootstrap(
    runs: list[BenchmarkRun], *, seed: int, resamples: int = 2000
) -> ClusteredBootstrapResult:
    """Bootstrap question-level paired deltas after mean repeat aggregation."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    deltas: dict[str, list[float]] = {}
    grouped: dict[tuple[str, int], dict[str, BenchmarkRun]] = {}
    for run in runs:
        label = run.identity.official_label
        if label not in {"naive", "agentic-v9"} or run.quality_score is None:
            continue
        grouped.setdefault(run.pair_key, {})[label] = run
    for (question_id, _repeat), arms in grouped.items():
        naive = arms.get("naive")
        v9 = arms.get("agentic-v9")
        if naive is not None and v9 is not None:
            deltas.setdefault(question_id, []).append(v9.quality_score - naive.quality_score)
    clusters = [mean(values) for _, values in sorted(deltas.items()) if values]
    if not clusters:
        return ClusteredBootstrapResult(None, None, None, 0, resamples, seed)
    observed = mean(clusters)
    rng = Random(seed)
    bootstrap = sorted(mean(rng.choice(clusters) for _ in clusters) for _ in range(resamples))
    lower_index = max(0, ceil(resamples * 0.025) - 1)
    upper_index = min(resamples - 1, ceil(resamples * 0.975) - 1)
    return ClusteredBootstrapResult(
        mean_delta=round(observed, 12),
        ci_lower=round(bootstrap[lower_index], 12),
        ci_upper=round(bootstrap[upper_index], 12),
        cluster_count=len(clusters),
        resamples=resamples,
        seed=seed,
    )


def ratio_of_sums(numerator_runs: list[BenchmarkRun], denominator_runs: list[BenchmarkRun]) -> float | None:
    """Return the official token ratio; no per-run-ratio averaging."""
    if any(run.runtime_tokens is None for run in [*numerator_runs, *denominator_runs]):
        return None
    denominator = sum(int(run.runtime_tokens or 0) for run in denominator_runs)
    if denominator <= 0:
        return None
    return sum(int(run.runtime_tokens or 0) for run in numerator_runs) / denominator


def successful_p95(runs: list[BenchmarkRun]) -> float | None:
    values = sorted(
        float(run.latency_ms)
        for run in runs
        if run.completed and not run.timed_out and run.accounting_complete and run.latency_ms is not None
    )
    if not values:
        return None
    return values[max(0, ceil(0.95 * len(values)) - 1)]


def stable_snapshot_fingerprint(payload: object) -> str:
    """Canonical digest used only for immutable snapshot comparisons."""
    import json

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _ordered_blocks(blocks: list[ManifestBlock], seed: int) -> list[ManifestBlock]:
    grouped: dict[tuple[str, int], list[ManifestBlock]] = {}
    for block in blocks:
        grouped.setdefault((block.question_id, block.repeat_number), []).append(block)
    ordered: list[ManifestBlock] = []
    for pair_key, pair_blocks in sorted(grouped.items()):
        rng = Random(f"{seed}:{pair_key[0]}:{pair_key[1]}")
        shuffled = list(pair_blocks)
        rng.shuffle(shuffled)
        ordered.extend(shuffled)
    return ordered
