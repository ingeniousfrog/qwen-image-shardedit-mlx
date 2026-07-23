"""Verdict helpers for affine group_size speed/quality triage."""

from __future__ import annotations

from collections.abc import Mapping

from shardedit_mlx.gemm_profile import relative_speedup


def decide_group_size_verdict(
    *,
    baseline_group_size: int,
    baseline_median: float,
    candidates: Mapping[int, float],
    speedup_threshold: float = 0.05,
) -> tuple[str, str, tuple[int, ...]]:
    """Return verdict, reason, and group sizes worth a quality gate.

    ``candidates`` maps group_size -> median seconds. Baseline is usually 64.
    """

    if speedup_threshold < 0:
        raise ValueError("speedup_threshold cannot be negative")
    if baseline_median <= 0:
        raise ValueError("baseline_median must be positive")

    worthwhile: list[tuple[int, float]] = []
    for group_size, median in sorted(candidates.items()):
        if group_size == baseline_group_size:
            continue
        if median <= 0:
            raise ValueError(f"candidate median must be positive: group_size={group_size}")
        speedup = relative_speedup(baseline_median, median)
        if speedup >= (1.0 + speedup_threshold):
            worthwhile.append((group_size, speedup))

    if worthwhile:
        worthwhile.sort(key=lambda item: item[1], reverse=True)
        sizes = tuple(group_size for group_size, _ in worthwhile)
        details = ", ".join(
            f"gs={group_size}:{speedup:.3f}x" for group_size, speedup in worthwhile
        )
        return (
            "group_size_worth_quality_gate",
            (
                f"relative to gs={baseline_group_size}, candidates exceed "
                f"{1.0 + speedup_threshold:.2f}x: {details}; run quality regression "
                "before changing the runtime default"
            ),
            sizes,
        )

    summaries = []
    for group_size, median in sorted(candidates.items()):
        if group_size == baseline_group_size:
            summaries.append(f"gs={group_size}:1.000x")
            continue
        summaries.append(
            f"gs={group_size}:{relative_speedup(baseline_median, median):.3f}x"
        )
    return (
        "group_size_no_speedup",
        (
            f"no candidate beat gs={baseline_group_size} by "
            f"{speedup_threshold:.0%}: {', '.join(summaries)}; "
            "do not change runtime group_size; escalate to fused Metal kernel "
            "only if still pursuing the dense/q6 gap"
        ),
        (),
    )


def best_candidate_speedup(
    *,
    baseline_median: float,
    candidates: Mapping[int, float],
    baseline_group_size: int,
) -> tuple[int | None, float | None]:
    """Return the fastest non-baseline group_size and its speedup, if any."""

    best_size: int | None = None
    best_speedup: float | None = None
    for group_size, median in candidates.items():
        if group_size == baseline_group_size:
            continue
        speedup = relative_speedup(baseline_median, median)
        if best_speedup is None or speedup > best_speedup:
            best_size = group_size
            best_speedup = speedup
    return best_size, best_speedup
