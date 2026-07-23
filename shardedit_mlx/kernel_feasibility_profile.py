"""Pure verdict helpers for img_ff kernel feasibility spikes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from shardedit_mlx.gemm_profile import relative_speedup


BASELINE_PATH = "mlx_q6"
DENSE_PATH = "mlx_dense_predequant"


class KernelTimingLike(Protocol):
    name: str
    median_seconds: float
    max_abs_error_vs_mlx_q6: float | None
    all_finite: bool


@dataclass(frozen=True)
class KernelPathSummary:
    name: str
    median_seconds: float
    max_abs_error_vs_mlx_q6: float | None
    all_finite: bool


def is_kernel_candidate(name: str) -> bool:
    return name not in {BASELINE_PATH, DENSE_PATH}


def decide_img_ff_kernel_spike_verdict(
    paths: Sequence[KernelTimingLike],
    *,
    speedup_threshold: float = 0.10,
    max_abs_error_tolerance: float = 32.0,
) -> tuple[str, str]:
    """Classify whether any non-dense kernel candidate deserves more work.

    ``mlx_dense_predequant`` is treated as an upper bound, not as an integration
    candidate, because it changes the weight residency and memory profile.
    """

    if speedup_threshold < 0:
        raise ValueError("speedup_threshold cannot be negative")
    if max_abs_error_tolerance < 0:
        raise ValueError("max_abs_error_tolerance cannot be negative")

    baseline = next((path for path in paths if path.name == BASELINE_PATH), None)
    if baseline is None:
        raise ValueError(f"missing {BASELINE_PATH} baseline path")
    gate = 1.0 + speedup_threshold
    dense = next((path for path in paths if path.name == DENSE_PATH), None)
    dense_note = "no dense upper bound measured"
    if dense is not None:
        dense_speedup = relative_speedup(baseline.median_seconds, dense.median_seconds)
        dense_note = f"dense upper bound is {dense_speedup:.3f}x vs MLX q6"

    valid_candidates: list[tuple[str, float]] = []
    invalid_candidates: list[str] = []
    for path in paths:
        if not is_kernel_candidate(path.name):
            continue
        error = path.max_abs_error_vs_mlx_q6
        if not path.all_finite:
            invalid_candidates.append(f"{path.name}: non-finite output")
            continue
        if error is None or error > max_abs_error_tolerance:
            invalid_candidates.append(
                f"{path.name}: err {error} > {max_abs_error_tolerance:g}"
            )
            continue
        valid_candidates.append(
            (path.name, relative_speedup(baseline.median_seconds, path.median_seconds))
        )

    if valid_candidates:
        best_name, best_speedup = max(valid_candidates, key=lambda item: item[1])
        if best_speedup >= gate:
            return (
                "kernel_candidate_promising",
                (
                    f"{best_name} is {best_speedup:.3f}x faster than MLX q6 "
                    f"(gate {gate:.2f}x); {dense_note}. Continue with a wider "
                    "block sweep and e2e proof before integration."
                ),
            )

    measured_note = "no valid kernel candidate measured"
    if valid_candidates:
        best_name, best_speedup = max(valid_candidates, key=lambda item: item[1])
        measured_note = f"best valid candidate {best_name} is {best_speedup:.3f}x"
    if invalid_candidates:
        measured_note = f"{measured_note}; invalid/skipped: {'; '.join(invalid_candidates)}"

    if dense is not None:
        dense_speedup = relative_speedup(baseline.median_seconds, dense.median_seconds)
        if dense_speedup >= gate:
            return (
                "dense_upper_bound_only",
                (
                    f"{measured_note}; {dense_note}. This says the shape has "
                    "headroom, but no measured open-source/fork kernel clears "
                    f"the {gate:.2f}x gate yet."
                ),
            )

    return (
        "kernel_candidate_not_enough",
        (
            f"{measured_note}; {dense_note}. Do not integrate a kernel path from "
            "this spike."
        ),
    )
