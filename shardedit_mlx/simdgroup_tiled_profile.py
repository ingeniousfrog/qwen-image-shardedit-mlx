"""Verdict helpers for simdgroup-MMA q6 Metal prototypes."""

from __future__ import annotations

from shardedit_mlx.gemm_profile import relative_speedup


def estimate_e2e_wallclock_improvement(
    *,
    gemm_speedup: float,
    gemm_fraction_of_block: float,
    block_compute_fraction_of_step: float = 0.94,
    full_miss_fraction_of_wall: float = 0.88,
) -> float:
    """Map a single-GEMM speedup into expected F1B2 wall-clock improvement.

    Defaults approximate the measured F1B2 stack:
    - image MLP ≈51% of block; mlp_in ≈ half of that → ~0.255
    - window compute ≈38s of ≈40s step → ~0.94
    - ~5 full-miss × ~40s of ~228s wall → ~0.88

    Returns a signed fraction: positive means faster wall-clock, negative means slower.
    """

    if gemm_speedup <= 0:
        raise ValueError("gemm_speedup must be positive")
    if not 0.0 < gemm_fraction_of_block <= 1.0:
        raise ValueError("gemm_fraction_of_block must be in (0, 1]")
    if not 0.0 < block_compute_fraction_of_step <= 1.0:
        raise ValueError("block_compute_fraction_of_step must be in (0, 1]")
    if not 0.0 < full_miss_fraction_of_wall <= 1.0:
        raise ValueError("full_miss_fraction_of_wall must be in (0, 1]")

    # Time remaining for the targeted GEMM after speedup: 1/speedup.
    gemm_time_factor = 1.0 / gemm_speedup
    block_factor = 1.0 - gemm_fraction_of_block * (1.0 - gemm_time_factor)
    step_factor = 1.0 - block_compute_fraction_of_step * (1.0 - block_factor)
    # Only full-miss steps benefit; hit steps are unchanged.
    wall_factor = 1.0 - full_miss_fraction_of_wall * (1.0 - step_factor)
    return 1.0 - wall_factor


def decide_simdgroup_verdict(
    *,
    eager_median: float,
    simdgroup_median: float,
    dense_median: float,
    simdgroup_f32_vs_dense_max_abs_error: float,
    simdgroup_all_finite: bool,
    gemm_fraction_of_block: float = 0.255,
    min_e2e_improvement: float = 0.10,
    f32_error_tolerance: float = 1e-2,
) -> tuple[str, str]:
    """Classify whether a simdgroup MMA prototype clears the go/no-go gate."""

    if min_e2e_improvement <= 0 or min_e2e_improvement >= 1:
        raise ValueError("min_e2e_improvement must be in (0, 1)")
    if f32_error_tolerance < 0:
        raise ValueError("f32_error_tolerance cannot be negative")
    if not simdgroup_all_finite:
        return (
            "invalid_simdgroup_output",
            "simdgroup Metal GEMM produced non-finite values; do not integrate",
        )
    if simdgroup_f32_vs_dense_max_abs_error > f32_error_tolerance:
        return (
            "invalid_simdgroup_output",
            (
                f"float32 simdgroup vs dense max abs error "
                f"{simdgroup_f32_vs_dense_max_abs_error:.6g} exceeds "
                f"tolerance {f32_error_tolerance:.6g}"
            ),
        )

    gemm_speedup = relative_speedup(eager_median, simdgroup_median)
    dense_speedup = relative_speedup(eager_median, dense_median)
    e2e = estimate_e2e_wallclock_improvement(
        gemm_speedup=gemm_speedup,
        gemm_fraction_of_block=gemm_fraction_of_block,
    )
    if gemm_speedup < 1.0:
        return (
            "simdgroup_not_enough",
            (
                f"simdgroup GEMM is slower than eager q6 ({gemm_speedup:.3f}x); "
                f"projected F1B2 wall delta {e2e:+.1%} "
                f"(dense GEMM upper bound {dense_speedup:.3f}x). Archive closed."
            ),
        )
    if e2e >= min_e2e_improvement:
        return (
            "simdgroup_go",
            (
                f"simdgroup GEMM is {gemm_speedup:.3f}x vs eager "
                f"(dense upper bound {dense_speedup:.3f}x); "
                f"projected F1B2 wall improvement {e2e:.1%} "
                f"meets the {min_e2e_improvement:.0%} go gate. Worth a full C plan."
            ),
        )
    return (
        "simdgroup_not_enough",
        (
            f"simdgroup GEMM is {gemm_speedup:.3f}x vs eager "
            f"(dense upper bound {dense_speedup:.3f}x); "
            f"projected F1B2 wall improvement {e2e:.1%} is below the "
            f"{min_e2e_improvement:.0%} go gate. Archive closed."
        ),
    )
