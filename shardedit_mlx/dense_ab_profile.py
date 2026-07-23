"""Verdict helpers for q6 vs dense single-block dequant overhead diagnosis."""

from __future__ import annotations

from shardedit_mlx.gemm_profile import relative_speedup


def decide_dense_ab_verdict(
    *,
    q6_median: float,
    dense_median: float,
    dense_vs_q6_max_abs_error: float,
    dense_vs_q6_all_finite: bool,
    speedup_threshold: float,
) -> tuple[str, str]:
    """Classify whether dequant unpacking appears to carry real runtime cost."""

    if speedup_threshold <= 0:
        raise ValueError("speedup_threshold must be positive")
    if not dense_vs_q6_all_finite:
        return (
            "invalid_dense_output",
            "dense bf16 block produced non-finite values; treat the timing as unusable",
        )
    # Quantization noise is expected. Reject only pathological magnitude errors.
    if dense_vs_q6_max_abs_error > 50.0:
        return (
            "invalid_dense_output",
            f"dense vs q6 max abs error {dense_vs_q6_max_abs_error:.3f} exceeds sanity gate",
        )

    speedup = relative_speedup(q6_median, dense_median)
    if speedup >= (1.0 + speedup_threshold):
        return (
            "dequant_has_overhead",
            (
                f"dense bf16 is {speedup:.3f}x faster than q6 "
                f"(threshold {1.0 + speedup_threshold:.2f}x); "
                "dequant unpacking appears to carry real cost, justifying a custom "
                "q6-packed fused Metal kernel experiment"
            ),
        )
    if speedup <= (1.0 - speedup_threshold):
        return (
            "gemm_bandwidth_bound",
            (
                f"dense bf16 is slower than q6 ({speedup:.3f}x relative); "
                "extra dense traffic outweighs any unpack savings, so quantized "
                "linear / image MLP kernel tuning is unlikely to help"
            ),
        )
    return (
        "gemm_bandwidth_bound",
        (
            f"dense vs q6 speedup {speedup:.3f}x is within the "
            f"±{speedup_threshold:.0%} noise gate; treat the paths as equivalent "
            "and stop investing in quantized-linear / image-MLP kernel retunes"
        ),
    )
