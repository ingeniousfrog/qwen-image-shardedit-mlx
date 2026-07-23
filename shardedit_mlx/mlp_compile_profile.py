"""Verdict helpers for scoped mx.compile of Qwen feed-forward modules."""

from __future__ import annotations

from shardedit_mlx.gemm_profile import relative_speedup


def decide_mlp_compile_verdict(
    *,
    eager_median: float,
    compiled_median: float,
    max_abs_error: float,
    all_finite: bool,
    speedup_threshold: float = 0.05,
    error_tolerance: float = 0.0,
) -> tuple[str, str]:
    """Classify whether scoped MLP compile is worth integrating.

    ``speedup_threshold`` is a relative gate (0.05 = require >= 1.05x).
    ``error_tolerance`` defaults to exact match; floating-point noise can raise it.
    """

    if speedup_threshold < 0:
        raise ValueError("speedup_threshold cannot be negative")
    if error_tolerance < 0:
        raise ValueError("error_tolerance cannot be negative")
    if not all_finite:
        return (
            "invalid_compiled_output",
            "compiled MLP produced non-finite values; do not integrate",
        )
    if max_abs_error > error_tolerance:
        return (
            "invalid_compiled_output",
            (
                f"compiled vs eager max abs error {max_abs_error:.6g} exceeds "
                f"tolerance {error_tolerance:.6g}"
            ),
        )

    speedup = relative_speedup(eager_median, compiled_median)
    if speedup >= (1.0 + speedup_threshold):
        return (
            "compile_helps",
            (
                f"compiled MLP is {speedup:.3f}x faster than eager "
                f"(threshold {1.0 + speedup_threshold:.2f}x) with matching outputs; "
                "worth integrating as a zero-risk opt-in"
            ),
        )
    return (
        "compile_not_enough",
        (
            f"compiled MLP speedup {speedup:.3f}x is below the "
            f"{1.0 + speedup_threshold:.2f}x gate; proceed to group_size sweep "
            "or accept that graph fusion alone does not close the dense/q6 gap"
        ),
    )
