"""Verdict helpers for Steel-inspired tiled q6 Metal MLP kernels."""

from __future__ import annotations

from shardedit_mlx.gemm_profile import relative_speedup


def decide_steel_tiled_verdict(
    *,
    eager_median: float,
    tiled_median: float,
    naive_median: float | None,
    dense_median: float,
    tiled_f32_vs_dense_max_abs_error: float,
    tiled_all_finite: bool,
    speedup_threshold: float = 0.05,
    f32_error_tolerance: float = 1e-2,
) -> tuple[str, str]:
    """Classify whether a tiled custom q6 kernel is worth integrating."""

    if speedup_threshold < 0:
        raise ValueError("speedup_threshold cannot be negative")
    if f32_error_tolerance < 0:
        raise ValueError("f32_error_tolerance cannot be negative")
    if not tiled_all_finite:
        return (
            "invalid_tiled_output",
            "tiled Metal MLP produced non-finite values; do not integrate",
        )
    if tiled_f32_vs_dense_max_abs_error > f32_error_tolerance:
        return (
            "invalid_tiled_output",
            (
                f"float32 tiled vs dense max abs error "
                f"{tiled_f32_vs_dense_max_abs_error:.6g} exceeds "
                f"tolerance {f32_error_tolerance:.6g}"
            ),
        )

    tiled_vs_eager = relative_speedup(eager_median, tiled_median)
    dense_vs_eager = relative_speedup(eager_median, dense_median)
    naive_note = ""
    if naive_median is not None:
        tiled_vs_naive = relative_speedup(naive_median, tiled_median)
        naive_note = f"; {tiled_vs_naive:.2f}x vs naive Metal"

    if tiled_vs_eager >= (1.0 + speedup_threshold):
        return (
            "tiled_helps",
            (
                f"tiled Metal MLP is {tiled_vs_eager:.3f}x faster than eager q6 "
                f"(threshold {1.0 + speedup_threshold:.2f}x; dense upper bound "
                f"{dense_vs_eager:.3f}x{naive_note}). Worth integrating behind a gate."
            ),
        )
    return (
        "tiled_not_enough",
        (
            f"tiled Metal MLP speedup {tiled_vs_eager:.3f}x is below the "
            f"{1.0 + speedup_threshold:.2f}x gate (dense upper bound "
            f"{dense_vs_eager:.3f}x{naive_note}). Threadgroup tiling alone cannot "
            "beat Steel simdgroup MMA; keep disabled and prefer window dense "
            "img_ff or an upstream MLX kernel contribution."
        ),
    )
