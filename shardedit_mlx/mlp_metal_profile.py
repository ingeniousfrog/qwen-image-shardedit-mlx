"""Verdict helpers for custom q6 Metal image-MLP fusion experiments."""

from __future__ import annotations

from shardedit_mlx.gemm_profile import relative_speedup


def decide_mlp_metal_verdict(
    *,
    dense_eager_median: float,
    dense_median: float,
    dense_max_abs_error: float,
    dense_all_finite: bool,
    metal_eager_median: float | None = None,
    metal_median: float | None = None,
    metal_f32_vs_dense_max_abs_error: float | None = None,
    metal_bf16_vs_eager_max_abs_error: float | None = None,
    metal_all_finite: bool | None = None,
    speedup_threshold: float = 0.05,
    f32_error_tolerance: float = 1e-2,
    bf16_error_tolerance: float = 32.0,
) -> tuple[str, str]:
    """Classify whether a custom Metal MLP prototype is worth integrating.

    Correctness primary gate: float32 Metal vs pre-dequant dense (packing/math).
    bf16 vs eager QuantizedLinear is informational (Steel vs custom acc order).
    """

    if speedup_threshold < 0:
        raise ValueError("speedup_threshold cannot be negative")
    if f32_error_tolerance < 0 or bf16_error_tolerance < 0:
        raise ValueError("error tolerances cannot be negative")
    if not dense_all_finite:
        return (
            "invalid_dense_output",
            "pre-dequant dense MLP produced non-finite values; upper bound unusable",
        )
    if dense_max_abs_error > max(bf16_error_tolerance, 1.0):
        return (
            "invalid_dense_output",
            (
                f"dense vs eager max abs error {dense_max_abs_error:.6g} exceeds "
                f"sanity gate {max(bf16_error_tolerance, 1.0):.6g}"
            ),
        )

    dense_speedup = relative_speedup(dense_eager_median, dense_median)

    if metal_median is None or metal_eager_median is None:
        return (
            "metal_skipped",
            (
                f"Metal timing skipped; dense upper bound is {dense_speedup:.3f}x vs eager. "
                "Re-run with a small --metal-tokens budget to validate the naive kernel, "
                "or invest in a Steel-class tiled q6 kernel before expecting full-miss wins."
            ),
        )
    if (
        metal_f32_vs_dense_max_abs_error is None
        or metal_bf16_vs_eager_max_abs_error is None
        or metal_all_finite is None
    ):
        raise ValueError("metal correctness fields required when metal ran")
    if not metal_all_finite:
        return (
            "invalid_metal_output",
            "Metal fused MLP produced non-finite values; do not integrate",
        )
    if metal_f32_vs_dense_max_abs_error > f32_error_tolerance:
        return (
            "invalid_metal_output",
            (
                f"float32 Metal vs dense max abs error "
                f"{metal_f32_vs_dense_max_abs_error:.6g} exceeds packing/math "
                f"tolerance {f32_error_tolerance:.6g}"
            ),
        )
    if metal_bf16_vs_eager_max_abs_error > bf16_error_tolerance:
        return (
            "invalid_metal_output",
            (
                f"bf16 Metal vs eager max abs error "
                f"{metal_bf16_vs_eager_max_abs_error:.6g} exceeds "
                f"tolerance {bf16_error_tolerance:.6g}"
            ),
        )

    metal_speedup = relative_speedup(metal_eager_median, metal_median)
    if metal_speedup >= (1.0 + speedup_threshold):
        return (
            "metal_helps",
            (
                f"Metal fused MLP is {metal_speedup:.3f}x faster than eager q6 "
                f"(threshold {1.0 + speedup_threshold:.2f}x); dense upper bound "
                f"{dense_speedup:.3f}x. Worth integrating behind a correctness gate."
            ),
        )
    return (
        "metal_not_enough",
        (
            f"naive/custom Metal MLP speedup {metal_speedup:.3f}x is below the "
            f"{1.0 + speedup_threshold:.2f}x gate (dense upper bound "
            f"{dense_speedup:.3f}x; f32 vs dense err "
            f"{metal_f32_vs_dense_max_abs_error:.3g}). Correctness of q6 unpack "
            "passes, but beating Steel affine_qmm requires a tiled/shared-memory "
            "kernel — not this per-output-thread prototype."
        ),
    )
