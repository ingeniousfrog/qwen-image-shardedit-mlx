"""Verdict helpers for window-local dense image-MLP residency."""

from __future__ import annotations

from shardedit_mlx.gemm_profile import relative_speedup


def decide_dense_prefetch_verdict(
    *,
    materialize_median: float,
    q6_img_ff_median: float,
    dense_img_ff_median: float,
    window_compute_median: float,
    min_mlp_savings: float = 0.02,
    overlap_slack: float = 1.0,
) -> tuple[str, str]:
    """Decide whether async prefetch / double-buffer is worth engineering.

    Sync path pays ``materialize`` every window. Prefetch can hide that cost behind
    the previous window's full compute when ``materialize <= window_compute``.
    The unlocked benefit is then approximately the MLP savings alone.
    """

    if min_mlp_savings < 0:
        raise ValueError("min_mlp_savings cannot be negative")
    if overlap_slack <= 0:
        raise ValueError("overlap_slack must be positive")
    if materialize_median <= 0 or q6_img_ff_median <= 0 or dense_img_ff_median <= 0:
        raise ValueError("timing medians must be positive")
    if window_compute_median <= 0:
        raise ValueError("window_compute_median must be positive")

    mlp_savings = q6_img_ff_median - dense_img_ff_median
    sync_net = mlp_savings - materialize_median
    overlap_budget = window_compute_median * overlap_slack

    if mlp_savings < min_mlp_savings:
        return (
            "prefetch_not_worth_it",
            (
                f"MLP savings {mlp_savings:+.3f}s is below {min_mlp_savings:.3f}s; "
                "prefetch cannot unlock a meaningful compute win."
            ),
        )
    if sync_net >= 0:
        return (
            "sync_already_wins",
            (
                f"sync net {sync_net:+.3f}s already positive "
                f"(savings {mlp_savings:+.3f}s - materialize {materialize_median:.3f}s); "
                "prefer sync dense before investing in prefetch."
            ),
        )
    if materialize_median <= overlap_budget:
        return (
            "prefetch_plausible",
            (
                f"materialize {materialize_median:.3f}s fits under window compute "
                f"{window_compute_median:.3f}s (slack {overlap_slack:.2f}x); "
                f"hiding it could unlock ~{mlp_savings:.3f}s/window "
                f"(sync net is {sync_net:+.3f}s)."
            ),
        )
    return (
        "prefetch_not_worth_it",
        (
            f"materialize {materialize_median:.3f}s exceeds overlap budget "
            f"{overlap_budget:.3f}s (window compute {window_compute_median:.3f}s); "
            f"pipeline would stall. MLP savings {mlp_savings:+.3f}s, "
            f"sync net {sync_net:+.3f}s."
        ),
    )


def decide_dense_img_ff_window_verdict(
    *,
    q6_median: float,
    dense_median: float,
    q6_peak_gib: float,
    dense_peak_gib: float,
    max_abs_error: float,
    all_finite: bool,
    speedup_threshold: float = 0.05,
    peak_budget_gib: float | None = None,
    error_tolerance: float = 32.0,
) -> tuple[str, str]:
    """Classify whether window-local dense img_ff is worth enabling."""

    if speedup_threshold < 0:
        raise ValueError("speedup_threshold cannot be negative")
    if error_tolerance < 0:
        raise ValueError("error_tolerance cannot be negative")
    if peak_budget_gib is not None and peak_budget_gib <= 0:
        raise ValueError("peak_budget_gib must be positive when set")
    if not all_finite:
        return (
            "invalid_dense_output",
            "dense img_ff window produced non-finite values; do not enable",
        )
    if max_abs_error > error_tolerance:
        return (
            "invalid_dense_output",
            (
                f"dense vs q6 max abs error {max_abs_error:.6g} exceeds "
                f"tolerance {error_tolerance:.6g}"
            ),
        )
    if peak_budget_gib is not None and dense_peak_gib > peak_budget_gib:
        return (
            "peak_over_budget",
            (
                f"dense peak {dense_peak_gib:.2f} GiB exceeds budget "
                f"{peak_budget_gib:.2f} GiB (q6 peak {q6_peak_gib:.2f} GiB)"
            ),
        )

    speedup = relative_speedup(q6_median, dense_median)
    peak_delta = dense_peak_gib - q6_peak_gib
    if speedup >= (1.0 + speedup_threshold):
        return (
            "dense_img_ff_helps",
            (
                f"window dense img_ff is {speedup:.3f}x faster than q6 "
                f"(threshold {1.0 + speedup_threshold:.2f}x); peak delta "
                f"{peak_delta:+.3f} GiB. Worth keeping as opt-in."
            ),
        )
    return (
        "dense_img_ff_not_enough",
        (
            f"window dense img_ff speedup {speedup:.3f}x is below the "
            f"{1.0 + speedup_threshold:.2f}x gate; peak delta {peak_delta:+.3f} GiB. "
            "Keep disabled; next bet is Steel-class tiled q6 kernels."
        ),
    )
