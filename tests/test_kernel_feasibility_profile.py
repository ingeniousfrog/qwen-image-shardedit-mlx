"""Tests for img_ff kernel feasibility verdicts."""

from __future__ import annotations

from shardedit_mlx.kernel_feasibility_profile import (
    KernelPathSummary,
    decide_img_ff_kernel_spike_verdict,
    is_kernel_candidate,
)


def test_dense_upper_bound_only_when_no_candidate_clears_gate() -> None:
    verdict, reason = decide_img_ff_kernel_spike_verdict(
        (
            KernelPathSummary("mlx_q6", 10.0, 0.0, True),
            KernelPathSummary("mlx_dense_predequant", 7.0, 0.0, True),
            KernelPathSummary("shardedit_mlx_fork_tiled_metal", 12.0, 0.0, True),
        ),
        speedup_threshold=0.10,
    )

    assert verdict == "dense_upper_bound_only"
    assert "headroom" in reason


def test_kernel_candidate_promising_when_candidate_clears_gate() -> None:
    verdict, reason = decide_img_ff_kernel_spike_verdict(
        (
            KernelPathSummary("mlx_q6", 10.0, 0.0, True),
            KernelPathSummary("mlx_dense_predequant", 8.0, 0.0, True),
            KernelPathSummary("mlx_kquant_q6_k", 8.5, 2.0, True),
        ),
        speedup_threshold=0.10,
        max_abs_error_tolerance=4.0,
    )

    assert verdict == "kernel_candidate_promising"
    assert "mlx_kquant_q6_k" in reason


def test_fast_candidate_with_large_error_is_not_promoted() -> None:
    verdict, reason = decide_img_ff_kernel_spike_verdict(
        (
            KernelPathSummary("mlx_q6", 10.0, 0.0, True),
            KernelPathSummary("mlx_dense_predequant", 7.0, 0.0, True),
            KernelPathSummary("mlx_kquant_q6_k", 5.0, 100.0, True),
        ),
        speedup_threshold=0.10,
        max_abs_error_tolerance=4.0,
    )

    assert verdict == "dense_upper_bound_only"
    assert "invalid/skipped" in reason


def test_path_name_classification() -> None:
    assert not is_kernel_candidate("mlx_q6")
    assert not is_kernel_candidate("mlx_dense_predequant")
    assert is_kernel_candidate("shardedit_mlx_fork_tiled_metal")
