"""Tests for simdgroup-MMA q6 Metal prototype helpers."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from shardedit_mlx.q6_metal_mlp import quantized_linear_spec
from shardedit_mlx.q6_simdgroup_mlp import (
    affine_q6_qmm_t_simdgroup,
    make_simdgroup_mlp_in_callables,
)
from shardedit_mlx.simdgroup_tiled_profile import (
    decide_simdgroup_verdict,
    estimate_e2e_wallclock_improvement,
)


def test_estimate_e2e_improvement_monotonic() -> None:
    low = estimate_e2e_wallclock_improvement(
        gemm_speedup=1.2, gemm_fraction_of_block=0.255
    )
    high = estimate_e2e_wallclock_improvement(
        gemm_speedup=2.0, gemm_fraction_of_block=0.255
    )
    assert 0.0 < low < high < 1.0


def test_verdict_go_when_projected_clears_gate() -> None:
    # ~2.5x on mlp_in (~25.5% of block) projects above 10% e2e.
    verdict, reason = decide_simdgroup_verdict(
        eager_median=1.0,
        simdgroup_median=0.35,
        dense_median=0.50,
        simdgroup_f32_vs_dense_max_abs_error=0.0,
        simdgroup_all_finite=True,
        gemm_fraction_of_block=0.255,
        min_e2e_improvement=0.10,
    )
    assert verdict == "simdgroup_go"
    assert "go gate" in reason


def test_verdict_not_enough_when_slow() -> None:
    verdict, reason = decide_simdgroup_verdict(
        eager_median=1.0,
        simdgroup_median=2.0,
        dense_median=0.6,
        simdgroup_f32_vs_dense_max_abs_error=0.0,
        simdgroup_all_finite=True,
    )
    assert verdict == "simdgroup_not_enough"
    assert "Archive closed" in reason


def test_simdgroup_qmm_matches_dense_float32() -> None:
    mx.random.seed(0)
    lin = nn.Linear(128, 256, bias=True)
    q = nn.QuantizedLinear.from_linear(lin, bits=6, group_size=64)
    spec = quantized_linear_spec(q)
    x = mx.random.normal((16, 128)).astype(mx.float32)
    w = mx.dequantize(
        q.weight,
        q.scales,
        q.biases,
        group_size=64,
        bits=6,
        mode="affine",
        dtype=mx.float32,
    )
    ref = x @ w.T + q.bias.astype(mx.float32)
    got = affine_q6_qmm_t_simdgroup(x, spec, dtype=mx.float32)
    mx.eval(ref, got)
    err = float(mx.max(mx.abs(ref - got)).item())
    assert bool(mx.isfinite(got).all().item())
    assert err < 1e-4


def test_simdgroup_mlp_in_matches_eager_small() -> None:
    mx.random.seed(1)

    class FF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp_in = nn.QuantizedLinear.from_linear(
                nn.Linear(64, 128, bias=True), bits=6, group_size=64
            )
            self.mlp_out = nn.QuantizedLinear.from_linear(
                nn.Linear(128, 64, bias=True), bits=6, group_size=64
            )

    ff = FF()
    eager, simd, dense = make_simdgroup_mlp_in_callables(ff)
    x = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
    ye, ys, yd = eager(x), simd(x), dense(x)
    mx.eval(ye, ys, yd)
    assert float(mx.max(mx.abs(ye.astype(mx.float32) - ys.astype(mx.float32))).item()) < 0.05
    assert float(mx.max(mx.abs(ye.astype(mx.float32) - yd.astype(mx.float32))).item()) < 0.05
