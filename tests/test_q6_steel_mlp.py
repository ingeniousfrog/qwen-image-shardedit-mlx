"""Tests for Steel-inspired tiled q6 Metal MLP helpers."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from shardedit_mlx.q6_metal_mlp import quantized_linear_spec
from shardedit_mlx.q6_steel_mlp import affine_q6_qmm_t_tiled, make_tiled_feed_forward_callables
from shardedit_mlx.steel_tiled_profile import decide_steel_tiled_verdict


def test_tiled_qmm_matches_dense_float32() -> None:
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
    got = affine_q6_qmm_t_tiled(x, spec, dtype=mx.float32)
    mx.eval(ref, got)
    err = float(mx.max(mx.abs(ref - got)).item())
    assert bool(mx.isfinite(got).all().item())
    assert err < 1e-5


def test_tiled_mlp_matches_eager_small() -> None:
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
    eager, tiled, dense = make_tiled_feed_forward_callables(ff)
    x = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
    ye, yt, yd = eager(x), tiled(x), dense(x)
    mx.eval(ye, yt, yd)
    assert float(mx.max(mx.abs(ye.astype(mx.float32) - yt.astype(mx.float32))).item()) < 0.05
    assert float(mx.max(mx.abs(ye.astype(mx.float32) - yd.astype(mx.float32))).item()) < 0.05


def test_verdict_not_enough() -> None:
    verdict, reason = decide_steel_tiled_verdict(
        eager_median=0.01,
        tiled_median=0.04,
        naive_median=0.20,
        dense_median=0.007,
        tiled_f32_vs_dense_max_abs_error=0.0,
        tiled_all_finite=True,
    )
    assert verdict == "tiled_not_enough"
    assert "simdgroup" in reason or "Steel" in reason
