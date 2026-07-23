"""Unit tests for affine q6 Metal MLP helpers."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from shardedit_mlx.mlp_metal_profile import decide_mlp_metal_verdict
from shardedit_mlx.q6_metal_mlp import (
    affine_q6_qmm_t,
    dequantize_linear,
    make_feed_forward_callables,
    quantized_linear_spec,
)


def _make_ff(in_features: int = 64, mid: int = 128, out_features: int = 64) -> nn.Module:
    class FF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp_in = nn.QuantizedLinear.from_linear(
                nn.Linear(in_features, mid, bias=True),
                bits=6,
                group_size=64,
            )
            self.mlp_out = nn.QuantizedLinear.from_linear(
                nn.Linear(mid, out_features, bias=True),
                bits=6,
                group_size=64,
            )

    return FF()


def test_affine_q6_qmm_matches_quantized_linear() -> None:
    mx.random.seed(0)
    ff = _make_ff()
    x = mx.random.normal((2, 8, 64)).astype(mx.bfloat16)
    spec = quantized_linear_spec(ff.mlp_in)
    got = affine_q6_qmm_t(x, spec, apply_gelu=False)
    ref = ff.mlp_in(x)
    mx.eval(got, ref)
    err = float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item())
    assert bool(mx.isfinite(got).all().item())
    assert err < 0.05


def test_fused_mlp_matches_eager() -> None:
    mx.random.seed(1)
    ff = _make_ff()
    x = mx.random.normal((1, 12, 64)).astype(mx.bfloat16)
    eager, metal, dense = make_feed_forward_callables(ff)
    y_eager = eager(x)
    y_metal = metal(x)
    y_dense = dense(x)
    mx.eval(y_eager, y_metal, y_dense)
    metal_err = float(
        mx.max(mx.abs(y_eager.astype(mx.float32) - y_metal.astype(mx.float32))).item()
    )
    dense_err = float(
        mx.max(mx.abs(y_eager.astype(mx.float32) - y_dense.astype(mx.float32))).item()
    )
    assert metal_err < 0.05
    assert dense_err < 0.05


def test_dequantize_linear_shapes() -> None:
    ff = _make_ff(64, 128, 64)
    spec = quantized_linear_spec(ff.mlp_in)
    dense = dequantize_linear(spec, dtype=mx.bfloat16)
    mx.eval(dense.weight)
    assert dense.weight.shape == (128, 64)
    assert dense.bias is not None and dense.bias.shape == (128,)


def test_metal_verdict_not_enough() -> None:
    verdict, reason = decide_mlp_metal_verdict(
        dense_eager_median=0.30,
        dense_median=0.20,
        dense_max_abs_error=0.01,
        dense_all_finite=True,
        metal_eager_median=0.01,
        metal_median=0.10,
        metal_f32_vs_dense_max_abs_error=1e-4,
        metal_bf16_vs_eager_max_abs_error=8.0,
        metal_all_finite=True,
    )
    assert verdict == "metal_not_enough"
    assert "Steel" in reason or "tiled" in reason


def test_metal_verdict_helps() -> None:
    verdict, _ = decide_mlp_metal_verdict(
        dense_eager_median=0.30,
        dense_median=0.20,
        dense_max_abs_error=0.01,
        dense_all_finite=True,
        metal_eager_median=0.10,
        metal_median=0.08,
        metal_f32_vs_dense_max_abs_error=1e-4,
        metal_bf16_vs_eager_max_abs_error=1.0,
        metal_all_finite=True,
    )
    assert verdict == "metal_helps"


def test_metal_verdict_invalid_packing() -> None:
    verdict, _ = decide_mlp_metal_verdict(
        dense_eager_median=0.30,
        dense_median=0.20,
        dense_max_abs_error=0.01,
        dense_all_finite=True,
        metal_eager_median=0.10,
        metal_median=0.08,
        metal_f32_vs_dense_max_abs_error=1.0,
        metal_bf16_vs_eager_max_abs_error=1.0,
        metal_all_finite=True,
        f32_error_tolerance=1e-2,
    )
    assert verdict == "invalid_metal_output"
