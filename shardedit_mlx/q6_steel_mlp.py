"""Tiled affine-q6 matmul inspired by MLX Steel QuantizedBlockLoader + BlockMMA.

Application-level prototype via ``mx.fast.metal_kernel`` (no Steel headers /
simdgroup MMA). Correct and much faster than the naive per-output kernel, but
still well behind MLX ``affine_qmm``.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
from mlx import nn

from shardedit_mlx.q6_metal_mlp import (
    Q6LinearSpec,
    dense_mlp,
    dequantize_linear,
    eager_q6_mlp,
    quantized_linear_spec,
)

HEADER = r"""
#include <metal_stdlib>
using namespace metal;

inline float gelu_approx_f(float x) {
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + precise::tanh(k * (x + c * x3)));
}

inline uint extract_q6(device const uint* packed, uint index) {
    uint bit_off = index * 6u;
    uint word = bit_off / 32u;
    uint shift = bit_off % 32u;
    if (shift <= 26u) {
        return (packed[word] >> shift) & 0x3fu;
    }
    uint lo_bits = 32u - shift;
    uint lo = packed[word] >> shift;
    uint hi = packed[word + 1u] & ((1u << (6u - lo_bits)) - 1u);
    return lo | (hi << lo_bits);
}

inline float dequant_q6_at(
    device const uint* packed,
    device const float* scales,
    device const float* qbiases,
    uint k,
    uint group_size
) {
    uint group = k / group_size;
    uint code = extract_q6(packed, k);
    return scales[group] * float(code) + qbiases[group];
}
"""

# BM=BN=BK=32, TM=TN=2, TG=16x16. Literal shared sizes (template ints are unreliable
# as C-array bounds).
SOURCE_TILED = r"""
    threadgroup float As[32 * 32];
    threadgroup float Bs[32 * 32];

    const uint tg_n = threadgroup_position_in_grid.x;
    const uint tg_m = threadgroup_position_in_grid.y;
    const uint tx = thread_position_in_threadgroup.x;
    const uint ty = thread_position_in_threadgroup.y;
    const uint lid = ty * TGX + tx;
    const uint tgp_size = TGX * TGY;

    const uint m0 = tg_m * BM;
    const uint n0 = tg_n * BN;

    float acc[TM][TN];
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint idx = lid; idx < BM * BK; idx += tgp_size) {
            uint row = idx / BK;
            uint col = idx - row * BK;
            uint gm = m0 + row;
            uint gk = k0 + col;
            float v = 0.0f;
            if (gm < M && gk < K) {
                v = float(x[gm * K + gk]);
            }
            As[row * BK + col] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = lid; idx < BN * BK; idx += tgp_size) {
            uint row = idx / BK;
            uint col = idx - row * BK;
            uint gn = n0 + row;
            uint gk = k0 + col;
            float v = 0.0f;
            if (gn < N && gk < K) {
                device const uint* w_row = weight + gn * weight_stride;
                device const float* scale_row = scales + gn * scale_stride;
                device const float* qbias_row = qbiases + gn * scale_stride;
                v = dequant_q6_at(w_row, scale_row, qbias_row, gk, group_size);
            }
            Bs[row * BK + col] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; ++kk) {
            float a_reg[TM];
            float b_reg[TN];
            for (uint i = 0; i < TM; ++i) {
                a_reg[i] = As[(ty * TM + i) * BK + kk];
            }
            for (uint j = 0; j < TN; ++j) {
                b_reg[j] = Bs[(tx * TN + j) * BK + kk];
            }
            for (uint i = 0; i < TM; ++i) {
                for (uint j = 0; j < TN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM; ++i) {
        uint gm = m0 + ty * TM + i;
        if (gm >= M) continue;
        for (uint j = 0; j < TN; ++j) {
            uint gn = n0 + tx * TN + j;
            if (gn >= N) continue;
            float v = acc[i][j];
            if (has_bias) v += float(bias[gn]);
            if (apply_gelu) v = gelu_approx_f(v);
            out[gm * N + gn] = T(v);
        }
    }
"""

_BM = 32
_BN = 32
_BK = 32
_TM = 2
_TN = 2
_TGX = _BN // _TN
_TGY = _BM // _TM

_kernel_tiled = mx.fast.metal_kernel(
    name="shardedit_mlx_affine_q6_qmm_t_tiled_v3",
    input_names=["x", "weight", "scales", "qbiases", "bias"],
    output_names=["out"],
    source=SOURCE_TILED,
    header=HEADER,
    ensure_row_contiguous=True,
)


def affine_q6_qmm_t_tiled(
    x: mx.array,
    spec: Q6LinearSpec,
    *,
    apply_gelu: bool = False,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Tiled affine q6 matmul (Linear layout: y = x @ W.T + b)."""

    if x.ndim not in (2, 3):
        raise ValueError(f"expected x with rank 2 or 3, got shape {x.shape}")
    out_dtype = x.dtype if dtype is None else dtype
    leading = x.shape[:-1]
    flat = x.reshape((-1, x.shape[-1]))
    m, k = int(flat.shape[0]), int(flat.shape[-1])
    if k != spec.in_features:
        raise ValueError(f"x in_features {k} != weight in_features {spec.in_features}")
    n = spec.out_features
    bias = (
        mx.zeros((n,), dtype=mx.float32)
        if spec.bias is None
        else spec.bias.astype(mx.float32)
    )
    grid_n = (n + _BN - 1) // _BN
    grid_m = (m + _BM - 1) // _BM
    outputs = _kernel_tiled(
        inputs=[
            flat.astype(out_dtype),
            spec.weight,
            spec.scales,
            spec.qbiases,
            bias,
        ],
        template=[
            ("T", out_dtype),
            ("M", m),
            ("N", n),
            ("K", k),
            ("BM", _BM),
            ("BN", _BN),
            ("BK", _BK),
            ("TM", _TM),
            ("TN", _TN),
            ("TGX", _TGX),
            ("TGY", _TGY),
            ("group_size", spec.group_size),
            ("weight_stride", int(spec.weight.shape[1])),
            ("scale_stride", int(spec.scales.shape[1])),
            ("has_bias", 0 if spec.bias is None else 1),
            ("apply_gelu", 1 if apply_gelu else 0),
        ],
        # metal_kernel grid == total threads (dispatchThreads), not threadgroups.
        grid=(grid_n * _TGX, grid_m * _TGY, 1),
        threadgroup=(_TGX, _TGY, 1),
        output_shapes=[(m, n)],
        output_dtypes=[out_dtype],
    )
    return outputs[0].reshape(leading + (n,))


def fused_q6_mlp_tiled(
    hidden: mx.array,
    mlp_in: Q6LinearSpec,
    mlp_out: Q6LinearSpec,
) -> mx.array:
    mid = affine_q6_qmm_t_tiled(hidden, mlp_in, apply_gelu=True)
    return affine_q6_qmm_t_tiled(mid, mlp_out, apply_gelu=False)


def make_tiled_feed_forward_callables(
    ff: nn.Module,
    *,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
]:
    """Return (eager_q6, tiled_metal, dense_predequant)."""

    mlp_in = quantized_linear_spec(ff.mlp_in)
    mlp_out = quantized_linear_spec(ff.mlp_out)
    dense_in = dequantize_linear(mlp_in, dtype=dtype)
    dense_out = dequantize_linear(mlp_out, dtype=dtype)
    mx.eval(dense_in.weight, dense_out.weight)
    if dense_in.bias is not None:
        mx.eval(dense_in.bias)
    if dense_out.bias is not None:
        mx.eval(dense_out.bias)
    return (
        lambda x, module=ff: eager_q6_mlp(module, x),
        lambda x, a=mlp_in, b=mlp_out: fused_q6_mlp_tiled(x, a, b),
        lambda x, a=dense_in, b=dense_out: dense_mlp(x, a, b),
    )
