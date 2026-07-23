"""Minimal simdgroup-MMA affine-q6 matmul prototype (single common shape).

Uses Metal ``simdgroup_matrix`` / ``simdgroup_multiply_accumulate`` via
``mx.fast.metal_kernel``. Scope is intentionally narrow: prove whether hardware
MMA beats MLX ``affine_qmm`` enough to justify a full kernel integration.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
from mlx import nn

from shardedit_mlx.q6_metal_mlp import (
    Q6LinearSpec,
    dequantize_linear,
    eager_q6_mlp,
    quantized_linear_spec,
)


HEADER = r"""
#include <metal_stdlib>
using namespace metal;

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

# One simdgroup (32 threads) owns one 8x8 C tile. Activations and dequantized
# weight tiles live in threadgroup memory; MMA does the 8x8x8 accumulate.
SOURCE_SIMDGROUP = r"""
    threadgroup float As[8 * 8];
    threadgroup float Bs[8 * 8];

    const uint tg_n = threadgroup_position_in_grid.x;
    const uint tg_m = threadgroup_position_in_grid.y;
    const uint lid = thread_index_in_threadgroup;
    const uint m0 = tg_m * 8u;
    const uint n0 = tg_n * 8u;

    simdgroup_matrix<float, 8, 8> acc;
    acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0u; k0 < K; k0 += 8u) {
        // Cooperative load of A[m0:m0+8, k0:k0+8] and B = W.T[k0:k0+8, n0:n0+8].
        for (uint idx = lid; idx < 64u; idx += 32u) {
            uint row = idx / 8u;
            uint col = idx - row * 8u;
            uint gm = m0 + row;
            uint gk = k0 + col;
            float av = 0.0f;
            if (gm < M && gk < K) {
                av = float(x[gm * K + gk]);
            }
            As[row * 8u + col] = av;

            uint gn = n0 + row;
            uint gk_w = k0 + col;
            float bv = 0.0f;
            if (gn < N && gk_w < K) {
                device const uint* w_row = weight + gn * weight_stride;
                device const float* scale_row = scales + gn * scale_stride;
                device const float* qbias_row = qbiases + gn * scale_stride;
                // Store as B[col, row] so Bs is [BK, BN] = W.T tile.
                bv = dequant_q6_at(w_row, scale_row, qbias_row, gk_w, group_size);
            }
            Bs[col * 8u + row] = bv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> a_tile;
        simdgroup_matrix<float, 8, 8> b_tile;
        simdgroup_load(a_tile, As, 8, ulong2(0, 0), false);
        simdgroup_load(b_tile, Bs, 8, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float Cs[8 * 8];
    simdgroup_store(acc, Cs, 8, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = lid; idx < 64u; idx += 32u) {
        uint row = idx / 8u;
        uint col = idx - row * 8u;
        uint gm = m0 + row;
        uint gn = n0 + col;
        if (gm >= M || gn >= N) {
            continue;
        }
        float v = Cs[row * 8u + col];
        if (has_bias) {
            v += float(bias[gn]);
        }
        out[gm * N + gn] = T(v);
    }
"""

_kernel_simdgroup = mx.fast.metal_kernel(
    name="shardedit_mlx_affine_q6_qmm_t_simdgroup_v1",
    input_names=["x", "weight", "scales", "qbiases", "bias"],
    output_names=["out"],
    source=SOURCE_SIMDGROUP,
    header=HEADER,
    ensure_row_contiguous=True,
)


def affine_q6_qmm_t_simdgroup(
    x: mx.array,
    spec: Q6LinearSpec,
    *,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Simdgroup-MMA affine q6 matmul prototype: y = x @ W.T (+ b)."""

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
    grid_n = (n + 7) // 8
    grid_m = (m + 7) // 8
    outputs = _kernel_simdgroup(
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
            ("group_size", spec.group_size),
            ("weight_stride", int(spec.weight.shape[1])),
            ("scale_stride", int(spec.scales.shape[1])),
            ("has_bias", 0 if spec.bias is None else 1),
        ],
        # One simdgroup (32 threads) per 8x8 C tile.
        grid=(grid_n * 32, grid_m, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[out_dtype],
    )
    return outputs[0].reshape(leading + (n,))


def make_simdgroup_mlp_in_callables(
    ff: nn.Module,
    *,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
]:
    """Return (eager_q6_mlp_in, simdgroup_mlp_in, dense_mlp_in) for image MLP."""

    mlp_in = quantized_linear_spec(ff.mlp_in)
    dense_in = dequantize_linear(mlp_in, dtype=dtype)
    mx.eval(dense_in.weight)
    if dense_in.bias is not None:
        mx.eval(dense_in.bias)

    def eager_in(hidden: mx.array, module: nn.Module = ff) -> mx.array:
        return module.mlp_in(hidden)

    def simd_in(hidden: mx.array, spec: Q6LinearSpec = mlp_in) -> mx.array:
        return affine_q6_qmm_t_simdgroup(hidden, spec, dtype=dtype)

    def dense_in_fn(hidden: mx.array, weights=dense_in) -> mx.array:
        out = hidden @ weights.weight.T
        if weights.bias is not None:
            out = out + weights.bias
        return out

    return eager_in, simd_in, dense_in_fn


def make_eager_vs_simdgroup_feed_forward(
    ff: nn.Module,
    *,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[Callable[[mx.array], mx.array], Callable[[mx.array], mx.array]]:
    """Full MLP wrappers used only for small correctness checks."""

    mlp_in = quantized_linear_spec(ff.mlp_in)
    mlp_out = quantized_linear_spec(ff.mlp_out)

    def simd_ff(hidden: mx.array) -> mx.array:
        mid = affine_q6_qmm_t_simdgroup(hidden, mlp_in, dtype=dtype)
        mid = nn.gelu_approx(mid)
        return affine_q6_qmm_t_simdgroup(mid, mlp_out, dtype=dtype)

    return (lambda x, module=ff: eager_q6_mlp(module, x), simd_ff)
