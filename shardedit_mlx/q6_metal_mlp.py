"""Affine q6 Metal helpers for image-MLP fusion experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


# Matches mlx.nn.gelu_approx
_GELU_COEFF = 0.044715
_GELU_SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi)


HEADER = r"""
#include <metal_stdlib>
using namespace metal;

inline float gelu_approx_f(float x) {
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    const float c = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + precise::tanh(k * (x + c * x3)));
}

// Continuous 6-bit packing across a uint32 stream (MLX affine bits=6).
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

inline float dequant_q6(
    device const uint* packed,
    device const float* scales,
    device const float* qbiases,
    uint k,
    uint group_size
) {
    uint group = k / group_size;
    uint code = extract_q6(packed, k);
    float scale = scales[group];
    float qbias = qbiases[group];
    return scale * float(code) + qbias;
}
"""


SOURCE_QMM_T = r"""
    uint m = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    if (m >= M || n >= N) {
        return;
    }

    device const T* x_row = x + m * K;
    device const uint* w_row = weight + n * weight_stride;
    device const float* scale_row = scales + n * scale_stride;
    device const float* qbias_row = qbiases + n * scale_stride;

    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float xv = float(x_row[k]);
        float wv = dequant_q6(w_row, scale_row, qbias_row, k, group_size);
        acc += xv * wv;
    }
    if (has_bias) {
        acc += float(bias[n]);
    }
    if (apply_gelu) {
        acc = gelu_approx_f(acc);
    }
    out[m * N + n] = T(acc);
"""


_kernel_qmm_t = mx.fast.metal_kernel(
    name="shardedit_mlx_affine_q6_qmm_t",
    input_names=["x", "weight", "scales", "qbiases", "bias"],
    output_names=["out"],
    source=SOURCE_QMM_T,
    header=HEADER,
    ensure_row_contiguous=True,
)


@dataclass(frozen=True)
class Q6LinearSpec:
    weight: mx.array  # uint32 [N, K*bits/32]
    scales: mx.array  # [N, K/group_size]
    qbiases: mx.array  # [N, K/group_size]
    bias: mx.array | None
    group_size: int
    bits: int

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1] * 32 // self.bits)


def quantized_linear_spec(layer: nn.Module) -> Q6LinearSpec:
    """Extract packed affine q6 tensors from a QuantizedLinear-like module."""

    weight = layer.weight
    scales = layer.scales
    qbiases = layer.biases
    bias = getattr(layer, "bias", None)
    bits = int(layer.bits)
    group_size = int(layer.group_size)
    if bits != 6:
        raise ValueError(f"only bits=6 is supported, got {bits}")
    if weight.dtype != mx.uint32:
        raise ValueError(f"expected packed uint32 weight, got {weight.dtype}")
    return Q6LinearSpec(
        weight=weight,
        scales=scales.astype(mx.float32),
        qbiases=qbiases.astype(mx.float32),
        bias=None if bias is None else bias.astype(mx.float32),
        group_size=group_size,
        bits=bits,
    )


@dataclass(frozen=True)
class DenseLinearWeights:
    weight: mx.array  # [N, K]
    bias: mx.array | None


def dequantize_linear(spec: Q6LinearSpec, *, dtype: mx.Dtype) -> DenseLinearWeights:
    weight = mx.dequantize(
        spec.weight,
        spec.scales,
        biases=spec.qbiases,
        group_size=spec.group_size,
        bits=spec.bits,
        mode="affine",
        dtype=dtype,
    )
    bias = None if spec.bias is None else spec.bias.astype(dtype)
    return DenseLinearWeights(weight=weight, bias=bias)


def affine_q6_qmm_t(
    x: mx.array,
    spec: Q6LinearSpec,
    *,
    apply_gelu: bool = False,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Naive per-output-thread affine q6 matmul (weight transposed / Linear layout)."""

    if x.ndim != 3 and x.ndim != 2:
        raise ValueError(f"expected x with rank 2 or 3, got shape {x.shape}")
    out_dtype = x.dtype if dtype is None else dtype
    leading = x.shape[:-1]
    flat = x.reshape((-1, x.shape[-1]))
    m, k = int(flat.shape[0]), int(flat.shape[1])
    if k != spec.in_features:
        raise ValueError(f"x in_features {k} != weight in_features {spec.in_features}")
    n = spec.out_features
    bias = (
        mx.zeros((n,), dtype=mx.float32)
        if spec.bias is None
        else spec.bias.astype(mx.float32)
    )
    outputs = _kernel_qmm_t(
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
            ("apply_gelu", 1 if apply_gelu else 0),
        ],
        grid=(m, n, 1),
        threadgroup=(16, 16, 1),
        output_shapes=[(m, n)],
        output_dtypes=[out_dtype],
    )
    return outputs[0].reshape(leading + (n,))


def fused_q6_mlp(
    hidden: mx.array,
    mlp_in: Q6LinearSpec,
    mlp_out: Q6LinearSpec,
) -> mx.array:
    """Image/text MLP: q6 mlp_in + GELU epilogue, then q6 mlp_out."""

    mid = affine_q6_qmm_t(hidden, mlp_in, apply_gelu=True)
    return affine_q6_qmm_t(mid, mlp_out, apply_gelu=False)


def dense_mlp(
    hidden: mx.array,
    mlp_in: DenseLinearWeights,
    mlp_out: DenseLinearWeights,
) -> mx.array:
    """Dense Linear math with already-dequantized weights (fair upper bound)."""

    mid = hidden @ mlp_in.weight.T
    if mlp_in.bias is not None:
        mid = mid + mlp_in.bias
    mid = nn.gelu_approx(mid)
    out = mid @ mlp_out.weight.T
    if mlp_out.bias is not None:
        out = out + mlp_out.bias
    return out


def dense_mlp_from_q6(
    hidden: mx.array,
    mlp_in: Q6LinearSpec,
    mlp_out: Q6LinearSpec,
) -> mx.array:
    """Dequantize each call then dense math (diagnostic only; includes unpack cost)."""

    return dense_mlp(
        hidden,
        dequantize_linear(mlp_in, dtype=hidden.dtype),
        dequantize_linear(mlp_out, dtype=hidden.dtype),
    )


def eager_q6_mlp(ff: nn.Module, hidden: mx.array) -> mx.array:
    hidden = ff.mlp_in(hidden)
    hidden = nn.gelu_approx(hidden)
    hidden = ff.mlp_out(hidden)
    return hidden


def make_feed_forward_callables(
    ff: nn.Module,
    *,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
]:
    """Return (eager_q6, metal_fused, dense_predequant) callables for one FF module."""

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
        lambda x, a=mlp_in, b=mlp_out: fused_q6_mlp(x, a, b),
        lambda x, a=dense_in, b=dense_out: dense_mlp(x, a, b),
    )
