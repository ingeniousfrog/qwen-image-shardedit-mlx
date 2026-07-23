"""Window-local dense image MLP: dequantize once after LoRA, free with the window."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import gc
from typing import Any

import mlx.core as mx
from mlx import nn

from shardedit_mlx.q6_metal_mlp import (
    DenseLinearWeights,
    dense_mlp,
    dequantize_linear,
    quantized_linear_spec,
)
from shardedit_mlx.qwen_block_loader import LoadedBlock


@dataclass(frozen=True)
class DenseImgFFHandle:
    """Per-block dense img_ff replacement installed for one residency window."""

    block_index: int
    mlp_in: DenseLinearWeights
    mlp_out: DenseLinearWeights
    bytes_materialized: int


class DenseFeedForward:
    """Drop-in img_ff replacement: dense Linear math matching QwenFeedForward."""

    def __init__(
        self,
        mlp_in: DenseLinearWeights,
        mlp_out: DenseLinearWeights,
        *,
        mlp_in_lora: tuple[mx.array, mx.array, float] | None = None,
        mlp_out_lora: tuple[mx.array, mx.array, float] | None = None,
    ) -> None:
        self.mlp_in = mlp_in
        self.mlp_out = mlp_out
        self.mlp_in_lora = mlp_in_lora
        self.mlp_out_lora = mlp_out_lora

    def _linear(
        self,
        hidden: mx.array,
        weights: DenseLinearWeights,
        lora: tuple[mx.array, mx.array, float] | None,
    ) -> mx.array:
        out = hidden @ weights.weight.T
        if weights.bias is not None:
            out = out + weights.bias
        if lora is not None:
            lora_a, lora_b, scale = lora
            out = out + scale * ((hidden @ lora_a) @ lora_b)
        return out

    def __call__(self, hidden: mx.array) -> mx.array:
        from mlx import nn

        hidden = self._linear(hidden, self.mlp_in, self.mlp_in_lora)
        hidden = nn.gelu_approx(hidden)
        return self._linear(hidden, self.mlp_out, self.mlp_out_lora)


def _tensor_nbytes(array: mx.array) -> int:
    return int(array.size) * array.dtype.size


def _dense_nbytes(weights: DenseLinearWeights) -> int:
    total = _tensor_nbytes(weights.weight)
    if weights.bias is not None:
        total += _tensor_nbytes(weights.bias)
    return total


def _base_linear(layer: Any) -> nn.Module:
    """Unwrap LoRA wrappers to the underlying QuantizedLinear / Linear."""

    nested = getattr(layer, "linear", None)
    if isinstance(nested, (nn.Linear, nn.QuantizedLinear)):
        return nested
    nested = getattr(layer, "base_linear", None)
    if isinstance(nested, (nn.Linear, nn.QuantizedLinear)):
        return nested
    return layer


def _extract_lora_adapters(
    layer: Any,
    *,
    dtype: mx.Dtype,
) -> tuple[mx.array, mx.array, float] | None:
    """Return (A, B, scale) for a single LoRALinear, keeping adapters additive."""

    if hasattr(layer, "lora_A") and hasattr(layer, "lora_B"):
        return (
            layer.lora_A.astype(dtype),
            layer.lora_B.astype(dtype),
            float(getattr(layer, "scale", 1.0)),
        )
    loras = getattr(layer, "loras", None)
    if not loras:
        return None
    if len(loras) == 1:
        lora = loras[0]
        return (
            lora.lora_A.astype(dtype),
            lora.lora_B.astype(dtype),
            float(lora.scale),
        )
    # Multiple adapters: fold into one effective delta via fused weight path.
    return None


def _lora_delta_weight(layer: Any, *, dtype: mx.Dtype) -> mx.array | None:
    """Fallback multi-LoRA fuse: scale * (A @ B).T in [out, in] layout."""

    loras = getattr(layer, "loras", None)
    if not loras or len(loras) < 2:
        return None
    delta = None
    for lora in loras:
        piece = float(lora.scale) * (lora.lora_A @ lora.lora_B)
        delta = piece if delta is None else delta + piece
    assert delta is not None
    return delta.T.astype(dtype)


def materialize_dense_linear(layer: Any, *, dtype: mx.Dtype) -> DenseLinearWeights:
    """Dequantize Linear/QuantizedLinear base weights (LoRA kept separate when possible)."""

    base = _base_linear(layer)
    if isinstance(base, nn.QuantizedLinear):
        dense = dequantize_linear(quantized_linear_spec(base), dtype=dtype)
    elif isinstance(base, nn.Linear):
        bias = None if base.bias is None else base.bias.astype(dtype)
        dense = DenseLinearWeights(weight=base.weight.astype(dtype), bias=bias)
    else:
        raise TypeError(f"unsupported linear module type: {type(base)!r}")

    delta = _lora_delta_weight(layer, dtype=dtype)
    if delta is not None:
        if delta.shape != dense.weight.shape:
            raise ValueError(
                f"LoRA delta shape {delta.shape} != weight shape {dense.weight.shape}"
            )
        dense = DenseLinearWeights(weight=dense.weight + delta, bias=dense.bias)
    return dense


def materialize_dense_img_ff(
    img_ff: Any,
    *,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[DenseFeedForward, int]:
    """Dequantize one QwenFeedForward's mlp_in/mlp_out into a dense module."""

    mlp_in = materialize_dense_linear(img_ff.mlp_in, dtype=dtype)
    mlp_out = materialize_dense_linear(img_ff.mlp_out, dtype=dtype)
    mlp_in_lora = _extract_lora_adapters(img_ff.mlp_in, dtype=dtype)
    mlp_out_lora = _extract_lora_adapters(img_ff.mlp_out, dtype=dtype)
    # If adapters were folded into weights (multi-LoRA), don't also apply additive.
    if _lora_delta_weight(img_ff.mlp_in, dtype=dtype) is not None:
        mlp_in_lora = None
    if _lora_delta_weight(img_ff.mlp_out, dtype=dtype) is not None:
        mlp_out_lora = None

    leaves = [mlp_in.weight, mlp_out.weight]
    if mlp_in.bias is not None:
        leaves.append(mlp_in.bias)
    if mlp_out.bias is not None:
        leaves.append(mlp_out.bias)
    if mlp_in_lora is not None:
        leaves.extend([mlp_in_lora[0], mlp_in_lora[1]])
    if mlp_out_lora is not None:
        leaves.extend([mlp_out_lora[0], mlp_out_lora[1]])
    mx.eval(*leaves)
    nbytes = _dense_nbytes(mlp_in) + _dense_nbytes(mlp_out)
    return (
        DenseFeedForward(
            mlp_in,
            mlp_out,
            mlp_in_lora=mlp_in_lora,
            mlp_out_lora=mlp_out_lora,
        ),
        nbytes,
    )

def prepare_dense_img_ff_window(
    blocks: Sequence[LoadedBlock],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    reclaim_quantized: bool = True,
    cache: dict[int, DenseFeedForward] | None = None,
    cache_max_blocks: int = 12,
) -> tuple[DenseImgFFHandle, ...]:
    """Replace each block's ``img_ff`` with a dense pre-dequantized module.

    When ``reclaim_quantized`` is true, the previous q6 ``img_ff`` is dropped so the
    window holds dense MLP weights instead of both copies. The whole block (including
    dense weights) is freed when the residency window releases ``blocks``.

    ``cache`` (optional) reuses already-materialized dense ``img_ff`` across windows /
    denoise steps for the same ``block_index``. This amortizes dequant: the first
    touch pays, later touches are pointer swaps. ``cache_max_blocks`` is a simple
    FIFO cap so peak stays bounded (about 150 MiB/block).
    """

    handles: list[DenseImgFFHandle] = []
    for loaded in blocks:
        dense_ff: DenseFeedForward | None = None
        nbytes = 0
        if cache is not None and loaded.block_index in cache:
            dense_ff = cache[loaded.block_index]
            nbytes = (
                _dense_nbytes(dense_ff.mlp_in) + _dense_nbytes(dense_ff.mlp_out)
            )
        else:
            dense_ff, nbytes = materialize_dense_img_ff(
                loaded.module.img_ff, dtype=dtype
            )
            if cache is not None:
                cache[loaded.block_index] = dense_ff
                while len(cache) > cache_max_blocks:
                    # FIFO eviction: dict preserves insertion order (Py3.7+).
                    cache.pop(next(iter(cache)))
        loaded.module.img_ff = dense_ff
        handles.append(
            DenseImgFFHandle(
                block_index=loaded.block_index,
                mlp_in=dense_ff.mlp_in,
                mlp_out=dense_ff.mlp_out,
                bytes_materialized=nbytes,
            )
        )
    if reclaim_quantized:
        # Do not mx.clear_cache() here: the residency window already clears on
        # release. Clearing after every prepare was paying a large sync tax.
        gc.collect()
    return tuple(handles)


def release_dense_img_ff_window(handles: Sequence[DenseImgFFHandle] | None) -> None:
    """Drop dense weight references (blocks themselves are deleted by the runtime)."""

    if not handles:
        return
    for handle in handles:
        del handle
