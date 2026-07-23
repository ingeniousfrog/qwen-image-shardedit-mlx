"""Window-local dense image attention output projection probe."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import gc
from typing import Any

import mlx.core as mx

from shardedit_mlx.dense_img_ff_window import (
    _dense_nbytes,
    _extract_lora_adapters,
    _lora_delta_weight,
    materialize_dense_linear,
)
from shardedit_mlx.q6_metal_mlp import DenseLinearWeights
from shardedit_mlx.qwen_block_loader import LoadedBlock


@dataclass(frozen=True)
class DenseImgAttnOutHandle:
    block_index: int
    bytes_materialized: int


class DenseLinearProjection:
    """Drop-in dense replacement for a Linear/QuantizedLinear projection."""

    def __init__(
        self,
        weights: DenseLinearWeights,
        *,
        lora: tuple[mx.array, mx.array, float] | None = None,
    ) -> None:
        self.weights = weights
        self.lora = lora

    def __call__(self, hidden: mx.array) -> mx.array:
        out = hidden @ self.weights.weight.T
        if self.weights.bias is not None:
            out = out + self.weights.bias
        if self.lora is not None:
            lora_a, lora_b, scale = self.lora
            out = out + scale * ((hidden @ lora_a) @ lora_b)
        return out


def materialize_dense_projection(
    layer: Any,
    *,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[DenseLinearProjection, int]:
    weights = materialize_dense_linear(layer, dtype=dtype)
    lora = _extract_lora_adapters(layer, dtype=dtype)
    if _lora_delta_weight(layer, dtype=dtype) is not None:
        lora = None

    leaves = [weights.weight]
    if weights.bias is not None:
        leaves.append(weights.bias)
    if lora is not None:
        leaves.extend([lora[0], lora[1]])
    mx.eval(*leaves)
    return DenseLinearProjection(weights, lora=lora), _dense_nbytes(weights)


def prepare_dense_img_attn_out_window(
    blocks: Sequence[LoadedBlock],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    reclaim_quantized: bool = True,
    cache: dict[int, DenseLinearProjection] | None = None,
    cache_max_blocks: int = 12,
) -> tuple[DenseImgAttnOutHandle, ...]:
    """Replace each block's image attention output projection with dense bf16."""

    handles: list[DenseImgAttnOutHandle] = []
    for loaded in blocks:
        dense_projection: DenseLinearProjection | None = None
        nbytes = 0
        if cache is not None and loaded.block_index in cache:
            dense_projection = cache[loaded.block_index]
            nbytes = _dense_nbytes(dense_projection.weights)
        else:
            dense_projection, nbytes = materialize_dense_projection(
                loaded.module.attn.attn_to_out[0],
                dtype=dtype,
            )
            if cache is not None:
                cache[loaded.block_index] = dense_projection
                while len(cache) > cache_max_blocks:
                    cache.pop(next(iter(cache)))
        loaded.module.attn.attn_to_out[0] = dense_projection
        handles.append(
            DenseImgAttnOutHandle(
                block_index=loaded.block_index,
                bytes_materialized=nbytes,
            )
        )
    if reclaim_quantized:
        gc.collect()
    return tuple(handles)


def release_dense_img_attn_out_window(
    handles: Sequence[DenseImgAttnOutHandle] | None,
) -> None:
    if not handles:
        return
    for handle in handles:
        del handle
