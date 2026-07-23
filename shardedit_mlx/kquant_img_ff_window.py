"""Window-local K-quant image MLP replacements for throughput spikes."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
import gc
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class KQuantLinearWeights:
    weight: mx.array
    scales: mx.array
    bias: mx.array | None
    codec: str
    in_features: int
    out_features: int
    bytes_materialized: int


@dataclass(frozen=True)
class KQuantImgFFHandle:
    block_index: int
    cache_hit: bool
    bytes_materialized: int


class KQuantFeedForward:
    """Drop-in img_ff replacement backed by mlx-kquant quantized matmuls."""

    def __init__(
        self,
        mlp_in: KQuantLinearWeights,
        mlp_out: KQuantLinearWeights,
        *,
        kquant: Any,
        mlp_in_lora: tuple[mx.array, mx.array, float] | None = None,
        mlp_out_lora: tuple[mx.array, mx.array, float] | None = None,
    ) -> None:
        self.mlp_in = mlp_in
        self.mlp_out = mlp_out
        self.kquant = kquant
        self.mlp_in_lora = mlp_in_lora
        self.mlp_out_lora = mlp_out_lora

    def _linear(
        self,
        hidden: mx.array,
        weights: KQuantLinearWeights,
        lora: tuple[mx.array, mx.array, float] | None,
    ) -> mx.array:
        leading = hidden.shape[:-1]
        flat = hidden.reshape((-1, hidden.shape[-1]))
        out = self.kquant.quantized_matmul(
            flat.astype(mx.float16),
            weights.weight,
            weights.scales,
            weights.codec,
            transpose=True,
        )
        if weights.bias is not None:
            out = out + weights.bias.astype(out.dtype)
        out = out.reshape(leading + (weights.out_features,))
        if lora is not None:
            lora_a, lora_b, scale = lora
            out = out + scale * ((hidden @ lora_a) @ lora_b)
        return out

    def __call__(self, hidden: mx.array) -> mx.array:
        from mlx import nn

        hidden = self._linear(hidden, self.mlp_in, self.mlp_in_lora)
        hidden = nn.gelu_approx(hidden)
        return self._linear(hidden, self.mlp_out, self.mlp_out_lora)


def import_mlx_kquant() -> Any:
    try:
        import mlx_kquant as kquant
    except ImportError as error:
        raise RuntimeError(
            "mlx-kquant is required for this spike; install it with "
            "`.venv/bin/pip install mlx-kquant`"
        ) from error
    return kquant


def _tensor_nbytes(array: mx.array | None) -> int:
    return 0 if array is None else int(array.nbytes)


def _quantize_dense_linear(
    layer: Any,
    *,
    codec: str,
    kquant: Any,
) -> KQuantLinearWeights:
    from shardedit_mlx.dense_img_ff_window import materialize_dense_linear

    dense = materialize_dense_linear(layer, dtype=mx.float16)
    mx.eval(dense.weight)
    quantized, scales = kquant.quantize(dense.weight, codec)
    bias = None if dense.bias is None else dense.bias.astype(mx.float32)
    leaves = [quantized, scales]
    if bias is not None:
        leaves.append(bias)
    mx.eval(*leaves)
    return KQuantLinearWeights(
        weight=quantized,
        scales=scales,
        bias=bias,
        codec=codec,
        in_features=int(dense.weight.shape[1]),
        out_features=int(dense.weight.shape[0]),
        bytes_materialized=(
            _tensor_nbytes(quantized) + _tensor_nbytes(scales) + _tensor_nbytes(bias)
        ),
    )


def materialize_kquant_img_ff(
    img_ff: Any,
    *,
    codec: str = "q6_k",
    kquant: Any | None = None,
) -> tuple[KQuantFeedForward, int]:
    """Re-encode one QwenFeedForward's mlp_in/mlp_out into K-quant weights."""

    from shardedit_mlx.dense_img_ff_window import _extract_lora_adapters, _lora_delta_weight

    kquant = import_mlx_kquant() if kquant is None else kquant
    mlp_in = _quantize_dense_linear(img_ff.mlp_in, codec=codec, kquant=kquant)
    mlp_out = _quantize_dense_linear(img_ff.mlp_out, codec=codec, kquant=kquant)
    mlp_in_lora = _extract_lora_adapters(img_ff.mlp_in, dtype=mx.bfloat16)
    mlp_out_lora = _extract_lora_adapters(img_ff.mlp_out, dtype=mx.bfloat16)
    if _lora_delta_weight(img_ff.mlp_in, dtype=mx.float16) is not None:
        mlp_in_lora = None
    if _lora_delta_weight(img_ff.mlp_out, dtype=mx.float16) is not None:
        mlp_out_lora = None

    lora_bytes = 0
    if mlp_in_lora is not None:
        lora_bytes += _tensor_nbytes(mlp_in_lora[0]) + _tensor_nbytes(mlp_in_lora[1])
    if mlp_out_lora is not None:
        lora_bytes += _tensor_nbytes(mlp_out_lora[0]) + _tensor_nbytes(mlp_out_lora[1])
    nbytes = mlp_in.bytes_materialized + mlp_out.bytes_materialized + lora_bytes
    return (
        KQuantFeedForward(
            mlp_in,
            mlp_out,
            kquant=kquant,
            mlp_in_lora=mlp_in_lora,
            mlp_out_lora=mlp_out_lora,
        ),
        nbytes,
    )


def prepare_kquant_img_ff_window(
    blocks: Sequence[Any],
    *,
    codec: str = "q6_k",
    kquant: Any | None = None,
    reclaim_quantized: bool = True,
    cache: MutableMapping[tuple[int, str], KQuantFeedForward] | None = None,
    cache_max_blocks: int = 60,
) -> tuple[KQuantImgFFHandle, ...]:
    """Replace each block's img_ff with a K-quant module, optionally cached."""

    kquant = import_mlx_kquant() if kquant is None else kquant
    handles: list[KQuantImgFFHandle] = []
    for loaded in blocks:
        key = (int(loaded.block_index), codec)
        cache_hit = cache is not None and key in cache
        if cache_hit:
            kquant_ff = cache[key]
            nbytes = (
                kquant_ff.mlp_in.bytes_materialized
                + kquant_ff.mlp_out.bytes_materialized
            )
        else:
            kquant_ff, nbytes = materialize_kquant_img_ff(
                loaded.module.img_ff,
                codec=codec,
                kquant=kquant,
            )
            if cache is not None:
                cache[key] = kquant_ff
                while len(cache) > cache_max_blocks:
                    cache.pop(next(iter(cache)))
        loaded.module.img_ff = kquant_ff
        handles.append(
            KQuantImgFFHandle(
                block_index=int(loaded.block_index),
                cache_hit=cache_hit,
                bytes_materialized=nbytes,
            )
        )
    if reclaim_quantized:
        gc.collect()
    return tuple(handles)


def kquant_cache_bytes(
    cache: MutableMapping[tuple[int, str], KQuantFeedForward],
) -> int:
    return sum(
        ff.mlp_in.bytes_materialized + ff.mlp_out.bytes_materialized
        for ff in cache.values()
    )
