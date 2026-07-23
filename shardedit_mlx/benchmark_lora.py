"""LoRA helpers for local benchmark scripts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def normalize_lora_args(
    lora_paths: Sequence[str | Path],
    lora_scales: Sequence[float],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    paths = tuple(str(Path(path).expanduser()) for path in lora_paths)
    scales = tuple(float(scale) for scale in lora_scales)
    if not paths:
        if scales:
            raise ValueError("LoRA scales were provided without LoRA paths")
        return (), ()
    if not scales:
        return paths, tuple(1.0 for _ in paths)
    if len(paths) != len(scales):
        raise ValueError("LoRA paths and scales must have the same length")
    return paths, scales


def apply_loras_to_loaded_blocks(
    blocks: Sequence[Any],
    *,
    lora_paths: Sequence[str | Path],
    lora_scales: Sequence[float],
    block_count: int,
) -> tuple[Any, ...]:
    paths, scales = normalize_lora_args(lora_paths, lora_scales)
    if not paths:
        return ()

    from mflux.models.common.lora.mapping.lora_loader import LoRALoader
    from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping

    from shardedit_mlx.shard_runtime import apply_window_loras, prepare_lora_sources

    sources = prepare_lora_sources(paths, scales, block_count=block_count)
    mappings = LoRALoader._build_pattern_mappings(QwenLoRAMapping.get_mapping())
    return apply_window_loras(
        tuple(blocks),
        sources,
        block_count=block_count,
        mappings=mappings,
    )
