"""Selective q6 Qwen Transformer block loading for residency experiments."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten

from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import (
    QwenTransformerBlock,
)

from shardedit_mlx.sweep_profile import BlockShardPlan, plan_transformer_block_shards


@dataclass(frozen=True)
class TransformerLayout:
    transformer_dir: Path
    plans: tuple[BlockShardPlan, ...]
    ordered_shards: tuple[str, ...]


@dataclass(frozen=True)
class LoadedBlock:
    block_index: int
    module: QwenTransformerBlock


def load_transformer_layout(model_dir: Path) -> TransformerLayout:
    transformer_dir = model_dir / "transformer"
    index_path = transformer_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing Transformer index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise ValueError(f"invalid weight_map in {index_path}")

    plans = plan_transformer_block_shards(weight_map)
    ordered_shards = tuple(
        dict.fromkeys(shard for plan in plans for shard in plan.shards)
    )
    missing_shards = tuple(
        shard
        for shard in ordered_shards
        if not (transformer_dir / shard).is_file()
    )
    if missing_shards:
        raise FileNotFoundError(f"missing Transformer shards: {missing_shards}")
    return TransformerLayout(
        transformer_dir=transformer_dir,
        plans=plans,
        ordered_shards=ordered_shards,
    )


def _instantiate_block(
    flat_weights: tuple[tuple[str, mx.array], ...],
) -> QwenTransformerBlock:
    block = QwenTransformerBlock()
    nn.quantize(block, group_size=64, bits=6, mode="affine")
    block.update(tree_unflatten(flat_weights), strict=True)
    return block


def _load_shard_parts(
    layout: TransformerLayout,
    shard_name: str,
    plans: tuple[BlockShardPlan, ...],
) -> tuple[tuple[int, tuple[tuple[str, mx.array], ...]], ...]:
    shard_weights: dict[str, mx.array] = mx.load(
        str(layout.transformer_dir / shard_name)
    )
    parts = tuple(
        (
            plan.block_index,
            tuple(
                (name.removeprefix(prefix), value)
                for name, value in shard_weights.items()
                if name.startswith(prefix)
            ),
        )
        for plan in plans
        for prefix in (f"transformer_blocks.{plan.block_index}.",)
        if shard_name in plan.shards
    )
    del shard_weights
    gc.collect()
    mx.clear_cache()
    return parts


def load_block_window(
    layout: TransformerLayout,
    block_indices: tuple[int, ...],
) -> tuple[LoadedBlock, ...]:
    """Instantiate only the selected consecutive q6 Transformer blocks."""

    if not block_indices:
        raise ValueError("block_indices cannot be empty")
    if block_indices != tuple(range(block_indices[0], block_indices[-1] + 1)):
        raise ValueError("block_indices must be unique, consecutive, and ordered")

    selected_plans = tuple(
        plan for plan in layout.plans if plan.block_index in block_indices
    )
    if tuple(plan.block_index for plan in selected_plans) != block_indices:
        raise ValueError(f"unknown Transformer block indices: {block_indices}")
    required_shards = tuple(
        shard
        for shard in layout.ordered_shards
        if any(shard in plan.shards for plan in selected_plans)
    )
    shard_parts = tuple(
        _load_shard_parts(layout, shard, selected_plans) for shard in required_shards
    )

    loaded = tuple(
        LoadedBlock(
            block_index=plan.block_index,
            module=_instantiate_block(flat_weights),
        )
        for plan in selected_plans
        if (
            flat_weights := tuple(
                weight
                for parts in shard_parts
                for part_block_index, part_weights in parts
                if part_block_index == plan.block_index
                for weight in part_weights
            )
        )
        and len(flat_weights) == plan.tensor_count
    )
    if tuple(block.block_index for block in loaded) != block_indices:
        loaded_counts = {
            plan.block_index: sum(
                len(part_weights)
                for parts in shard_parts
                for part_block_index, part_weights in parts
                if part_block_index == plan.block_index
            )
            for plan in selected_plans
        }
        raise RuntimeError(
            "failed to load complete Transformer blocks: "
            f"expected={block_indices}, loaded_counts={loaded_counts}"
        )
    return loaded
