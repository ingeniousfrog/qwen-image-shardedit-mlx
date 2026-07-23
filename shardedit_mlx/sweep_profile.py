"""Pure model-layout helpers for long-running Transformer sweep benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockShardPlan:
    block_index: int
    tensor_count: int
    shards: tuple[str, ...]


def plan_transformer_block_shards(
    weight_map: Mapping[str, str],
    *,
    block_count: int = 60,
) -> tuple[BlockShardPlan, ...]:
    """Describe which safetensor shards own every Transformer block."""

    if block_count <= 0:
        raise ValueError("block_count must be positive")
    if not weight_map:
        raise ValueError("weight_map cannot be empty")

    plans = tuple(
        BlockShardPlan(
            block_index=block_index,
            tensor_count=len(block_items),
            shards=tuple(sorted({shard for _, shard in block_items})),
        )
        for block_index in range(block_count)
        if (
            block_items := tuple(
                (name, shard)
                for name, shard in weight_map.items()
                if name.startswith(f"transformer_blocks.{block_index}.")
            )
        )
    )
    present = {plan.block_index for plan in plans}
    missing = tuple(index for index in range(block_count) if index not in present)
    if missing:
        raise ValueError(f"weight_map is missing Transformer blocks: {missing}")
    return plans
