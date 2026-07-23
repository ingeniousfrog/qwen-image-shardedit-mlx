"""Pure planning helpers for controlled Transformer weight residency."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from shardedit_mlx.sweep_profile import BlockShardPlan


@dataclass(frozen=True)
class ResidencyWindow:
    index: int
    block_indices: tuple[int, ...]
    shards: tuple[str, ...]


def _validate_plans(plans: tuple[BlockShardPlan, ...]) -> None:
    if not plans:
        raise ValueError("plans cannot be empty")
    indices = tuple(plan.block_index for plan in plans)
    if indices != tuple(range(len(plans))):
        raise ValueError("Transformer block plans must be contiguous and ordered")


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _make_window(
    index: int,
    plans: tuple[BlockShardPlan, ...],
) -> ResidencyWindow:
    return ResidencyWindow(
        index=index,
        block_indices=tuple(plan.block_index for plan in plans),
        shards=_ordered_unique(shard for plan in plans for shard in plan.shards),
    )


def fixed_block_windows(
    plans: tuple[BlockShardPlan, ...],
    window_size: int,
) -> tuple[ResidencyWindow, ...]:
    """Split ordered block plans into fixed-size residency windows."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    _validate_plans(plans)
    return tuple(
        _make_window(window_index, plans[start : start + window_size])
        for window_index, start in enumerate(range(0, len(plans), window_size))
    )


def shard_block_windows(
    plans: tuple[BlockShardPlan, ...],
    ordered_shards: tuple[str, ...],
) -> tuple[ResidencyWindow, ...]:
    """Group blocks by the last safetensor shard required to instantiate them."""

    _validate_plans(plans)
    if not ordered_shards or len(set(ordered_shards)) != len(ordered_shards):
        raise ValueError("ordered_shards must be non-empty and unique")
    shard_positions = {shard: index for index, shard in enumerate(ordered_shards)}
    unknown = tuple(
        sorted(
            {
                shard
                for plan in plans
                for shard in plan.shards
                if shard not in shard_positions
            }
        )
    )
    if unknown:
        raise ValueError(f"shards not present in ordered_shards: {unknown}")

    assigned_positions = tuple(
        max(shard_positions[shard] for shard in plan.shards) for plan in plans
    )
    if assigned_positions != tuple(sorted(assigned_positions)):
        raise ValueError("block shard assignments must be monotonic")

    return tuple(
        _make_window(
            window_index,
            tuple(
                plan
                for plan, assigned_position in zip(
                    plans, assigned_positions, strict=True
                )
                if assigned_position == shard_position
            ),
        )
        for window_index, shard_position in enumerate(
            position
            for position in range(len(ordered_shards))
            if position in assigned_positions
        )
    )
