from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from shardedit_mlx.residency_plan import fixed_block_windows, shard_block_windows
from shardedit_mlx.sweep_profile import BlockShardPlan


def make_plan(block_index: int, *shards: str) -> BlockShardPlan:
    return BlockShardPlan(
        block_index=block_index,
        tensor_count=60,
        shards=tuple(shards),
    )


def test_fixed_windows_cover_blocks_once_in_order() -> None:
    plans = tuple(make_plan(index, f"{index // 3}.safetensors") for index in range(7))

    windows = fixed_block_windows(plans, window_size=3)

    assert tuple(window.block_indices for window in windows) == (
        (0, 1, 2),
        (3, 4, 5),
        (6,),
    )
    assert windows[0].shards == ("0.safetensors",)
    assert windows[1].shards == ("1.safetensors",)
    assert windows[2].shards == ("2.safetensors",)


def test_shard_windows_assign_split_block_to_last_required_shard() -> None:
    plans = (
        make_plan(0, "0.safetensors"),
        make_plan(1, "0.safetensors", "1.safetensors"),
        make_plan(2, "1.safetensors"),
        make_plan(3, "1.safetensors", "2.safetensors"),
        make_plan(4, "2.safetensors"),
    )

    windows = shard_block_windows(
        plans,
        ordered_shards=("0.safetensors", "1.safetensors", "2.safetensors"),
    )

    assert tuple(window.block_indices for window in windows) == (
        (0,),
        (1, 2),
        (3, 4),
    )
    assert windows[1].shards == ("0.safetensors", "1.safetensors")
    assert windows[2].shards == ("1.safetensors", "2.safetensors")


def test_residency_window_is_immutable() -> None:
    window = fixed_block_windows((make_plan(0, "0.safetensors"),), window_size=1)[0]

    with pytest.raises(FrozenInstanceError):
        window.block_indices = ()  # type: ignore[misc]


@pytest.mark.parametrize("window_size", (0, -1))
def test_fixed_windows_reject_non_positive_size(window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size must be positive"):
        fixed_block_windows((make_plan(0, "0.safetensors"),), window_size)


def test_window_planners_reject_missing_or_reordered_blocks() -> None:
    plans = (make_plan(0, "0.safetensors"), make_plan(2, "1.safetensors"))

    with pytest.raises(ValueError, match="contiguous and ordered"):
        fixed_block_windows(plans, window_size=1)


def test_shard_windows_reject_unknown_shards() -> None:
    plans = (make_plan(0, "missing.safetensors"),)

    with pytest.raises(ValueError, match="not present in ordered_shards"):
        shard_block_windows(plans, ordered_shards=("0.safetensors",))
