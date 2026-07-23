from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from shardedit_mlx.sweep_profile import plan_transformer_block_shards


def test_block_shard_plan_counts_tensors_and_shards() -> None:
    plans = plan_transformer_block_shards(
        {
            "transformer_blocks.0.attn.weight": "part-1.safetensors",
            "transformer_blocks.0.mlp.weight": "part-1.safetensors",
            "transformer_blocks.1.attn.weight": "part-1.safetensors",
            "transformer_blocks.1.mlp.weight": "part-2.safetensors",
            "proj_out.weight": "part-2.safetensors",
        },
        block_count=2,
    )

    assert plans[0].tensor_count == 2
    assert plans[0].shards == ("part-1.safetensors",)
    assert plans[1].shards == ("part-1.safetensors", "part-2.safetensors")


def test_block_shard_plan_is_immutable() -> None:
    plan = plan_transformer_block_shards(
        {"transformer_blocks.0.weight": "part.safetensors"}, block_count=1
    )[0]

    with pytest.raises(FrozenInstanceError):
        plan.tensor_count = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("weight_map", "block_count"),
    [
        ({}, 1),
        ({"transformer_blocks.0.weight": "part.safetensors"}, 0),
        ({"transformer_blocks.0.weight": "part.safetensors"}, 2),
    ],
)
def test_block_shard_plan_rejects_incomplete_layouts(
    weight_map: dict[str, str], block_count: int
) -> None:
    with pytest.raises(ValueError):
        plan_transformer_block_shards(weight_map, block_count=block_count)
