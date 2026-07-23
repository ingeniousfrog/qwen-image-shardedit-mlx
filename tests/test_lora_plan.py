from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from shardedit_mlx.lora_plan import plan_qwen_lora_keys, select_lora_keys


def test_qwen_lora_plan_groups_supported_namespaces_by_block() -> None:
    plan = plan_qwen_lora_keys(
        (
            "transformer_blocks.0.attn.to_q.lora_up.weight",
            "transformer.transformer_blocks.1.attn.to_q.lora.down.weight",
            "diffusion_model.transformer_blocks.1.attn.to_q.lora_B.weight",
            "lora_unet_transformer_blocks_2_attn_to_q.alpha",
        ),
        block_count=3,
    )

    assert plan.key_count == 4
    assert plan.keys_by_block[0] == (
        "transformer_blocks.0.attn.to_q.lora_up.weight",
    )
    assert len(plan.keys_by_block[1]) == 2
    assert plan.keys_by_block[2] == (
        "lora_unet_transformer_blocks_2_attn_to_q.alpha",
    )


def test_select_lora_keys_preserves_block_then_key_order() -> None:
    plan = plan_qwen_lora_keys(
        (
            "transformer_blocks.1.attn.to_v.lora_up.weight",
            "transformer_blocks.0.attn.to_q.lora_up.weight",
            "transformer_blocks.1.attn.to_q.lora_up.weight",
        ),
        block_count=2,
    )

    selected = select_lora_keys(plan, (0, 1))

    assert selected == (
        "transformer_blocks.0.attn.to_q.lora_up.weight",
        "transformer_blocks.1.attn.to_q.lora_up.weight",
        "transformer_blocks.1.attn.to_v.lora_up.weight",
    )


def test_qwen_lora_plan_is_immutable() -> None:
    plan = plan_qwen_lora_keys(
        ("transformer_blocks.0.attn.to_q.alpha",),
        block_count=1,
    )

    with pytest.raises(FrozenInstanceError):
        plan.key_count = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("keys", "block_count", "message"),
    [
        ((), 1, "cannot be empty"),
        (("transformer_blocks.0.attn.to_q.alpha",), 0, "must be positive"),
        (("unrecognized.key",), 1, "without a Transformer block index"),
        (("transformer_blocks.2.attn.to_q.alpha",), 2, "outside the model"),
        (
            (
                "transformer_blocks.0.attn.to_q.alpha",
                "transformer_blocks.0.attn.to_q.alpha",
            ),
            1,
            "must be unique",
        ),
    ],
)
def test_qwen_lora_plan_rejects_invalid_keys(
    keys: tuple[str, ...], block_count: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_qwen_lora_keys(keys, block_count=block_count)


def test_select_lora_keys_rejects_unknown_blocks() -> None:
    plan = plan_qwen_lora_keys(
        ("transformer_blocks.0.attn.to_q.alpha",),
        block_count=1,
    )

    with pytest.raises(ValueError, match="outside the LoRA plan"):
        select_lora_keys(plan, (1,))
