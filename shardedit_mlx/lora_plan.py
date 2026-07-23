"""Pure key indexing for windowed Qwen Transformer LoRA loading."""

from __future__ import annotations

from dataclasses import dataclass
import re


_BLOCK_PATTERNS = (
    re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\."),
    re.compile(r"(?:^|_)transformer_blocks_(\d+)_"),
)


@dataclass(frozen=True)
class LoRAKeyPlan:
    block_count: int
    keys_by_block: tuple[tuple[str, ...], ...]
    key_count: int


def _block_index(key: str) -> int | None:
    for pattern in _BLOCK_PATTERNS:
        if match := pattern.search(key):
            return int(match.group(1))
    return None


def plan_qwen_lora_keys(
    keys: tuple[str, ...],
    *,
    block_count: int = 60,
) -> LoRAKeyPlan:
    """Index supported Qwen LoRA keys without loading their tensor values."""

    if block_count <= 0:
        raise ValueError("block_count must be positive")
    if not keys:
        raise ValueError("LoRA keys cannot be empty")
    if len(set(keys)) != len(keys):
        raise ValueError("LoRA keys must be unique")

    indexed = tuple((key, _block_index(key)) for key in keys)
    unrecognized = tuple(key for key, block_index in indexed if block_index is None)
    if unrecognized:
        raise ValueError(
            "LoRA keys without a Transformer block index: "
            f"{unrecognized[:5]}"
        )
    outside = tuple(
        key
        for key, block_index in indexed
        if block_index is not None and not 0 <= block_index < block_count
    )
    if outside:
        raise ValueError(f"LoRA keys outside the model block range: {outside[:5]}")

    keys_by_block = tuple(
        tuple(
            sorted(
                key
                for key, key_block_index in indexed
                if key_block_index == block_index
            )
        )
        for block_index in range(block_count)
    )
    return LoRAKeyPlan(
        block_count=block_count,
        keys_by_block=keys_by_block,
        key_count=len(keys),
    )


def select_lora_keys(
    plan: LoRAKeyPlan,
    block_indices: tuple[int, ...],
) -> tuple[str, ...]:
    """Return the deterministic LoRA key list for selected blocks."""

    if any(index < 0 or index >= plan.block_count for index in block_indices):
        raise ValueError("block index is outside the LoRA plan")
    if len(set(block_indices)) != len(block_indices):
        raise ValueError("block indices must be unique")
    return tuple(
        key for block_index in block_indices for key in plan.keys_by_block[block_index]
    )
