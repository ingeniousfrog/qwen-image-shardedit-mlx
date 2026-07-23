"""Pure planning helpers for condition-only image token merging.

The runtime keeps target tokens untouched and only shortens the static
reference-condition suffix while a middle Transformer block runs. These helpers
describe that geometry without importing MLX, so the risky parts are easy to
unit-test before the tensor path is wired in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any


@dataclass(frozen=True)
class ConditionGrid:
    frames: int
    height: int
    width: int

    @property
    def token_count(self) -> int:
        return self.frames * self.height * self.width

    def merged_width(self, *, stride: int) -> int:
        if stride < 2:
            raise ValueError("condition merge stride must be >= 2")
        return ceil(self.width / stride)

    def merged_token_count(self, *, stride: int) -> int:
        return self.frames * self.height * self.merged_width(stride=stride)


@dataclass(frozen=True)
class ConditionMergePlan:
    target_token_count: int
    image_token_count: int
    grids: tuple[ConditionGrid, ...]
    stride: int

    @property
    def condition_token_count(self) -> int:
        return condition_token_count(self.grids)

    @property
    def merged_condition_token_count(self) -> int:
        return sum(grid.merged_token_count(stride=self.stride) for grid in self.grids)

    @property
    def merged_image_token_count(self) -> int:
        return self.target_token_count + self.merged_condition_token_count

    @property
    def merged_widths(self) -> tuple[int, ...]:
        return tuple(grid.merged_width(stride=self.stride) for grid in self.grids)

    @property
    def reduction_ratio(self) -> float:
        return self.merged_image_token_count / self.image_token_count


@dataclass(frozen=True)
class TextMergePlan:
    text_token_count: int
    valid_text_token_count: int
    stride: int

    @property
    def merged_valid_text_token_count(self) -> int:
        return ceil(self.valid_text_token_count / self.stride)

    @property
    def merged_text_token_count(self) -> int:
        return self.merged_valid_text_token_count

    @property
    def reduction_ratio(self) -> float:
        return self.merged_text_token_count / self.text_token_count


def normalize_condition_grids(raw_grid: Any) -> tuple[ConditionGrid, ...]:
    """Normalize mflux's `cond_image_grid` tuple/list shape into ConditionGrid."""

    if raw_grid is None:
        return ()
    if _is_grid_sequence(raw_grid):
        return (_grid_from_sequence(raw_grid),)
    if isinstance(raw_grid, list | tuple):
        grids = tuple(_grid_from_sequence(item) for item in raw_grid)
        return grids
    raise ValueError("condition image grid must be a 3-tuple or a list of 3-tuples")


def condition_token_count(grids: Sequence[ConditionGrid]) -> int:
    return sum(grid.token_count for grid in grids)


def build_condition_merge_plan(
    *,
    target_token_count: int,
    total_image_tokens: int,
    cond_image_grid: Any,
    stride: int,
) -> ConditionMergePlan | None:
    """Return a V0 local-width merge plan, or None when merging is not applicable."""

    if stride < 2:
        raise ValueError("condition merge stride must be >= 2")
    if target_token_count <= 0 or total_image_tokens <= target_token_count:
        return None
    grids = normalize_condition_grids(cond_image_grid)
    if not grids:
        return None
    if any(grid.frames <= 0 or grid.height <= 0 or grid.width <= 1 for grid in grids):
        return None
    if condition_token_count(grids) != total_image_tokens - target_token_count:
        return None
    return ConditionMergePlan(
        target_token_count=target_token_count,
        image_token_count=total_image_tokens,
        grids=grids,
        stride=stride,
    )


def build_text_merge_plan(
    *,
    total_text_tokens: int,
    valid_text_tokens: int,
    stride: int,
) -> TextMergePlan | None:
    """Return a V0 local text-token merge plan, or None when it is not applicable."""

    if stride < 2:
        raise ValueError("text token merge stride must be >= 2")
    if total_text_tokens <= 0 or valid_text_tokens <= 1:
        return None
    if valid_text_tokens > total_text_tokens:
        return None
    plan = TextMergePlan(
        text_token_count=total_text_tokens,
        valid_text_token_count=valid_text_tokens,
        stride=stride,
    )
    if plan.merged_text_token_count >= total_text_tokens:
        return None
    return plan


def should_merge_condition_block(
    *,
    enabled: bool,
    cache_hit: bool,
    block_index: int,
    block_count: int,
    start_block: int,
    back_blocks: int,
) -> bool:
    """Return whether this zero-based block should run with condition merge."""

    if not enabled or cache_hit:
        return False
    if block_count <= 0 or block_index < 0 or block_index >= block_count:
        return False
    if start_block < 1:
        raise ValueError("condition merge start block must be one-based and >= 1")
    if back_blocks < 0:
        raise ValueError("condition merge back blocks must be >= 0")
    first_merge_index = start_block - 1
    last_merge_index = block_count - back_blocks - 1
    return first_merge_index <= block_index <= last_merge_index


def _is_grid_sequence(value: Any) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) == 3
        and all(isinstance(part, int) for part in value)
    )


def _grid_from_sequence(value: Any) -> ConditionGrid:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError("condition image grid entries must have three integers")
    frames, height, width = value
    if not all(isinstance(part, int) for part in (frames, height, width)):
        raise ValueError("condition image grid entries must be integers")
    return ConditionGrid(frames=frames, height=height, width=width)
