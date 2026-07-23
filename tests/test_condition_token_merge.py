from __future__ import annotations

import pytest

from shardedit_mlx.condition_token_merge import (
    ConditionGrid,
    TextMergePlan,
    build_condition_merge_plan,
    build_text_merge_plan,
    condition_token_count,
    normalize_condition_grids,
    should_merge_condition_block,
)


def test_normalize_condition_grids_accepts_single_and_multi_image() -> None:
    assert normalize_condition_grids((1, 28, 20)) == (ConditionGrid(1, 28, 20),)
    assert normalize_condition_grids([1, 28, 20]) == (ConditionGrid(1, 28, 20),)
    assert normalize_condition_grids([(1, 28, 20), (1, 32, 24)]) == (
        ConditionGrid(1, 28, 20),
        ConditionGrid(1, 32, 24),
    )


def test_build_condition_merge_plan_keeps_target_and_halves_even_condition_width() -> None:
    plan = build_condition_merge_plan(
        target_token_count=6,
        total_image_tokens=14,
        cond_image_grid=(1, 2, 4),
        stride=2,
    )

    assert plan is not None
    assert plan.target_token_count == 6
    assert plan.condition_token_count == 8
    assert plan.merged_condition_token_count == 4
    assert plan.merged_image_token_count == 10
    assert plan.reduction_ratio == pytest.approx(10 / 14)


def test_build_condition_merge_plan_keeps_tail_columns_inside_each_row() -> None:
    plan = build_condition_merge_plan(
        target_token_count=6,
        total_image_tokens=16,
        cond_image_grid=(1, 2, 5),
        stride=2,
    )

    assert plan is not None
    assert plan.condition_token_count == 10
    assert plan.merged_condition_token_count == 6
    assert plan.merged_widths == (3,)


def test_build_condition_merge_plan_never_crosses_reference_images() -> None:
    plan = build_condition_merge_plan(
        target_token_count=4,
        total_image_tokens=16,
        cond_image_grid=[(1, 2, 4), (1, 1, 4)],
        stride=2,
    )

    assert plan is not None
    assert plan.condition_token_count == 12
    assert plan.merged_condition_token_count == 6
    assert plan.merged_widths == (2, 2)
    assert condition_token_count(plan.grids) == 12


def test_build_condition_merge_plan_rejects_mismatched_or_unmergeable_inputs() -> None:
    assert (
        build_condition_merge_plan(
            target_token_count=6,
            total_image_tokens=15,
            cond_image_grid=(1, 2, 4),
            stride=2,
        )
        is None
    )
    assert (
        build_condition_merge_plan(
            target_token_count=6,
            total_image_tokens=14,
            cond_image_grid=(1, 2, 1),
            stride=2,
        )
        is None
    )


def test_build_text_merge_plan_halves_valid_tokens_and_drops_padding() -> None:
    plan = build_text_merge_plan(
        total_text_tokens=10,
        valid_text_tokens=8,
        stride=2,
    )

    assert plan == TextMergePlan(
        text_token_count=10,
        valid_text_token_count=8,
        stride=2,
    )
    assert plan.merged_valid_text_token_count == 4
    assert plan.merged_text_token_count == 4
    assert plan.reduction_ratio == pytest.approx(0.4)


def test_build_text_merge_plan_keeps_odd_tail_token() -> None:
    plan = build_text_merge_plan(
        total_text_tokens=9,
        valid_text_tokens=7,
        stride=2,
    )

    assert plan is not None
    assert plan.merged_valid_text_token_count == 4
    assert plan.merged_text_token_count == 4


def test_build_text_merge_plan_rejects_unmergeable_inputs() -> None:
    assert build_text_merge_plan(total_text_tokens=8, valid_text_tokens=1, stride=2) is None
    assert build_text_merge_plan(total_text_tokens=8, valid_text_tokens=9, stride=2) is None
    with pytest.raises(ValueError):
        build_text_merge_plan(total_text_tokens=8, valid_text_tokens=8, stride=1)


def test_should_merge_condition_block_targets_full_middle_blocks_only() -> None:
    assert not should_merge_condition_block(
        enabled=True,
        cache_hit=False,
        block_index=0,
        block_count=60,
        start_block=2,
        back_blocks=2,
    )
    assert should_merge_condition_block(
        enabled=True,
        cache_hit=False,
        block_index=1,
        block_count=60,
        start_block=2,
        back_blocks=2,
    )
    assert should_merge_condition_block(
        enabled=True,
        cache_hit=False,
        block_index=57,
        block_count=60,
        start_block=2,
        back_blocks=2,
    )
    assert not should_merge_condition_block(
        enabled=True,
        cache_hit=False,
        block_index=58,
        block_count=60,
        start_block=2,
        back_blocks=2,
    )
    assert not should_merge_condition_block(
        enabled=True,
        cache_hit=True,
        block_index=10,
        block_count=60,
        start_block=2,
        back_blocks=2,
    )
