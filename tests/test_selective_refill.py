from __future__ import annotations

import pytest

from shardedit_mlx.selective_refill import (
    build_image_gather_indices,
    dampened_token_value,
    scaled_residual_token_value,
    select_unique_even_indices,
    should_apply_selective_refill,
    uniqueness_scaled_dampens,
    uniqueness_scaled_residual_scales,
)


def test_select_unique_even_indices_picks_lowest_similarity() -> None:
    sims = (0.9, 0.2, 0.8, 0.1)
    assert select_unique_even_indices(sims, fraction=0.5) == (2, 6)


def test_select_unique_even_indices_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        select_unique_even_indices((0.5, 0.5), fraction=0.0)


def test_build_image_gather_indices_appends_condition() -> None:
    assert build_image_gather_indices((0, 4), target_token_count=6, total_image_tokens=10) == (
        0,
        4,
        6,
        7,
        8,
        9,
    )


def test_build_image_gather_indices_rejects_oob_target() -> None:
    with pytest.raises(ValueError, match="out of range"):
        build_image_gather_indices((6,), target_token_count=6, total_image_tokens=10)


def test_dampened_token_value_endpoints() -> None:
    assert dampened_token_value(f1_value=1.0, bridged_value=5.0, dampen=0.0) == 5.0
    assert dampened_token_value(f1_value=1.0, bridged_value=5.0, dampen=1.0) == 1.0
    assert dampened_token_value(f1_value=1.0, bridged_value=5.0, dampen=0.5) == pytest.approx(3.0)


def test_scaled_residual_token_value_boost_and_identity() -> None:
    assert scaled_residual_token_value(f1_value=1.0, bridged_value=5.0, scale=1.0) == 5.0
    assert scaled_residual_token_value(f1_value=1.0, bridged_value=5.0, scale=0.0) == 1.0
    assert scaled_residual_token_value(f1_value=1.0, bridged_value=5.0, scale=1.5) == pytest.approx(
        7.0
    )


def test_should_apply_selective_refill_respects_min_step() -> None:
    assert (
        should_apply_selective_refill(
            fraction=0.15, current_step=3, min_step=7, cache_hit=True
        )
        is False
    )
    assert (
        should_apply_selective_refill(
            fraction=0.15, current_step=7, min_step=7, cache_hit=True
        )
        is True
    )


def test_uniqueness_scaled_dampens_proportional_to_uniqueness() -> None:
    sims = (0.9, 0.2, 0.8, 0.1)
    indices, dampens = uniqueness_scaled_dampens(sims, fraction=0.5, dampen_max=0.5)
    assert indices == (2, 6)
    assert dampens[0] == pytest.approx(0.5 * (0.8 / 0.9))
    assert dampens[1] == pytest.approx(0.5)


def test_uniqueness_boost_scales_amplify_residual() -> None:
    sims = (0.9, 0.2, 0.8, 0.1)
    indices, scales = uniqueness_scaled_residual_scales(
        sims, fraction=0.5, amount=0.5, direction="boost"
    )
    assert indices == (2, 6)
    assert scales[0] == pytest.approx(1.0 + 0.5 * (0.8 / 0.9))
    assert scales[1] == pytest.approx(1.5)
