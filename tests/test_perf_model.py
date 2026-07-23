from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from shardedit_mlx.perf_model import (
    EditTokenPlan,
    RuntimeMemoryPlan,
    classifier_free_guidance_passes,
)


def test_768_edit_with_1024_condition_has_6400_image_tokens() -> None:
    plan = EditTokenPlan.from_dimensions(
        target_width=768,
        target_height=768,
        condition_width=1024,
        condition_height=1024,
    )

    assert plan == EditTokenPlan(target_tokens=2304, condition_tokens=4096)
    assert plan.image_tokens == 6400


def test_384_condition_reduces_attention_work_to_about_one_fifth() -> None:
    baseline = EditTokenPlan.from_dimensions(
        target_width=768,
        target_height=768,
        condition_width=1024,
        condition_height=1024,
    )
    reduced = EditTokenPlan.from_dimensions(
        target_width=768,
        target_height=768,
        condition_width=384,
        condition_height=384,
    )

    assert reduced.image_tokens == 2880
    assert reduced.attention_cost / baseline.attention_cost == pytest.approx(0.2025)


def test_multiple_condition_images_append_their_tokens() -> None:
    plan = EditTokenPlan.from_dimensions(
        target_width=768,
        target_height=768,
        condition_width=384,
        condition_height=384,
        condition_count=4,
    )

    assert plan.target_tokens == 2304
    assert plan.condition_tokens == 2304
    assert plan.image_tokens == 4608


def test_token_plan_rejects_zero_condition_images() -> None:
    with pytest.raises(ValueError, match="condition count must be positive"):
        EditTokenPlan.from_dimensions(
            target_width=768,
            target_height=768,
            condition_width=384,
            condition_height=384,
            condition_count=0,
        )


@pytest.mark.parametrize("width,height", [(0, 768), (768, -16), (767, 768)])
def test_token_plan_rejects_invalid_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="positive multiples of 16"):
        EditTokenPlan.from_dimensions(
            target_width=width,
            target_height=height,
            condition_width=1024,
            condition_height=1024,
        )


def test_guidance_one_needs_only_the_positive_transformer_pass() -> None:
    assert classifier_free_guidance_passes(1.0) == 1


def test_non_unit_guidance_needs_positive_and_negative_passes() -> None:
    assert classifier_free_guidance_passes(1.01) == 2


def test_24_gib_plan_keeps_q6_transformer() -> None:
    plan = RuntimeMemoryPlan(
        physical_memory_gib=24,
        system_reserve_gib=3,
        activation_reserve_gib=2.5,
        transformer_size_gib=15.5,
        transformer_bits=6,
    )

    assert plan.required_memory_gib == pytest.approx(21.0)
    assert plan.fits is True
    assert plan.recommended_bits() == 6


def test_16_gib_plan_requires_q4_runtime_cache() -> None:
    plan = RuntimeMemoryPlan(
        physical_memory_gib=16,
        system_reserve_gib=3,
        activation_reserve_gib=2.5,
        transformer_size_gib=15.5,
        transformer_bits=6,
    )

    assert plan.fits is False
    assert plan.recommended_bits() == 4


def test_memory_plan_is_immutable() -> None:
    plan = RuntimeMemoryPlan(24, 3, 2.5, 15.5, 6)

    with pytest.raises(FrozenInstanceError):
        plan.transformer_bits = 4  # type: ignore[misc]
