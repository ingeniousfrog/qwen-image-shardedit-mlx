from __future__ import annotations

import pytest

from shardedit_mlx.cache_model import CacheSchedule


def test_no_cache_matches_eight_full_steps() -> None:
    schedule = CacheSchedule(total_steps=8, full_steps=8, cached_steps=0)

    assert schedule.equivalent_block_evaluations == 480
    assert schedule.relative_compute == 1.0
    assert schedule.projected_seconds(51.33) == pytest.approx(410.64)


def test_ideal_middle_step_cache_still_exceeds_one_minute() -> None:
    schedule = CacheSchedule(total_steps=8, full_steps=2, cached_steps=6)

    assert schedule.equivalent_block_evaluations == 120
    assert schedule.maximum_speedup == 4.0
    assert schedule.projected_seconds(51.33) == pytest.approx(102.66)


def test_qwen_lightning_style_16_plus_16_probe_has_small_eight_step_gain() -> None:
    schedule = CacheSchedule(
        total_steps=8,
        full_steps=5,
        cached_steps=3,
        probe_blocks_per_cached_step=32,
    )

    assert schedule.equivalent_block_evaluations == 396
    assert schedule.maximum_speedup == pytest.approx(480 / 396)


def test_aggressive_one_block_probe_needs_about_108_seconds() -> None:
    schedule = CacheSchedule(
        total_steps=8,
        full_steps=2,
        cached_steps=6,
        probe_blocks_per_cached_step=1,
    )

    assert schedule.projected_seconds(51.33) == pytest.approx(107.793)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_steps": 8, "full_steps": 1, "cached_steps": 6},
        {"total_steps": 8, "full_steps": -1, "cached_steps": 9},
        {"total_steps": 8, "full_steps": 2, "cached_steps": 6, "total_blocks": 0},
        {
            "total_steps": 8,
            "full_steps": 2,
            "cached_steps": 6,
            "probe_blocks_per_cached_step": 61,
        },
    ],
)
def test_invalid_cache_schedule_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        CacheSchedule(**kwargs)
