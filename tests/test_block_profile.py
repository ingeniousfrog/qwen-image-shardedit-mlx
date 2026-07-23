from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from shardedit_mlx.block_profile import summarize_component_durations


def test_component_shares_use_median_stage_times() -> None:
    timings = summarize_component_durations(
        {
            "attention": (0.4, 0.6, 0.5),
            "image_mlp": (0.8, 1.0, 0.9),
        }
    )

    assert [timing.name for timing in timings] == ["attention", "image_mlp"]
    assert timings[0].median_seconds == pytest.approx(0.5)
    assert timings[0].share == pytest.approx(0.5 / 1.4)
    assert sum(timing.share for timing in timings) == pytest.approx(1.0)


def test_component_timings_are_immutable() -> None:
    timing = summarize_component_durations({"attention": (0.5,)})[0]

    with pytest.raises(FrozenInstanceError):
        timing.share = 0.0  # type: ignore[misc]


@pytest.mark.parametrize("durations", [{}, {"attention": ()}, {"attention": (-0.1,)}])
def test_component_summary_rejects_invalid_durations(
    durations: dict[str, tuple[float, ...]],
) -> None:
    with pytest.raises(ValueError):
        summarize_component_durations(durations)
