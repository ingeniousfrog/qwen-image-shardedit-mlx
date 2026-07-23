from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations


def test_duration_summary_reports_ordered_drift() -> None:
    summary = summarize_durations((1.0, 1.2, 1.4, 1.6), window=2)

    assert summary.count == 4
    assert summary.median_seconds == pytest.approx(1.3)
    assert summary.first_window_mean_seconds == pytest.approx(1.1)
    assert summary.last_window_mean_seconds == pytest.approx(1.5)
    assert summary.drift_ratio == pytest.approx(1.5 / 1.1)


def test_duration_summary_is_immutable() -> None:
    summary = summarize_durations((1.0,))

    with pytest.raises(FrozenInstanceError):
        summary.count = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("durations", "window"),
    [
        ((), 1),
        ((0.0,), 1),
        ((-1.0,), 1),
        ((1.0,), 0),
    ],
)
def test_duration_summary_rejects_invalid_values(
    durations: tuple[float, ...], window: int
) -> None:
    with pytest.raises(ValueError):
        summarize_durations(durations, window=window)


def test_relative_speedup_uses_baseline_over_candidate() -> None:
    assert relative_speedup(2.0, 1.6) == pytest.approx(1.25)


@pytest.mark.parametrize("baseline,candidate", [(0.0, 1.0), (1.0, 0.0)])
def test_relative_speedup_rejects_non_positive_values(
    baseline: float, candidate: float
) -> None:
    with pytest.raises(ValueError):
        relative_speedup(baseline, candidate)
