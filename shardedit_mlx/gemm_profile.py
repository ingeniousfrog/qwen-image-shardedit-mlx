"""Pure summaries for fixed-shape GEMM and long-run timing experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import statistics


@dataclass(frozen=True)
class DurationSummary:
    count: int
    median_seconds: float
    min_seconds: float
    max_seconds: float
    mean_seconds: float
    first_window_mean_seconds: float
    last_window_mean_seconds: float
    drift_ratio: float


def summarize_durations(
    durations: Sequence[float],
    *,
    window: int = 3,
) -> DurationSummary:
    """Summarize latency and end-to-start drift for an ordered run."""

    if not durations:
        raise ValueError("at least one duration is required")
    if window <= 0:
        raise ValueError("window must be positive")
    if any(duration <= 0.0 for duration in durations):
        raise ValueError("durations must be positive")

    actual_window = min(window, len(durations))
    first_mean = statistics.fmean(durations[:actual_window])
    last_mean = statistics.fmean(durations[-actual_window:])
    return DurationSummary(
        count=len(durations),
        median_seconds=statistics.median(durations),
        min_seconds=min(durations),
        max_seconds=max(durations),
        mean_seconds=statistics.fmean(durations),
        first_window_mean_seconds=first_mean,
        last_window_mean_seconds=last_mean,
        drift_ratio=last_mean / first_mean,
    )


def relative_speedup(baseline_seconds: float, candidate_seconds: float) -> float:
    """Return baseline/candidate, where values above one are faster."""

    if baseline_seconds <= 0.0 or candidate_seconds <= 0.0:
        raise ValueError("timings must be positive")
    return baseline_seconds / candidate_seconds
