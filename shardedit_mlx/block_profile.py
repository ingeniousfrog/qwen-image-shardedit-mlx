"""Pure data helpers for Qwen Transformer component profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import statistics


@dataclass(frozen=True)
class ComponentTiming:
    name: str
    median_seconds: float
    min_seconds: float
    share: float


def summarize_component_durations(
    durations: Mapping[str, Sequence[float]],
) -> tuple[ComponentTiming, ...]:
    """Summarize measured stage durations without changing their order."""

    if not durations:
        raise ValueError("at least one component duration is required")

    medians: list[tuple[str, float, float]] = []
    for name, values in durations.items():
        if not values:
            raise ValueError(f"component has no durations: {name}")
        if any(value < 0.0 for value in values):
            raise ValueError(f"component has a negative duration: {name}")
        medians.append((name, statistics.median(values), min(values)))

    total = sum(median for _, median, _ in medians)
    if total <= 0.0:
        raise ValueError("component duration total must be positive")

    return tuple(
        ComponentTiming(
            name=name,
            median_seconds=median,
            min_seconds=minimum,
            share=median / total,
        )
        for name, median, minimum in medians
    )
