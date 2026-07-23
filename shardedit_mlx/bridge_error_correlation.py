"""Pure logic for correlating F1B2 middle-bridge error with token uniqueness.

Phase-0 diagnostic for selective token refill
(`docs/experiments/2026-07-17-m2-qwen-edit.md` §39 plan): before changing the
forward path, ask whether the residual-bridge approximation error on a full
pass is concentrated on low-redundancy ("unique") target tokens -- the same
tokens §38 showed aligning with image edge density.

The bipartite uniqueness signal only covers even token indices (src half of
the ToMe-style split). Bridge error is therefore also summarized on that
same even-index subset so the two vectors are aligned for correlation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shardedit_mlx.token_redundancy_edge_correlation import (
    pearson_correlation,
    spearman_correlation,
)


@dataclass(frozen=True)
class BridgeErrorSummary:
    token_count: int
    mean_abs_error: float
    mean_uniqueness: float
    pearson: float
    spearman: float
    # go when spearman(uniqueness, abs_error) is clearly positive
    go: bool


def even_index_values(values: Sequence[float]) -> tuple[float, ...]:
    """Keep values at even flat indices (matches bipartite src tokens)."""

    if not values:
        raise ValueError("values must be non-empty")
    return tuple(values[i] for i in range(0, len(values), 2))


def uniqueness_from_similarities(similarities: Sequence[float]) -> tuple[float, ...]:
    """Map best-match cosine similarity in [-1, 1] to uniqueness in [0, 2]."""

    if not similarities:
        raise ValueError("similarities must be non-empty")
    return tuple(1.0 - float(value) for value in similarities)


def correlate_bridge_error_with_uniqueness(
    *,
    abs_errors: Sequence[float],
    uniqueness: Sequence[float],
    go_spearman_threshold: float = 0.25,
) -> BridgeErrorSummary:
    """Correlate per-token bridge |error| with uniqueness (both same length)."""

    if len(abs_errors) != len(uniqueness):
        raise ValueError(
            f"abs_errors length {len(abs_errors)} != uniqueness length {len(uniqueness)}"
        )
    if len(abs_errors) < 2:
        raise ValueError("need at least 2 tokens to correlate")
    for value in abs_errors:
        if value < 0:
            raise ValueError("abs_errors must be non-negative")

    error_grid = [list(abs_errors)]
    uniq_grid = [list(uniqueness)]
    pearson = pearson_correlation(error_grid, uniq_grid)
    spearman = spearman_correlation(error_grid, uniq_grid)
    return BridgeErrorSummary(
        token_count=len(abs_errors),
        mean_abs_error=sum(abs_errors) / len(abs_errors),
        mean_uniqueness=sum(uniqueness) / len(uniqueness),
        pearson=pearson,
        spearman=spearman,
        go=spearman >= go_spearman_threshold,
    )


def decide_phase0_go(
    step_spearmans: Sequence[float],
    *,
    min_steps: int = 2,
    go_spearman_threshold: float = 0.25,
    min_go_fraction: float = 0.5,
    late_half_only: bool = False,
) -> bool:
    """Aggregate per-step spearmans into a phase-0 go/no-go decision.

    When ``late_half_only`` is True, only the last ceil(n/2) steps are counted.
    That matches F1B2 practice: early steps are usually full-miss / near-noise,
    while cache hits (and softening) concentrate in the later half.
    """

    values = list(step_spearmans)
    if late_half_only and values:
        start = len(values) - max(1, (len(values) + 1) // 2)
        values = values[start:]
    if len(values) < min_steps:
        return False
    hits = sum(1 for value in values if value >= go_spearman_threshold)
    return hits / len(values) >= min_go_fraction
