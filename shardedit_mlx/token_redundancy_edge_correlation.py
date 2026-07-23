"""Pure logic for correlating token-redundancy heatmaps with image edge density.

Follow-up to `shardedit_mlx.token_redundancy_heatmap` (§38 in
`docs/experiments/2026-07-17-m2-qwen-edit.md`): a single case's heatmap showed
low-similarity ("unique") target tokens correlating with local edge/detail
density in the generated image, more strongly at deeper blocks. This module
holds the reusable, pure-Python math (block-average downsampling + Pearson/
Spearman correlation) so that check can be repeated across more cases from a
CLI tool without hand-rolled analysis each time. No PIL/numpy/MLX here --
those live in the `tools/analyze_token_redundancy_heatmap.py` CLI, which is
the only place that needs to load actual images.
"""

from __future__ import annotations

from collections.abc import Sequence
import math


def downsample_grid(
    values: Sequence[Sequence[float]],
    *,
    grid_height: int,
    grid_width: int,
) -> list[list[float]]:
    """Block-average-pool a 2D grid down to `(grid_height, grid_width)`.

    `values` must have dimensions that are exact multiples of the target
    shape (e.g. a 768x768 edge map downsampled to 48x48 patches, cell size
    16x16). Raises ValueError otherwise.
    """

    source_height = len(values)
    source_width = len(values[0]) if source_height else 0
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("grid_height and grid_width must be positive")
    if source_height == 0 or source_width == 0:
        raise ValueError("values must be a non-empty 2D grid")
    if source_height % grid_height != 0 or source_width % grid_width != 0:
        raise ValueError(
            f"source grid {source_height}x{source_width} is not an exact multiple of "
            f"target grid {grid_height}x{grid_width}"
        )

    cell_height = source_height // grid_height
    cell_width = source_width // grid_width
    result: list[list[float]] = []
    for row in range(grid_height):
        row_values: list[float] = []
        for col in range(grid_width):
            total = 0.0
            count = 0
            for dy in range(cell_height):
                source_row = values[row * cell_height + dy]
                for dx in range(cell_width):
                    total += source_row[col * cell_width + dx]
                    count += 1
            row_values.append(total / count)
        result.append(row_values)
    return result


def _flatten(grid: Sequence[Sequence[float]]) -> list[float]:
    return [value for row in grid for value in row]


def pearson_correlation(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Pearson correlation between two equal-shape grids, flattened."""

    values_a = _flatten(a)
    values_b = _flatten(b)
    if len(values_a) != len(values_b):
        raise ValueError("grids must have the same total number of cells")
    if len(values_a) < 2:
        raise ValueError("need at least 2 cells to correlate")

    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [v - mean_a for v in values_a]
    centered_b = [v - mean_b for v in values_b]
    numerator = sum(x * y for x, y in zip(centered_a, centered_b))
    denom_a = math.sqrt(sum(x * x for x in centered_a))
    denom_b = math.sqrt(sum(y * y for y in centered_b))
    if denom_a == 0.0 or denom_b == 0.0:
        return 0.0
    return numerator / (denom_a * denom_b)


def _ranks(values: Sequence[float]) -> list[float]:
    """Average-tie ranking (0-indexed), matching pandas/scipy's default."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def spearman_correlation(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Spearman rank correlation between two equal-shape grids, flattened."""

    values_a = _flatten(a)
    values_b = _flatten(b)
    if len(values_a) != len(values_b):
        raise ValueError("grids must have the same total number of cells")
    ranked_a = [_ranks(values_a)]
    ranked_b = [_ranks(values_b)]
    return pearson_correlation(ranked_a, ranked_b)
