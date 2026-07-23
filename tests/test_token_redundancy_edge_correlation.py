from __future__ import annotations

import pytest

from shardedit_mlx.token_redundancy_edge_correlation import (
    downsample_grid,
    pearson_correlation,
    spearman_correlation,
)


def test_downsample_grid_block_averages() -> None:
    # 4x4 source -> 2x2 target, cell size 2x2.
    source = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]

    result = downsample_grid(source, grid_height=2, grid_width=2)

    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_downsample_grid_rejects_non_multiple_shape() -> None:
    source = [[1, 2, 3]]
    with pytest.raises(ValueError, match="not an exact multiple"):
        downsample_grid(source, grid_height=1, grid_width=2)


def test_downsample_grid_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError, match="positive"):
        downsample_grid([[1.0]], grid_height=0, grid_width=1)


def test_downsample_grid_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        downsample_grid([], grid_height=1, grid_width=1)


def test_pearson_correlation_perfect_positive() -> None:
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[2.0, 4.0], [6.0, 8.0]]

    assert pearson_correlation(a, b) == pytest.approx(1.0)


def test_pearson_correlation_perfect_negative() -> None:
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[8.0, 6.0], [4.0, 2.0]]

    assert pearson_correlation(a, b) == pytest.approx(-1.0)


def test_pearson_correlation_zero_variance_is_zero() -> None:
    a = [[5.0, 5.0], [5.0, 5.0]]
    b = [[1.0, 2.0], [3.0, 4.0]]

    assert pearson_correlation(a, b) == 0.0


def test_pearson_correlation_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same total number"):
        pearson_correlation([[1.0, 2.0]], [[1.0]])


def test_spearman_correlation_perfect_monotonic_nonlinear() -> None:
    # b is a monotonic (but non-linear) function of a -> spearman should be
    # 1.0 even though the pearson correlation would be less than 1.0.
    a = [[1.0, 2.0, 3.0, 4.0]]
    b = [[1.0, 8.0, 27.0, 64.0]]

    assert spearman_correlation(a, b) == pytest.approx(1.0)
    assert pearson_correlation(a, b) < 1.0


def test_spearman_correlation_handles_ties() -> None:
    a = [[1.0, 1.0, 2.0, 3.0]]
    b = [[1.0, 1.0, 2.0, 3.0]]

    assert spearman_correlation(a, b) == pytest.approx(1.0)
