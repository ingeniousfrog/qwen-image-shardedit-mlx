from __future__ import annotations

import pytest

from shardedit_mlx.token_redundancy_heatmap import (
    grid_to_rgb_rows,
    similarities_to_grid,
    similarity_to_rgb,
)


def test_similarities_to_grid_reshapes_row_major() -> None:
    # 2x4 grid -> src width 2, 2 rows -> 4 values total.
    grid = similarities_to_grid((0.1, 0.2, 0.3, 0.4), grid_height=2, grid_width=4)

    assert grid == [[0.1, 0.2], [0.3, 0.4]]


def test_similarities_to_grid_rejects_odd_width() -> None:
    with pytest.raises(ValueError, match="even"):
        similarities_to_grid((0.1,), grid_height=1, grid_width=3)


def test_similarities_to_grid_rejects_non_positive_dims() -> None:
    with pytest.raises(ValueError, match="positive"):
        similarities_to_grid((0.1,), grid_height=0, grid_width=2)


def test_similarities_to_grid_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        similarities_to_grid((0.1, 0.2, 0.3), grid_height=2, grid_width=4)


def test_similarity_to_rgb_endpoints() -> None:
    assert similarity_to_rgb(0.0) == (178, 24, 43)
    assert similarity_to_rgb(1.0) == (33, 102, 172)


def test_similarity_to_rgb_clamps_out_of_range() -> None:
    assert similarity_to_rgb(-1.0) == similarity_to_rgb(0.0)
    assert similarity_to_rgb(2.0) == similarity_to_rgb(1.0)


def test_similarity_to_rgb_high_similarity_is_bluer_than_low() -> None:
    # "blueness" = blue channel minus red channel; should flip sign from the
    # red (low-similarity) endpoint to the blue (high-similarity) endpoint.
    def blueness(rgb: tuple[int, int, int]) -> int:
        return rgb[2] - rgb[0]

    assert blueness(similarity_to_rgb(0.0)) < 0
    assert blueness(similarity_to_rgb(1.0)) > 0


def test_grid_to_rgb_rows_applies_elementwise() -> None:
    grid = [[0.0, 1.0]]

    rows = grid_to_rgb_rows(grid)

    assert rows == [[similarity_to_rgb(0.0), similarity_to_rgb(1.0)]]
