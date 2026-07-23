"""Pure logic for turning per-token best-match similarities into a spatial map.

Follow-up to `shardedit_mlx.token_redundancy`: that module answers "how much
redundancy is there overall"; this one answers "*where* in the image does the
redundancy live" so a human can eyeball whether low-similarity ("unique")
target tokens cluster on high-frequency detail (hair, lace, eyes) rather than
being scattered uniformly. Still training-free and read-only: this only
reshapes/colors numbers that the runtime hook already computed, it does not
change any Transformer computation.

The bipartite split in `mflux_fast_edit.bipartite_best_match_similarities`
takes every *even* flattened token index as "src" (`slice(0, None, 2)` along
the token axis). For a row-major `(grid_height, grid_width)` patch grid with
an even `grid_width`, `row * grid_width` is always even, so "even flat index"
reduces to "even column index" -- i.e. the src tokens are exactly the
even-numbered columns of every row. That means `similarities_to_grid` can
reshape the flat `best_similarities` list directly into a
`(grid_height, grid_width // 2)` grid without any gaps, at the cost of the
heatmap being column-compressed 2x relative to the real image (nearest-
neighbor upscaling the caller does before saving corrects this for visual
comparison).
"""

from __future__ import annotations

from collections.abc import Sequence


def similarities_to_grid(
    similarities: Sequence[float],
    *,
    grid_height: int,
    grid_width: int,
) -> list[list[float]]:
    """Reshape a flat best-similarity list into a (grid_height, grid_width // 2) grid.

    Raises ValueError if the grid shape is invalid or does not match the
    number of similarities (grid_height * (grid_width // 2)).
    """

    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("grid_height and grid_width must be positive")
    if grid_width % 2 != 0:
        raise ValueError("grid_width must be even for the even/odd bipartite split")

    src_width = grid_width // 2
    expected = grid_height * src_width
    values = list(similarities)
    if len(values) != expected:
        raise ValueError(
            f"expected {expected} similarities for a {grid_height}x{grid_width} grid "
            f"(src width {src_width}), got {len(values)}"
        )

    return [values[row * src_width : (row + 1) * src_width] for row in range(grid_height)]


_COLOR_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (178, 24, 43)),  # red: least redundant / most "unique"
    (0.5, (253, 219, 199)),  # pale
    (0.7, (247, 247, 247)),  # white: mid
    (0.9, (146, 197, 222)),  # light blue
    (1.0, (33, 102, 172)),  # blue: most redundant / safest to merge
)


def similarity_to_rgb(value: float) -> tuple[int, int, int]:
    """Map a [0, 1] best-match similarity to a diverging red-white-blue color.

    Red = low similarity (unique, likely high-frequency detail token).
    Blue = high similarity (redundant, a safe merge candidate).
    """

    clamped = max(0.0, min(1.0, value))
    for (lo_pos, lo_color), (hi_pos, hi_color) in zip(_COLOR_STOPS, _COLOR_STOPS[1:]):
        if lo_pos <= clamped <= hi_pos:
            span = hi_pos - lo_pos
            t = (clamped - lo_pos) / span if span > 0 else 0.0
            return tuple(
                round(lo_channel + (hi_channel - lo_channel) * t)
                for lo_channel, hi_channel in zip(lo_color, hi_color)
            )
    return _COLOR_STOPS[-1][1]


def grid_to_rgb_rows(grid: Sequence[Sequence[float]]) -> list[list[tuple[int, int, int]]]:
    """Apply `similarity_to_rgb` element-wise to a 2D similarity grid."""

    return [[similarity_to_rgb(value) for value in row] for row in grid]
