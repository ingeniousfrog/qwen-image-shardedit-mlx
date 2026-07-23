from __future__ import annotations

import pytest

from shardedit_mlx.token_redundancy import (
    RedundancySummary,
    best_match_similarities,
    redundancy_summary,
)


def test_best_match_similarities_takes_row_maxima() -> None:
    matrix = (
        (0.1, 0.9, 0.4),
        (0.99, 0.2, 0.3),
    )

    assert best_match_similarities(matrix) == (0.9, 0.99)


def test_best_match_similarities_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        best_match_similarities(())


def test_best_match_similarities_rejects_empty_row() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        best_match_similarities(((),))


def test_redundancy_summary_computes_mean_median_and_fractions() -> None:
    summary = redundancy_summary((0.5, 0.9, 0.96, 0.99), thresholds=(0.9, 0.95, 0.995))

    assert summary == RedundancySummary(
        token_count=4,
        mean_best_similarity=pytest.approx((0.5 + 0.9 + 0.96 + 0.99) / 4),
        median_best_similarity=pytest.approx((0.9 + 0.96) / 2),
        fraction_above_threshold={
            0.9: pytest.approx(0.75),
            0.95: pytest.approx(0.5),
            0.995: pytest.approx(0.0),
        },
    )


def test_redundancy_summary_handles_odd_length_median() -> None:
    summary = redundancy_summary((0.1, 0.5, 0.9), thresholds=(0.5,))

    assert summary.median_best_similarity == pytest.approx(0.5)
    assert summary.fraction_above_threshold[0.5] == pytest.approx(2 / 3)


def test_redundancy_summary_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        redundancy_summary(())


def test_redundancy_summary_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        redundancy_summary((float("nan"), 0.5))


def test_redundancy_summary_rejects_empty_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        redundancy_summary((0.5,), thresholds=())
