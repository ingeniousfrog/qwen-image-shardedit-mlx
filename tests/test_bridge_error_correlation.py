from __future__ import annotations

import pytest

from shardedit_mlx.bridge_error_correlation import (
    BridgeErrorSummary,
    correlate_bridge_error_with_uniqueness,
    decide_phase0_go,
    even_index_values,
    uniqueness_from_similarities,
)


def test_even_index_values_keeps_even_positions() -> None:
    assert even_index_values((0.1, 0.2, 0.3, 0.4, 0.5)) == (0.1, 0.3, 0.5)


def test_even_index_values_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        even_index_values(())


def test_uniqueness_from_similarities() -> None:
    assert uniqueness_from_similarities((0.9, 0.5)) == pytest.approx((0.1, 0.5))


def test_correlate_perfect_alignment_is_go() -> None:
    # Higher uniqueness -> higher abs error
    abs_errors = (0.1, 0.2, 0.3, 0.4)
    uniqueness = (0.1, 0.2, 0.3, 0.4)

    summary = correlate_bridge_error_with_uniqueness(
        abs_errors=abs_errors,
        uniqueness=uniqueness,
        go_spearman_threshold=0.25,
    )

    assert summary == BridgeErrorSummary(
        token_count=4,
        mean_abs_error=pytest.approx(0.25),
        mean_uniqueness=pytest.approx(0.25),
        pearson=pytest.approx(1.0),
        spearman=pytest.approx(1.0),
        go=True,
    )


def test_correlate_anti_aligned_is_not_go() -> None:
    abs_errors = (0.1, 0.2, 0.3, 0.4)
    uniqueness = (0.4, 0.3, 0.2, 0.1)

    summary = correlate_bridge_error_with_uniqueness(
        abs_errors=abs_errors,
        uniqueness=uniqueness,
        go_spearman_threshold=0.25,
    )

    assert summary.spearman == pytest.approx(-1.0)
    assert summary.go is False


def test_correlate_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        correlate_bridge_error_with_uniqueness(abs_errors=(0.1, 0.2), uniqueness=(0.1,))


def test_correlate_rejects_negative_errors() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        correlate_bridge_error_with_uniqueness(abs_errors=(-0.1, 0.2), uniqueness=(0.1, 0.2))


def test_decide_phase0_go_requires_enough_late_steps() -> None:
    assert decide_phase0_go((0.5,), min_steps=2) is False
    assert decide_phase0_go((0.5, 0.6), min_steps=2, go_spearman_threshold=0.25) is True
    assert decide_phase0_go((0.1, 0.1, 0.6), min_steps=2, min_go_fraction=0.5) is False
    assert decide_phase0_go((0.1, 0.3, 0.6), min_steps=2, min_go_fraction=0.5) is True


def test_decide_phase0_go_late_half_only() -> None:
    # Early steps fail, late half clears → go with late_half_only
    spearmans = (0.03, 0.04, 0.05, 0.07, 0.13, 0.29, 0.73)
    assert decide_phase0_go(spearmans, min_steps=2) is False
    assert decide_phase0_go(spearmans, min_steps=2, late_half_only=True) is True
