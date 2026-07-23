"""Pure helpers for selective middle residual refill on F1B2 cache hits.

Modes:

- ``subset`` / ``subset-f1`` (1A): gather unique target tokens (+ condition), run
  one deep middle block, scatter back. ``subset`` gathers the bridged state
  (legacy); ``subset-f1`` gathers the F1 state so the block output is a true
  recomputed middle residual for those tokens.
- ``residual-dampen`` (1B): fixed residual scale ``1 - dampen`` on unique tokens.
- ``uniqueness-scale``: residual scale ``1 - dampen_max * (u_i / max_u)``
  (unique tokens keep less residual).
- ``uniqueness-boost``: residual scale ``1 + boost_max * (u_i / max_u)``
  (unique tokens amplify the predicted middle residual).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

SELECTIVE_REFILL_MODES = (
    "subset",
    "subset-f1",
    "residual-dampen",
    "uniqueness-scale",
    "uniqueness-boost",
)
DEFAULT_SELECTIVE_REFILL_MODE = "residual-dampen"
RESIDUAL_ADJUST_MODES = frozenset(
    {"residual-dampen", "uniqueness-scale", "uniqueness-boost"}
)
SUBSET_REFILL_MODES = frozenset({"subset", "subset-f1"})


def select_unique_even_indices(
    best_similarities: Sequence[float],
    *,
    fraction: float,
) -> tuple[int, ...]:
    """Select the most unique even flat indices from bipartite src similarities.

    `best_similarities[i]` corresponds to flat token index `2*i`. Returns flat
    (even) indices into the full target sequence, sorted ascending.
    """

    indices, _ = uniqueness_scaled_residual_scales(
        best_similarities,
        fraction=fraction,
        amount=0.0,
        direction="boost",
    )
    return indices


def uniqueness_scaled_residual_scales(
    best_similarities: Sequence[float],
    *,
    fraction: float,
    amount: float,
    direction: Literal["dampen", "boost"],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Select top unique even tokens and assign residual scales ∝ uniqueness.

    uniqueness = 1 - best_similarity. Among the top-``fraction`` most unique
    even tokens, weight ``w_i = u_i / max_u``. Then:

    - dampen: ``scale_i = 1 - amount * w_i`` (unique keep less residual)
    - boost: ``scale_i = 1 + amount * w_i`` (unique amplify residual)

    Returns (flat_even_indices, scales) sorted by ascending index.
    """

    if not best_similarities:
        raise ValueError("best_similarities must be non-empty")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be in [0, 1]")
    if direction not in ("dampen", "boost"):
        raise ValueError("direction must be 'dampen' or 'boost'")

    count = max(1, int(round(len(best_similarities) * fraction)))
    count = min(count, len(best_similarities))
    ranked = sorted(range(len(best_similarities)), key=lambda i: best_similarities[i])
    selected_src = ranked[:count]
    uniqueness = [1.0 - float(best_similarities[i]) for i in selected_src]
    max_u = max(uniqueness)
    if max_u <= 0.0:
        weights = [1.0] * len(selected_src)
    else:
        weights = [u / max_u for u in uniqueness]

    if direction == "dampen":
        scales = [1.0 - amount * w for w in weights]
    else:
        scales = [1.0 + amount * w for w in weights]

    paired = sorted(
        ((2 * src_i, scale) for src_i, scale in zip(selected_src, scales, strict=True)),
        key=lambda item: item[0],
    )
    indices = tuple(index for index, _ in paired)
    scale_values = tuple(scale for _, scale in paired)
    return indices, scale_values


def uniqueness_scaled_dampens(
    best_similarities: Sequence[float],
    *,
    fraction: float,
    dampen_max: float,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Backward-compatible dampen amounts: ``dampen_i = dampen_max * (u_i / max_u)``."""

    indices, scales = uniqueness_scaled_residual_scales(
        best_similarities,
        fraction=fraction,
        amount=dampen_max,
        direction="dampen",
    )
    dampens = tuple(1.0 - scale for scale in scales)
    return indices, dampens


def build_image_gather_indices(
    selected_target_indices: Sequence[int],
    *,
    target_token_count: int,
    total_image_tokens: int,
) -> tuple[int, ...]:
    """Gather selected target tokens plus the full condition suffix."""

    if target_token_count <= 0 or total_image_tokens <= target_token_count:
        raise ValueError("need a non-empty target prefix and condition suffix")
    for index in selected_target_indices:
        if index < 0 or index >= target_token_count:
            raise ValueError(f"target index {index} out of range [0, {target_token_count})")
    selected = tuple(sorted(set(int(i) for i in selected_target_indices)))
    condition = tuple(range(target_token_count, total_image_tokens))
    return selected + condition


def dampened_token_value(
    *,
    f1_value: float,
    bridged_value: float,
    dampen: float,
) -> float:
    """Blend bridged token back toward F1: dampen=1 keeps F1, dampen=0 keeps bridge."""

    if not 0.0 <= dampen <= 1.0:
        raise ValueError("dampen must be in [0, 1]")
    return (1.0 - dampen) * bridged_value + dampen * f1_value


def scaled_residual_token_value(
    *,
    f1_value: float,
    bridged_value: float,
    scale: float,
) -> float:
    """Apply ``f1 + scale * (bridged - f1)``. scale=1 keeps bridge; scale=0 keeps F1."""

    return f1_value + scale * (bridged_value - f1_value)


def should_apply_selective_refill(
    *,
    fraction: float,
    current_step: int | None,
    min_step: int,
    cache_hit: bool,
) -> bool:
    """Return whether selective refill should run for this denoise step."""

    if fraction <= 0.0 or not cache_hit or current_step is None:
        return False
    if min_step < 0:
        raise ValueError("min_step must be >= 0")
    return current_step >= min_step
