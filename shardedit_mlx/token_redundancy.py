"""Training-free diagnostics for how mergeable the image token stream is.

This does not change any computation. It answers, for a chosen set of block
boundaries, "how much near-duplicate information is already sitting in the
token stream at this point?" as a cheap signal for whether ToMe-style
bipartite token merging inside skippable middle blocks would have real
headroom, *before* writing any merge/unmerge runtime code.

The bipartite split mirrors Token Merging (Bolya et al.): tokens are
partitioned into two halves (even/odd along the sequence -- a simplification
of ToMe's checkerboard partition, adequate for a diagnostic) and each "src"
token is matched to its most similar "dst" token by cosine similarity. A
high best-match similarity means that token's information is close to
redundant with another token already present in the stream -- a candidate
for merging. The actual cosine-similarity matrix computation needs real
tensors (MLX) and lives in the `mflux_fast_edit` runtime hook; this module
only covers the pure, testable summarization math.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RedundancySummary:
    token_count: int
    mean_best_similarity: float
    median_best_similarity: float
    fraction_above_threshold: Mapping[float, float]


def best_match_similarities(similarity_matrix: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Return each row's maximum value from a [n_src, n_dst] similarity matrix."""

    if not similarity_matrix:
        raise ValueError("similarity_matrix must have at least one row")
    best: list[float] = []
    for row in similarity_matrix:
        if not row:
            raise ValueError("similarity_matrix rows must be non-empty")
        best.append(max(row))
    return tuple(best)


def redundancy_summary(
    best_similarities: Sequence[float],
    *,
    thresholds: Sequence[float] = (0.90, 0.95, 0.98, 0.995),
) -> RedundancySummary:
    """Summarize per-token best-match similarities into headroom statistics."""

    values = tuple(best_similarities)
    if not values:
        raise ValueError("best_similarities must have at least one value")
    for value in values:
        if not math.isfinite(value):
            raise ValueError("best_similarities must be finite")
    if not thresholds:
        raise ValueError("thresholds must have at least one value")

    sorted_values = sorted(values)
    count = len(sorted_values)
    mid = count // 2
    median = (
        sorted_values[mid]
        if count % 2 == 1
        else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    )

    fractions = {
        threshold: sum(1 for value in values if value >= threshold) / count
        for threshold in thresholds
    }

    return RedundancySummary(
        token_count=count,
        mean_best_similarity=sum(values) / count,
        median_best_similarity=median,
        fraction_above_threshold=fractions,
    )
