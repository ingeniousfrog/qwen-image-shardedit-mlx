"""Offline analysis for escalating the F1B2 front-probe depth on a cache miss.

This answers one question cheaply, without touching the runtime cache
decision: if a denoise step would miss the fixed 1-block ("F1") residual
check, would checking a deeper front-block residual (2, 3, 4, ... blocks)
have brought the relative L1 back under threshold?

It works purely from `--shardedit-probe-blocks` timing logs (parsed with the
shared SHARDEDIT_TIMING helpers in `shardedit_mlx.full_miss_profile`) and does not
require the real cache to be enabled, since probing and caching are mutually
exclusive at runtime (see `parse_runtime_options` in `shardedit_mlx.mflux_fast_edit`).

The runtime cache decision compares this step's block-0 residual against a
*predicted anchor* derived from the last full-miss step (see
`decide_residual_cache` / `select_predicted_anchor`). The probe mechanism
instead compares this step's residual at a given depth against the
*immediately preceding* step's residual at the same depth. With the default
`cache-predictor last` and `cache-max-consecutive 1`, these coincide exactly
whenever the preceding step was itself a full miss, which is the common case
right after a miss. Treat results here as a first-pass signal, not a
byte-for-byte replay of the runtime decision.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeStepDepths:
    """Per-depth image relative L1 for one denoise step, from residual probes."""

    step: int
    relative_l1_by_depth: Mapping[int, float]


@dataclass(frozen=True)
class EscalationOutcome:
    """Whether escalating the front-probe depth would have rescued one step."""

    step: int
    baseline_depth: int
    baseline_relative_l1: float | None
    baseline_hit: bool
    rescue_depth: int | None
    rescue_relative_l1: float | None


@dataclass(frozen=True)
class EscalationSummary:
    threshold: float
    baseline_depth: int
    escalation_depths: tuple[int, ...]
    outcomes: tuple[EscalationOutcome, ...]

    @property
    def baseline_misses(self) -> tuple[EscalationOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if not outcome.baseline_hit)

    @property
    def rescued_misses(self) -> tuple[EscalationOutcome, ...]:
        return tuple(
            outcome for outcome in self.baseline_misses if outcome.rescue_depth is not None
        )

    @property
    def rescue_rate(self) -> float:
        misses = self.baseline_misses
        if not misses:
            return 0.0
        return len(self.rescued_misses) / len(misses)


def probe_depths_by_step(events: Sequence[Mapping]) -> tuple[ProbeStepDepths, ...]:
    """Group `residual_probe` timing events into per-step, per-depth relative L1."""

    by_step: dict[int, dict[int, float]] = defaultdict(dict)
    for event in events:
        if event.get("name") != "residual_probe" or not event.get("has_previous", False):
            continue
        step = event.get("step")
        block = event.get("block")
        relative_l1 = event.get("image_relative_l1")
        if not isinstance(step, int):
            raise ValueError("residual_probe event is missing an integer step")
        if not isinstance(block, int):
            raise ValueError("residual_probe event is missing an integer block")
        if not isinstance(relative_l1, int | float):
            raise ValueError("residual_probe event is missing image_relative_l1")
        by_step[step][block] = float(relative_l1)
    return tuple(
        ProbeStepDepths(step=step, relative_l1_by_depth=dict(depths))
        for step, depths in sorted(by_step.items())
    )


def simulate_escalation(
    steps: Sequence[ProbeStepDepths],
    *,
    threshold: float,
    baseline_depth: int = 1,
    escalation_depths: Sequence[int] = (),
) -> EscalationSummary:
    """Simulate "expand F on a miss" from recorded multi-depth probe residuals.

    For each step, the baseline decision uses `baseline_depth` (F1 by
    default). If that misses (relative_l1 >= threshold), depths in
    `escalation_depths` greater than `baseline_depth` are tried in ascending
    order; the first one under threshold is the rescue depth, if any.
    """

    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    ordered_depths = tuple(
        sorted(depth for depth in set(escalation_depths) if depth > baseline_depth)
    )

    outcomes: list[EscalationOutcome] = []
    for step in steps:
        baseline_relative_l1 = step.relative_l1_by_depth.get(baseline_depth)
        baseline_hit = baseline_relative_l1 is not None and baseline_relative_l1 < threshold

        rescue_depth: int | None = None
        rescue_relative_l1: float | None = None
        if not baseline_hit:
            for depth in ordered_depths:
                candidate = step.relative_l1_by_depth.get(depth)
                if candidate is not None and candidate < threshold:
                    rescue_depth = depth
                    rescue_relative_l1 = candidate
                    break

        outcomes.append(
            EscalationOutcome(
                step=step.step,
                baseline_depth=baseline_depth,
                baseline_relative_l1=baseline_relative_l1,
                baseline_hit=baseline_hit,
                rescue_depth=rescue_depth,
                rescue_relative_l1=rescue_relative_l1,
            )
        )

    return EscalationSummary(
        threshold=threshold,
        baseline_depth=baseline_depth,
        escalation_depths=ordered_depths,
        outcomes=tuple(outcomes),
    )
