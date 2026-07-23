"""Compute-only projections for training-free diffusion block caching."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CacheSchedule:
    """A projection of full and cached denoise steps for one Transformer."""

    total_steps: int
    full_steps: int
    cached_steps: int
    total_blocks: int = 60
    probe_blocks_per_cached_step: int = 0

    def __post_init__(self) -> None:
        if self.total_steps <= 0 or self.total_blocks <= 0:
            raise ValueError("total steps and total blocks must be positive")
        if self.full_steps <= 0 or self.cached_steps < 0:
            raise ValueError("at least one full step is required and cached steps cannot be negative")
        if self.full_steps + self.cached_steps != self.total_steps:
            raise ValueError("full and cached steps must add up to total steps")
        if not 0 <= self.probe_blocks_per_cached_step <= self.total_blocks:
            raise ValueError("cached-step probe blocks must fit within the Transformer")

    @property
    def equivalent_block_evaluations(self) -> int:
        return (
            self.full_steps * self.total_blocks
            + self.cached_steps * self.probe_blocks_per_cached_step
        )

    @property
    def relative_compute(self) -> float:
        baseline_blocks = self.total_steps * self.total_blocks
        return self.equivalent_block_evaluations / baseline_blocks

    @property
    def maximum_speedup(self) -> float:
        return 1.0 / self.relative_compute

    def projected_seconds(self, full_step_seconds: float) -> float:
        if not math.isfinite(full_step_seconds) or full_step_seconds <= 0:
            raise ValueError("full-step latency must be a finite positive number")
        equivalent_full_steps = self.equivalent_block_evaluations / self.total_blocks
        return full_step_seconds * equivalent_full_steps
