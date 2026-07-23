#!/usr/bin/env python3
"""Check whether escalating the F1 front-probe depth would rescue cache misses.

This is an offline, no-runtime-change way to test one idea for F1B2: "if this
step would miss with F=1, would running 2/3/4/.. front blocks instead have
brought the residual back under threshold?" It reads a qwen-image-shardedit-mlx
`--shardedit-probe-blocks` timing log (cache disabled, since probing and
caching are mutually exclusive at runtime) and simulates the decision using
the recorded per-depth relative L1 values. See
`shardedit_mlx/probe_escalation_analysis.py` for the caveats of this
approximation.

Typical usage:

  # 1. Collect a probe-only run (real cache stays off) at several depths:
  SHARDEDIT_PROBE_BLOCKS="1,2,3,4,5,8" SHARDEDIT_CACHE_THRESHOLD=0 \\
    benchmarks/run_qwen_edit_benchmark.sh --runtime shardedit ...

  # 2. Analyze it against the F1B2 threshold:
  python3 tools/analyze_probe_escalation.py \\
    benchmark-runs/<date>/<run>/shardedit-1/stdout.log --threshold 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shardedit_mlx.full_miss_profile import load_timing_events
from shardedit_mlx.probe_escalation_analysis import probe_depths_by_step, simulate_escalation


def _format_l1(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _parse_depths(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="relative L1 hit threshold to simulate (matches --shardedit-cache-threshold)",
    )
    parser.add_argument(
        "--baseline-depth",
        type=int,
        default=1,
        help="front-block depth used for the baseline F1 decision",
    )
    parser.add_argument(
        "--escalation-depths",
        type=str,
        default="2,3,4,5,8",
        help="comma-separated front-block depths to try after a baseline miss",
    )
    args = parser.parse_args()

    events = load_timing_events(args.log_path)
    steps = probe_depths_by_step(events)
    if not steps:
        raise SystemExit(f"no residual_probe events with a previous step found in {args.log_path}")

    summary = simulate_escalation(
        steps,
        threshold=args.threshold,
        baseline_depth=args.baseline_depth,
        escalation_depths=_parse_depths(args.escalation_depths),
    )

    print(f"# {args.log_path}")
    print(f"threshold: {summary.threshold}")
    print(f"baseline depth (F): {summary.baseline_depth}")
    print(f"escalation depths tried: {summary.escalation_depths}")
    print()
    print("step\tbaseline_l1\tbaseline_hit\trescue_depth\trescue_l1")
    for outcome in summary.outcomes:
        print(
            "\t".join(
                (
                    str(outcome.step),
                    _format_l1(outcome.baseline_relative_l1),
                    str(outcome.baseline_hit),
                    "-" if outcome.rescue_depth is None else str(outcome.rescue_depth),
                    _format_l1(outcome.rescue_relative_l1),
                )
            )
        )

    misses = summary.baseline_misses
    rescued = summary.rescued_misses
    print()
    print(f"baseline misses: {len(misses)}/{len(summary.outcomes)}")
    print(f"rescued by escalation: {len(rescued)} ({summary.rescue_rate:.0%} of misses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
