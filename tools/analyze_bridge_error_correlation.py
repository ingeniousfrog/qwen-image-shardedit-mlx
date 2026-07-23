#!/usr/bin/env python3
"""Aggregate phase-0 bridge-error vs uniqueness events from a qwen-image-shardedit-mlx log.

Typical usage (full pass, F1B2 geometry, cache off):

  SHARDEDIT_CACHE_THRESHOLD=0 SHARDEDIT_CACHE_BACK_BLOCKS=2 \\
  SHARDEDIT_BRIDGE_ERROR_DIAGNOSE=1 \\
    benchmarks/run_qwen_edit_benchmark.sh --runtime shardedit --residency shard --steps 8

  python3 tools/analyze_bridge_error_correlation.py \\
    benchmark-runs/<date>/<run>/shardedit-1/stdout.log
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shardedit_mlx.bridge_error_correlation import decide_phase0_go
from shardedit_mlx.full_miss_profile import load_timing_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument(
        "--go-spearman-threshold",
        type=float,
        default=0.25,
        help="per-step spearman threshold for a go vote (default 0.25)",
    )
    parser.add_argument(
        "--min-go-fraction",
        type=float,
        default=0.5,
        help="fraction of steps that must clear the threshold (default 0.5)",
    )
    args = parser.parse_args()

    events = [
        event
        for event in load_timing_events(args.log_path)
        if event.get("name") == "bridge_error_vs_redundancy"
    ]
    if not events:
        raise SystemExit(f"no bridge_error_vs_redundancy events in {args.log_path}")

    print(f"# {args.log_path}")
    print("step\tblock\tmean_abs_error\tmean_uniqueness\tpearson\tspearman\tgo")
    spearmans: list[float] = []
    for event in events:
        spearman = float(event["spearman"])
        spearmans.append(spearman)
        print(
            f"{event.get('step')}\t{event.get('block')}\t"
            f"{float(event['mean_abs_error']):.6f}\t"
            f"{float(event['mean_uniqueness']):.6f}\t"
            f"{float(event['pearson']):+.3f}\t"
            f"{spearman:+.3f}\t"
            f"{event.get('go')}"
        )

    decision = decide_phase0_go(
        spearmans,
        go_spearman_threshold=args.go_spearman_threshold,
        min_go_fraction=args.min_go_fraction,
    )
    late_decision = decide_phase0_go(
        spearmans,
        go_spearman_threshold=args.go_spearman_threshold,
        min_go_fraction=args.min_go_fraction,
        late_half_only=True,
    )
    print()
    print(f"steps: {len(spearmans)}")
    print(f"mean spearman: {sum(spearmans) / len(spearmans):+.3f}")
    print(f"phase0 decision (all steps): {'go' if decision else 'no-go'}")
    print(f"phase0 decision (late half): {'go' if late_decision else 'no-go'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
