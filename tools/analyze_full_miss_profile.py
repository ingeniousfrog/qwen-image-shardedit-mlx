#!/usr/bin/env python3
"""Analyze qwen-image-shardedit-mlx full-miss, cache-hit, and anchor timing logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shardedit_mlx.full_miss_profile import (
    analyze_run_directory,
    load_timing_events,
    stdout_log_for_run_dir,
)
from shardedit_mlx.q6_linear_profile import summarize_q6_linear_events


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}s"


def _print_profile(run_dir: Path) -> None:
    profile = analyze_run_directory(run_dir)
    events = load_timing_events(stdout_log_for_run_dir(run_dir))
    q6_summaries = summarize_q6_linear_events(events)
    print(f"# {run_dir}")
    print(f"process: {_format_seconds(profile.process_seconds)}")
    print(f"generate: {_format_seconds(profile.generate_seconds)}")
    print(
        "peak_memory: "
        + ("-" if profile.peak_memory_gb is None else f"{profile.peak_memory_gb:.2f}GB")
    )
    print(
        "category\tsteps\tmean_step\ttotal_step\tmean_blocks\t"
        "mean_anchor\tmean_compute\tmean_load\tmean_lora\tmean_prepare\t"
        "mean_release\tlora_cache_hits\tpatched_hits\tmax_patched_cache\t"
        "kquant_hits\tkquant_misses\tmax_kquant_cache\tmax_kquant_cache_bytes"
    )
    for category in profile.categories:
        print(
            "\t".join(
                (
                    category.category,
                    str(category.steps),
                    f"{category.mean_seconds:.2f}s",
                    f"{category.total_seconds:.2f}s",
                    f"{category.mean_blocks:.1f}",
                    f"{category.mean_anchor_seconds:.2f}s",
                    f"{category.mean_window_compute_seconds:.2f}s",
                    f"{category.mean_window_load_seconds:.2f}s",
                    f"{category.mean_window_lora_seconds:.2f}s",
                    f"{category.mean_window_prepare_seconds:.2f}s",
                    f"{category.mean_window_release_seconds:.2f}s",
                    str(category.total_lora_weight_cache_hits),
                    str(category.total_patched_window_cache_hits),
                    str(category.max_patched_window_cache_size),
                    str(category.total_kquant_img_ff_cache_hits),
                    str(category.total_kquant_img_ff_cache_misses),
                    str(category.max_kquant_img_ff_cache_size),
                    str(category.max_kquant_img_ff_cache_bytes),
                )
            )
        )
    if q6_summaries:
        print()
        print("q6_linear_profile full-miss/non-hit top callsites")
        print(
            "category\tsite\tcalls\ttotal\tmean_step\tblocks\t"
            "input_shape\toutput_shape\tbits\tgroup_size"
        )
        for summary in q6_summaries:
            print(
                "\t".join(
                    (
                        summary.category,
                        summary.site,
                        str(summary.call_count),
                        f"{summary.total_seconds:.2f}s",
                        f"{summary.mean_seconds_per_step:.2f}s",
                        summary.blocks,
                        "x".join(str(part) for part in summary.input_shape),
                        "x".join(str(part) for part in summary.output_shape),
                        "-" if summary.bits is None else str(summary.bits),
                        "-" if summary.group_size is None else str(summary.group_size),
                    )
                )
            )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    profiles = [analyze_run_directory(path) for path in args.run_dirs]
    if args.json:
        payload = []
        for path, profile in zip(args.run_dirs, profiles, strict=True):
            events = load_timing_events(stdout_log_for_run_dir(path))
            payload.append(
                {
                    **profile.to_json_dict(),
                    "q6_linear_profile": [
                        {
                            "category": summary.category,
                            "site": summary.site,
                            "total_seconds": summary.total_seconds,
                            "mean_seconds_per_step": summary.mean_seconds_per_step,
                            "call_count": summary.call_count,
                            "steps": list(summary.steps),
                            "blocks": summary.blocks,
                            "input_shape": list(summary.input_shape),
                            "output_shape": list(summary.output_shape),
                            "bits": summary.bits,
                            "group_size": summary.group_size,
                            "mode": summary.mode,
                        }
                        for summary in summarize_q6_linear_events(events)
                    ],
                }
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for path in args.run_dirs:
        _print_profile(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
