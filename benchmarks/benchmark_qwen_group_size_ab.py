#!/usr/bin/env python3
"""Interleaved q6 group_size 32/64/128 single-block A/B."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx

from benchmark_qwen_block import DEFAULT_MODEL, load_block, make_inputs, positive_int
from benchmark_qwen_block_dense_ab import (
    all_finite,
    max_abs_error,
    measure_round_robin,
    time_once,
)
from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations
from shardedit_mlx.group_size_profile import decide_group_size_verdict


GROUP_SIZES = (32, 64, 128)


@dataclass(frozen=True)
class VariantTiming:
    name: str
    group_size: int
    median_seconds: float
    min_seconds: float
    mean_seconds: float
    durations_seconds: tuple[float, ...]
    peak_memory_gib: float
    relative_to_gs64: float | None
    max_abs_error_vs_gs64: float | None
    all_finite: bool


@dataclass(frozen=True)
class GroupSizeAbResult:
    mlx_version: str
    platform: str
    model: str
    block_index: int
    bits: int
    image_tokens: int
    text_tokens: int
    warmup_runs: int
    measured_runs: int
    variants: tuple[VariantTiming, ...]
    verdict: str
    verdict_reason: str
    quality_gate_candidates: tuple[int, ...]
    proceed_to_metal_kernel: bool


def run_block(
    block: Any,
    inputs: dict[str, Any],
    block_index: int,
) -> tuple[mx.array, mx.array]:
    return block(**inputs, block_idx=block_index)


def run_benchmark(args: argparse.Namespace) -> GroupSizeAbResult:
    model_dir = args.model.expanduser().resolve()
    inputs = make_inputs(args.image_tokens, args.text_tokens)

    blocks: dict[int, Any] = {}
    for group_size in GROUP_SIZES:
        print(f"loading q6 group_size={group_size}", file=sys.stderr, flush=True)
        blocks[group_size] = load_block(
            model_dir,
            args.block_index,
            args.bits,
            group_size=group_size,
        )

    operations = tuple(
        (
            f"gs{group_size}",
            (
                lambda block=blocks[group_size]: run_block(
                    block, inputs, args.block_index
                )
            ),
        )
        for group_size in GROUP_SIZES
    )

    measured = measure_round_robin(
        operations,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    peak_by_group: dict[int, float] = {}
    for group_size in GROUP_SIZES:
        mx.reset_peak_memory()
        mx.clear_cache()
        time_once(
            lambda block=blocks[group_size]: run_block(block, inputs, args.block_index)
        )
        peak_by_group[group_size] = mx.get_peak_memory() / 1024**3

    baseline_output, baseline_times = measured["gs64"]
    baseline_summary = summarize_durations(baseline_times)
    medians: dict[int, float] = {64: baseline_summary.median_seconds}

    variants: list[VariantTiming] = [
        VariantTiming(
            name="gs64",
            group_size=64,
            median_seconds=baseline_summary.median_seconds,
            min_seconds=baseline_summary.min_seconds,
            mean_seconds=baseline_summary.mean_seconds,
            durations_seconds=baseline_times,
            peak_memory_gib=peak_by_group[64],
            relative_to_gs64=None,
            max_abs_error_vs_gs64=None,
            all_finite=all_finite(baseline_output),
        )
    ]

    for group_size in (32, 128):
        output, times = measured[f"gs{group_size}"]
        summary = summarize_durations(times)
        medians[group_size] = summary.median_seconds
        variants.append(
            VariantTiming(
                name=f"gs{group_size}",
                group_size=group_size,
                median_seconds=summary.median_seconds,
                min_seconds=summary.min_seconds,
                mean_seconds=summary.mean_seconds,
                durations_seconds=times,
                peak_memory_gib=peak_by_group[group_size],
                relative_to_gs64=relative_speedup(
                    baseline_summary.median_seconds,
                    summary.median_seconds,
                ),
                max_abs_error_vs_gs64=max_abs_error(baseline_output, output),
                all_finite=all_finite(output),
            )
        )

    verdict, reason, candidates = decide_group_size_verdict(
        baseline_group_size=64,
        baseline_median=baseline_summary.median_seconds,
        candidates=medians,
        speedup_threshold=args.speedup_threshold,
    )
    return GroupSizeAbResult(
        mlx_version=version("mlx"),
        platform=platform.platform(),
        model=str(model_dir),
        block_index=args.block_index,
        bits=args.bits,
        image_tokens=args.image_tokens,
        text_tokens=args.text_tokens,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        variants=tuple(variants),
        verdict=verdict,
        verdict_reason=reason,
        quality_gate_candidates=candidates,
        proceed_to_metal_kernel=verdict == "group_size_no_speedup",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--bits", type=int, choices=(4, 5, 6), default=6)
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument(
        "--speedup-threshold",
        type=float,
        default=0.05,
        help="Require at least this relative speedup vs gs64 before quality gating",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.block_index < 0 or args.block_index >= 60:
        parser.error("--block-index must be between 0 and 59")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    result = run_benchmark(args)
    payload = json.dumps(asdict(result), indent=2) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")

    print(
        f"verdict={result.verdict} candidates={list(result.quality_gate_candidates)} "
        f"proceed_metal={result.proceed_to_metal_kernel}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
