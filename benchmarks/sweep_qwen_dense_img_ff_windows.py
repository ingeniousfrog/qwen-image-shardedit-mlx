#!/usr/bin/env python3
"""Sweep dense image MLP window A/B across Transformer block windows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_qwen_block import DEFAULT_MODEL, positive_int
from benchmark_qwen_dense_img_ff_window import run_benchmark
from shardedit_mlx.gemm_profile import relative_speedup
from shardedit_mlx.qwen_block_loader import load_transformer_layout
from shardedit_mlx.residency_plan import fixed_block_windows, shard_block_windows


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one integer")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return values


def benchmark_namespace(args: argparse.Namespace, *, block_start: int, block_count: int) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        block_start=block_start,
        block_count=block_count,
        image_tokens=args.image_tokens,
        text_tokens=args.text_tokens,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
        warmup=args.warmup,
        runs=args.runs,
        speedup_threshold=args.speedup_threshold,
        error_tolerance=args.error_tolerance,
        peak_budget_gib=args.peak_budget_gib,
        output=None,
    )


def selected_windows(args: argparse.Namespace) -> tuple[tuple[int, tuple[int, ...]], ...]:
    layout = load_transformer_layout(args.model)
    if args.window_mode == "shard":
        windows = shard_block_windows(layout.plans, layout.ordered_shards)
        return tuple((window.index, window.block_indices) for window in windows)
    if args.window_mode == "fixed":
        windows = fixed_block_windows(layout.plans, args.block_count)
        return tuple((window.index, window.block_indices) for window in windows)

    starts = args.block_starts
    if starts is None:
        raise ValueError("--block-starts is required when --window-mode starts")
    block_count = len(layout.plans)
    selected: tuple[tuple[int, tuple[int, ...]], ...] = ()
    for window_index, start in enumerate(starts):
        end = start + args.block_count
        if end > block_count:
            raise ValueError(
                f"window {start}-{end - 1} exceeds block count {block_count}"
            )
        selected = (*selected, (window_index, tuple(range(start, end))))
    return selected


def path_by_name(result: Any, name: str) -> Any:
    for path in result.paths:
        if path.name == name:
            return path
    raise KeyError(name)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    windows = selected_windows(args)
    results: list[dict[str, Any]] = []
    q6_total = 0.0
    dense_total = 0.0
    max_q6_peak = 0.0
    max_dense_peak = 0.0
    max_dense_active_after_prepare = 0.0
    total_dense_bytes = 0
    all_finite = True
    max_error = 0.0

    for window_index, block_indices in windows:
        print(
            f"window {window_index}: blocks {block_indices[0]}-{block_indices[-1]}",
            file=sys.stderr,
            flush=True,
        )
        result = run_benchmark(
            benchmark_namespace(
                args,
                block_start=block_indices[0],
                block_count=len(block_indices),
            )
        )
        q6_path = path_by_name(result, "q6_img_ff")
        dense_path = path_by_name(result, "dense_img_ff_window")
        q6_total += q6_path.median_seconds
        dense_total += dense_path.median_seconds
        max_q6_peak = max(max_q6_peak, q6_path.peak_gib)
        max_dense_peak = max(max_dense_peak, dense_path.peak_gib)
        max_dense_active_after_prepare = max(
            max_dense_active_after_prepare,
            dense_path.active_after_prepare_gib or 0.0,
        )
        total_dense_bytes += dense_path.bytes_materialized or 0
        all_finite = all_finite and dense_path.all_finite
        max_error = max(max_error, dense_path.max_abs_error_vs_q6 or 0.0)
        results.append(
            {
                "window_index": window_index,
                "block_indices": block_indices,
                "result": asdict(result),
            }
        )

    saved_per_full_pass = q6_total - dense_total
    return {
        "model": str(args.model),
        "window_mode": args.window_mode,
        "image_tokens": args.image_tokens,
        "text_tokens": args.text_tokens,
        "lora_paths": tuple(str(path) for path in args.lora_paths),
        "lora_scales": tuple(float(scale) for scale in args.lora_scales),
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "full_miss_steps": args.full_miss_steps,
        "summary": {
            "q6_seconds_per_full_pass": q6_total,
            "dense_seconds_per_full_pass": dense_total,
            "saved_seconds_per_full_pass": saved_per_full_pass,
            "projected_saved_seconds": saved_per_full_pass * args.full_miss_steps,
            "relative_to_q6": relative_speedup(q6_total, dense_total),
            "max_q6_peak_gib": max_q6_peak,
            "max_dense_peak_gib": max_dense_peak,
            "max_dense_active_after_prepare_gib": max_dense_active_after_prepare,
            "total_dense_bytes_materialized": total_dense_bytes,
            "max_abs_error_vs_q6": max_error,
            "all_finite": all_finite,
        },
        "windows": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image-tokens", type=positive_int, default=3456)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--lora-paths", nargs="*", default=(), help="Optional LoRA paths")
    parser.add_argument("--lora-scales", nargs="*", type=float, default=(), help="Optional LoRA scales")
    parser.add_argument(
        "--window-mode",
        choices=("shard", "fixed", "starts"),
        default="shard",
        help="Which windows to sweep. shard matches the runtime residency windows.",
    )
    parser.add_argument("--block-count", type=positive_int, default=4)
    parser.add_argument("--block-starts", type=parse_int_list, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--speedup-threshold", type=float, default=0.05)
    parser.add_argument("--error-tolerance", type=float, default=32.0)
    parser.add_argument("--peak-budget-gib", type=float, default=None)
    parser.add_argument("--full-miss-steps", type=positive_int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    result = run_sweep(args)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    summary = result["summary"]
    print(
        "dense/q6="
        f"{summary['relative_to_q6']:.3f}x "
        f"saved_per_full_pass={summary['saved_seconds_per_full_pass']:.3f}s "
        f"projected_saved={summary['projected_saved_seconds']:.3f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
