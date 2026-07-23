#!/usr/bin/env python3
"""A/B: residency-window image MLP as q6 vs window-local dense bf16."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import gc
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx

from benchmark_qwen_block import DEFAULT_MODEL, make_inputs, positive_int
from benchmark_qwen_block_dense_ab import all_finite, max_abs_error
from shardedit_mlx.benchmark_lora import apply_loras_to_loaded_blocks
from shardedit_mlx.dense_img_ff_profile import decide_dense_img_ff_window_verdict
from shardedit_mlx.dense_img_ff_window import prepare_dense_img_ff_window
from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations
from shardedit_mlx.qwen_block_loader import load_block_window, load_transformer_layout


@dataclass(frozen=True)
class PathTiming:
    name: str
    median_seconds: float
    min_seconds: float
    mean_seconds: float
    durations_seconds: tuple[float, ...]
    peak_gib: float
    active_after_prepare_gib: float | None
    bytes_materialized: int | None
    lora_seconds: float
    lora_selected_keys: int
    lora_applied_layers: int
    relative_to_q6: float | None
    max_abs_error_vs_q6: float | None
    all_finite: bool


@dataclass(frozen=True)
class DenseImgFFWindowResult:
    mlx_version: str
    platform: str
    model: str
    block_indices: tuple[int, ...]
    image_tokens: int
    text_tokens: int
    lora_paths: tuple[str, ...]
    lora_scales: tuple[float, ...]
    warmup_runs: int
    measured_runs: int
    paths: tuple[PathTiming, ...]
    verdict: str
    verdict_reason: str


def _reset_memory() -> None:
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def _run_blocks(blocks: Any, inputs: dict[str, Any]) -> Any:
    hidden = inputs["hidden_states"]
    encoder = inputs["encoder_hidden_states"]
    for loaded in blocks:
        encoder, hidden = loaded.module(
            **{
                **inputs,
                "hidden_states": hidden,
                "encoder_hidden_states": encoder,
            },
            block_idx=loaded.block_index,
        )
    mx.eval(encoder, hidden)
    return {"encoder_hidden_states": encoder, "hidden_states": hidden}


def time_path(
    *,
    name: str,
    layout: Any,
    block_indices: tuple[int, ...],
    inputs: dict[str, Any],
    dense: bool,
    lora_paths: tuple[str, ...],
    lora_scales: tuple[float, ...],
    block_count: int,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[Any, PathTiming]:
    def once() -> Any:
        _reset_memory()
        blocks = load_block_window(layout, block_indices)
        active_after_prepare = None
        bytes_materialized = None
        lora_started = time.perf_counter()
        lora_results = apply_loras_to_loaded_blocks(
            blocks,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            block_count=block_count,
        )
        lora_seconds = time.perf_counter() - lora_started
        if dense:
            handles = prepare_dense_img_ff_window(blocks)
            bytes_materialized = sum(handle.bytes_materialized for handle in handles)
            active_after_prepare = mx.get_active_memory() / 1024**3
        output = _run_blocks(blocks, inputs)
        peak_gib = mx.get_peak_memory() / 1024**3
        del blocks
        _reset_memory()
        return (
            output,
            peak_gib,
            active_after_prepare,
            bytes_materialized,
            lora_seconds,
            sum(result.selected_keys for result in lora_results),
            sum(result.applied_layers for result in lora_results),
        )

    # Warmup + measured without round-robin across variants (each reload is costly).
    for _ in range(warmup_runs):
        once()

    outputs: list[Any] = []
    durations: list[float] = []
    peaks: list[float] = []
    actives: list[float | None] = []
    nbytes_list: list[int | None] = []
    lora_durations: list[float] = []
    lora_selected_keys = 0
    lora_applied_layers = 0
    for _ in range(measured_runs):
        started = time.perf_counter()
        (
            output,
            peak_gib,
            active_after_prepare,
            bytes_materialized,
            lora_seconds,
            selected_keys,
            applied_layers,
        ) = once()
        durations.append(time.perf_counter() - started)
        outputs.append(output)
        peaks.append(peak_gib)
        actives.append(active_after_prepare)
        nbytes_list.append(bytes_materialized)
        lora_durations.append(lora_seconds)
        lora_selected_keys = selected_keys
        lora_applied_layers = applied_layers

    summary = summarize_durations(durations)
    lora_summary = summarize_durations(lora_durations)
    return outputs[-1], PathTiming(
        name=name,
        median_seconds=summary.median_seconds,
        min_seconds=summary.min_seconds,
        mean_seconds=summary.mean_seconds,
        durations_seconds=tuple(durations),
        peak_gib=float(sorted(peaks)[len(peaks) // 2]),
        active_after_prepare_gib=actives[-1],
        bytes_materialized=nbytes_list[-1],
        lora_seconds=lora_summary.median_seconds,
        lora_selected_keys=lora_selected_keys,
        lora_applied_layers=lora_applied_layers,
        relative_to_q6=None,
        max_abs_error_vs_q6=None,
        all_finite=all_finite(outputs[-1]),
    )


def run_benchmark(args: argparse.Namespace) -> DenseImgFFWindowResult:
    layout = load_transformer_layout(args.model)
    block_indices = tuple(range(args.block_start, args.block_start + args.block_count))
    inputs = make_inputs(args.image_tokens, args.text_tokens)
    lora_paths = tuple(str(path) for path in args.lora_paths)
    lora_scales = tuple(float(scale) for scale in args.lora_scales)
    block_count = len(layout.plans)

    q6_output, q6_path = time_path(
        name="q6_img_ff",
        layout=layout,
        block_indices=block_indices,
        inputs=inputs,
        dense=False,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        block_count=block_count,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    dense_output, dense_path = time_path(
        name="dense_img_ff_window",
        layout=layout,
        block_indices=block_indices,
        inputs=inputs,
        dense=True,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        block_count=block_count,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    error = max_abs_error(q6_output, dense_output)
    q6_path = PathTiming(
        **{
            **asdict(q6_path),
            "relative_to_q6": 1.0,
            "max_abs_error_vs_q6": 0.0,
        }
    )
    dense_path = PathTiming(
        **{
            **asdict(dense_path),
            "relative_to_q6": relative_speedup(
                q6_path.median_seconds, dense_path.median_seconds
            ),
            "max_abs_error_vs_q6": error,
            "all_finite": all_finite(dense_output) and dense_path.all_finite,
        }
    )

    verdict, reason = decide_dense_img_ff_window_verdict(
        q6_median=q6_path.median_seconds,
        dense_median=dense_path.median_seconds,
        q6_peak_gib=q6_path.peak_gib,
        dense_peak_gib=dense_path.peak_gib,
        max_abs_error=error,
        all_finite=dense_path.all_finite,
        speedup_threshold=args.speedup_threshold,
        peak_budget_gib=args.peak_budget_gib,
        error_tolerance=args.error_tolerance,
    )
    return DenseImgFFWindowResult(
        mlx_version=version("mlx"),
        platform=platform.platform(),
        model=str(args.model),
        block_indices=block_indices,
        image_tokens=args.image_tokens,
        text_tokens=args.text_tokens,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        paths=(q6_path, dense_path),
        verdict=verdict,
        verdict_reason=reason,
    )


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-start", type=non_negative_int, default=0)
    parser.add_argument("--block-count", type=positive_int, default=4)
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--lora-paths", nargs="*", default=(), help="Optional LoRA paths")
    parser.add_argument("--lora-scales", nargs="*", type=float, default=(), help="Optional LoRA scales")
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--speedup-threshold", type=float, default=0.05)
    parser.add_argument("--error-tolerance", type=float, default=32.0)
    parser.add_argument(
        "--peak-budget-gib",
        type=float,
        default=None,
        help="Optional hard peak memory gate for dense path",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    payload = asdict(result)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    q6, dense = result.paths
    print(
        f"verdict={result.verdict} "
        f"q6={q6.median_seconds:.3f}s peak={q6.peak_gib:.2f}GiB "
        f"dense={dense.median_seconds:.3f}s({dense.relative_to_q6:.3f}x) "
        f"peak={dense.peak_gib:.2f}GiB err={dense.max_abs_error_vs_q6}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
