#!/usr/bin/env python3
"""Probe: per-window dense materialize wall vs image-MLP savings.

Answers whether async prefetch / double-buffer is worth engineering by splitting:
  1) cold materialize seconds for one residency window
  2) q6 vs dense img_ff-only seconds for that window
  3) full-block window compute (overlap budget for hiding materialize)
"""

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
from shardedit_mlx.dense_img_ff_profile import decide_dense_prefetch_verdict
from shardedit_mlx.dense_img_ff_window import prepare_dense_img_ff_window
from shardedit_mlx.gemm_profile import summarize_durations
from shardedit_mlx.qwen_block_loader import load_block_window, load_transformer_layout
from shardedit_mlx.residency_plan import shard_block_windows


@dataclass(frozen=True)
class ProbeSample:
    load_seconds: float
    q6_img_ff_seconds: float
    materialize_seconds: float
    dense_img_ff_seconds: float
    q6_window_compute_seconds: float
    dense_window_compute_seconds: float
    bytes_materialized: int
    mlp_savings_seconds: float
    sync_net_seconds: float


@dataclass(frozen=True)
class PrefetchProbeResult:
    mlx_version: str
    platform: str
    model: str
    window_index: int
    block_indices: tuple[int, ...]
    shards: tuple[str, ...]
    image_tokens: int
    text_tokens: int
    warmup_runs: int
    measured_runs: int
    samples: tuple[ProbeSample, ...]
    materialize_median: float
    q6_img_ff_median: float
    dense_img_ff_median: float
    mlp_savings_median: float
    sync_net_median: float
    q6_window_compute_median: float
    dense_window_compute_median: float
    bytes_materialized: int
    verdict: str
    verdict_reason: str


def _reset_memory() -> None:
    gc.collect()
    mx.clear_cache()


def _run_img_ffs(blocks: Any, image_hidden: mx.array) -> None:
    outs = [loaded.module.img_ff(image_hidden) for loaded in blocks]
    mx.eval(*outs)


def _run_blocks(blocks: Any, inputs: dict[str, Any]) -> None:
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


def measure_once(
    *,
    layout: Any,
    block_indices: tuple[int, ...],
    inputs: dict[str, Any],
) -> ProbeSample:
    _reset_memory()
    image_hidden = inputs["hidden_states"]

    load_started = time.perf_counter()
    blocks = load_block_window(layout, block_indices)
    load_seconds = time.perf_counter() - load_started

    # Full-window q6 compute = overlap budget for hiding next-window materialize.
    started = time.perf_counter()
    _run_blocks(blocks, inputs)
    q6_window_compute = time.perf_counter() - started

    started = time.perf_counter()
    _run_img_ffs(blocks, image_hidden)
    q6_img_ff = time.perf_counter() - started

    started = time.perf_counter()
    handles = prepare_dense_img_ff_window(blocks, reclaim_quantized=True)
    materialize = time.perf_counter() - started
    bytes_materialized = sum(handle.bytes_materialized for handle in handles)

    started = time.perf_counter()
    _run_img_ffs(blocks, image_hidden)
    dense_img_ff = time.perf_counter() - started

    started = time.perf_counter()
    _run_blocks(blocks, inputs)
    dense_window_compute = time.perf_counter() - started

    del blocks, handles
    _reset_memory()

    mlp_savings = q6_img_ff - dense_img_ff
    return ProbeSample(
        load_seconds=load_seconds,
        q6_img_ff_seconds=q6_img_ff,
        materialize_seconds=materialize,
        dense_img_ff_seconds=dense_img_ff,
        q6_window_compute_seconds=q6_window_compute,
        dense_window_compute_seconds=dense_window_compute,
        bytes_materialized=bytes_materialized,
        mlp_savings_seconds=mlp_savings,
        sync_net_seconds=mlp_savings - materialize,
    )


def run_probe(args: argparse.Namespace) -> PrefetchProbeResult:
    layout = load_transformer_layout(args.model)
    windows = shard_block_windows(layout.plans, layout.ordered_shards)
    if args.window_index >= len(windows):
        raise SystemExit(
            f"--window-index {args.window_index} out of range; "
            f"model has {len(windows)} shard windows"
        )
    window = windows[args.window_index]
    block_indices = window.block_indices
    inputs = make_inputs(args.image_tokens, args.text_tokens)

    for _ in range(args.warmup):
        measure_once(layout=layout, block_indices=block_indices, inputs=inputs)

    samples = tuple(
        measure_once(layout=layout, block_indices=block_indices, inputs=inputs)
        for _ in range(args.runs)
    )

    materialize_median = summarize_durations(
        [sample.materialize_seconds for sample in samples]
    ).median_seconds
    q6_img_ff_median = summarize_durations(
        [sample.q6_img_ff_seconds for sample in samples]
    ).median_seconds
    dense_img_ff_median = summarize_durations(
        [sample.dense_img_ff_seconds for sample in samples]
    ).median_seconds
    q6_window_median = summarize_durations(
        [sample.q6_window_compute_seconds for sample in samples]
    ).median_seconds
    dense_window_median = summarize_durations(
        [sample.dense_window_compute_seconds for sample in samples]
    ).median_seconds
    mlp_savings_median = q6_img_ff_median - dense_img_ff_median
    sync_net_median = mlp_savings_median - materialize_median

    verdict, reason = decide_dense_prefetch_verdict(
        materialize_median=materialize_median,
        q6_img_ff_median=q6_img_ff_median,
        dense_img_ff_median=dense_img_ff_median,
        window_compute_median=q6_window_median,
        min_mlp_savings=args.min_mlp_savings,
        overlap_slack=args.overlap_slack,
    )
    return PrefetchProbeResult(
        mlx_version=version("mlx"),
        platform=platform.platform(),
        model=str(args.model),
        window_index=window.index,
        block_indices=block_indices,
        shards=window.shards,
        image_tokens=args.image_tokens,
        text_tokens=args.text_tokens,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        samples=samples,
        materialize_median=materialize_median,
        q6_img_ff_median=q6_img_ff_median,
        dense_img_ff_median=dense_img_ff_median,
        mlp_savings_median=mlp_savings_median,
        sync_net_median=sync_net_median,
        q6_window_compute_median=q6_window_median,
        dense_window_compute_median=dense_window_median,
        bytes_materialized=samples[-1].bytes_materialized,
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
    parser.add_argument(
        "--window-index",
        type=non_negative_int,
        default=1,
        help="Shard residency window index (default 1 ≈ 8 blocks)",
    )
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument(
        "--min-mlp-savings",
        type=float,
        default=0.02,
        help="Minimum q6-dense img_ff savings (seconds) to consider prefetch",
    )
    parser.add_argument(
        "--overlap-slack",
        type=float,
        default=1.0,
        help="materialize must be <= window_compute * slack to fully hide",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    payload = asdict(result)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(
        f"verdict={result.verdict}\n"
        f"  materialize={result.materialize_median:.3f}s  "
        f"q6_img_ff={result.q6_img_ff_median:.3f}s  "
        f"dense_img_ff={result.dense_img_ff_median:.3f}s\n"
        f"  mlp_savings={result.mlp_savings_median:+.3f}s  "
        f"sync_net={result.sync_net_median:+.3f}s\n"
        f"  q6_window_compute={result.q6_window_compute_median:.3f}s  "
        f"(overlap budget)\n"
        f"  {result.verdict_reason}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
