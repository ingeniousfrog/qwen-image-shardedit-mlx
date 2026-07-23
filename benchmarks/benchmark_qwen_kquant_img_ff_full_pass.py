#!/usr/bin/env python3
"""Full-pass img_ff-only K-quant conversion-cache spike across 60 blocks."""

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
from shardedit_mlx.benchmark_lora import apply_loras_to_loaded_blocks, normalize_lora_args
from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations
from shardedit_mlx.kquant_img_ff_window import (
    KQuantFeedForward,
    import_mlx_kquant,
    kquant_cache_bytes,
    prepare_kquant_img_ff_window,
)
from shardedit_mlx.qwen_block_loader import load_block_window, load_transformer_layout
from shardedit_mlx.residency_plan import shard_block_windows


@dataclass(frozen=True)
class PassTiming:
    pass_index: int
    q6_img_ff_seconds: float
    kquant_prepare_seconds: float
    kquant_img_ff_seconds: float
    load_seconds: float
    lora_seconds: float
    lora_selected_keys: int
    lora_applied_layers: int
    cache_hits: int
    cache_misses: int
    cache_blocks: int
    cache_bytes: int
    active_after_prepare_gib: float
    peak_gib: float
    max_abs_error_vs_q6: float
    all_finite: bool


@dataclass(frozen=True)
class AggregateTiming:
    q6_img_ff_median: float
    kquant_warm_img_ff_median: float
    kquant_cold_prepare_seconds: float
    kquant_warm_prepare_median: float
    relative_to_q6_warm: float
    saved_seconds_per_full_pass: float
    projected_saved_seconds: float
    max_abs_error_vs_q6: float
    all_finite: bool
    max_cache_bytes: int
    max_peak_gib: float


@dataclass(frozen=True)
class KQuantFullPassResult:
    mlx_version: str
    mlx_kquant_version: str | None
    platform: str
    model: str
    block_count: int
    window_count: int
    image_tokens: int
    text_tokens: int
    lora_paths: tuple[str, ...]
    lora_scales: tuple[float, ...]
    codec: str
    passes: int
    full_miss_steps: int
    pass_timings: tuple[PassTiming, ...]
    aggregate: AggregateTiming
    verdict: str
    verdict_reason: str


def _reset_memory() -> None:
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def _run_img_ffs(blocks: Any, image_hidden: mx.array) -> Any:
    outs = tuple(loaded.module.img_ff(image_hidden) for loaded in blocks)
    mx.eval(*outs)
    return outs


def _window_error(reference: Any, candidate: Any) -> float:
    return max_abs_error(reference, candidate)


def measure_full_pass(
    *,
    layout: Any,
    windows: Any,
    image_hidden: mx.array,
    cache: dict[tuple[int, str], KQuantFeedForward],
    codec: str,
    cache_max_blocks: int,
    lora_paths: tuple[str, ...],
    lora_scales: tuple[float, ...],
    block_count: int,
    pass_index: int,
    kquant: Any,
) -> PassTiming:
    _reset_memory()
    q6_seconds = 0.0
    kquant_seconds = 0.0
    prepare_seconds = 0.0
    load_seconds = 0.0
    lora_seconds = 0.0
    lora_selected_keys = 0
    lora_applied_layers = 0
    cache_hits = 0
    cache_misses = 0
    max_error = 0.0
    finite = True
    active_after_prepare = 0.0

    for window in windows:
        load_started = time.perf_counter()
        blocks = load_block_window(layout, window.block_indices)
        load_seconds += time.perf_counter() - load_started

        lora_started = time.perf_counter()
        lora_results = apply_loras_to_loaded_blocks(
            blocks,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            block_count=block_count,
        )
        lora_seconds += time.perf_counter() - lora_started
        lora_selected_keys += sum(result.selected_keys for result in lora_results)
        lora_applied_layers += sum(result.applied_layers for result in lora_results)

        started = time.perf_counter()
        q6_out = _run_img_ffs(blocks, image_hidden)
        q6_seconds += time.perf_counter() - started

        started = time.perf_counter()
        handles = prepare_kquant_img_ff_window(
            blocks,
            codec=codec,
            kquant=kquant,
            cache=cache,
            cache_max_blocks=cache_max_blocks,
        )
        prepare_seconds += time.perf_counter() - started
        cache_hits += sum(1 for handle in handles if handle.cache_hit)
        cache_misses += sum(1 for handle in handles if not handle.cache_hit)
        active_after_prepare = max(active_after_prepare, mx.get_active_memory() / 1024**3)

        started = time.perf_counter()
        kquant_out = _run_img_ffs(blocks, image_hidden)
        kquant_seconds += time.perf_counter() - started
        max_error = max(max_error, _window_error(q6_out, kquant_out))
        finite = finite and all_finite(kquant_out)

        del blocks, handles, q6_out, kquant_out
        gc.collect()

    return PassTiming(
        pass_index=pass_index,
        q6_img_ff_seconds=q6_seconds,
        kquant_prepare_seconds=prepare_seconds,
        kquant_img_ff_seconds=kquant_seconds,
        load_seconds=load_seconds,
        lora_seconds=lora_seconds,
        lora_selected_keys=lora_selected_keys,
        lora_applied_layers=lora_applied_layers,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_blocks=len(cache),
        cache_bytes=kquant_cache_bytes(cache),
        active_after_prepare_gib=active_after_prepare,
        peak_gib=mx.get_peak_memory() / 1024**3,
        max_abs_error_vs_q6=max_error,
        all_finite=finite,
    )


def decide_verdict(
    *,
    aggregate: AggregateTiming,
    speedup_threshold: float,
    error_tolerance: float,
) -> tuple[str, str]:
    gate = 1.0 + speedup_threshold
    if not aggregate.all_finite:
        return "invalid_kquant_output", "K-quant img_ff produced non-finite output"
    if aggregate.max_abs_error_vs_q6 > error_tolerance:
        return (
            "invalid_kquant_output",
            (
                f"K-quant img_ff max abs error {aggregate.max_abs_error_vs_q6:.6g} "
                f"exceeds tolerance {error_tolerance:.6g}"
            ),
        )
    if aggregate.relative_to_q6_warm >= gate:
        return (
            "kquant_img_ff_full_pass_promising",
            (
                f"warm K-quant img_ff is {aggregate.relative_to_q6_warm:.3f}x "
                f"faster than MLX q6 (gate {gate:.2f}x), saving "
                f"{aggregate.saved_seconds_per_full_pass:.3f}s per 60-block "
                "img_ff pass before full Transformer/e2e overhead."
            ),
        )
    return (
        "kquant_img_ff_full_pass_not_enough",
        (
            f"warm K-quant img_ff speedup {aggregate.relative_to_q6_warm:.3f}x "
            f"is below the {gate:.2f}x gate."
        ),
    )


def run_benchmark(args: argparse.Namespace) -> KQuantFullPassResult:
    layout = load_transformer_layout(args.model)
    windows = shard_block_windows(layout.plans, layout.ordered_shards)
    inputs = make_inputs(args.image_tokens, args.text_tokens)
    image_hidden = inputs["hidden_states"]
    lora_paths, lora_scales = normalize_lora_args(args.lora_paths, args.lora_scales)
    kquant = import_mlx_kquant()
    try:
        kquant_version = version("mlx-kquant")
    except Exception:
        kquant_version = None
    cache: dict[tuple[int, str], KQuantFeedForward] = {}
    timings = tuple(
        measure_full_pass(
            layout=layout,
            windows=windows,
            image_hidden=image_hidden,
            cache=cache,
            codec=args.codec,
            cache_max_blocks=args.cache_max_blocks,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            block_count=len(layout.plans),
            pass_index=pass_index,
            kquant=kquant,
        )
        for pass_index in range(args.passes)
    )
    warm_timings = timings[1:] if len(timings) > 1 else timings
    q6_summary = summarize_durations(
        [timing.q6_img_ff_seconds for timing in warm_timings]
    )
    kquant_summary = summarize_durations(
        [timing.kquant_img_ff_seconds for timing in warm_timings]
    )
    warm_prepare_summary = summarize_durations(
        [max(timing.kquant_prepare_seconds, 1e-9) for timing in warm_timings]
    )
    relative = relative_speedup(
        q6_summary.median_seconds,
        kquant_summary.median_seconds,
    )
    saved = q6_summary.median_seconds - kquant_summary.median_seconds
    aggregate = AggregateTiming(
        q6_img_ff_median=q6_summary.median_seconds,
        kquant_warm_img_ff_median=kquant_summary.median_seconds,
        kquant_cold_prepare_seconds=timings[0].kquant_prepare_seconds,
        kquant_warm_prepare_median=warm_prepare_summary.median_seconds,
        relative_to_q6_warm=relative,
        saved_seconds_per_full_pass=saved,
        projected_saved_seconds=saved * args.full_miss_steps,
        max_abs_error_vs_q6=max(timing.max_abs_error_vs_q6 for timing in timings),
        all_finite=all(timing.all_finite for timing in timings),
        max_cache_bytes=max(timing.cache_bytes for timing in timings),
        max_peak_gib=max(timing.peak_gib for timing in timings),
    )
    verdict, reason = decide_verdict(
        aggregate=aggregate,
        speedup_threshold=args.speedup_threshold,
        error_tolerance=args.error_tolerance,
    )
    return KQuantFullPassResult(
        mlx_version=version("mlx"),
        mlx_kquant_version=kquant_version,
        platform=platform.platform(),
        model=str(args.model),
        block_count=len(layout.plans),
        window_count=len(windows),
        image_tokens=args.image_tokens,
        text_tokens=args.text_tokens,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        codec=args.codec,
        passes=args.passes,
        full_miss_steps=args.full_miss_steps,
        pass_timings=timings,
        aggregate=aggregate,
        verdict=verdict,
        verdict_reason=reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image-tokens", type=positive_int, default=3456)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--codec", default="q6_k")
    parser.add_argument("--lora-paths", nargs="*", default=(), help="Optional LoRA paths")
    parser.add_argument("--lora-scales", nargs="*", type=float, default=(), help="Optional LoRA scales")
    parser.add_argument("--passes", type=positive_int, default=2)
    parser.add_argument("--cache-max-blocks", type=positive_int, default=60)
    parser.add_argument("--full-miss-steps", type=positive_int, default=5)
    parser.add_argument("--speedup-threshold", type=float, default=0.10)
    parser.add_argument("--error-tolerance", type=float, default=128.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.speedup_threshold < 0:
        parser.error("--speedup-threshold cannot be negative")
    if args.error_tolerance < 0:
        parser.error("--error-tolerance cannot be negative")
    try:
        normalize_lora_args(args.lora_paths, args.lora_scales)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    payload = asdict(result)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    aggregate = result.aggregate
    print(
        f"verdict={result.verdict} "
        f"q6={aggregate.q6_img_ff_median:.3f}s "
        f"kquant={aggregate.kquant_warm_img_ff_median:.3f}s "
        f"({aggregate.relative_to_q6_warm:.3f}x) "
        f"saved={aggregate.saved_seconds_per_full_pass:.3f}s/full-pass "
        f"projected={aggregate.projected_saved_seconds:.3f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
