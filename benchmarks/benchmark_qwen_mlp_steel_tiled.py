#!/usr/bin/env python3
"""A/B: eager q6 vs tiled Metal q6 vs naive Metal vs dense image MLP."""

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

from benchmark_qwen_block import DEFAULT_MODEL, load_block, positive_int
from benchmark_qwen_block_dense_ab import all_finite, max_abs_error, measure_round_robin
from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations
from shardedit_mlx.q6_metal_mlp import (
    dequantize_linear,
    fused_q6_mlp,
    make_feed_forward_callables,
    quantized_linear_spec,
)
from shardedit_mlx.q6_steel_mlp import affine_q6_qmm_t_tiled, make_tiled_feed_forward_callables
from shardedit_mlx.steel_tiled_profile import decide_steel_tiled_verdict


@dataclass(frozen=True)
class PathTiming:
    name: str
    median_seconds: float
    min_seconds: float
    mean_seconds: float
    durations_seconds: tuple[float, ...]
    relative_to_eager: float | None
    max_abs_error_vs_eager: float | None
    all_finite: bool
    tokens: int


@dataclass(frozen=True)
class SteelTiledResult:
    mlx_version: str
    platform: str
    model: str
    block_index: int
    image_tokens: int
    warmup_runs: int
    measured_runs: int
    tiled_f32_vs_dense_max_abs_error: float
    paths: tuple[PathTiming, ...]
    verdict: str
    verdict_reason: str


def make_random_hidden(tokens: int, dim: int = 3072, *, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)


def tiled_f32_error(ff: Any, tokens: int, *, seed: int) -> float:
    mx.random.seed(seed)
    hidden = mx.random.normal((1, tokens, 3072)).astype(mx.float32)
    mlp_in = quantized_linear_spec(ff.mlp_in)
    mlp_out = quantized_linear_spec(ff.mlp_out)
    dense_in = dequantize_linear(mlp_in, dtype=mx.float32)
    dense_out = dequantize_linear(mlp_out, dtype=mx.float32)
    from shardedit_mlx.q6_metal_mlp import dense_mlp

    metal = affine_q6_qmm_t_tiled(
        affine_q6_qmm_t_tiled(hidden, mlp_in, apply_gelu=True, dtype=mx.float32),
        mlp_out,
        apply_gelu=False,
        dtype=mx.float32,
    )
    dense = dense_mlp(hidden, dense_in, dense_out)
    mx.eval(metal, dense)
    return float(mx.max(mx.abs(metal - dense)).item())


def run_benchmark(args: argparse.Namespace) -> SteelTiledResult:
    block = load_block(args.model, args.block_index, bits=6)
    ff = block.img_ff
    eager_fn, _naive_unused, dense_fn = make_feed_forward_callables(ff)
    _, tiled_fn, _ = make_tiled_feed_forward_callables(ff)
    mlp_in = quantized_linear_spec(ff.mlp_in)
    mlp_out = quantized_linear_spec(ff.mlp_out)

    def naive_fn(hidden: mx.array) -> mx.array:
        return fused_q6_mlp(hidden, mlp_in, mlp_out)

    hidden = make_random_hidden(args.image_tokens, seed=args.seed)
    f32_err = tiled_f32_error(ff, min(args.image_tokens, 32), seed=args.seed + 1)

    ops: list[tuple[str, Any]] = [
        ("eager_q6", lambda: eager_fn(hidden)),
        ("tiled_metal", lambda: tiled_fn(hidden)),
        ("dense_predequant", lambda: dense_fn(hidden)),
    ]
    include_naive = args.image_tokens <= 64 and not args.skip_naive
    if include_naive:
        ops.insert(2, ("naive_metal", lambda: naive_fn(hidden)))

    measured = measure_round_robin(
        tuple(ops),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    paths: list[PathTiming] = []
    eager_out, eager_times = measured["eager_q6"]
    eager_summary = summarize_durations(eager_times)
    paths.append(
        PathTiming(
            name="eager_q6",
            median_seconds=eager_summary.median_seconds,
            min_seconds=eager_summary.min_seconds,
            mean_seconds=eager_summary.mean_seconds,
            durations_seconds=tuple(eager_times),
            relative_to_eager=1.0,
            max_abs_error_vs_eager=0.0,
            all_finite=all_finite(eager_out),
            tokens=args.image_tokens,
        )
    )
    ordered = ["tiled_metal"]
    if include_naive:
        ordered.append("naive_metal")
    ordered.append("dense_predequant")
    for name in ordered:
        out, times = measured[name]
        summary = summarize_durations(times)
        paths.append(
            PathTiming(
                name=name,
                median_seconds=summary.median_seconds,
                min_seconds=summary.min_seconds,
                mean_seconds=summary.mean_seconds,
                durations_seconds=tuple(times),
                relative_to_eager=relative_speedup(
                    eager_summary.median_seconds, summary.median_seconds
                ),
                max_abs_error_vs_eager=max_abs_error(eager_out, out),
                all_finite=all_finite(out),
                tokens=args.image_tokens,
            )
        )

    by_name = {path.name: path for path in paths}
    verdict, reason = decide_steel_tiled_verdict(
        eager_median=by_name["eager_q6"].median_seconds,
        tiled_median=by_name["tiled_metal"].median_seconds,
        naive_median=(
            None if "naive_metal" not in by_name else by_name["naive_metal"].median_seconds
        ),
        dense_median=by_name["dense_predequant"].median_seconds,
        tiled_f32_vs_dense_max_abs_error=f32_err,
        tiled_all_finite=by_name["tiled_metal"].all_finite,
        speedup_threshold=args.speedup_threshold,
        f32_error_tolerance=args.f32_error_tolerance,
    )
    return SteelTiledResult(
        mlx_version=version("mlx"),
        platform=platform.platform(),
        model=str(args.model),
        block_index=args.block_index,
        image_tokens=args.image_tokens,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        tiled_f32_vs_dense_max_abs_error=f32_err,
        paths=tuple(paths),
        verdict=verdict,
        verdict_reason=reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-index", type=positive_int, default=0)
    parser.add_argument("--image-tokens", type=positive_int, default=256)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speedup-threshold", type=float, default=0.05)
    parser.add_argument("--f32-error-tolerance", type=float, default=1e-2)
    parser.add_argument(
        "--skip-naive",
        action="store_true",
        help="Skip naive Metal timing even when image-tokens <= 64",
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
    print(
        f"verdict={result.verdict} f32_err={result.tiled_f32_vs_dense_max_abs_error:.4g} "
        + " ".join(
            f"{p.name}={p.median_seconds:.4f}s"
            + (f"({p.relative_to_eager:.3f}x)" if p.relative_to_eager else "")
            for p in result.paths
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
