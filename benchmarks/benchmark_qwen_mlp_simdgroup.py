#!/usr/bin/env python3
"""A/B: eager q6 vs simdgroup-MMA q6 vs dense for image mlp_in (single shape)."""

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


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed
from benchmark_qwen_block_dense_ab import all_finite, max_abs_error, measure_round_robin
from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations
from shardedit_mlx.q6_metal_mlp import dequantize_linear, quantized_linear_spec
from shardedit_mlx.q6_simdgroup_mlp import (
    affine_q6_qmm_t_simdgroup,
    make_simdgroup_mlp_in_callables,
)
from shardedit_mlx.simdgroup_tiled_profile import (
    decide_simdgroup_verdict,
    estimate_e2e_wallclock_improvement,
)


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


@dataclass(frozen=True)
class SimdgroupResult:
    mlx_version: str
    platform: str
    model: str
    block_index: int
    image_tokens: int
    in_features: int
    out_features: int
    warmup_runs: int
    measured_runs: int
    simdgroup_f32_vs_dense_max_abs_error: float
    gemm_fraction_of_block: float
    projected_e2e_improvement: float
    paths: tuple[PathTiming, ...]
    verdict: str
    verdict_reason: str


def make_random_hidden(tokens: int, dim: int, *, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)


def simdgroup_f32_error(ff: Any, tokens: int, *, seed: int) -> float:
    mx.random.seed(seed)
    mlp_in = quantized_linear_spec(ff.mlp_in)
    hidden = mx.random.normal((1, tokens, mlp_in.in_features)).astype(mx.float32)
    dense = dequantize_linear(mlp_in, dtype=mx.float32)
    metal = affine_q6_qmm_t_simdgroup(hidden, mlp_in, dtype=mx.float32)
    ref = hidden @ dense.weight.T
    if dense.bias is not None:
        ref = ref + dense.bias
    mx.eval(metal, ref)
    return float(mx.max(mx.abs(metal - ref)).item())


def run_benchmark(args: argparse.Namespace) -> SimdgroupResult:
    block = load_block(args.model, args.block_index, bits=6)
    ff = block.img_ff
    eager_fn, simd_fn, dense_fn = make_simdgroup_mlp_in_callables(ff)
    mlp_in = quantized_linear_spec(ff.mlp_in)
    hidden = make_random_hidden(args.image_tokens, mlp_in.in_features, seed=args.seed)
    f32_err = simdgroup_f32_error(ff, min(args.image_tokens, 32), seed=args.seed + 1)

    measured = measure_round_robin(
        (
            ("eager_q6", lambda: eager_fn(hidden)),
            ("simdgroup_metal", lambda: simd_fn(hidden)),
            ("dense_predequant", lambda: dense_fn(hidden)),
        ),
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
            all_finite=all_finite({"out": eager_out}),
        )
    )

    for name in ("simdgroup_metal", "dense_predequant"):
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
                max_abs_error_vs_eager=max_abs_error({"out": eager_out}, {"out": out}),
                all_finite=all_finite({"out": out}),
            )
        )

    by_name = {path.name: path for path in paths}
    simd = by_name["simdgroup_metal"]
    dense = by_name["dense_predequant"]
    projected = estimate_e2e_wallclock_improvement(
        gemm_speedup=max(simd.relative_to_eager or 1e-9, 1e-9),
        gemm_fraction_of_block=args.gemm_fraction_of_block,
    )
    verdict, reason = decide_simdgroup_verdict(
        eager_median=eager_summary.median_seconds,
        simdgroup_median=simd.median_seconds,
        dense_median=dense.median_seconds,
        simdgroup_f32_vs_dense_max_abs_error=f32_err,
        simdgroup_all_finite=simd.all_finite,
        gemm_fraction_of_block=args.gemm_fraction_of_block,
        min_e2e_improvement=args.min_e2e_improvement,
        f32_error_tolerance=args.error_tolerance,
    )
    return SimdgroupResult(
        mlx_version=version("mlx"),
        platform=platform.platform(),
        model=str(args.model),
        block_index=args.block_index,
        image_tokens=args.image_tokens,
        in_features=mlp_in.in_features,
        out_features=mlp_in.out_features,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        simdgroup_f32_vs_dense_max_abs_error=f32_err,
        gemm_fraction_of_block=args.gemm_fraction_of_block,
        projected_e2e_improvement=projected,
        paths=tuple(paths),
        verdict=verdict,
        verdict_reason=reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-index", type=non_negative_int, default=0)
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gemm-fraction-of-block",
        type=float,
        default=0.255,
        help="mlp_in share of block compute (~half of image MLP ≈51%)",
    )
    parser.add_argument(
        "--min-e2e-improvement",
        type=float,
        default=0.10,
        help="Go gate: projected F1B2 wall-clock improvement",
    )
    parser.add_argument("--error-tolerance", type=float, default=1e-2)
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
    by_name = {path.name: path for path in result.paths}
    print(
        f"verdict={result.verdict} "
        f"eager={by_name['eager_q6'].median_seconds:.4f}s "
        f"simd={by_name['simdgroup_metal'].median_seconds:.4f}s"
        f"({by_name['simdgroup_metal'].relative_to_eager:.3f}x) "
        f"dense={by_name['dense_predequant'].median_seconds:.4f}s "
        f"projected_e2e={result.projected_e2e_improvement:.1%} "
        f"f32_err={result.simdgroup_f32_vs_dense_max_abs_error:.3g}\n"
        f"  {result.verdict_reason}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
