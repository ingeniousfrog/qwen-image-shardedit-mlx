#!/usr/bin/env python3
"""Benchmark custom Metal fused q6 image MLP vs eager q6 and dense upper bound."""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
from benchmark_qwen_block_dense_ab import (
    all_finite,
    max_abs_error,
    measure_round_robin,
)
from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations
from shardedit_mlx.mlp_metal_profile import decide_mlp_metal_verdict
from shardedit_mlx.q6_metal_mlp import (
    affine_q6_qmm_t,
    dense_mlp,
    dequantize_linear,
    make_feed_forward_callables,
    quantized_linear_spec,
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
    tokens: int


@dataclass(frozen=True)
class MlpMetalResult:
    mlx_version: str
    platform: str
    model: str
    block_index: int
    bits: int
    group_size: int
    image_tokens: int
    metal_tokens: int
    warmup_runs: int
    measured_runs: int
    f32_error_tolerance: float
    bf16_error_tolerance: float
    metal_f32_vs_dense_max_abs_error: float | None
    metal_bf16_vs_eager_max_abs_error: float | None
    paths: tuple[PathTiming, ...]
    verdict: str
    verdict_reason: str
    metal_kernel: str


def make_random_hidden(tokens: int, dim: int = 3072, *, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)


def metal_f32_vs_dense_error(
    ff: Any,
    tokens: int,
    *,
    seed: int,
) -> float:
    """Primary packing/math gate: float32 fused Metal vs pre-dequant dense."""

    mx.random.seed(seed)
    hidden = mx.random.normal((1, tokens, 3072)).astype(mx.float32)
    mlp_in = quantized_linear_spec(ff.mlp_in)
    mlp_out = quantized_linear_spec(ff.mlp_out)
    dense_in = dequantize_linear(mlp_in, dtype=mx.float32)
    dense_out = dequantize_linear(mlp_out, dtype=mx.float32)
    metal = affine_q6_qmm_t(
        affine_q6_qmm_t(hidden, mlp_in, apply_gelu=True, dtype=mx.float32),
        mlp_out,
        apply_gelu=False,
        dtype=mx.float32,
    )
    dense = dense_mlp(hidden, dense_in, dense_out)
    mx.eval(metal, dense)
    return float(mx.max(mx.abs(metal - dense)).item())


def run_benchmark(args: argparse.Namespace) -> MlpMetalResult:
    block = load_block(args.model, args.block_index, bits=args.bits)
    ff = block.img_ff
    eager_fn, metal_fn, dense_fn = make_feed_forward_callables(ff)

    hidden_full = make_random_hidden(args.image_tokens, seed=args.seed)
    hidden_metal = make_random_hidden(args.metal_tokens, seed=args.seed + 1)

    paths: list[PathTiming] = []

    # Speed A/B: eager + dense at full image tokens (pre-dequant dense is fair upper bound).
    measured_full = measure_round_robin(
        (
            ("eager_q6", lambda: eager_fn(hidden_full)),
            ("dense_predequant", lambda: dense_fn(hidden_full)),
        ),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    eager_output, eager_times = measured_full["eager_q6"]
    dense_output, dense_times = measured_full["dense_predequant"]
    eager_summary = summarize_durations(eager_times)
    dense_summary = summarize_durations(dense_times)
    eager_path = PathTiming(
        name="eager_q6",
        median_seconds=eager_summary.median_seconds,
        min_seconds=eager_summary.min_seconds,
        mean_seconds=eager_summary.mean_seconds,
        durations_seconds=tuple(eager_times),
        relative_to_eager=1.0,
        max_abs_error_vs_eager=0.0,
        all_finite=all_finite(eager_output),
        tokens=args.image_tokens,
    )
    dense_path = PathTiming(
        name="dense_predequant",
        median_seconds=dense_summary.median_seconds,
        min_seconds=dense_summary.min_seconds,
        mean_seconds=dense_summary.mean_seconds,
        durations_seconds=tuple(dense_times),
        relative_to_eager=relative_speedup(
            eager_path.median_seconds, dense_summary.median_seconds
        ),
        max_abs_error_vs_eager=max_abs_error(eager_output, dense_output),
        all_finite=all_finite(dense_output),
        tokens=args.image_tokens,
    )
    paths.extend([eager_path, dense_path])

    metal_path: PathTiming | None = None
    metal_eager_median = eager_path.median_seconds
    metal_f32_err: float | None = None
    metal_bf16_err: float | None = None
    if not args.skip_metal_timing:
        metal_f32_err = metal_f32_vs_dense_error(
            ff, args.metal_tokens, seed=args.seed + 2
        )
        # Naive Metal is O(MNK); time at --metal-tokens against the same-shape eager.
        measured_metal = measure_round_robin(
            (
                ("eager_q6_metal_tokens", lambda: eager_fn(hidden_metal)),
                ("metal_fused", lambda: metal_fn(hidden_metal)),
            ),
            warmup_runs=args.warmup,
            measured_runs=args.runs,
        )
        eager_m_out, eager_m_times = measured_metal["eager_q6_metal_tokens"]
        metal_m_out, metal_m_times = measured_metal["metal_fused"]
        eager_m_summary = summarize_durations(eager_m_times)
        metal_summary = summarize_durations(metal_m_times)
        metal_eager_median = eager_m_summary.median_seconds
        metal_bf16_err = max_abs_error(eager_m_out, metal_m_out)
        paths.append(
            PathTiming(
                name="eager_q6_metal_tokens",
                median_seconds=eager_m_summary.median_seconds,
                min_seconds=eager_m_summary.min_seconds,
                mean_seconds=eager_m_summary.mean_seconds,
                durations_seconds=tuple(eager_m_times),
                relative_to_eager=1.0,
                max_abs_error_vs_eager=0.0,
                all_finite=all_finite(eager_m_out),
                tokens=args.metal_tokens,
            )
        )
        metal_path = PathTiming(
            name="metal_fused",
            median_seconds=metal_summary.median_seconds,
            min_seconds=metal_summary.min_seconds,
            mean_seconds=metal_summary.mean_seconds,
            durations_seconds=tuple(metal_m_times),
            relative_to_eager=relative_speedup(
                eager_m_summary.median_seconds, metal_summary.median_seconds
            ),
            max_abs_error_vs_eager=metal_bf16_err,
            all_finite=all_finite(metal_m_out),
            tokens=args.metal_tokens,
        )
        paths.append(metal_path)

    verdict, reason = decide_mlp_metal_verdict(
        dense_eager_median=eager_path.median_seconds,
        dense_median=dense_path.median_seconds,
        dense_max_abs_error=dense_path.max_abs_error_vs_eager or 0.0,
        dense_all_finite=dense_path.all_finite,
        metal_eager_median=None if metal_path is None else metal_eager_median,
        metal_median=None if metal_path is None else metal_path.median_seconds,
        metal_f32_vs_dense_max_abs_error=metal_f32_err,
        metal_bf16_vs_eager_max_abs_error=metal_bf16_err,
        metal_all_finite=None if metal_path is None else metal_path.all_finite,
        speedup_threshold=args.speedup_threshold,
        f32_error_tolerance=args.f32_error_tolerance,
        bf16_error_tolerance=args.bf16_error_tolerance,
    )

    return MlpMetalResult(
        mlx_version=version("mlx"),
        platform=platform.platform(),
        model=str(args.model),
        block_index=args.block_index,
        bits=args.bits,
        group_size=int(ff.mlp_in.group_size),
        image_tokens=args.image_tokens,
        metal_tokens=args.metal_tokens,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        f32_error_tolerance=args.f32_error_tolerance,
        bf16_error_tolerance=args.bf16_error_tolerance,
        metal_f32_vs_dense_max_abs_error=metal_f32_err,
        metal_bf16_vs_eager_max_abs_error=metal_bf16_err,
        paths=tuple(paths),
        verdict=verdict,
        verdict_reason=reason,
        metal_kernel="shardedit_mlx_affine_q6_qmm_t+gelu_epilogue",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--block-index", type=positive_int, default=0)
    parser.add_argument("--bits", type=int, default=6, choices=[4, 6, 8])
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument(
        "--metal-tokens",
        type=positive_int,
        default=32,
        help="Token count for naive Metal timing/correctness (O(MNK); keep small)",
    )
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speedup-threshold", type=float, default=0.05)
    parser.add_argument("--f32-error-tolerance", type=float, default=1e-2)
    parser.add_argument("--bf16-error-tolerance", type=float, default=32.0)
    parser.add_argument(
        "--skip-metal-timing",
        action="store_true",
        help="Only measure eager/dense upper bound (skip slow naive Metal timing)",
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
        f"verdict={result.verdict} "
        + " ".join(
            f"{p.name}@{p.tokens}tok={p.median_seconds:.4f}s"
            + (
                f"({p.relative_to_eager:.3f}x)"
                if p.relative_to_eager is not None and p.name != "eager_q6"
                else ""
            )
            for p in result.paths
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
