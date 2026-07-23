#!/usr/bin/env python3
"""Interleaved q6 / q4 / dense-bf16 single-block A/B for dequant overhead diagnosis."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
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

from benchmark_qwen_block import DEFAULT_MODEL, load_block, make_inputs, positive_int
from shardedit_mlx.dense_ab_profile import decide_dense_ab_verdict
from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations


VARIANT_BITS = (("q6", 6), ("q4", 4), ("dense_bf16", 16))


@dataclass(frozen=True)
class VariantTiming:
    name: str
    bits: int
    median_seconds: float
    min_seconds: float
    mean_seconds: float
    durations_seconds: tuple[float, ...]
    peak_memory_gib: float
    relative_to_q6: float | None


@dataclass(frozen=True)
class DenseAbResult:
    mlx_version: str
    platform: str
    model: str
    block_index: int
    image_tokens: int
    text_tokens: int
    warmup_runs: int
    measured_runs: int
    variants: tuple[VariantTiming, ...]
    dense_vs_q6_max_abs_error: float
    dense_vs_q6_all_finite: bool
    q4_vs_q6_max_abs_error: float
    q4_vs_q6_all_finite: bool
    verdict: str
    verdict_reason: str


def array_leaves(value: Any) -> list[mx.array]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in array_leaves(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in array_leaves(item)]
    if value is None:
        return []
    return [value]


def evaluate(value: Any) -> None:
    leaves = array_leaves(value)
    if leaves:
        mx.eval(*leaves)


def time_once(operation: Callable[[], Any]) -> tuple[Any, float]:
    started_at = time.perf_counter()
    output = operation()
    evaluate(output)
    return output, time.perf_counter() - started_at


def max_abs_error(reference: Any, candidate: Any) -> float:
    reference_leaves = array_leaves(reference)
    candidate_leaves = array_leaves(candidate)
    if len(reference_leaves) != len(candidate_leaves):
        raise ValueError("reference and candidate output structures differ")
    errors = [
        mx.max(mx.abs(reference_value - candidate_value))
        for reference_value, candidate_value in zip(
            reference_leaves, candidate_leaves, strict=True
        )
    ]
    mx.eval(*errors)
    return max(float(error.item()) for error in errors)


def all_finite(value: Any) -> bool:
    flags = [mx.all(mx.isfinite(leaf)) for leaf in array_leaves(value)]
    mx.eval(*flags)
    return all(bool(flag.item()) for flag in flags)


def run_block(
    block: Any,
    inputs: dict[str, Any],
    block_index: int,
) -> tuple[mx.array, mx.array]:
    return block(**inputs, block_idx=block_index)


def measure_round_robin(
    operations: Sequence[tuple[str, Callable[[], Any]]],
    *,
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, tuple[Any, tuple[float, ...]]]:
    """Warm up then alternate variants so thermal drift hits each path evenly."""

    last_outputs: dict[str, Any] = {}
    for _ in range(warmup_runs):
        for name, operation in operations:
            last_outputs[name], _ = time_once(operation)

    times: dict[str, list[float]] = {name: [] for name, _ in operations}
    for run_index in range(measured_runs):
        order = list(operations)
        if run_index % 2 == 1:
            order.reverse()
        for name, operation in order:
            last_outputs[name], duration = time_once(operation)
            times[name].append(duration)
    return {
        name: (last_outputs[name], tuple(times[name])) for name, _ in operations
    }


def run_benchmark(args: argparse.Namespace) -> DenseAbResult:
    model_dir = args.model.expanduser().resolve()
    inputs = make_inputs(args.image_tokens, args.text_tokens)

    blocks: dict[str, Any] = {}
    for name, bits in VARIANT_BITS:
        print(f"loading {name} (bits={bits})", file=sys.stderr, flush=True)
        blocks[name] = load_block(model_dir, args.block_index, bits)

    operations = tuple(
        (
            name,
            (lambda block=blocks[name]: run_block(block, inputs, args.block_index)),
        )
        for name, _ in VARIANT_BITS
    )

    mx.reset_peak_memory()
    measured = measure_round_robin(
        operations,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    peak_by_variant: dict[str, float] = {}
    # Re-measure peak memory per variant with a short dedicated pass after the
    # interleaved timing so each path gets an isolated peak reading.
    for name, bits in VARIANT_BITS:
        mx.reset_peak_memory()
        mx.clear_cache()
        _, _ = time_once(lambda block=blocks[name]: run_block(block, inputs, args.block_index))
        peak_by_variant[name] = mx.get_peak_memory() / 1024**3

    q6_output, q6_times = measured["q6"]
    q4_output, q4_times = measured["q4"]
    dense_output, dense_times = measured["dense_bf16"]

    dense_error = max_abs_error(q6_output, dense_output)
    q4_error = max_abs_error(q6_output, q4_output)
    dense_finite = all_finite(dense_output)
    q4_finite = all_finite(q4_output)

    summaries = {
        name: summarize_durations(times)
        for name, (_, times) in measured.items()
    }
    q6_median = summaries["q6"].median_seconds
    variants = tuple(
        VariantTiming(
            name=name,
            bits=bits,
            median_seconds=summaries[name].median_seconds,
            min_seconds=summaries[name].min_seconds,
            mean_seconds=summaries[name].mean_seconds,
            durations_seconds=measured[name][1],
            peak_memory_gib=peak_by_variant[name],
            relative_to_q6=(
                None
                if name == "q6"
                else relative_speedup(q6_median, summaries[name].median_seconds)
            ),
        )
        for name, bits in VARIANT_BITS
    )

    verdict, reason = decide_dense_ab_verdict(
        q6_median=q6_median,
        dense_median=summaries["dense_bf16"].median_seconds,
        dense_vs_q6_max_abs_error=dense_error,
        dense_vs_q6_all_finite=dense_finite,
        speedup_threshold=args.speedup_threshold,
    )
    return DenseAbResult(
        mlx_version=version("mlx"),
        platform=platform.platform(),
        model=str(model_dir),
        block_index=args.block_index,
        image_tokens=args.image_tokens,
        text_tokens=args.text_tokens,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        variants=variants,
        dense_vs_q6_max_abs_error=dense_error,
        dense_vs_q6_all_finite=dense_finite,
        q4_vs_q6_max_abs_error=q4_error,
        q4_vs_q6_all_finite=q4_finite,
        verdict=verdict,
        verdict_reason=reason,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument(
        "--speedup-threshold",
        type=float,
        default=0.15,
        help="Relative speedup gate vs q6 before calling dequant overhead real (default 0.15 = 15%)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.block_index < 0 or args.block_index >= 60:
        parser.error("--block-index must be between 0 and 59")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.speedup_threshold <= 0:
        parser.error("--speedup-threshold must be positive")

    result = run_benchmark(args)
    payload = json.dumps(asdict(result), indent=2) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(
        f"verdict={result.verdict} dense/q6={next(v.relative_to_q6 for v in result.variants if v.name == 'dense_bf16'):.3f}x "
        f"dense_err={result.dense_vs_q6_max_abs_error:.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
