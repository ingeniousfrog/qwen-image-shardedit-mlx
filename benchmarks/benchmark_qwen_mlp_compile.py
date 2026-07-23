#!/usr/bin/env python3
"""Benchmark scoped mx.compile on q6 image/text MLP (no attention/mask)."""

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
from mlx import nn

from benchmark_qwen_block import DEFAULT_MODEL, load_block, positive_int
from benchmark_qwen_block_dense_ab import (
    all_finite,
    max_abs_error,
    measure_round_robin,
    time_once,
)
from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations
from shardedit_mlx.mlp_compile_profile import decide_mlp_compile_verdict


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
class MlpCompileResult:
    mlx_version: str
    platform: str
    model: str
    block_index: int
    bits: int
    image_tokens: int
    text_tokens: int
    warmup_runs: int
    measured_runs: int
    image_mlp: tuple[PathTiming, ...]
    text_mlp: tuple[PathTiming, ...]
    whole_block_mask_none: tuple[PathTiming, ...] | None
    image_mlp_verdict: str
    image_mlp_verdict_reason: str
    proceed_to_group_size_sweep: bool


def make_random_hidden(tokens: int, dim: int = 3072, *, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)


def make_block_inputs(image_tokens: int, text_tokens: int, *, seed: int) -> dict[str, Any]:
    mx.random.seed(seed)
    dtype = mx.bfloat16
    return {
        "hidden_states": mx.random.normal((1, image_tokens, 3072)).astype(dtype),
        "encoder_hidden_states": mx.random.normal((1, text_tokens, 3072)).astype(dtype),
        "encoder_hidden_states_mask": None,
        "text_embeddings": mx.random.normal((1, 3072)).astype(dtype),
        "image_rotary_emb": (
            (
                mx.ones((image_tokens, 64), dtype=mx.float32),
                mx.zeros((image_tokens, 64), dtype=mx.float32),
            ),
            (
                mx.ones((text_tokens, 64), dtype=mx.float32),
                mx.zeros((text_tokens, 64), dtype=mx.float32),
            ),
        ),
    }


def feed_forward_call(ff: Any, hidden: mx.array) -> mx.array:
    hidden = ff.mlp_in(hidden)
    hidden = nn.gelu_approx(hidden)
    hidden = ff.mlp_out(hidden)
    return hidden


def compile_feed_forward(ff: Any) -> Callable[[mx.array], mx.array]:
    def function(hidden: mx.array) -> mx.array:
        return feed_forward_call(ff, hidden)

    return mx.compile(function)


def timing_pair(
    *,
    label: str,
    eager: Callable[[], Any],
    compiled: Callable[[], Any],
    warmup_runs: int,
    measured_runs: int,
) -> tuple[PathTiming, PathTiming]:
    measured = measure_round_robin(
        (("eager", eager), ("compiled", compiled)),
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    eager_output, eager_times = measured["eager"]
    compiled_output, compiled_times = measured["compiled"]
    eager_summary = summarize_durations(eager_times)
    compiled_summary = summarize_durations(compiled_times)
    error = max_abs_error(eager_output, compiled_output)
    finite = all_finite(compiled_output)
    return (
        PathTiming(
            name=f"{label}_eager",
            median_seconds=eager_summary.median_seconds,
            min_seconds=eager_summary.min_seconds,
            mean_seconds=eager_summary.mean_seconds,
            durations_seconds=eager_times,
            relative_to_eager=None,
            max_abs_error_vs_eager=None,
            all_finite=all_finite(eager_output),
        ),
        PathTiming(
            name=f"{label}_compiled",
            median_seconds=compiled_summary.median_seconds,
            min_seconds=compiled_summary.min_seconds,
            mean_seconds=compiled_summary.mean_seconds,
            durations_seconds=compiled_times,
            relative_to_eager=relative_speedup(
                eager_summary.median_seconds,
                compiled_summary.median_seconds,
            ),
            max_abs_error_vs_eager=error,
            all_finite=finite,
        ),
    )


def measure_whole_block_mask_none(
    block: Any,
    inputs: dict[str, Any],
    *,
    block_index: int,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[PathTiming, PathTiming]:
    rotary = inputs["image_rotary_emb"]

    def eager() -> tuple[mx.array, mx.array]:
        return block(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            encoder_hidden_states_mask=None,
            text_embeddings=inputs["text_embeddings"],
            image_rotary_emb=rotary,
            block_idx=block_index,
        )

    def block_function(
        image_hidden: mx.array,
        text_hidden: mx.array,
        embeddings: mx.array,
        image_cos: mx.array,
        image_sin: mx.array,
        text_cos: mx.array,
        text_sin: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return block(
            hidden_states=image_hidden,
            encoder_hidden_states=text_hidden,
            encoder_hidden_states_mask=None,
            text_embeddings=embeddings,
            image_rotary_emb=((image_cos, image_sin), (text_cos, text_sin)),
            block_idx=block_index,
        )

    compiled_function = mx.compile(block_function)

    def compiled() -> tuple[mx.array, mx.array]:
        return compiled_function(
            inputs["hidden_states"],
            inputs["encoder_hidden_states"],
            inputs["text_embeddings"],
            rotary[0][0],
            rotary[0][1],
            rotary[1][0],
            rotary[1][1],
        )

    return timing_pair(
        label="whole_block_mask_none",
        eager=eager,
        compiled=compiled,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )


def run_benchmark(args: argparse.Namespace) -> MlpCompileResult:
    model_dir = args.model.expanduser().resolve()
    print(f"loading q6 block {args.block_index}", file=sys.stderr, flush=True)
    block = load_block(model_dir, args.block_index, args.bits)

    image_hidden = make_random_hidden(args.image_tokens, seed=args.seed)
    text_hidden = make_random_hidden(args.text_tokens, seed=args.seed + 1)
    block_inputs = make_block_inputs(
        args.image_tokens,
        args.text_tokens,
        seed=args.seed + 2,
    )

    # Touch once so first-compile cost is outside the measured window.
    compiled_img = compile_feed_forward(block.img_ff)
    compiled_txt = compile_feed_forward(block.txt_ff)
    time_once(lambda: compiled_img(image_hidden))
    time_once(lambda: compiled_txt(text_hidden))

    image_eager, image_compiled = timing_pair(
        label="image_mlp",
        eager=lambda: feed_forward_call(block.img_ff, image_hidden),
        compiled=lambda: compiled_img(image_hidden),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    text_eager, text_compiled = timing_pair(
        label="text_mlp",
        eager=lambda: feed_forward_call(block.txt_ff, text_hidden),
        compiled=lambda: compiled_txt(text_hidden),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    whole: tuple[PathTiming, ...] | None = None
    if args.include_whole_block:
        print("measuring whole-block compile with mask=None", file=sys.stderr, flush=True)
        whole_eager, whole_compiled = measure_whole_block_mask_none(
            block,
            block_inputs,
            block_index=args.block_index,
            warmup_runs=args.warmup,
            measured_runs=args.runs,
        )
        whole = (whole_eager, whole_compiled)

    assert image_compiled.max_abs_error_vs_eager is not None
    verdict, reason = decide_mlp_compile_verdict(
        eager_median=image_eager.median_seconds,
        compiled_median=image_compiled.median_seconds,
        max_abs_error=image_compiled.max_abs_error_vs_eager,
        all_finite=image_compiled.all_finite,
        speedup_threshold=args.speedup_threshold,
        error_tolerance=args.error_tolerance,
    )
    proceed = verdict == "compile_not_enough"
    return MlpCompileResult(
        mlx_version=version("mlx"),
        platform=platform.platform(),
        model=str(model_dir),
        block_index=args.block_index,
        bits=args.bits,
        image_tokens=args.image_tokens,
        text_tokens=args.text_tokens,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        image_mlp=(image_eager, image_compiled),
        text_mlp=(text_eager, text_compiled),
        whole_block_mask_none=whole,
        image_mlp_verdict=verdict,
        image_mlp_verdict_reason=reason,
        proceed_to_group_size_sweep=proceed,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--speedup-threshold",
        type=float,
        default=0.05,
        help="Require at least this relative speedup (0.05 => 1.05x) to call compile helpful",
    )
    parser.add_argument(
        "--error-tolerance",
        type=float,
        default=0.0,
        help="Max allowed abs error between eager and compiled outputs",
    )
    parser.add_argument(
        "--include-whole-block",
        action="store_true",
        help="Also time whole-block mx.compile with mask=None as a reference upper bound",
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

    image_compiled = result.image_mlp[1]
    print(
        f"verdict={result.image_mlp_verdict} "
        f"image_mlp={image_compiled.relative_to_eager:.3f}x "
        f"err={image_compiled.max_abs_error_vs_eager} "
        f"proceed_group_size={result.proceed_to_group_size_sweep}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
