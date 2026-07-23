#!/usr/bin/env python3
"""Benchmark fixed-shape q6 QKV fusion and image-MLP row tiles on MLX Metal."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict
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

from benchmark_qwen_block import DEFAULT_MODEL, load_block, positive_int
from shardedit_mlx.benchmark_lora import apply_loras_to_loaded_blocks
from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations
from shardedit_mlx.qwen_block_loader import LoadedBlock


def array_leaves(value: Any) -> list[mx.array]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in array_leaves(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in array_leaves(item)]
    return [] if value is None else [value]


def evaluate(value: Any) -> None:
    leaves = array_leaves(value)
    if leaves:
        mx.eval(*leaves)


def time_once(operation: Callable[[], Any]) -> tuple[Any, float]:
    started_at = time.perf_counter()
    output = operation()
    evaluate(output)
    return output, time.perf_counter() - started_at


def measure_pair(
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[Any, Any, tuple[float, ...], tuple[float, ...]]:
    baseline_output: Any = None
    candidate_output: Any = None
    for _ in range(warmup_runs):
        baseline_output, _ = time_once(baseline)
        candidate_output, _ = time_once(candidate)

    baseline_times: list[float] = []
    candidate_times: list[float] = []
    for run_index in range(measured_runs):
        if run_index % 2 == 0:
            baseline_output, duration = time_once(baseline)
            baseline_times.append(duration)
            candidate_output, duration = time_once(candidate)
            candidate_times.append(duration)
        else:
            candidate_output, duration = time_once(candidate)
            candidate_times.append(duration)
            baseline_output, duration = time_once(baseline)
            baseline_times.append(duration)
    return (
        baseline_output,
        candidate_output,
        tuple(baseline_times),
        tuple(candidate_times),
    )


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


def parse_row_tiles(value: str) -> tuple[int, ...]:
    try:
        tiles = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("row tiles must be comma-separated integers") from error
    if not tiles or any(tile <= 0 for tile in tiles):
        raise argparse.ArgumentTypeError("row tiles must be positive")
    return tiles


def base_linear(layer: Any) -> Any:
    nested = getattr(layer, "linear", None)
    if nested is not None:
        return nested
    nested = getattr(layer, "base_linear", None)
    if nested is not None:
        return nested
    return layer


def lora_delta(layer: Any, values: mx.array, template: mx.array) -> mx.array:
    if hasattr(layer, "lora_A") and hasattr(layer, "lora_B"):
        return float(getattr(layer, "scale", 1.0)) * (
            (values @ layer.lora_A) @ layer.lora_B
        )
    loras = getattr(layer, "loras", None)
    if not loras:
        return mx.zeros_like(template)
    total = mx.zeros_like(template)
    for lora in loras:
        total = total + float(lora.scale) * ((values @ lora.lora_A) @ lora.lora_B)
    return total


def add_lora_delta(layer: Any, values: mx.array, base: mx.array) -> mx.array:
    if not (
        hasattr(layer, "lora_A")
        or hasattr(layer, "lora_B")
        or getattr(layer, "loras", None)
    ):
        return base
    return base + lora_delta(layer, values, base)


class FusedQuantizedLinears:
    """Prepacked view of compatible quantized linear projections."""

    def __init__(self, layers: Sequence[Any]):
        if not layers:
            raise ValueError("at least one layer is required")
        first = layers[0]
        attributes = ("group_size", "bits", "mode")
        if any(
            getattr(layer, attribute) != getattr(first, attribute)
            for layer in layers[1:]
            for attribute in attributes
        ):
            raise ValueError("quantized linear configurations differ")
        if any(layer.weight.shape[1] != first.weight.shape[1] for layer in layers[1:]):
            raise ValueError("quantized linear input dimensions differ")

        self.group_size = first.group_size
        self.bits = first.bits
        self.mode = first.mode
        self.output_dims = tuple(layer.weight.shape[0] for layer in layers)
        self.weight = mx.concatenate(tuple(layer.weight for layer in layers), axis=0)
        self.scales = mx.concatenate(tuple(layer.scales for layer in layers), axis=0)
        layer_biases = tuple(layer.get("biases") for layer in layers)
        if any(bias is None for bias in layer_biases) and any(
            bias is not None for bias in layer_biases
        ):
            raise ValueError("quantization bias availability differs")
        self.biases = (
            mx.concatenate(layer_biases, axis=0)
            if all(bias is not None for bias in layer_biases)
            else None
        )
        output_biases = tuple(layer.get("bias") for layer in layers)
        if any(bias is None for bias in output_biases) and any(
            bias is not None for bias in output_biases
        ):
            raise ValueError("output bias availability differs")
        self.output_bias = (
            mx.concatenate(output_biases, axis=0)
            if all(bias is not None for bias in output_biases)
            else None
        )
        evaluate((self.weight, self.scales, self.biases, self.output_bias))

    def __call__(self, values: mx.array) -> tuple[mx.array, ...]:
        output = mx.quantized_matmul(
            values,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if self.output_bias is not None:
            output = output + self.output_bias
        offsets: list[int] = []
        running_total = 0
        for output_dim in self.output_dims[:-1]:
            running_total += output_dim
            offsets.append(running_total)
        starts = (0, *offsets)
        ends = (*offsets, output.shape[-1])
        return tuple(output[..., start:end] for start, end in zip(starts, ends, strict=True))


def timing_payload(durations: Sequence[float], *, window: int = 3) -> dict[str, Any]:
    return {
        **asdict(summarize_durations(durations, window=window)),
        "durations_seconds": list(durations),
    }


def pair_payload(
    baseline_times: Sequence[float],
    candidate_times: Sequence[float],
    *,
    error: float,
) -> dict[str, Any]:
    baseline = summarize_durations(baseline_times)
    candidate = summarize_durations(candidate_times)
    return {
        "baseline": timing_payload(baseline_times),
        "candidate": timing_payload(candidate_times),
        "median_speedup": relative_speedup(
            baseline.median_seconds, candidate.median_seconds
        ),
        "max_abs_error": error,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model.expanduser().resolve()
    block = load_block(model_dir, args.block_index, bits=6)
    lora_paths = tuple(str(path) for path in args.lora_paths)
    lora_scales = tuple(float(scale) for scale in args.lora_scales)
    apply_loras_to_loaded_blocks(
        (LoadedBlock(block_index=args.block_index, module=block),),
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        block_count=args.lora_block_count,
    )
    mx.random.seed(args.seed)
    image_hidden = mx.random.normal((1, args.image_tokens, 3072)).astype(mx.bfloat16)
    text_hidden = mx.random.normal((1, args.text_tokens, 3072)).astype(mx.bfloat16)
    evaluate((image_hidden, text_hidden))

    attention = block.attn
    image_layers = (attention.to_q, attention.to_k, attention.to_v)
    text_layers = (attention.add_q_proj, attention.add_k_proj, attention.add_v_proj)
    fused_image = FusedQuantizedLinears(tuple(base_linear(layer) for layer in image_layers))
    fused_text = FusedQuantizedLinears(tuple(base_linear(layer) for layer in text_layers))

    def separate_qkv() -> tuple[mx.array, ...]:
        return tuple(layer(image_hidden) for layer in image_layers) + tuple(
            layer(text_hidden) for layer in text_layers
        )

    def fused_qkv() -> tuple[mx.array, ...]:
        image_base = fused_image(image_hidden)
        text_base = fused_text(text_hidden)
        return tuple(
            add_lora_delta(layer, image_hidden, base)
            for layer, base in zip(image_layers, image_base, strict=True)
        ) + tuple(
            add_lora_delta(layer, text_hidden, base)
            for layer, base in zip(text_layers, text_base, strict=True)
        )

    qkv_reference, qkv_candidate, qkv_baseline_times, qkv_fused_times = measure_pair(
        separate_qkv,
        fused_qkv,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    qkv_result = pair_payload(
        qkv_baseline_times,
        qkv_fused_times,
        error=max_abs_error(qkv_reference, qkv_candidate),
    )

    mlp_input = image_hidden

    def full_mlp() -> mx.array:
        return block.img_ff(mlp_input)

    mlp_tiles: dict[str, Any] = {}
    for row_tile in args.row_tiles:
        if row_tile > args.image_tokens:
            continue

        def tiled_mlp(tile: int = row_tile) -> mx.array:
            chunks = tuple(
                block.img_ff(mlp_input[:, start : start + tile, :])
                for start in range(0, args.image_tokens, tile)
            )
            return mx.concatenate(chunks, axis=1)

        reference, candidate, baseline_times, tiled_times = measure_pair(
            full_mlp,
            tiled_mlp,
            warmup_runs=args.warmup,
            measured_runs=args.runs,
        )
        mlp_tiles[str(row_tile)] = pair_payload(
            baseline_times,
            tiled_times,
            error=max_abs_error(reference, candidate),
        )

    long_baseline: list[float] = []
    long_fused: list[float] = []
    for run_index in range(args.long_runs):
        operations = (
            ((separate_qkv, long_baseline), (fused_qkv, long_fused))
            if run_index % 2 == 0
            else ((fused_qkv, long_fused), (separate_qkv, long_baseline))
        )
        for operation, durations in operations:
            _, duration = time_once(operation)
            durations.append(duration)

    return {
        "environment": {
            "mlx_version": version("mlx"),
            "platform": platform.platform(),
            "model": str(model_dir),
            "block_index": args.block_index,
            "bits": 6,
            "group_size": 64,
            "seed": args.seed,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
        },
        "shape": {
            "image_tokens": args.image_tokens,
            "text_tokens": args.text_tokens,
            "hidden_size": 3072,
            "mlp_hidden_size": 12288,
            "qkv_fused_output_size": 9216,
        },
        "mlx_m2_dispatch": {
            "kernel": "qmm_t_impl",
            "bm": args.dispatch_bm,
            "bn": args.dispatch_bn,
            "bk": args.dispatch_bk,
            "wm": 2,
            "wn": 2,
            "shape_autotuning": False,
        },
        "qkv_fusion": qkv_result,
        "image_mlp_row_tiles": mlp_tiles,
        "long_qkv": {
            "separate": timing_payload(long_baseline, window=args.drift_window),
            "fused": timing_payload(long_fused, window=args.drift_window),
            "median_speedup": relative_speedup(
                summarize_durations(long_baseline).median_seconds,
                summarize_durations(long_fused).median_seconds,
            ),
        },
        "memory": {
            "active_gib": mx.get_active_memory() / 1024**3,
            "cache_gib": mx.get_cache_memory() / 1024**3,
            "peak_gib": mx.get_peak_memory() / 1024**3,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--lora-paths", nargs="*", default=(), help="Optional LoRA paths")
    parser.add_argument("--lora-scales", nargs="*", type=float, default=(), help="Optional LoRA scales")
    parser.add_argument("--lora-block-count", type=positive_int, default=60)
    parser.add_argument("--row-tiles", type=parse_row_tiles, default=(256, 512, 1024, 1432))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--long-runs", type=positive_int, default=24)
    parser.add_argument("--drift-window", type=positive_int, default=5)
    parser.add_argument("--dispatch-bm", type=positive_int, default=32)
    parser.add_argument("--dispatch-bn", type=positive_int, default=32)
    parser.add_argument("--dispatch-bk", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.block_index < 0 or args.block_index >= 60:
        parser.error("--block-index must be between 0 and 59")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    result = run_benchmark(args)
    payload = json.dumps(result, indent=2) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
