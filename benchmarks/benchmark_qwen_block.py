#!/usr/bin/env python3
"""Benchmark one real q6 Qwen Transformer block on MLX Metal."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten

from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import (
    QwenTransformerBlock,
)
from shardedit_mlx.perf_model import EditTokenPlan, classifier_free_guidance_passes


DEFAULT_MODEL = Path("models/qwen-edit-2511-q6")
BLOCK_PREFIX = "transformer_blocks.{block_index}."


@dataclass(frozen=True)
class BlockBenchmarkResult:
    block_index: int
    bits: int
    group_size: int
    reference_count: int
    target_tokens: int
    condition_tokens: int
    text_tokens: int
    warmup_runs: int
    measured_runs: int
    median_block_seconds: float
    min_block_seconds: float
    projected_denoise_seconds: float
    peak_memory_gib: float


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def transformer_shard(model_dir: Path, block_index: int) -> Path:
    index_path = model_dir / "transformer" / "model.safetensors.index.json"
    if not index_path.exists():
        fail(f"missing Transformer index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        fail(f"invalid weight_map in {index_path}")
    probe_key = f"transformer_blocks.{block_index}.attn.to_q.weight"
    shard_name = weight_map.get(probe_key)
    if not isinstance(shard_name, str):
        fail(f"block {block_index} is not present in {index_path}")
    shard = model_dir / "transformer" / shard_name
    if not shard.exists():
        fail(f"missing Transformer shard: {shard}")
    return shard


def dequantize_weights_to_dense(
    flat_weights: list[tuple[str, mx.array]],
    *,
    source_bits: int,
    source_group_size: int = 64,
) -> list[tuple[str, mx.array]]:
    """Expand packed qN weights into dense bf16 Linear tensors.

    Quantization scales / biases are dropped. Ordinary Linear ``.bias`` tensors
    are preserved unchanged. This is a diagnostic helper only: a full 60-block
    dense Transformer would not fit in 24 GB unified memory.
    """

    source = dict(flat_weights)
    converted: list[tuple[str, mx.array]] = []
    for name, value in flat_weights:
        if name.endswith((".scales", ".biases")):
            base_name = name.rsplit(".", 1)[0]
            if source.get(f"{base_name}.weight") is not None:
                continue
        if not name.endswith(".weight") or value.dtype != mx.uint32:
            converted.append((name, value))
            continue

        base_name = name.removesuffix(".weight")
        scales = source.get(f"{base_name}.scales")
        biases = source.get(f"{base_name}.biases")
        if scales is None:
            fail(f"quantized tensor has no scales: {name}")
        dense = mx.dequantize(
            value,
            scales,
            biases=biases,
            group_size=source_group_size,
            bits=source_bits,
            mode="affine",
            dtype=mx.bfloat16,
        )
        mx.eval(dense)
        converted.append((name, dense))
    return converted


def requantize_weights(
    flat_weights: list[tuple[str, mx.array]],
    *,
    source_bits: int,
    target_bits: int,
    source_group_size: int = 64,
    target_group_size: int = 64,
) -> list[tuple[str, mx.array]]:
    if target_bits == 16:
        return dequantize_weights_to_dense(
            flat_weights,
            source_bits=source_bits,
            source_group_size=source_group_size,
        )
    if source_bits == target_bits and source_group_size == target_group_size:
        return flat_weights

    source = dict(flat_weights)
    converted: list[tuple[str, mx.array]] = []
    for name, value in flat_weights:
        if name.endswith((".scales", ".biases")):
            base_name = name.rsplit(".", 1)[0]
            if source.get(f"{base_name}.weight") is not None:
                continue
        if not name.endswith(".weight") or value.dtype != mx.uint32:
            converted.append((name, value))
            continue

        base_name = name.removesuffix(".weight")
        scales = source.get(f"{base_name}.scales")
        biases = source.get(f"{base_name}.biases")
        if scales is None:
            fail(f"quantized tensor has no scales: {name}")
        dense = mx.dequantize(
            value,
            scales,
            biases=biases,
            group_size=source_group_size,
            bits=source_bits,
            mode="affine",
            dtype=mx.bfloat16,
        )
        quantized = mx.quantize(
            dense,
            group_size=target_group_size,
            bits=target_bits,
            mode="affine",
        )
        mx.eval(*quantized)
        converted.extend(
            [
                (name, quantized[0]),
                (f"{base_name}.scales", quantized[1]),
                (f"{base_name}.biases", quantized[2]),
            ]
        )
    return converted


def load_block(
    model_dir: Path,
    block_index: int,
    bits: int,
    *,
    group_size: int = 64,
) -> QwenTransformerBlock:
    if bits not in (4, 5, 6, 16):
        fail(f"unsupported bits={bits}; expected 4, 5, 6, or 16 (dense)")
    if group_size not in (32, 64, 128):
        fail(f"unsupported group_size={group_size}; expected 32, 64, or 128")
    if bits == 16 and group_size != 64:
        fail("dense bf16 path ignores group_size; leave it at the default 64")
    shard = transformer_shard(model_dir, block_index)
    prefix = BLOCK_PREFIX.format(block_index=block_index)
    shard_weights: dict[str, mx.array] = mx.load(str(shard))
    flat_weights = [
        (name.removeprefix(prefix), value)
        for name, value in shard_weights.items()
        if name.startswith(prefix)
    ]
    if not flat_weights:
        fail(f"no weights for block {block_index} in {shard}")

    runtime_weights = requantize_weights(
        flat_weights,
        source_bits=6,
        target_bits=bits,
        source_group_size=64,
        target_group_size=group_size,
    )
    block = QwenTransformerBlock()
    if bits != 16:
        nn.quantize(block, group_size=group_size, bits=bits, mode="affine")
    block.update(tree_unflatten(runtime_weights), strict=True)

    del shard_weights
    del flat_weights
    del runtime_weights
    gc.collect()
    mx.clear_cache()
    return block


def make_inputs(image_tokens: int, text_tokens: int) -> dict[str, Any]:
    dtype = mx.bfloat16
    hidden_states = mx.zeros((1, image_tokens, 3072), dtype=dtype)
    encoder_hidden_states = mx.zeros((1, text_tokens, 3072), dtype=dtype)
    text_embeddings = mx.zeros((1, 3072), dtype=dtype)
    mask = mx.ones((1, text_tokens), dtype=dtype)
    image_cos = mx.ones((image_tokens, 64), dtype=mx.float32)
    image_sin = mx.zeros((image_tokens, 64), dtype=mx.float32)
    text_cos = mx.ones((text_tokens, 64), dtype=mx.float32)
    text_sin = mx.zeros((text_tokens, 64), dtype=mx.float32)
    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_hidden_states_mask": mask,
        "text_embeddings": text_embeddings,
        "image_rotary_emb": ((image_cos, image_sin), (text_cos, text_sin)),
    }


def evaluate_block(block: QwenTransformerBlock, inputs: dict[str, Any], block_index: int) -> None:
    encoder_output, image_output = block(**inputs, block_idx=block_index)
    mx.eval(encoder_output, image_output)


def run_benchmark(args: argparse.Namespace) -> BlockBenchmarkResult:
    token_plan = EditTokenPlan.from_dimensions(
        target_width=args.target_width,
        target_height=args.target_height,
        condition_width=args.condition_width,
        condition_height=args.condition_height,
        condition_count=args.condition_count,
    )
    model_dir = args.model.expanduser().resolve()
    block = load_block(
        model_dir,
        args.block_index,
        args.bits,
        group_size=args.group_size,
    )
    inputs = make_inputs(token_plan.image_tokens, args.text_tokens)

    for _ in range(args.warmup):
        evaluate_block(block, inputs, args.block_index)

    mx.reset_peak_memory()
    durations: list[float] = []
    for _ in range(args.runs):
        started = time.perf_counter()
        evaluate_block(block, inputs, args.block_index)
        durations.append(time.perf_counter() - started)

    median_seconds = statistics.median(durations)
    passes = classifier_free_guidance_passes(args.guidance)
    projected = median_seconds * 60 * args.steps * passes
    return BlockBenchmarkResult(
        block_index=args.block_index,
        bits=args.bits,
        group_size=args.group_size,
        reference_count=args.condition_count,
        target_tokens=token_plan.target_tokens,
        condition_tokens=token_plan.condition_tokens,
        text_tokens=args.text_tokens,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        median_block_seconds=median_seconds,
        min_block_seconds=min(durations),
        projected_denoise_seconds=projected,
        peak_memory_gib=mx.get_peak_memory() / 1024**3,
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument(
        "--bits",
        type=int,
        choices=(4, 5, 6, 16),
        default=6,
        help="4/5/6 keep quantized Linear; 16 dequantizes the packed q6 weights to dense bf16",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        choices=(32, 64, 128),
        default=64,
        help="Affine quantization group size (source checkpoint is always group_size=64)",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Alias for --bits 16 (diagnostic dense bf16 Linear path)",
    )
    parser.add_argument("--target-width", type=positive_int, default=768)
    parser.add_argument("--target-height", type=positive_int, default=768)
    parser.add_argument("--condition-width", type=positive_int, default=384)
    parser.add_argument("--condition-height", type=positive_int, default=384)
    parser.add_argument("--condition-count", type=positive_int, default=1)
    parser.add_argument("--text-tokens", type=positive_int, default=128)
    parser.add_argument("--steps", type=positive_int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=positive_int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.dense:
        args.bits = 16
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
