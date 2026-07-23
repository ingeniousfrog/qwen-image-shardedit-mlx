#!/usr/bin/env python3
"""Measure same-weight and 60-block q6 Transformer latency over long runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
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

from benchmark_qwen_block import DEFAULT_MODEL, make_inputs, positive_int
from shardedit_mlx.gemm_profile import summarize_durations
from shardedit_mlx.sweep_profile import BlockShardPlan, plan_transformer_block_shards


def read_command(args: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def memory_snapshot(label: str) -> dict[str, Any]:
    rss_kib = read_command(("ps", "-o", "rss=", "-p", str(os.getpid())))
    return {
        "label": label,
        "active_gib": mx.get_active_memory() / 1024**3,
        "cache_gib": mx.get_cache_memory() / 1024**3,
        "peak_gib": mx.get_peak_memory() / 1024**3,
        "process_rss_gib": float(rss_kib) / 1024**2 if rss_kib else None,
        "swapusage": read_command(("sysctl", "-n", "vm.swapusage")),
    }


def load_index(model_dir: Path) -> tuple[dict[str, str], Path]:
    transformer_dir = model_dir / "transformer"
    index_path = transformer_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing Transformer index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise ValueError(f"invalid weight_map in {index_path}")
    return weight_map, transformer_dir


def instantiate_block(flat_weights: tuple[tuple[str, mx.array], ...]) -> QwenTransformerBlock:
    block = QwenTransformerBlock()
    nn.quantize(block, group_size=64, bits=6, mode="affine")
    block.update(tree_unflatten(flat_weights), strict=True)
    return block


def load_split_block(
    transformer_dir: Path, plan: BlockShardPlan
) -> QwenTransformerBlock:
    prefix = f"transformer_blocks.{plan.block_index}."
    flat_weights: tuple[tuple[str, mx.array], ...] = ()
    for shard_name in plan.shards:
        shard_weights: dict[str, mx.array] = mx.load(str(transformer_dir / shard_name))
        shard_part = tuple(
            (name.removeprefix(prefix), value)
            for name, value in shard_weights.items()
            if name.startswith(prefix)
        )
        flat_weights = (*flat_weights, *shard_part)
        del shard_weights
        gc.collect()
        mx.clear_cache()
    if len(flat_weights) != plan.tensor_count:
        raise RuntimeError(
            f"block {plan.block_index} expected {plan.tensor_count} tensors, "
            f"loaded {len(flat_weights)}"
        )
    return instantiate_block(flat_weights)


def load_all_blocks(
    model_dir: Path,
) -> tuple[tuple[QwenTransformerBlock, ...], tuple[BlockShardPlan, ...], float]:
    weight_map, transformer_dir = load_index(model_dir)
    plans = plan_transformer_block_shards(weight_map)

    ordered_shards = tuple(
        sorted(
            {plan.shards[0] for plan in plans if len(plan.shards) == 1},
            key=lambda shard: min(
                plan.block_index for plan in plans if plan.shards == (shard,)
            ),
        )
    )
    loaded: tuple[tuple[int, QwenTransformerBlock], ...] = ()
    started_at = time.perf_counter()
    for shard_position, shard_name in enumerate(ordered_shards, start=1):
        print(
            f"loading shard {shard_position}/{len(ordered_shards)}: {shard_name}",
            file=sys.stderr,
            flush=True,
        )
        shard_weights: dict[str, mx.array] = mx.load(str(transformer_dir / shard_name))
        shard_plans = tuple(plan for plan in plans if plan.shards == (shard_name,))
        shard_blocks = tuple(
            (
                plan.block_index,
                instantiate_block(
                    tuple(
                        (name.removeprefix(prefix), value)
                        for name, value in shard_weights.items()
                        if name.startswith(prefix)
                    )
                ),
            )
            for plan in shard_plans
            for prefix in (f"transformer_blocks.{plan.block_index}.",)
        )
        loaded = (*loaded, *shard_blocks)
        del shard_weights
        gc.collect()
        mx.clear_cache()

    split_plans = tuple(plan for plan in plans if len(plan.shards) > 1)
    for split_position, plan in enumerate(split_plans, start=1):
        print(
            f"loading split block {split_position}/{len(split_plans)}: {plan.block_index}",
            file=sys.stderr,
            flush=True,
        )
        loaded = (*loaded, (plan.block_index, load_split_block(transformer_dir, plan)))

    ordered = tuple(block for _, block in sorted(loaded, key=lambda item: item[0]))
    if len(ordered) != 60:
        raise RuntimeError(f"expected 60 blocks, loaded {len(ordered)}")
    return ordered, plans, time.perf_counter() - started_at


def evaluate_block(
    block: QwenTransformerBlock,
    inputs: dict[str, Any],
    block_index: int,
) -> tuple[tuple[mx.array, mx.array], float]:
    started_at = time.perf_counter()
    output = block(**inputs, block_idx=block_index)
    mx.eval(*output)
    return output, time.perf_counter() - started_at


def output_as_inputs(
    base_inputs: dict[str, Any], output: tuple[mx.array, mx.array]
) -> dict[str, Any]:
    return {
        **base_inputs,
        "encoder_hidden_states": output[0],
        "hidden_states": output[1],
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model.expanduser().resolve()
    before_load = memory_snapshot("before_load")
    blocks, plans, load_seconds = load_all_blocks(model_dir)
    after_load = memory_snapshot("after_load")
    base_inputs = make_inputs(args.image_tokens, args.text_tokens)

    for _ in range(args.warmup):
        evaluate_block(blocks[0], base_inputs, 0)

    same_block_times: tuple[float, ...] = ()
    for run_index in range(args.repeat_runs):
        _, duration = evaluate_block(blocks[0], base_inputs, 0)
        same_block_times = (*same_block_times, duration)
        if (run_index + 1) % 10 == 0:
            print(
                f"same block {run_index + 1}/{args.repeat_runs}: {duration:.3f}s",
                file=sys.stderr,
                flush=True,
            )
    after_repeat = memory_snapshot("after_same_block")

    sweep_results: tuple[dict[str, Any], ...] = ()
    for sweep_index in range(args.sweeps):
        current_inputs = base_inputs
        block_times: tuple[float, ...] = ()
        sweep_started_at = time.perf_counter()
        for block_index, block in enumerate(blocks):
            output, duration = evaluate_block(block, current_inputs, block_index)
            current_inputs = output_as_inputs(base_inputs, output)
            block_times = (*block_times, duration)
        sweep_seconds = time.perf_counter() - sweep_started_at
        summary = summarize_durations(block_times, window=10)
        sweep_result = {
            "sweep_index": sweep_index + 1,
            "total_seconds": sweep_seconds,
            "block_summary": asdict(summary),
            "block_durations_seconds": list(block_times),
            "memory": memory_snapshot(f"after_sweep_{sweep_index + 1}"),
        }
        sweep_results = (*sweep_results, sweep_result)
        print(
            f"sweep {sweep_index + 1}/{args.sweeps}: {sweep_seconds:.3f}s",
            file=sys.stderr,
            flush=True,
        )

    return {
        "environment": {
            "mlx_version": version("mlx"),
            "platform": platform.platform(),
            "model": str(model_dir),
            "bits": 6,
            "group_size": 64,
            "image_tokens": args.image_tokens,
            "text_tokens": args.text_tokens,
            "block_count": len(blocks),
            "single_shard_per_block": all(len(plan.shards) == 1 for plan in plans),
        },
        "load_seconds": load_seconds,
        "memory": {
            "before_load": before_load,
            "after_load": after_load,
            "after_same_block": after_repeat,
        },
        "same_block": {
            **asdict(summarize_durations(same_block_times, window=10)),
            "durations_seconds": list(same_block_times),
        },
        "sweeps": list(sweep_results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat-runs", type=positive_int, default=30)
    parser.add_argument("--sweeps", type=positive_int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
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
