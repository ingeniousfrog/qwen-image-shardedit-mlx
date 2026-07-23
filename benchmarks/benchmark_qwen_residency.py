#!/usr/bin/env python3
"""Compare resident, shard, and fixed-window q6 Transformer execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
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
import numpy as np

from benchmark_qwen_block import DEFAULT_MODEL, make_inputs, positive_int
from shardedit_mlx.gemm_profile import summarize_durations
from shardedit_mlx.qwen_block_loader import (
    LoadedBlock,
    TransformerLayout,
    load_block_window,
    load_transformer_layout,
)
from shardedit_mlx.residency_plan import (
    ResidencyWindow,
    fixed_block_windows,
    shard_block_windows,
)
from shardedit_mlx.system_memory import (
    SwapUsage,
    VmStatSnapshot,
    counter_deltas,
    parse_memory_pressure_free_percentage,
    parse_swap_usage,
    parse_vm_stat,
)


VM_COUNTERS = (
    "pages_free",
    "pages_active",
    "pages_inactive",
    "pages_speculative",
    "pages_wired_down",
    "pages_stored_in_compressor",
    "pages_occupied_by_compressor",
    "decompressions",
    "compressions",
    "pageins",
    "pageouts",
    "swapins",
    "swapouts",
)


@dataclass(frozen=True)
class CommandCapture:
    output: str | None
    error: str | None


@dataclass(frozen=True)
class MemorySample:
    label: str
    active_gib: float
    cache_gib: float
    peak_gib: float
    process_rss_gib: float | None
    vm_stat: VmStatSnapshot | None
    free_percentage: int | None
    swap_usage: SwapUsage | None
    errors: tuple[str, ...]


def run_command(args: tuple[str, ...]) -> CommandCapture:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except OSError as error:
        return CommandCapture(None, f"{' '.join(args)}: {error}")
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        return CommandCapture(None, f"{' '.join(args)}: {detail}")
    return CommandCapture(result.stdout.strip(), None)


def capture_memory(label: str, *, detailed: bool = False) -> MemorySample:
    rss_capture = run_command(("ps", "-o", "rss=", "-p", str(os.getpid())))
    vm_capture = run_command(("vm_stat",))
    pressure_capture = (
        run_command(("memory_pressure",)) if detailed else CommandCapture(None, None)
    )
    swap_capture = (
        run_command(("sysctl", "-n", "vm.swapusage"))
        if detailed
        else CommandCapture(None, None)
    )

    errors = tuple(
        error
        for error in (
            rss_capture.error,
            vm_capture.error,
            pressure_capture.error,
            swap_capture.error,
        )
        if error is not None
    )
    vm_snapshot: VmStatSnapshot | None = None
    if vm_capture.output is not None:
        try:
            vm_snapshot = parse_vm_stat(vm_capture.output)
        except ValueError as error:
            errors = (*errors, f"vm_stat parse: {error}")

    rss_gib: float | None = None
    if rss_capture.output is not None:
        try:
            rss_gib = float(rss_capture.output) / 1024**2
        except ValueError:
            errors = (*errors, f"ps returned invalid RSS: {rss_capture.output!r}")

    free_percentage = (
        parse_memory_pressure_free_percentage(pressure_capture.output)
        if pressure_capture.output is not None
        else None
    )
    swap_usage = (
        parse_swap_usage(swap_capture.output)
        if swap_capture.output is not None
        else None
    )
    return MemorySample(
        label=label,
        active_gib=mx.get_active_memory() / 1024**3,
        cache_gib=mx.get_cache_memory() / 1024**3,
        peak_gib=mx.get_peak_memory() / 1024**3,
        process_rss_gib=rss_gib,
        vm_stat=vm_snapshot,
        free_percentage=free_percentage,
        swap_usage=swap_usage,
        errors=errors,
    )


def memory_as_dict(sample: MemorySample) -> dict[str, Any]:
    available_counters = dict(sample.vm_stat.counters) if sample.vm_stat else {}
    vm_payload = (
        {
            "page_size_bytes": sample.vm_stat.page_size_bytes,
            "counters": {
                name: available_counters[name]
                for name in VM_COUNTERS
                if name in available_counters
            },
        }
        if sample.vm_stat is not None
        else None
    )
    return {
        "label": sample.label,
        "active_gib": sample.active_gib,
        "cache_gib": sample.cache_gib,
        "peak_gib": sample.peak_gib,
        "process_rss_gib": sample.process_rss_gib,
        "vm_stat": vm_payload,
        "free_percentage": sample.free_percentage,
        "swap_usage": asdict(sample.swap_usage) if sample.swap_usage else None,
        "errors": list(sample.errors),
    }


def paging_delta(
    before: MemorySample,
    after: MemorySample,
) -> dict[str, int] | None:
    if before.vm_stat is None or after.vm_stat is None:
        return None
    return counter_deltas(before.vm_stat, after.vm_stat)


def mlx_memory(label: str) -> dict[str, float | str]:
    return {
        "label": label,
        "active_gib": mx.get_active_memory() / 1024**3,
        "cache_gib": mx.get_cache_memory() / 1024**3,
        "peak_gib": mx.get_peak_memory() / 1024**3,
    }


def output_as_inputs(
    base_inputs: dict[str, Any],
    output: tuple[mx.array, mx.array],
) -> dict[str, Any]:
    return {
        **base_inputs,
        "encoder_hidden_states": output[0],
        "hidden_states": output[1],
    }


def execute_blocks(
    blocks: tuple[LoadedBlock, ...],
    current_inputs: dict[str, Any],
    base_inputs: dict[str, Any],
) -> tuple[tuple[mx.array, mx.array], dict[str, Any], tuple[float, ...]]:
    output: tuple[mx.array, mx.array] | None = None
    durations: tuple[float, ...] = ()
    next_inputs = current_inputs
    for loaded_block in blocks:
        started_at = time.perf_counter()
        output = loaded_block.module(**next_inputs, block_idx=loaded_block.block_index)
        mx.eval(*output)
        durations = (*durations, time.perf_counter() - started_at)
        next_inputs = output_as_inputs(base_inputs, output)
    if output is None:
        raise RuntimeError("cannot execute an empty block window")
    return output, next_inputs, durations


def output_fingerprint(output: tuple[mx.array, mx.array]) -> dict[str, Any]:
    names = ("text", "image")
    arrays = tuple(np.asarray(value.astype(mx.float32)) for value in output)
    payload = {
        name: {
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "sum_float64": float(array.sum(dtype=np.float64)),
            "all_finite": bool(np.isfinite(array).all()),
        }
        for name, array in zip(names, arrays, strict=True)
    }
    return payload


def release_runtime() -> float:
    started_at = time.perf_counter()
    gc.collect()
    mx.clear_cache()
    return time.perf_counter() - started_at


def load_timed(
    layout: TransformerLayout,
    block_indices: tuple[int, ...],
) -> tuple[tuple[LoadedBlock, ...], float]:
    started_at = time.perf_counter()
    blocks = load_block_window(layout, block_indices)
    return blocks, time.perf_counter() - started_at


def run_streaming_sweep(
    layout: TransformerLayout,
    windows: tuple[ResidencyWindow, ...],
    base_inputs: dict[str, Any],
    sweep_index: int,
) -> dict[str, Any]:
    sweep_before = capture_memory(f"sweep_{sweep_index}_before", detailed=True)
    mx.reset_peak_memory()
    current_inputs = base_inputs
    output: tuple[mx.array, mx.array] | None = None
    window_results: tuple[dict[str, Any], ...] = ()
    all_block_durations: tuple[float, ...] = ()
    observed_started_at = time.perf_counter()

    for window in windows:
        block_range = f"{window.block_indices[0]}-{window.block_indices[-1]}"
        print(
            f"sweep {sweep_index}: loading window {window.index + 1}/{len(windows)} "
            f"blocks {block_range}",
            file=sys.stderr,
            flush=True,
        )
        before_load = capture_memory(
            f"sweep_{sweep_index}_window_{window.index}_before_load"
        )
        blocks, load_seconds = load_timed(layout, window.block_indices)
        after_load = mlx_memory(
            f"sweep_{sweep_index}_window_{window.index}_after_load"
        )
        output, current_inputs, block_durations = execute_blocks(
            blocks, current_inputs, base_inputs
        )
        after_compute = capture_memory(
            f"sweep_{sweep_index}_window_{window.index}_after_compute"
        )
        del blocks
        release_seconds = release_runtime()
        after_release = capture_memory(
            f"sweep_{sweep_index}_window_{window.index}_after_release"
        )
        compute_seconds = sum(block_durations)
        all_block_durations = (*all_block_durations, *block_durations)
        window_results = (
            *window_results,
            {
                "window_index": window.index,
                "block_indices": list(window.block_indices),
                "shards": list(window.shards),
                "load_seconds": load_seconds,
                "compute_seconds": compute_seconds,
                "release_seconds": release_seconds,
                "block_durations_seconds": list(block_durations),
                "memory": {
                    "before_load": memory_as_dict(before_load),
                    "after_load": after_load,
                    "after_compute": memory_as_dict(after_compute),
                    "after_release": memory_as_dict(after_release),
                },
                "paging": {
                    "load_and_compute": paging_delta(before_load, after_compute),
                    "release": paging_delta(after_compute, after_release),
                },
            },
        )

    observed_wall_seconds = time.perf_counter() - observed_started_at
    if output is None:
        raise RuntimeError("sweep did not execute any Transformer blocks")
    sweep_after = capture_memory(f"sweep_{sweep_index}_after", detailed=True)
    fingerprint = output_fingerprint(output)
    summary = summarize_durations(all_block_durations, window=10)
    load_seconds = sum(result["load_seconds"] for result in window_results)
    compute_seconds = sum(result["compute_seconds"] for result in window_results)
    release_seconds = sum(result["release_seconds"] for result in window_results)
    managed_seconds = load_seconds + compute_seconds + release_seconds
    return {
        "sweep_index": sweep_index,
        "managed_seconds": managed_seconds,
        "observed_wall_seconds": observed_wall_seconds,
        "telemetry_seconds": observed_wall_seconds - managed_seconds,
        "load_seconds": load_seconds,
        "compute_seconds": compute_seconds,
        "release_seconds": release_seconds,
        "block_summary": asdict(summary),
        "block_durations_seconds": list(all_block_durations),
        "fingerprint": fingerprint,
        "memory": {
            "before": memory_as_dict(sweep_before),
            "after": memory_as_dict(sweep_after),
        },
        "paging": paging_delta(sweep_before, sweep_after),
        "windows": list(window_results),
    }


def run_resident_sweep(
    blocks: tuple[LoadedBlock, ...],
    base_inputs: dict[str, Any],
    sweep_index: int,
) -> dict[str, Any]:
    sweep_before = capture_memory(f"sweep_{sweep_index}_before", detailed=True)
    mx.reset_peak_memory()
    started_at = time.perf_counter()
    output, _, block_durations = execute_blocks(blocks, base_inputs, base_inputs)
    compute_seconds = time.perf_counter() - started_at
    sweep_after = capture_memory(f"sweep_{sweep_index}_after", detailed=True)
    fingerprint = output_fingerprint(output)
    summary = summarize_durations(block_durations, window=10)
    return {
        "sweep_index": sweep_index,
        "managed_seconds": compute_seconds,
        "observed_wall_seconds": compute_seconds,
        "telemetry_seconds": 0.0,
        "load_seconds": 0.0,
        "compute_seconds": compute_seconds,
        "release_seconds": 0.0,
        "block_summary": asdict(summary),
        "block_durations_seconds": list(block_durations),
        "fingerprint": fingerprint,
        "memory": {
            "before": memory_as_dict(sweep_before),
            "after": memory_as_dict(sweep_after),
        },
        "paging": paging_delta(sweep_before, sweep_after),
        "windows": [
            {
                "window_index": 0,
                "block_indices": [block.block_index for block in blocks],
                "load_seconds": 0.0,
                "compute_seconds": compute_seconds,
                "release_seconds": 0.0,
                "block_durations_seconds": list(block_durations),
                "memory": {"after_compute": memory_as_dict(sweep_after)},
                "paging": paging_delta(sweep_before, sweep_after),
            }
        ],
    }


def experiment_layout(layout: TransformerLayout, block_count: int) -> TransformerLayout:
    if block_count > len(layout.plans):
        raise ValueError(
            f"block_count cannot exceed the model layout ({len(layout.plans)})"
        )
    plans = layout.plans[:block_count]
    shards = tuple(
        shard
        for shard in layout.ordered_shards
        if any(shard in plan.shards for plan in plans)
    )
    return TransformerLayout(layout.transformer_dir, plans, shards)


def make_windows(
    layout: TransformerLayout,
    mode: str,
    window_size: int | None,
) -> tuple[ResidencyWindow, ...]:
    if mode == "resident":
        return fixed_block_windows(layout.plans, len(layout.plans))
    if mode == "shard":
        return shard_block_windows(layout.plans, layout.ordered_shards)
    if mode == "window" and window_size is not None:
        return fixed_block_windows(layout.plans, window_size)
    raise ValueError("window mode requires --window-size")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model.expanduser().resolve()
    layout = experiment_layout(load_transformer_layout(model_dir), args.block_count)
    windows = make_windows(layout, args.mode, args.window_size)
    base_inputs = make_inputs(args.image_tokens, args.text_tokens)
    before_run = capture_memory("before_run", detailed=True)
    resident_initialization: dict[str, Any] | None = None
    sweep_results: tuple[dict[str, Any], ...] = ()

    if args.mode == "resident":
        blocks, load_seconds = load_timed(
            layout, tuple(plan.block_index for plan in layout.plans)
        )
        resident_after_load = capture_memory("resident_after_load", detailed=True)
        resident_initialization = {
            "load_seconds": load_seconds,
            "memory_after_load": memory_as_dict(resident_after_load),
        }
        sweep_results = tuple(
            run_resident_sweep(blocks, base_inputs, sweep_index)
            for sweep_index in range(1, args.sweeps + 1)
        )
        del blocks
        final_release_seconds = release_runtime()
    else:
        sweep_results = tuple(
            run_streaming_sweep(
                layout,
                windows,
                base_inputs,
                sweep_index,
            )
            for sweep_index in range(1, args.sweeps + 1)
        )
        final_release_seconds = release_runtime()

    after_run = capture_memory("after_run", detailed=True)
    fingerprints = tuple(
        json.dumps(result["fingerprint"], sort_keys=True) for result in sweep_results
    )
    return {
        "environment": {
            "mlx_version": version("mlx"),
            "mflux_version": version("mflux"),
            "platform": platform.platform(),
            "model": str(model_dir),
            "bits": 6,
            "group_size": 64,
            "image_tokens": args.image_tokens,
            "text_tokens": args.text_tokens,
            "block_count": len(layout.plans),
            "lora_applied": False,
        },
        "policy": {
            "mode": args.mode,
            "window_size": args.window_size,
            "window_count": len(windows),
            "windows": [
                {
                    "window_index": window.index,
                    "block_indices": list(window.block_indices),
                    "shards": list(window.shards),
                }
                for window in windows
            ],
        },
        "resident_initialization": resident_initialization,
        "before_run": memory_as_dict(before_run),
        "sweeps": list(sweep_results),
        "fingerprints_match_across_sweeps": len(set(fingerprints)) == 1,
        "final_release_seconds": final_release_seconds,
        "after_run": memory_as_dict(after_run),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--mode", choices=("resident", "shard", "window"), required=True
    )
    parser.add_argument("--window-size", type=positive_int)
    parser.add_argument("--block-count", type=positive_int, default=60)
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--sweeps", type=positive_int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "window" and args.window_size is None:
        parser.error("--window-size is required when --mode=window")
    if args.mode != "window" and args.window_size is not None:
        parser.error("--window-size is only valid when --mode=window")
    if args.window_size is not None and args.window_size > args.block_count:
        parser.error("--window-size cannot exceed --block-count")

    result = run_benchmark(args)
    payload = json.dumps(result, indent=2) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
