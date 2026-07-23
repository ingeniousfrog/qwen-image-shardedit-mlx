#!/usr/bin/env python3
"""Open-source/fork kernel spike for real Qwen img_ff.mlp_in/out shapes."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import importlib
from importlib.metadata import PackageNotFoundError, version
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
from benchmark_qwen_block_dense_ab import all_finite, max_abs_error, measure_round_robin
from shardedit_mlx.gemm_profile import relative_speedup, summarize_durations
from shardedit_mlx.kernel_feasibility_profile import (
    BASELINE_PATH,
    DENSE_PATH,
    KernelPathSummary,
    decide_img_ff_kernel_spike_verdict,
)
from shardedit_mlx.q6_metal_mlp import (
    Q6LinearSpec,
    affine_q6_qmm_t,
    dequantize_linear,
    quantized_linear_spec,
)
from shardedit_mlx.q6_steel_mlp import affine_q6_qmm_t_tiled


DTYPES = {
    "bf16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}
FORK_TILED_PATH = "shardedit_mlx_fork_tiled_metal"
FORK_NAIVE_PATH = "shardedit_mlx_fork_naive_metal"
KQUANT_PATH_PREFIX = "mlx_kquant"


@dataclass(frozen=True)
class BackendStatus:
    name: str
    status: str
    reason: str
    version: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidateOp:
    name: str
    operation: Callable[[], mx.array]
    prepare_seconds: float
    materialized_bytes: int | None
    note: str | None = None


@dataclass(frozen=True)
class PathTiming:
    name: str
    block_index: int
    layer: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    weight_shape: tuple[int, int]
    bits: int
    group_size: int
    median_seconds: float
    min_seconds: float
    mean_seconds: float
    durations_seconds: tuple[float, ...]
    relative_to_mlx_q6: float | None
    max_abs_error_vs_mlx_q6: float | None
    all_finite: bool
    prepare_seconds: float
    materialized_bytes: int | None
    note: str | None = None


@dataclass(frozen=True)
class AggregateTiming:
    name: str
    measured_layers: int
    expected_layers: int
    median_seconds: float
    relative_to_mlx_q6: float | None
    max_abs_error_vs_mlx_q6: float | None
    all_finite: bool
    materialized_bytes: int | None
    complete: bool


@dataclass(frozen=True)
class KernelSpikeResult:
    mlx_version: str
    platform: str
    model: str
    block_indices: tuple[int, ...]
    image_tokens: int
    dtype: str
    warmup_runs: int
    measured_runs: int
    kquant_codec: str
    backend_statuses: tuple[BackendStatus, ...]
    paths: tuple[PathTiming, ...]
    aggregate_paths: tuple[AggregateTiming, ...]
    verdict: str
    verdict_reason: str


def parse_block_indices(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one block index is required")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("block indices must be unique")
    invalid = [index for index in parsed if index < 0 or index >= 60]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"block indices must be between 0 and 59, got {invalid}"
        )
    return parsed


def array_nbytes(values: Sequence[mx.array | None]) -> int:
    return sum(int(value.nbytes) for value in values if value is not None)


def make_random_hidden(tokens: int, dim: int, *, seed: int, dtype: mx.Dtype) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, dim)).astype(dtype)


def prepare_dense_candidate(
    hidden: mx.array,
    spec: Q6LinearSpec,
    *,
    dtype: mx.Dtype,
) -> CandidateOp:
    started = time.perf_counter()
    dense = dequantize_linear(spec, dtype=dtype)
    leaves = [dense.weight, dense.bias]
    mx.eval(*(leaf for leaf in leaves if leaf is not None))
    prepare_seconds = time.perf_counter() - started

    def operation(
        dense_weight: mx.array = dense.weight,
        dense_bias: mx.array | None = dense.bias,
    ) -> mx.array:
        out = hidden @ dense_weight.T
        if dense_bias is not None:
            out = out + dense_bias
        return out

    return CandidateOp(
        name=DENSE_PATH,
        operation=operation,
        prepare_seconds=prepare_seconds,
        materialized_bytes=array_nbytes(leaves),
        note="pre-dequantized dense weight upper bound; not a q6 checkpoint layout",
    )


def probe_mlx_kquant(codec: str, *, skip: bool) -> tuple[Any | None, BackendStatus]:
    if skip:
        return (
            None,
            BackendStatus(
                name="mlx_kquant",
                status="skipped",
                reason="disabled by --skip-mlx-kquant",
            ),
        )
    try:
        module = importlib.import_module("mlx_kquant")
    except ImportError as exc:
        return (
            None,
            BackendStatus(
                name="mlx_kquant",
                status="missing",
                reason=(
                    "Python package is not installed; install mlx-kquant in the "
                    "Metal benchmark environment to measure this backend"
                ),
                details={"error": str(exc)},
            ),
        )
    try:
        package_version = version("mlx-kquant")
    except PackageNotFoundError:
        package_version = None
    details: dict[str, Any] = {}
    try:
        codecs = tuple(module.codecs())
        details = {**details, "codecs": codecs}
    except Exception as exc:  # pragma: no cover - depends on optional extension
        return (
            None,
            BackendStatus(
                name="mlx_kquant",
                status="error",
                reason="mlx_kquant imported but codecs() failed",
                version=package_version,
                details={"error": repr(exc)},
            ),
        )
    if codec not in codecs:
        return (
            None,
            BackendStatus(
                name="mlx_kquant",
                status="skipped",
                reason=f"codec {codec!r} is not available",
                version=package_version,
                details=details,
            ),
        )
    try:
        details = {**details, "metallib_loads": bool(module.metallib_loads())}
    except Exception as exc:  # pragma: no cover - depends on optional extension
        details = {**details, "metallib_loads_error": repr(exc)}
    return (
        module,
        BackendStatus(
            name="mlx_kquant",
            status="ready",
            reason=(
                "will requantize affine q6 weights into K-quant for kernel "
                "throughput comparison; this is not a drop-in checkpoint path"
            ),
            version=package_version,
            details=details,
        ),
    )


def prepare_kquant_candidate(
    hidden: mx.array,
    spec: Q6LinearSpec,
    *,
    dtype: mx.Dtype,
    codec: str,
    kquant: Any,
) -> CandidateOp:
    started = time.perf_counter()
    dense = dequantize_linear(spec, dtype=mx.float16 if dtype == mx.bfloat16 else dtype)
    mx.eval(dense.weight)
    quantized_weight, scales = kquant.quantize(dense.weight, codec)
    mx.eval(quantized_weight, scales)
    bias = None if spec.bias is None else spec.bias.astype(mx.float32)
    if bias is not None:
        mx.eval(bias)
    prepare_seconds = time.perf_counter() - started
    path_name = f"{KQUANT_PATH_PREFIX}_{codec}"

    def operation() -> mx.array:
        leading = hidden.shape[:-1]
        flat = hidden.reshape((-1, hidden.shape[-1]))
        matmul_input = flat.astype(mx.float16 if dtype == mx.bfloat16 else dtype)
        out = kquant.quantized_matmul(
            matmul_input,
            quantized_weight,
            scales,
            codec,
            transpose=True,
        )
        if bias is not None:
            out = out + bias.astype(out.dtype)
        return out.reshape(leading + (spec.out_features,))

    return CandidateOp(
        name=path_name,
        operation=operation,
        prepare_seconds=prepare_seconds,
        materialized_bytes=array_nbytes([quantized_weight, scales, bias]),
        note=(
            f"mlx-kquant {codec}; requantized from current affine q6 dense "
            "weights, so error includes codec/layout change"
        ),
    )


def make_candidates(
    layer: Any,
    hidden: mx.array,
    *,
    dtype: mx.Dtype,
    include_fork_tiled: bool,
    include_naive_fork: bool,
    kquant: Any | None,
    kquant_codec: str,
) -> tuple[CandidateOp, ...]:
    spec = quantized_linear_spec(layer)
    candidates = (
        CandidateOp(
            name=BASELINE_PATH,
            operation=lambda module=layer: module(hidden),
            prepare_seconds=0.0,
            materialized_bytes=None,
            note="stock mlx.nn.QuantizedLinear affine q6",
        ),
        prepare_dense_candidate(hidden, spec, dtype=dtype),
    )
    if include_fork_tiled:
        candidates = candidates + (
            CandidateOp(
                name=FORK_TILED_PATH,
                operation=lambda s=spec: affine_q6_qmm_t_tiled(hidden, s),
                prepare_seconds=0.0,
                materialized_bytes=None,
                note="local fork/prototype mx.fast.metal_kernel tiled affine q6",
            ),
        )
    if include_naive_fork:
        candidates = candidates + (
            CandidateOp(
                name=FORK_NAIVE_PATH,
                operation=lambda s=spec: affine_q6_qmm_t(hidden, s),
                prepare_seconds=0.0,
                materialized_bytes=None,
                note="local fork/prototype naive per-output Metal affine q6",
            ),
        )
    if kquant is not None:
        candidates = candidates + (
            prepare_kquant_candidate(
                hidden,
                spec,
                dtype=dtype,
                codec=kquant_codec,
                kquant=kquant,
            ),
        )
    return candidates


def run_layer(
    *,
    block_index: int,
    layer_name: str,
    layer: Any,
    image_tokens: int,
    seed: int,
    dtype: mx.Dtype,
    warmup_runs: int,
    measured_runs: int,
    include_fork_tiled: bool,
    include_naive_fork: bool,
    kquant: Any | None,
    kquant_codec: str,
) -> tuple[PathTiming, ...]:
    spec = quantized_linear_spec(layer)
    hidden = make_random_hidden(
        image_tokens,
        spec.in_features,
        seed=seed,
        dtype=dtype,
    )
    candidates = make_candidates(
        layer,
        hidden,
        dtype=dtype,
        include_fork_tiled=include_fork_tiled,
        include_naive_fork=include_naive_fork,
        kquant=kquant,
        kquant_codec=kquant_codec,
    )
    measured = measure_round_robin(
        tuple((candidate.name, candidate.operation) for candidate in candidates),
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    by_candidate = {candidate.name: candidate for candidate in candidates}
    q6_output, q6_times = measured[BASELINE_PATH]
    q6_summary = summarize_durations(q6_times)
    paths: tuple[PathTiming, ...] = ()
    for candidate in candidates:
        output, durations = measured[candidate.name]
        summary = summarize_durations(durations)
        output_shape = tuple(int(dim) for dim in output.shape)
        paths = paths + (
            PathTiming(
                name=candidate.name,
                block_index=block_index,
                layer=layer_name,
                input_shape=tuple(int(dim) for dim in hidden.shape),
                output_shape=output_shape,
                weight_shape=(spec.out_features, spec.in_features),
                bits=spec.bits,
                group_size=spec.group_size,
                median_seconds=summary.median_seconds,
                min_seconds=summary.min_seconds,
                mean_seconds=summary.mean_seconds,
                durations_seconds=tuple(durations),
                relative_to_mlx_q6=(
                    1.0
                    if candidate.name == BASELINE_PATH
                    else relative_speedup(q6_summary.median_seconds, summary.median_seconds)
                ),
                max_abs_error_vs_mlx_q6=(
                    0.0 if candidate.name == BASELINE_PATH else max_abs_error(q6_output, output)
                ),
                all_finite=all_finite(output),
                prepare_seconds=by_candidate[candidate.name].prepare_seconds,
                materialized_bytes=by_candidate[candidate.name].materialized_bytes,
                note=by_candidate[candidate.name].note,
            ),
        )
    return paths


def aggregate_paths(
    paths: Sequence[PathTiming],
    expected_layers: int,
) -> tuple[AggregateTiming, ...]:
    names = tuple(dict.fromkeys(path.name for path in paths))
    baseline_total = sum(
        path.median_seconds for path in paths if path.name == BASELINE_PATH
    )
    aggregates: tuple[AggregateTiming, ...] = ()
    for name in names:
        selected = tuple(path for path in paths if path.name == name)
        total = sum(path.median_seconds for path in selected)
        materialized = (
            None
            if any(path.materialized_bytes is None for path in selected)
            else sum(int(path.materialized_bytes or 0) for path in selected)
        )
        aggregates = aggregates + (
            AggregateTiming(
                name=name,
                measured_layers=len(selected),
                expected_layers=expected_layers,
                median_seconds=total,
                relative_to_mlx_q6=(
                    1.0 if name == BASELINE_PATH else relative_speedup(baseline_total, total)
                ),
                max_abs_error_vs_mlx_q6=max(
                    path.max_abs_error_vs_mlx_q6 or 0.0 for path in selected
                ),
                all_finite=all(path.all_finite for path in selected),
                materialized_bytes=materialized,
                complete=len(selected) == expected_layers,
            ),
        )
    return aggregates


def run_benchmark(args: argparse.Namespace) -> KernelSpikeResult:
    dtype = DTYPES[args.dtype]
    model_dir = args.model.expanduser().resolve()
    kquant, kquant_status = probe_mlx_kquant(
        args.kquant_codec,
        skip=args.skip_mlx_kquant,
    )
    backend_statuses = (
        BackendStatus(
            name=BASELINE_PATH,
            status="enabled",
            reason="stock MLX affine QuantizedLinear baseline",
            version=version("mlx"),
        ),
        BackendStatus(
            name=DENSE_PATH,
            status="enabled",
            reason="pre-dequantized dense upper bound",
        ),
        BackendStatus(
            name=FORK_TILED_PATH,
            status="skipped" if args.skip_fork_tiled else "enabled",
            reason=(
                "disabled by --skip-fork-tiled"
                if args.skip_fork_tiled
                else "local fork/prototype tiled Metal affine-q6 kernel"
            ),
        ),
        kquant_status,
    )
    if args.include_naive_fork:
        backend_statuses = backend_statuses + (
            BackendStatus(
                name=FORK_NAIVE_PATH,
                status="enabled",
                reason="slow local naive Metal kernel; diagnostic only",
            ),
        )

    all_paths: tuple[PathTiming, ...] = ()
    for block_index in args.block_indices:
        print(f"loading q6 block {block_index}", file=sys.stderr, flush=True)
        block = load_block(model_dir, block_index, bits=6)
        layer_items = (
            ("img_ff.mlp_in", block.img_ff.mlp_in),
            ("img_ff.mlp_out", block.img_ff.mlp_out),
        )
        for layer_offset, (layer_name, layer) in enumerate(layer_items):
            print(
                f"benchmarking block {block_index} {layer_name}",
                file=sys.stderr,
                flush=True,
            )
            layer_seed = args.seed + block_index * 10 + layer_offset
            all_paths = all_paths + run_layer(
                block_index=block_index,
                layer_name=layer_name,
                layer=layer,
                image_tokens=args.image_tokens,
                seed=layer_seed,
                dtype=dtype,
                warmup_runs=args.warmup,
                measured_runs=args.runs,
                include_fork_tiled=not args.skip_fork_tiled,
                include_naive_fork=args.include_naive_fork,
                kquant=kquant,
                kquant_codec=args.kquant_codec,
            )
    expected_layers = len(args.block_indices) * 2
    aggregates = aggregate_paths(all_paths, expected_layers)
    complete_summaries = tuple(
        KernelPathSummary(
            name=aggregate.name,
            median_seconds=aggregate.median_seconds,
            max_abs_error_vs_mlx_q6=aggregate.max_abs_error_vs_mlx_q6,
            all_finite=aggregate.all_finite,
        )
        for aggregate in aggregates
        if aggregate.complete
    )
    verdict, reason = decide_img_ff_kernel_spike_verdict(
        complete_summaries,
        speedup_threshold=args.speedup_threshold,
        max_abs_error_tolerance=args.max_abs_error_tolerance,
    )
    return KernelSpikeResult(
        mlx_version=version("mlx"),
        platform=platform.platform(),
        model=str(model_dir),
        block_indices=args.block_indices,
        image_tokens=args.image_tokens,
        dtype=args.dtype,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        kquant_codec=args.kquant_codec,
        backend_statuses=backend_statuses,
        paths=all_paths,
        aggregate_paths=aggregates,
        verdict=verdict,
        verdict_reason=reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-indices", type=parse_block_indices, default=(0,))
    parser.add_argument("--image-tokens", type=positive_int, default=3456)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--speedup-threshold",
        type=float,
        default=0.10,
        help="Candidate kernel must be at least 1+threshold faster than MLX q6",
    )
    parser.add_argument(
        "--max-abs-error-tolerance",
        type=float,
        default=32.0,
        help="Max abs error gate vs MLX q6 for non-dense kernel candidates",
    )
    parser.add_argument(
        "--skip-fork-tiled",
        action="store_true",
        help="Do not measure the local tiled Metal q6 prototype",
    )
    parser.add_argument(
        "--include-naive-fork",
        action="store_true",
        help="Also measure the very slow naive per-output Metal q6 prototype",
    )
    parser.add_argument(
        "--skip-mlx-kquant",
        action="store_true",
        help="Do not import or measure optional mlx-kquant",
    )
    parser.add_argument(
        "--kquant-codec",
        default="q6_k",
        help="mlx-kquant codec to test when the optional package is installed",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.speedup_threshold < 0:
        parser.error("--speedup-threshold cannot be negative")
    if args.max_abs_error_tolerance < 0:
        parser.error("--max-abs-error-tolerance cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    payload = json.dumps(asdict(result), indent=2, ensure_ascii=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(
        f"verdict={result.verdict} "
        + " ".join(
            f"{path.name}={path.median_seconds:.4f}s"
            + (
                f"({path.relative_to_mlx_q6:.3f}x)"
                if path.relative_to_mlx_q6 is not None and path.name != BASELINE_PATH
                else ""
            )
            for path in result.aggregate_paths
            if path.complete
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
