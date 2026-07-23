#!/usr/bin/env python3
"""Profile the logical stages inside one real q6 Qwen Transformer block."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
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

from benchmark_qwen_block import DEFAULT_MODEL, load_block, make_inputs, positive_int
from shardedit_mlx.block_profile import ComponentTiming, summarize_component_durations


@dataclass(frozen=True)
class BlockComponentProfile:
    mlx_version: str
    block_index: int
    bits: int
    image_tokens: int
    text_tokens: int
    warmup_runs: int
    measured_runs: int
    whole_block_median_seconds: float
    whole_block_min_seconds: float
    compiled_block_median_seconds: float | None
    compiled_speedup: float | None
    component_median_sum_seconds: float
    component_barrier_ratio: float
    reconstructed_max_abs_error: float
    peak_memory_gib: float
    components: tuple[ComponentTiming, ...]


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


def measure(
    operation: Callable[[], Any],
    *,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[Any, tuple[float, ...]]:
    output: Any = None
    for _ in range(warmup_runs):
        output = operation()
        evaluate(output)

    durations: list[float] = []
    for _ in range(measured_runs):
        started_at = time.perf_counter()
        output = operation()
        evaluate(output)
        durations.append(time.perf_counter() - started_at)
    return output, tuple(durations)


def profile_block(args: argparse.Namespace) -> BlockComponentProfile:
    block = load_block(args.model.expanduser().resolve(), args.block_index, args.bits)
    inputs = make_inputs(args.image_tokens, args.text_tokens)
    hidden_states = inputs["hidden_states"]
    encoder_hidden_states = inputs["encoder_hidden_states"]
    encoder_mask = inputs["encoder_hidden_states_mask"]
    text_embeddings = inputs["text_embeddings"]
    image_rotary_emb = inputs["image_rotary_emb"]

    def whole_block() -> tuple[mx.array, mx.array]:
        return block(**inputs, block_idx=args.block_index)

    whole_output, whole_durations = measure(
        whole_block,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    compiled_median: float | None = None
    compiled_speedup: float | None = None
    if args.compare_compile:
        mask_is_all_ones = mx.all(encoder_mask >= 0.999)
        mx.eval(mask_is_all_ones)
        if not bool(mask_is_all_ones.item()):
            raise RuntimeError("compiled Block comparison currently requires an all-ones text mask")

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
                block_idx=args.block_index,
            )

        compiled_function = mx.compile(block_function)
        compiled_output, compiled_durations = measure(
            lambda: compiled_function(
                hidden_states,
                encoder_hidden_states,
                text_embeddings,
                image_rotary_emb[0][0],
                image_rotary_emb[0][1],
                image_rotary_emb[1][0],
                image_rotary_emb[1][1],
            ),
            warmup_runs=max(1, args.warmup),
            measured_runs=args.runs,
        )
        compile_error = mx.maximum(
            mx.max(mx.abs(whole_output[0] - compiled_output[0])),
            mx.max(mx.abs(whole_output[1] - compiled_output[1])),
        )
        mx.eval(compile_error)
        if float(compile_error.item()) != 0.0:
            raise RuntimeError(f"compiled Block output differs by {float(compile_error.item())}")
        compiled_median = statistics.median(compiled_durations)
        compiled_speedup = statistics.median(whole_durations) / compiled_median

    if args.metal_capture is not None:
        capture_path = args.metal_capture.expanduser().resolve()
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        mx.metal.start_capture(str(capture_path))
        try:
            evaluate(whole_block())
        finally:
            mx.metal.stop_capture()

    mx.reset_peak_memory()
    stage_durations: dict[str, tuple[float, ...]] = {}

    modulation, stage_durations["modulation_linears"] = measure(
        lambda: (
            block.img_mod_linear(block.img_mod_silu(text_embeddings)),
            block.txt_mod_linear(block.txt_mod_silu(text_embeddings)),
        ),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    img_mod1, img_mod2 = mx.split(modulation[0], 2, axis=-1)
    txt_mod1, txt_mod2 = mx.split(modulation[1], 2, axis=-1)

    pre_attention, stage_durations["pre_attention_norm_modulate"] = measure(
        lambda: (
            *block._modulate(block.img_norm1(hidden_states), img_mod1),
            *block._modulate(block.txt_norm1(encoder_hidden_states), txt_mod1),
        ),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    img_modulated, img_gate1, txt_modulated, txt_gate1 = pre_attention

    def project_qkv() -> tuple[mx.array, ...]:
        attention = block.attn
        tensors = (
            attention.to_q(img_modulated),
            attention.to_k(img_modulated),
            attention.to_v(img_modulated),
            attention.add_q_proj(txt_modulated),
            attention.add_k_proj(txt_modulated),
            attention.add_v_proj(txt_modulated),
        )
        return tuple(
            mx.reshape(tensor, (tensor.shape[0], tensor.shape[1], attention.num_heads, attention.head_dim))
            for tensor in tensors
        )

    qkv, stage_durations["attention_qkv_projections"] = measure(
        project_qkv,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    def normalize_rope_and_join() -> tuple[mx.array, mx.array, mx.array]:
        attention = block.attn
        img_query = attention.norm_q(qkv[0])
        img_key = attention.norm_k(qkv[1])
        txt_query = attention.norm_added_q(qkv[3])
        txt_key = attention.norm_added_k(qkv[4])
        (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
        img_query = attention._apply_rope_qwen(img_query, img_cos, img_sin)
        img_key = attention._apply_rope_qwen(img_key, img_cos, img_sin)
        txt_query = attention._apply_rope_qwen(txt_query, txt_cos, txt_sin)
        txt_key = attention._apply_rope_qwen(txt_key, txt_cos, txt_sin)
        return (
            mx.concatenate([txt_query, img_query], axis=1),
            mx.concatenate([txt_key, img_key], axis=1),
            mx.concatenate([qkv[5], qkv[2]], axis=1),
        )

    joint_qkv, stage_durations["attention_qk_norm_rope_join"] = measure(
        normalize_rope_and_join,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    attention_mask, stage_durations["attention_mask_conversion"] = measure(
        lambda: block.attn._convert_mask_for_qwen(
            mask=encoder_mask,
            joint_seq_len=joint_qkv[0].shape[1],
            txt_seq_len=args.text_tokens,
        ),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    attention_hidden, stage_durations["attention_sdpa"] = measure(
        lambda: block.attn._compute_attention_qwen(
            query=joint_qkv[0],
            key=joint_qkv[1],
            value=joint_qkv[2],
            mask=attention_mask,
            block_idx=args.block_index,
        ),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    def attention_outputs() -> tuple[mx.array, mx.array]:
        text_output = attention_hidden[:, : args.text_tokens, :]
        image_output = attention_hidden[:, args.text_tokens :, :]
        return block.attn.attn_to_out[0](image_output), block.attn.to_add_out(text_output)

    projected_attention, stage_durations["attention_output_projections"] = measure(
        attention_outputs,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    def post_attention() -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        image_hidden = hidden_states + img_gate1 * projected_attention[0]
        text_hidden = encoder_hidden_states + txt_gate1 * projected_attention[1]
        image_mlp_input, image_gate = block._modulate(block.img_norm2(image_hidden), img_mod2)
        text_mlp_input, text_gate = block._modulate(block.txt_norm2(text_hidden), txt_mod2)
        return image_hidden, text_hidden, image_mlp_input, image_gate, text_mlp_input, text_gate

    post_attention_values, stage_durations["post_attention_norm_modulate"] = measure(
        post_attention,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    image_mlp_hidden, stage_durations["image_mlp_in_gelu"] = measure(
        lambda: nn.gelu_approx(block.img_ff.mlp_in(post_attention_values[2])),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    image_mlp, stage_durations["image_mlp_out"] = measure(
        lambda: block.img_ff.mlp_out(image_mlp_hidden),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    text_mlp_hidden, stage_durations["text_mlp_in_gelu"] = measure(
        lambda: nn.gelu_approx(block.txt_ff.mlp_in(post_attention_values[4])),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    text_mlp, stage_durations["text_mlp_out"] = measure(
        lambda: block.txt_ff.mlp_out(text_mlp_hidden),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    reconstructed, stage_durations["residual_outputs"] = measure(
        lambda: (
            post_attention_values[1] + post_attention_values[5] * text_mlp,
            post_attention_values[0] + post_attention_values[3] * image_mlp,
        ),
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )

    max_error = mx.maximum(
        mx.max(mx.abs(whole_output[0] - reconstructed[0])),
        mx.max(mx.abs(whole_output[1] - reconstructed[1])),
    )
    mx.eval(max_error)

    components = summarize_component_durations(stage_durations)
    component_sum = sum(component.median_seconds for component in components)
    whole_median = statistics.median(whole_durations)
    return BlockComponentProfile(
        mlx_version=version("mlx"),
        block_index=args.block_index,
        bits=args.bits,
        image_tokens=args.image_tokens,
        text_tokens=args.text_tokens,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        whole_block_median_seconds=whole_median,
        whole_block_min_seconds=min(whole_durations),
        compiled_block_median_seconds=compiled_median,
        compiled_speedup=compiled_speedup,
        component_median_sum_seconds=component_sum,
        component_barrier_ratio=component_sum / whole_median,
        reconstructed_max_abs_error=float(max_error.item()),
        peak_memory_gib=mx.get_peak_memory() / 1024**3,
        components=components,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--bits", type=int, choices=(4, 5, 6), default=6)
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--compare-compile", action="store_true")
    parser.add_argument("--metal-capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.block_index < 0 or args.block_index >= 60:
        parser.error("--block-index must be between 0 and 59")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    result = profile_block(args)
    payload = json.dumps(asdict(result), indent=2) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
