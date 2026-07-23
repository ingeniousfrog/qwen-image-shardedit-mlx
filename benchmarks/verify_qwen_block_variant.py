#!/usr/bin/env python3
"""Save or compare a real q6 Block output across isolated MLX builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import numpy as np

from benchmark_qwen_block import DEFAULT_MODEL, load_block, make_inputs, positive_int


def random_inputs(image_tokens: int, text_tokens: int, seed: int) -> dict[str, Any]:
    mx.random.seed(seed)
    base = make_inputs(image_tokens, text_tokens)
    return {
        **base,
        "hidden_states": mx.random.normal((1, image_tokens, 3072)).astype(mx.bfloat16),
        "encoder_hidden_states": mx.random.normal((1, text_tokens, 3072)).astype(
            mx.bfloat16
        ),
        "text_embeddings": mx.random.normal((1, 3072)).astype(mx.bfloat16),
    }


def to_numpy(values: mx.array) -> np.ndarray:
    return np.asarray(values.astype(mx.float32))


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model.expanduser().resolve()
    block = load_block(model_dir, args.block_index, bits=6)
    inputs = random_inputs(args.image_tokens, args.text_tokens, args.seed)
    output = block(**inputs, block_idx=args.block_index)
    mx.eval(*output)
    text_output, image_output = (to_numpy(value) for value in output)

    reference_path = args.reference.expanduser().resolve()
    if args.write_reference:
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(reference_path, text_output=text_output, image_output=image_output)
        return {
            "mode": "write_reference",
            "reference": str(reference_path),
            "text_shape": list(text_output.shape),
            "image_shape": list(image_output.shape),
            "text_sum": float(text_output.sum(dtype=np.float64)),
            "image_sum": float(image_output.sum(dtype=np.float64)),
        }

    if not reference_path.exists():
        raise FileNotFoundError(f"reference does not exist: {reference_path}")
    with np.load(reference_path) as reference:
        text_error = np.abs(text_output - reference["text_output"])
        image_error = np.abs(image_output - reference["image_output"])
    return {
        "mode": "compare",
        "reference": str(reference_path),
        "text_max_abs_error": float(text_error.max()),
        "text_mean_abs_error": float(text_error.mean()),
        "image_max_abs_error": float(image_error.max()),
        "image_mean_abs_error": float(image_error.mean()),
        "all_finite": bool(
            np.isfinite(text_output).all() and np.isfinite(image_output).all()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--image-tokens", type=positive_int, default=2864)
    parser.add_argument("--text-tokens", type=positive_int, default=206)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("benchmark-runs/qwen-block-q6-2511-reference.npz"),
    )
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()
    if args.block_index < 0 or args.block_index >= 60:
        parser.error("--block-index must be between 0 and 59")

    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
