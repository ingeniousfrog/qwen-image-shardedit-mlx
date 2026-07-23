#!/usr/bin/env python3
"""Print qwen-image-shardedit-mlx token and memory budgets for 24 GiB and 16 GiB Macs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shardedit_mlx.perf_model import EditTokenPlan, RuntimeMemoryPlan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-size", type=int, default=768)
    parser.add_argument("--condition-size", type=int, default=384)
    parser.add_argument("--condition-count", type=int, default=1)
    parser.add_argument("--transformer-gib", type=float, default=15.5)
    parser.add_argument("--activation-reserve-gib", type=float, default=2.5)
    parser.add_argument("--system-reserve-gib", type=float, default=3.0)
    args = parser.parse_args()

    tokens = EditTokenPlan.from_dimensions(
        target_width=args.target_size,
        target_height=args.target_size,
        condition_width=args.condition_size,
        condition_height=args.condition_size,
        condition_count=args.condition_count,
    )
    print(
        f"image tokens: {tokens.image_tokens} "
        f"({tokens.target_tokens} target + {tokens.condition_tokens} condition "
        f"from {args.condition_count} reference image(s))"
    )
    for physical_gib in (24.0, 16.0):
        memory = RuntimeMemoryPlan(
            physical_memory_gib=physical_gib,
            system_reserve_gib=args.system_reserve_gib,
            activation_reserve_gib=args.activation_reserve_gib,
            transformer_size_gib=args.transformer_gib,
            transformer_bits=6,
        )
        bits = memory.recommended_bits()
        recommendation = f"q{bits}" if bits is not None else "below q4 or more staging"
        print(
            f"{physical_gib:.0f} GiB: source q6 requires {memory.required_memory_gib:.1f} GiB; "
            f"recommended Transformer cache: {recommendation}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
