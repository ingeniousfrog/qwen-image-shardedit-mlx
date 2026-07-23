#!/usr/bin/env python3
"""Compare two generated images with qwen-image-shardedit-mlx pixel-level quality metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from shardedit_mlx.quality_metrics import compare_image_files, metrics_pass_thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Baseline image path")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate image path")
    parser.add_argument("--max-mae", type=float, default=None)
    parser.add_argument("--min-psnr-db", type=float, default=None)
    parser.add_argument("--max-changed-channel-ratio", type=float, default=None)
    args = parser.parse_args()

    metrics = compare_image_files(args.reference, args.candidate)
    passed = metrics_pass_thresholds(
        metrics,
        max_mae=args.max_mae,
        min_psnr_db=args.min_psnr_db,
        max_changed_channel_ratio=args.max_changed_channel_ratio,
    )
    print(
        json.dumps(
            {
                "reference": str(args.reference),
                "candidate": str(args.candidate),
                "passed": passed,
                "metrics": asdict(metrics),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
