#!/usr/bin/env python3
"""Check whether low-redundancy target tokens track real image edge density.

Follow-up to the §37/§38 token-redundancy diagnostics
(`docs/experiments/2026-07-17-m2-qwen-edit.md`): a single cafe-portrait case
showed low-similarity ("unique") target tokens correlating with local edge/
detail density in the generated image, more strongly at deeper blocks (close
to F1B2's B2 back-blocks). This is an offline, read-only check to repeat that
finding across more cases before committing to any "selective B-block refill"
runtime change -- it does not change any Transformer computation.

Typical usage:

  # 1. Collect heatmaps for a case (see --shardedit-token-redundancy-blocks /
  #    --shardedit-token-redundancy-heatmap-dir in mflux_fast_edit.py):
  SHARDEDIT_IMAGE_PATH=/path/to/other-reference.png \\
  SHARDEDIT_TOKEN_REDUNDANCY_BLOCKS="1,58" \\
  SHARDEDIT_TOKEN_REDUNDANCY_HEATMAP_DIR="/tmp/heatmaps-case2" \\
    benchmarks/run_qwen_edit_benchmark.sh --runtime shardedit --residency shard --steps 8

  # 2. Correlate every step*_block*_target.png in that directory against the
  #    run's generated output image, and save highlight overlays:
  python3 tools/analyze_token_redundancy_heatmap.py \\
    /tmp/heatmaps-case2 \\
    benchmark-runs/<date>/<run>/shardedit-1.png \\
    --overlay-dir /tmp/heatmaps-case2/overlays
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from PIL import Image, ImageFilter

from shardedit_mlx.token_redundancy_edge_correlation import (
    downsample_grid,
    pearson_correlation,
    spearman_correlation,
)

_HEATMAP_NAME = re.compile(r"^step(?P<step>\d+)_block(?P<block>\d+)_(?P<region>\w+)\.png$")


def _redness_grid(heatmap: Image.Image) -> list[list[float]]:
    rgb = heatmap.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    return [[pixels[x, y][0] - pixels[x, y][2] for x in range(width)] for y in range(height)]


def _edge_energy_grid(image: Image.Image, *, grid_height: int, grid_width: int) -> list[list[float]]:
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    width, height = edges.size
    pixels = edges.load()
    raw = [[float(pixels[x, y]) for x in range(width)] for y in range(height)]
    return downsample_grid(raw, grid_height=grid_height, grid_width=grid_width)


def _save_overlay(
    *,
    image: Image.Image,
    redness: list[list[float]],
    top_fraction: float,
    out_path: Path,
) -> None:
    flat = sorted(value for row in redness for value in row)
    threshold_index = max(0, int(len(flat) * (1.0 - top_fraction)) - 1)
    threshold = flat[threshold_index]

    grid_height = len(redness)
    grid_width = len(redness[0])
    width, height = image.size
    cell_height = height / grid_height
    cell_width = width / grid_width

    overlay = image.convert("RGB").copy()
    pixels = overlay.load()
    for row in range(grid_height):
        for col in range(grid_width):
            if redness[row][col] < threshold:
                continue
            y0, y1 = int(row * cell_height), int((row + 1) * cell_height)
            x0, x1 = int(col * cell_width), int((col + 1) * cell_width)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (
                        round(r * 0.45 + 255 * 0.55),
                        round(g * 0.45 + 255 * 0.55),
                        round(b * 0.45 + 0 * 0.55),
                    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("heatmap_dir", type=Path, help="directory of step*_block*_target.png files")
    parser.add_argument("generated_image", type=Path, help="the run's final generated output image")
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="if set, save a highlight overlay per heatmap (top-K%% redness cells on the real image)",
    )
    parser.add_argument(
        "--overlay-top-fraction",
        type=float,
        default=0.15,
        help="fraction of cells (by redness) to highlight in the overlay (default 0.15)",
    )
    args = parser.parse_args()

    heatmap_paths = sorted(args.heatmap_dir.glob("step*_block*_*.png"))
    if not heatmap_paths:
        raise SystemExit(f"no step*_block*_*.png heatmaps found in {args.heatmap_dir}")

    generated = Image.open(args.generated_image)

    print(f"# heatmaps: {args.heatmap_dir}")
    print(f"# generated image: {args.generated_image} ({generated.size[0]}x{generated.size[1]})")
    print()
    print("step\tblock\tregion\tmean_redness\tpearson\tspearman")
    for path in heatmap_paths:
        match = _HEATMAP_NAME.match(path.name)
        if not match:
            continue
        heatmap = Image.open(path)
        redness = _redness_grid(heatmap)
        grid_height, grid_width = len(redness), len(redness[0])
        edge_energy = _edge_energy_grid(generated, grid_height=grid_height, grid_width=grid_width)
        pearson = pearson_correlation(redness, edge_energy)
        spearman = spearman_correlation(redness, edge_energy)
        mean_redness = sum(v for row in redness for v in row) / (grid_height * grid_width)
        print(
            f"{match['step']}\t{match['block']}\t{match['region']}\t"
            f"{mean_redness:+.1f}\t{pearson:+.3f}\t{spearman:+.3f}"
        )
        if args.overlay_dir is not None:
            _save_overlay(
                image=generated,
                redness=redness,
                top_fraction=args.overlay_top_fraction,
                out_path=args.overlay_dir / f"{path.stem}_overlay.png",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
