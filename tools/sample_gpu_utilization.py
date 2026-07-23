#!/usr/bin/env python3
"""Run a command while sampling GPU utilization and paging counters."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shardedit_mlx.gpu_utilization_sampler import GpuUtilizationSampler


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "benchmark-runs" / "gpu-utilization" / stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for gpu_utilization.jsonl and metadata (default: timestamped)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=0.2,
        help="Sampling interval in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--jsonl-name",
        default="gpu_utilization.jsonl",
        help="Filename for the JSONL sample stream",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after -- (required)",
    )
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide a command after --")

    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / args.jsonl_name
    metadata_path = output_dir / "sample_metadata.json"

    started_wall = time.time()
    started_mono = time.monotonic()
    with GpuUtilizationSampler(
        jsonl_path,
        interval_seconds=args.interval_seconds,
    ) as sampler:
        completed = subprocess.run(command)
        stopped_wall = time.time()
        stopped_mono = time.monotonic()

    metadata = {
        "command": command,
        "output_dir": str(output_dir),
        "jsonl_path": str(jsonl_path),
        "interval_seconds": args.interval_seconds,
        "started_wall_time": started_wall,
        "stopped_wall_time": stopped_wall,
        "started_monotonic": started_mono,
        "stopped_monotonic": stopped_mono,
        "elapsed_seconds": stopped_mono - started_mono,
        "sample_count": sampler.sample_count,
        "returncode": completed.returncode,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"sampled {sampler.sample_count} points -> {jsonl_path} "
        f"(exit={completed.returncode})",
        file=sys.stderr,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
