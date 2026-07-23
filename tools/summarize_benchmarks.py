#!/usr/bin/env python3
"""Summarize qwen-image-shardedit-mlx benchmark run directories."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


TIME_REAL_RE = re.compile(r"^\s*(?P<value>[0-9.]+)\s+real(?:\s|$)", re.MULTILINE)
MAX_RSS_RE = re.compile(r"^\s*(?P<value>[0-9]+)\s+maximum resident set size$", re.MULTILINE)
STATUS_RE = re.compile(r"^status:\s*(?P<value>\d+)$", re.MULTILINE)
ELAPSED_RE = re.compile(r"^elapsed_seconds:\s*(?P<value>\d+)$", re.MULTILINE)
SWIFT_PROFILE_RE = re.compile(
    r"\[profile\]\s*(?P<label>.+?)\s*\(\+(?P<delta>[0-9.]+)s,\s*total\s*(?P<total>[0-9.]+)s"
)
PROGRESS_RE = re.compile(
    r"(?P<done>\d+)/(?P<total>\d+)\s+\[[^\]]+,\s*(?P<seconds>[0-9.]+)s/it\]"
)
SHARDEDIT_TIMING_PREFIX = "SHARDEDIT_TIMING "


@dataclass(frozen=True)
class RunSummary:
    name: str
    status: str
    elapsed_seconds: str
    time_real_seconds: str
    max_rss: str
    output: str
    last_profile: str
    last_progress: str


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def first_match(pattern: re.Pattern[str], text: str, group: str) -> str:
    match = pattern.search(text)
    if not match:
        return "-"
    return match.group(group)


def format_bytes(raw: str) -> str:
    if raw == "-":
        return raw
    value = int(raw)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return raw


def shardedit_mlx_profile(stdout: str) -> str:
    events: list[dict] = []
    for line in stdout.splitlines():
        if not line.startswith(SHARDEDIT_TIMING_PREFIX):
            continue
        try:
            events.append(json.loads(line.removeprefix(SHARDEDIT_TIMING_PREFIX)))
        except json.JSONDecodeError:
            continue
    if not events:
        return "-"

    process = next((event for event in events if event.get("name") == "process_total"), None)
    steps = [event for event in events if event.get("name") == "denoise_transformer"]
    parts: list[str] = []
    if process is not None:
        parts.append(f"process={process['seconds']:.2f}s")
    if steps:
        step_values = ",".join(f"{event['step']}:{event['seconds']:.2f}" for event in steps)
        parts.append(f"steps=[{step_values}]")
        cache_events = [event for event in steps if isinstance(event.get("cache_hit"), bool)]
        if cache_events:
            hits = sum(1 for event in cache_events if event["cache_hit"])
            parts.append(f"cache_hits={hits}/{len(cache_events)}")
            executed_blocks = [event.get("blocks_executed") for event in cache_events]
            if all(isinstance(value, int) for value in executed_blocks):
                parts.append(f"blocks={sum(executed_blocks)}")
    peak = max((float(event.get("peak_memory_gb", 0.0)) for event in events), default=0.0)
    if peak:
        parts.append(f"peak={peak:.2f}GB")
    return " ".join(parts)


def summarize_run(run_dir: Path) -> RunSummary:
    result = read_text(run_dir / "result.txt")
    stdout = read_text(run_dir / "stdout.log")
    stderr = read_text(run_dir / "stderr.log")
    profiles = list(SWIFT_PROFILE_RE.finditer(stderr))
    profile = "-"
    if profiles:
        last = profiles[-1]
        profile = f"{last.group('label')} total={last.group('total')}s"
    shardedit = shardedit_mlx_profile(stdout)
    if shardedit != "-":
        profile = shardedit_mlx
    progress_matches = list(PROGRESS_RE.finditer(stderr.replace("\r", "\n")))
    progress = "-"
    if progress_matches:
        last_progress = progress_matches[-1]
        progress = (
            f"{last_progress.group('done')}/{last_progress.group('total')} "
            f"{last_progress.group('seconds')}s/it"
        )

    outputs = sorted(run_dir.glob("*.png"))
    sibling_output = run_dir.parent / f"{run_dir.name}.png"
    if not outputs and sibling_output.exists():
        outputs = [sibling_output]
    return RunSummary(
        name=run_dir.name,
        status=first_match(STATUS_RE, result, "value"),
        elapsed_seconds=first_match(ELAPSED_RE, result, "value"),
        time_real_seconds=first_match(TIME_REAL_RE, stderr, "value"),
        max_rss=format_bytes(first_match(MAX_RSS_RE, stderr, "value")),
        output=outputs[0].name if outputs else "-",
        last_profile=profile,
        last_progress=progress,
    )


def print_table(rows: list[RunSummary]) -> None:
    columns = [
        ("run", "name"),
        ("status", "status"),
        ("elapsed_s", "elapsed_seconds"),
        ("time_real_s", "time_real_seconds"),
        ("max_rss", "max_rss"),
        ("output", "output"),
        ("last_profile", "last_profile"),
        ("last_progress", "last_progress"),
    ]
    headers = [header for header, _field in columns]
    values = [[getattr(row, field) for _header, field in columns] for row in rows]
    widths = [
        max(len(headers[index]), *(len(value[index]) for value in values)) if values else len(headers[index])
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for value in values:
        print("  ".join(value[index].ljust(widths[index]) for index in range(len(headers))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Benchmark directory, e.g. benchmark-runs/20260717-210000")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        parser.error(f"run directory does not exist: {run_dir}")

    metadata = run_dir / "metadata.txt"
    if metadata.exists():
        print(f"# {run_dir}")
        for line in read_text(metadata).splitlines():
            if line.startswith(("date:", "runtime:", "model:", "lora:", "image:", "width:", "height:", "steps:")):
                print(line)
        print()

    rows = [summarize_run(path) for path in sorted(run_dir.iterdir()) if path.is_dir()]
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
