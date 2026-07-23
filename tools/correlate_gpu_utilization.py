#!/usr/bin/env python3
"""Correlate GPU utilization samples with SHARDEDIT_TIMING intervals."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shardedit_mlx.full_miss_profile import (
    load_timing_events,
    parse_timing_events,
    stdout_log_for_run_dir,
)
from shardedit_mlx.gpu_utilization_sampler import GpuUtilizationSample


@dataclass(frozen=True)
class IntervalSummary:
    name: str
    step: int | None
    start_wall: float
    end_wall: float
    duration_seconds: float
    sample_count: int
    mean_device_utilization_percent: float | None
    mean_renderer_utilization_percent: float | None
    mean_tiler_utilization_percent: float | None
    pageins_pages: int | None
    pageouts_pages: int | None
    swapins_pages: int | None
    swapouts_pages: int | None
    pageins_bytes: int | None
    swapins_bytes: int | None
    swap_used_delta_bytes: int | None


def load_gpu_samples(path: Path) -> tuple[GpuUtilizationSample, ...]:
    samples: list[GpuUtilizationSample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        samples.append(
            GpuUtilizationSample(
                timestamp_monotonic=float(payload["timestamp_monotonic"]),
                wall_time=float(payload["wall_time"]),
                device_utilization_percent=payload.get("device_utilization_percent"),
                renderer_utilization_percent=payload.get("renderer_utilization_percent"),
                tiler_utilization_percent=payload.get("tiler_utilization_percent"),
                alloc_system_memory_bytes=payload.get("alloc_system_memory_bytes"),
                in_use_system_memory_bytes=payload.get("in_use_system_memory_bytes"),
                pageins=payload.get("pageins"),
                pageouts=payload.get("pageouts"),
                swapins=payload.get("swapins"),
                swapouts=payload.get("swapouts"),
                page_size_bytes=payload.get("page_size_bytes"),
                swap_used_bytes=payload.get("swap_used_bytes"),
                swap_total_bytes=payload.get("swap_total_bytes"),
                errors=tuple(payload.get("errors") or ()),
            )
        )
    return tuple(samples)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def reconstruct_timing_intervals(
    events: Sequence[dict],
    *,
    wall_anchor: float,
    monotonic_anchor: float,
) -> tuple[dict, ...]:
    """Recover approximate wall intervals from timing events.

    SHARDEDIT_TIMING events do not embed absolute timestamps. Callers should pass
    the sampler metadata anchors (``started_wall_time`` / ``started_monotonic``)
    and any optional ``timestamp_monotonic`` fields on events. When events lack
    monotonic timestamps, intervals are reconstructed by walking events in order
    and assigning contiguous wall ranges from recorded ``seconds``.
    """

    intervals: list[dict] = []
    cursor = wall_anchor
    for event in events:
        name = event.get("name")
        if not isinstance(name, str):
            continue
        seconds = _optional_float(event.get("seconds"))
        if seconds is None:
            continue
        event_mono = _optional_float(event.get("timestamp_monotonic"))
        if event_mono is not None:
            end_wall = wall_anchor + (event_mono - monotonic_anchor)
            start_wall = end_wall - seconds
        else:
            start_wall = cursor
            end_wall = cursor + seconds
            cursor = end_wall
        intervals.append(
            {
                "name": name,
                "step": _optional_int(event.get("step")),
                "start_wall": start_wall,
                "end_wall": end_wall,
                "duration_seconds": seconds,
                "cache_hit": event.get("cache_hit"),
                "compute_seconds": _optional_float(event.get("compute_seconds")),
                "load_seconds": _optional_float(event.get("load_seconds")),
                "lora_seconds": _optional_float(event.get("lora_seconds")),
                "release_seconds": _optional_float(event.get("release_seconds")),
            }
        )
    return tuple(intervals)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _delta(first: int | None, last: int | None) -> int | None:
    if first is None or last is None:
        return None
    return last - first


def summarize_interval(
    interval: dict,
    samples: Sequence[GpuUtilizationSample],
) -> IntervalSummary:
    start = float(interval["start_wall"])
    end = float(interval["end_wall"])
    covered = tuple(
        sample for sample in samples if start <= sample.wall_time <= end
    )
    if not covered and samples:
        # Fall back to nearest samples when reconstruction is coarse.
        covered = tuple(
            sample
            for sample in samples
            if abs(sample.wall_time - ((start + end) / 2.0))
            <= max(end - start, 0.5)
        )

    device = [
        float(sample.device_utilization_percent)
        for sample in covered
        if sample.device_utilization_percent is not None
    ]
    renderer = [
        float(sample.renderer_utilization_percent)
        for sample in covered
        if sample.renderer_utilization_percent is not None
    ]
    tiler = [
        float(sample.tiler_utilization_percent)
        for sample in covered
        if sample.tiler_utilization_percent is not None
    ]

    first = covered[0] if covered else None
    last = covered[-1] if covered else None
    page_size = first.page_size_bytes if first is not None else None
    pageins = _delta(
        None if first is None else first.pageins,
        None if last is None else last.pageins,
    )
    pageouts = _delta(
        None if first is None else first.pageouts,
        None if last is None else last.pageouts,
    )
    swapins = _delta(
        None if first is None else first.swapins,
        None if last is None else last.swapins,
    )
    swapouts = _delta(
        None if first is None else first.swapouts,
        None if last is None else last.swapouts,
    )
    swap_used_delta = _delta(
        None if first is None else first.swap_used_bytes,
        None if last is None else last.swap_used_bytes,
    )
    return IntervalSummary(
        name=str(interval["name"]),
        step=interval.get("step"),
        start_wall=start,
        end_wall=end,
        duration_seconds=float(interval["duration_seconds"]),
        sample_count=len(covered),
        mean_device_utilization_percent=_mean(device),
        mean_renderer_utilization_percent=_mean(renderer),
        mean_tiler_utilization_percent=_mean(tiler),
        pageins_pages=pageins,
        pageouts_pages=pageouts,
        swapins_pages=swapins,
        swapouts_pages=swapouts,
        pageins_bytes=None if pageins is None or page_size is None else pageins * page_size,
        swapins_bytes=None if swapins is None or page_size is None else swapins * page_size,
        swap_used_delta_bytes=swap_used_delta,
    )


def summarize_by_wall_window(
    *,
    label: str,
    start_wall: float,
    end_wall: float,
    samples: Sequence[GpuUtilizationSample],
) -> IntervalSummary:
    return summarize_interval(
        {
            "name": label,
            "step": None,
            "start_wall": start_wall,
            "end_wall": end_wall,
            "duration_seconds": end_wall - start_wall,
        },
        samples,
    )


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_optional(value: float | int | None, *, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _print_summaries(summaries: Sequence[IntervalSummary]) -> None:
    print(
        "name\tstep\tduration\tsamples\tmean_device%\tmean_renderer%\t"
        "pageins_pages\tswapins_pages\tswap_used_delta_bytes"
    )
    for item in summaries:
        print(
            "\t".join(
                (
                    item.name,
                    "-" if item.step is None else str(item.step),
                    f"{item.duration_seconds:.2f}s",
                    str(item.sample_count),
                    _format_optional(item.mean_device_utilization_percent),
                    _format_optional(item.mean_renderer_utilization_percent),
                    _format_optional(item.pageins_pages),
                    _format_optional(item.swapins_pages),
                    _format_optional(item.swap_used_delta_bytes),
                )
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="gpu_utilization.jsonl from tools/sample_gpu_utilization.py",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="sample_metadata.json (provides wall/monotonic anchors)",
    )
    parser.add_argument(
        "--timing-log",
        type=Path,
        help="stdout.log or text containing SHARDEDIT_TIMING lines",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Benchmark run directory; loads stdout.log via full_miss_profile",
    )
    parser.add_argument(
        "--names",
        default="denoise_transformer,residency_window",
        help="Comma-separated timing event names to summarize",
    )
    parser.add_argument(
        "--window-label",
        action="append",
        default=[],
        metavar="LABEL:START:END",
        help="Manual wall-time window (repeatable). START/END are unix epochs.",
    )
    parser.add_argument(
        "--align-denoise-from-end",
        action="store_true",
        help=(
            "Place denoise_transformer intervals contiguously ending at the last "
            "sample timestamp. Prefer this when SHARDEDIT_TIMING events lack absolute "
            "timestamps and forward reconstruction drifts."
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    samples = load_gpu_samples(args.samples)
    if not samples:
        raise SystemExit(f"no samples in {args.samples}")

    metadata = load_metadata(args.metadata) if args.metadata else {}
    wall_anchor = float(metadata.get("started_wall_time", samples[0].wall_time))
    monotonic_anchor = float(
        metadata.get("started_monotonic", samples[0].timestamp_monotonic)
    )

    summaries: list[IntervalSummary] = []

    for window_spec in args.window_label:
        try:
            label, start_text, end_text = window_spec.split(":", 2)
            start_wall = float(start_text)
            end_wall = float(end_text)
        except ValueError as error:
            raise SystemExit(
                f"invalid --window-label {window_spec!r}; expected LABEL:START:END"
            ) from error
        summaries.append(
            summarize_by_wall_window(
                label=label,
                start_wall=start_wall,
                end_wall=end_wall,
                samples=samples,
            )
        )

    timing_text: str | None = None
    if args.timing_log is not None:
        timing_text = args.timing_log.read_text(encoding="utf-8", errors="replace")
    elif args.run_dir is not None:
        timing_text = stdout_log_for_run_dir(args.run_dir).read_text(
            encoding="utf-8", errors="replace"
        )

    if timing_text is not None:
        events = parse_timing_events(timing_text)
        wanted = {name.strip() for name in args.names.split(",") if name.strip()}
        if args.align_denoise_from_end and "denoise_transformer" in wanted:
            denoise_events = [
                event for event in events if event.get("name") == "denoise_transformer"
            ]
            cursor = samples[-1].wall_time
            aligned: list[dict] = []
            for event in reversed(denoise_events):
                seconds = _optional_float(event.get("seconds"))
                if seconds is None:
                    continue
                end_wall = cursor
                start_wall = end_wall - seconds
                aligned.append(
                    {
                        "name": "denoise_transformer",
                        "step": _optional_int(event.get("step")),
                        "start_wall": start_wall,
                        "end_wall": end_wall,
                        "duration_seconds": seconds,
                        "cache_hit": event.get("cache_hit"),
                    }
                )
                cursor = start_wall
            for interval in reversed(aligned):
                summaries.append(summarize_interval(interval, samples))
            wanted = wanted - {"denoise_transformer"}
        intervals = reconstruct_timing_intervals(
            events,
            wall_anchor=wall_anchor,
            monotonic_anchor=monotonic_anchor,
        )
        for interval in intervals:
            if interval["name"] in wanted:
                summaries.append(summarize_interval(interval, samples))

    if not summaries:
        # Whole-run summary when no timing intervals were provided.
        summaries.append(
            summarize_by_wall_window(
                label="full_sample_window",
                start_wall=samples[0].wall_time,
                end_wall=samples[-1].wall_time,
                samples=samples,
            )
        )

    if args.json:
        print(json.dumps([asdict(item) for item in summaries], indent=2, sort_keys=True))
    else:
        _print_summaries(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
