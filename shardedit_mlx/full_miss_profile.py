"""Summaries for qwen-image-shardedit-mlx full-miss and cache-hit denoise timing logs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path


SHARDEDIT_TIMING_PREFIX = "SHARDEDIT_TIMING "


@dataclass(frozen=True)
class StepTiming:
    step: int
    category: str
    seconds: float
    blocks_executed: int
    cache_reason: str | None
    anchor_seconds: float
    window_compute_seconds: float
    window_load_seconds: float
    window_lora_seconds: float
    window_prepare_seconds: float
    window_release_seconds: float
    window_lora_weight_cache_hits: int
    window_patched_cache_hits: int
    window_patched_cache_size_max: int
    window_kquant_img_ff_cache_hits: int
    window_kquant_img_ff_cache_misses: int
    window_kquant_img_ff_cache_size_max: int
    window_kquant_img_ff_cache_bytes_max: int


@dataclass(frozen=True)
class CategoryTiming:
    category: str
    steps: int
    total_seconds: float
    mean_seconds: float
    total_blocks: int
    mean_blocks: float
    total_anchor_seconds: float
    mean_anchor_seconds: float
    mean_window_compute_seconds: float
    mean_window_load_seconds: float
    mean_window_lora_seconds: float
    mean_window_prepare_seconds: float
    mean_window_release_seconds: float
    total_lora_weight_cache_hits: int
    total_patched_window_cache_hits: int
    max_patched_window_cache_size: int
    total_kquant_img_ff_cache_hits: int
    total_kquant_img_ff_cache_misses: int
    max_kquant_img_ff_cache_size: int
    max_kquant_img_ff_cache_bytes: int


@dataclass(frozen=True)
class FullMissRunProfile:
    run_dir: str
    process_seconds: float | None
    generate_seconds: float | None
    peak_memory_gb: float | None
    steps: tuple[StepTiming, ...]
    categories: tuple[CategoryTiming, ...]

    def to_json_dict(self) -> dict:
        return asdict(self)


def parse_timing_events(text: str) -> tuple[dict, ...]:
    events: list[dict] = []
    for line in text.splitlines():
        if not line.startswith(SHARDEDIT_TIMING_PREFIX):
            continue
        payload = line.removeprefix(SHARDEDIT_TIMING_PREFIX)
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid SHARDEDIT_TIMING JSON: {payload}") from exc
        if not isinstance(event, dict):
            raise ValueError("SHARDEDIT_TIMING payload must be an object")
        events.append(event)
    return tuple(events)


def load_timing_events(path: Path) -> tuple[dict, ...]:
    return parse_timing_events(path.read_text(encoding="utf-8", errors="replace"))


def stdout_log_for_run_dir(run_dir: Path) -> Path:
    direct = run_dir / "stdout.log"
    if direct.exists():
        return direct

    nested = tuple(sorted(run_dir.glob("*/stdout.log")))
    if len(nested) == 1:
        return nested[0]
    if not nested:
        raise FileNotFoundError(f"no stdout.log found under {run_dir}")
    raise ValueError(f"multiple stdout.log files found under {run_dir}")


def analyze_run_directory(run_dir: Path) -> FullMissRunProfile:
    return analyze_timing_events(
        load_timing_events(stdout_log_for_run_dir(run_dir)),
        run_dir=str(run_dir),
    )


def analyze_timing_events(
    events: Sequence[Mapping],
    *,
    run_dir: str = "",
) -> FullMissRunProfile:
    denoise_events = [
        event
        for event in events
        if event.get("name") == "denoise_transformer"
    ]
    if not denoise_events:
        raise ValueError("no denoise_transformer timing events found")

    anchors_by_step = _sum_by_step(
        event
        for event in events
        if event.get("name") == "residual_anchor_materialize"
    )
    windows_by_step = _windows_by_step(
        event
        for event in events
        if event.get("name") == "residency_window"
    )

    steps: list[StepTiming] = []
    for event in denoise_events:
        step = _positive_int(event.get("step"), "denoise step")
        seconds = _non_negative_float(event.get("seconds"), "denoise seconds")
        cache_hit = event.get("cache_hit")
        category = _step_category(cache_hit)
        blocks = event.get("blocks_executed")
        blocks_executed = int(blocks) if isinstance(blocks, int) else 0
        windows = windows_by_step.get(step, _WindowTotals())
        steps.append(
            StepTiming(
                step=step,
                category=category,
                seconds=seconds,
                blocks_executed=blocks_executed,
                cache_reason=_optional_string(event.get("cache_reason")),
                anchor_seconds=anchors_by_step.get(step, 0.0),
                window_compute_seconds=windows.compute_seconds,
                window_load_seconds=windows.load_seconds,
                window_lora_seconds=windows.lora_seconds,
                window_prepare_seconds=windows.prepare_seconds,
                window_release_seconds=windows.release_seconds,
                window_lora_weight_cache_hits=windows.lora_weight_cache_hits,
                window_patched_cache_hits=windows.patched_window_cache_hits,
                window_patched_cache_size_max=windows.patched_window_cache_size_max,
                window_kquant_img_ff_cache_hits=windows.kquant_img_ff_cache_hits,
                window_kquant_img_ff_cache_misses=windows.kquant_img_ff_cache_misses,
                window_kquant_img_ff_cache_size_max=windows.kquant_img_ff_cache_size_max,
                window_kquant_img_ff_cache_bytes_max=windows.kquant_img_ff_cache_bytes_max,
            )
        )

    return FullMissRunProfile(
        run_dir=run_dir,
        process_seconds=_event_seconds(events, "process_total"),
        generate_seconds=_event_seconds(events, "generate_total"),
        peak_memory_gb=_peak_memory(events),
        steps=tuple(steps),
        categories=summarize_categories(steps),
    )


def summarize_categories(steps: Sequence[StepTiming]) -> tuple[CategoryTiming, ...]:
    if not steps:
        raise ValueError("at least one step is required")

    grouped: dict[str, list[StepTiming]] = defaultdict(list)
    for step in steps:
        grouped[step.category].append(step)

    order = ("no_cache_full", "cache_full_miss", "cache_hit")
    categories = [category for category in order if category in grouped]
    categories.extend(sorted(category for category in grouped if category not in order))
    return tuple(_category_summary(category, grouped[category]) for category in categories)


def _category_summary(category: str, steps: Sequence[StepTiming]) -> CategoryTiming:
    count = len(steps)
    total_seconds = sum(step.seconds for step in steps)
    total_blocks = sum(step.blocks_executed for step in steps)
    return CategoryTiming(
        category=category,
        steps=count,
        total_seconds=total_seconds,
        mean_seconds=total_seconds / count,
        total_blocks=total_blocks,
        mean_blocks=total_blocks / count,
        total_anchor_seconds=sum(step.anchor_seconds for step in steps),
        mean_anchor_seconds=_mean(step.anchor_seconds for step in steps),
        mean_window_compute_seconds=_mean(step.window_compute_seconds for step in steps),
        mean_window_load_seconds=_mean(step.window_load_seconds for step in steps),
        mean_window_lora_seconds=_mean(step.window_lora_seconds for step in steps),
        mean_window_prepare_seconds=_mean(step.window_prepare_seconds for step in steps),
        mean_window_release_seconds=_mean(step.window_release_seconds for step in steps),
        total_lora_weight_cache_hits=sum(
            step.window_lora_weight_cache_hits for step in steps
        ),
        total_patched_window_cache_hits=sum(
            step.window_patched_cache_hits for step in steps
        ),
        max_patched_window_cache_size=max(
            step.window_patched_cache_size_max for step in steps
        ),
        total_kquant_img_ff_cache_hits=sum(
            step.window_kquant_img_ff_cache_hits for step in steps
        ),
        total_kquant_img_ff_cache_misses=sum(
            step.window_kquant_img_ff_cache_misses for step in steps
        ),
        max_kquant_img_ff_cache_size=max(
            step.window_kquant_img_ff_cache_size_max for step in steps
        ),
        max_kquant_img_ff_cache_bytes=max(
            step.window_kquant_img_ff_cache_bytes_max for step in steps
        ),
    )


@dataclass
class _WindowAccumulator:
    compute_seconds: float = 0.0
    load_seconds: float = 0.0
    lora_seconds: float = 0.0
    prepare_seconds: float = 0.0
    release_seconds: float = 0.0
    lora_weight_cache_hits: int = 0
    patched_window_cache_hits: int = 0
    patched_window_cache_size_max: int = 0
    kquant_img_ff_cache_hits: int = 0
    kquant_img_ff_cache_misses: int = 0
    kquant_img_ff_cache_size_max: int = 0
    kquant_img_ff_cache_bytes_max: int = 0


@dataclass(frozen=True)
class _WindowTotals:
    compute_seconds: float = 0.0
    load_seconds: float = 0.0
    lora_seconds: float = 0.0
    prepare_seconds: float = 0.0
    release_seconds: float = 0.0
    lora_weight_cache_hits: int = 0
    patched_window_cache_hits: int = 0
    patched_window_cache_size_max: int = 0
    kquant_img_ff_cache_hits: int = 0
    kquant_img_ff_cache_misses: int = 0
    kquant_img_ff_cache_size_max: int = 0
    kquant_img_ff_cache_bytes_max: int = 0


def _windows_by_step(events: Iterable[Mapping]) -> dict[int, _WindowTotals]:
    raw: dict[int, _WindowAccumulator] = defaultdict(_WindowAccumulator)
    for event in events:
        step = _positive_int(event.get("step"), "residency window step")
        window = raw[step]
        window.compute_seconds += _optional_non_negative_float(
            event.get("compute_seconds")
        )
        window.load_seconds += _optional_non_negative_float(event.get("load_seconds"))
        window.lora_seconds += _optional_non_negative_float(event.get("lora_seconds"))
        window.prepare_seconds += _optional_non_negative_float(
            event.get("prepare_seconds")
        )
        window.release_seconds += _optional_non_negative_float(
            event.get("release_seconds")
        )
        window.lora_weight_cache_hits += _optional_non_negative_int(
            event.get("lora_weight_cache_hits")
        )
        if event.get("patched_window_cache_hit") is True:
            window.patched_window_cache_hits += 1
        window.patched_window_cache_size_max = max(
            window.patched_window_cache_size_max,
            _optional_non_negative_int(event.get("patched_window_cache_size")),
        )
        window.kquant_img_ff_cache_hits += _optional_non_negative_int(
            event.get("kquant_img_ff_cache_hits")
        )
        window.kquant_img_ff_cache_misses += _optional_non_negative_int(
            event.get("kquant_img_ff_cache_misses")
        )
        window.kquant_img_ff_cache_size_max = max(
            window.kquant_img_ff_cache_size_max,
            _optional_non_negative_int(event.get("kquant_img_ff_cache_size")),
        )
        window.kquant_img_ff_cache_bytes_max = max(
            window.kquant_img_ff_cache_bytes_max,
            _optional_non_negative_int(event.get("kquant_img_ff_cache_bytes")),
        )
    return {
        step: _WindowTotals(
            compute_seconds=values.compute_seconds,
            load_seconds=values.load_seconds,
            lora_seconds=values.lora_seconds,
            prepare_seconds=values.prepare_seconds,
            release_seconds=values.release_seconds,
            lora_weight_cache_hits=values.lora_weight_cache_hits,
            patched_window_cache_hits=values.patched_window_cache_hits,
            patched_window_cache_size_max=values.patched_window_cache_size_max,
            kquant_img_ff_cache_hits=values.kquant_img_ff_cache_hits,
            kquant_img_ff_cache_misses=values.kquant_img_ff_cache_misses,
            kquant_img_ff_cache_size_max=values.kquant_img_ff_cache_size_max,
            kquant_img_ff_cache_bytes_max=values.kquant_img_ff_cache_bytes_max,
        )
        for step, values in raw.items()
    }


def _sum_by_step(events: Iterable[Mapping]) -> dict[int, float]:
    totals: dict[int, float] = defaultdict(float)
    for event in events:
        step = _positive_int(event.get("step"), "anchor step")
        totals[step] += _optional_non_negative_float(event.get("seconds"))
    return dict(totals)


def _step_category(cache_hit: object) -> str:
    if cache_hit is True:
        return "cache_hit"
    if cache_hit is False:
        return "cache_full_miss"
    return "no_cache_full"


def _event_seconds(events: Sequence[Mapping], name: str) -> float | None:
    for event in events:
        if event.get("name") == name:
            return _optional_non_negative_float(event.get("seconds"))
    return None


def _peak_memory(events: Sequence[Mapping]) -> float | None:
    peaks = [
        _optional_non_negative_float(event.get("peak_memory_gb"))
        for event in events
        if event.get("peak_memory_gb") is not None
    ]
    return max(peaks) if peaks else None


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_float(value: object, label: str) -> float:
    if not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{label} must be a non-negative number")
    return float(value)


def _optional_non_negative_float(value: object) -> float:
    if value is None:
        return 0.0
    return _non_negative_float(value, "optional timing")


def _optional_non_negative_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("optional count must be a non-negative integer")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    if not items:
        return 0.0
    return sum(items) / len(items)
