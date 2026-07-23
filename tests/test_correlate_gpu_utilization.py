from __future__ import annotations

import json
from pathlib import Path

from tools.correlate_gpu_utilization import (
    load_gpu_samples,
    reconstruct_timing_intervals,
    summarize_interval,
)
from shardedit_mlx.gpu_utilization_sampler import GpuUtilizationSample


def _sample(
    *,
    wall: float,
    device: int,
    pageins: int,
    swapins: int,
) -> GpuUtilizationSample:
    return GpuUtilizationSample(
        timestamp_monotonic=wall,
        wall_time=wall,
        device_utilization_percent=device,
        renderer_utilization_percent=device - 1,
        tiler_utilization_percent=device - 2,
        alloc_system_memory_bytes=1000,
        in_use_system_memory_bytes=500,
        pageins=pageins,
        pageouts=0,
        swapins=swapins,
        swapouts=0,
        page_size_bytes=16384,
        swap_used_bytes=1024**3,
        swap_total_bytes=3 * 1024**3,
        errors=(),
    )


def test_summarize_interval_averages_gpu_and_paging_deltas() -> None:
    samples = (
        _sample(wall=10.0, device=40, pageins=100, swapins=10),
        _sample(wall=10.2, device=80, pageins=110, swapins=12),
        _sample(wall=10.4, device=90, pageins=130, swapins=15),
        _sample(wall=12.0, device=20, pageins=200, swapins=40),
    )
    summary = summarize_interval(
        {
            "name": "denoise_transformer",
            "step": 1,
            "start_wall": 10.0,
            "end_wall": 10.5,
            "duration_seconds": 0.5,
        },
        samples,
    )

    assert summary.sample_count == 3
    assert summary.mean_device_utilization_percent == 70.0
    assert summary.pageins_pages == 30
    assert summary.swapins_pages == 5
    assert summary.pageins_bytes == 30 * 16384


def test_reconstruct_timing_intervals_walks_contiguous_seconds() -> None:
    intervals = reconstruct_timing_intervals(
        (
            {"name": "denoise_transformer", "step": 1, "seconds": 2.0, "cache_hit": False},
            {"name": "denoise_transformer", "step": 2, "seconds": 1.0, "cache_hit": True},
        ),
        wall_anchor=100.0,
        monotonic_anchor=0.0,
    )

    assert intervals[0]["start_wall"] == 100.0
    assert intervals[0]["end_wall"] == 102.0
    assert intervals[1]["start_wall"] == 102.0
    assert intervals[1]["end_wall"] == 103.0


def test_load_gpu_samples(tmp_path: Path) -> None:
    path = tmp_path / "gpu_utilization.jsonl"
    path.write_text(
        json.dumps(_sample(wall=1.0, device=50, pageins=1, swapins=0).to_json_dict())
        + "\n",
        encoding="utf-8",
    )
    samples = load_gpu_samples(path)
    assert len(samples) == 1
    assert samples[0].device_utilization_percent == 50
