from __future__ import annotations

import json

import pytest

from shardedit_mlx.full_miss_profile import (
    analyze_timing_events,
    parse_timing_events,
    summarize_categories,
)


def _timing(event: dict) -> str:
    return "SHARDEDIT_TIMING " + json.dumps(event, sort_keys=True)


def test_parse_timing_events_ignores_non_timing_lines() -> None:
    events = parse_timing_events(
        "\n".join(
            (
                "loading...",
                _timing({"name": "process_total", "seconds": 12.0}),
            )
        )
    )

    assert events == ({"name": "process_total", "seconds": 12.0},)


def test_analyze_timing_events_groups_cache_misses_hits_and_anchors() -> None:
    events = parse_timing_events(
        "\n".join(
            (
                _timing(
                    {
                        "name": "residency_window",
                        "step": 1,
                        "compute_seconds": 8.0,
                        "lora_weight_cache_hits": 0,
                        "patched_window_cache_hit": False,
                        "patched_window_cache_size": 1,
                        "kquant_img_ff_cache_hits": 0,
                        "kquant_img_ff_cache_misses": 8,
                        "kquant_img_ff_cache_size": 8,
                        "kquant_img_ff_cache_bytes": 1234,
                    }
                ),
                _timing({"name": "residency_window", "step": 1, "load_seconds": 1.0}),
                _timing({"name": "residual_anchor_materialize", "step": 1, "seconds": 2.5}),
                _timing(
                    {
                        "name": "denoise_transformer",
                        "step": 1,
                        "seconds": 42.0,
                        "cache_hit": False,
                        "cache_reason": "warmup",
                        "blocks_executed": 60,
                    }
                ),
                _timing({"name": "residency_window", "step": 2, "compute_seconds": 0.7}),
                _timing(
                    {
                        "name": "residency_window",
                        "step": 2,
                        "lora_seconds": 0.2,
                        "lora_weight_cache_hits": 1,
                        "patched_window_cache_hit": True,
                        "patched_window_cache_size": 2,
                        "kquant_img_ff_cache_hits": 3,
                        "kquant_img_ff_cache_misses": 0,
                        "kquant_img_ff_cache_size": 8,
                        "kquant_img_ff_cache_bytes": 1234,
                    }
                ),
                _timing(
                    {
                        "name": "denoise_transformer",
                        "step": 2,
                        "seconds": 2.2,
                        "cache_hit": True,
                        "cache_reason": "diff_hit",
                        "blocks_executed": 3,
                    }
                ),
                _timing({"name": "generate_total", "seconds": 50.0}),
                _timing({"name": "process_total", "seconds": 51.0}),
            )
        )
    )

    profile = analyze_timing_events(events, run_dir="run-a")

    assert profile.run_dir == "run-a"
    assert profile.process_seconds == 51.0
    assert profile.generate_seconds == 50.0
    assert [step.category for step in profile.steps] == ["cache_full_miss", "cache_hit"]
    assert profile.steps[0].anchor_seconds == pytest.approx(2.5)
    assert profile.steps[0].window_compute_seconds == pytest.approx(8.0)
    assert profile.steps[0].window_load_seconds == pytest.approx(1.0)
    assert profile.steps[0].window_lora_weight_cache_hits == 0
    assert profile.steps[0].window_patched_cache_hits == 0
    assert profile.steps[0].window_patched_cache_size_max == 1
    assert profile.steps[0].window_kquant_img_ff_cache_hits == 0
    assert profile.steps[0].window_kquant_img_ff_cache_misses == 8
    assert profile.steps[0].window_kquant_img_ff_cache_size_max == 8
    assert profile.steps[0].window_kquant_img_ff_cache_bytes_max == 1234
    assert profile.steps[1].window_lora_seconds == pytest.approx(0.2)
    assert profile.steps[1].window_lora_weight_cache_hits == 1
    assert profile.steps[1].window_patched_cache_hits == 1
    assert profile.steps[1].window_patched_cache_size_max == 2
    assert profile.steps[1].window_kquant_img_ff_cache_hits == 3
    assert profile.steps[1].window_kquant_img_ff_cache_misses == 0

    miss, hit = profile.categories
    assert miss.category == "cache_full_miss"
    assert miss.mean_seconds == pytest.approx(42.0)
    assert miss.mean_anchor_seconds == pytest.approx(2.5)
    assert miss.mean_blocks == pytest.approx(60.0)
    assert miss.total_lora_weight_cache_hits == 0
    assert miss.total_patched_window_cache_hits == 0
    assert miss.max_patched_window_cache_size == 1
    assert miss.total_kquant_img_ff_cache_hits == 0
    assert miss.total_kquant_img_ff_cache_misses == 8
    assert miss.max_kquant_img_ff_cache_size == 8
    assert miss.max_kquant_img_ff_cache_bytes == 1234
    assert hit.category == "cache_hit"
    assert hit.mean_seconds == pytest.approx(2.2)
    assert hit.mean_blocks == pytest.approx(3.0)
    assert hit.total_lora_weight_cache_hits == 1
    assert hit.total_patched_window_cache_hits == 1
    assert hit.max_patched_window_cache_size == 2
    assert hit.total_kquant_img_ff_cache_hits == 3
    assert hit.total_kquant_img_ff_cache_misses == 0


def test_analyze_timing_events_classifies_no_cache_full_steps() -> None:
    profile = analyze_timing_events(
        (
            {"name": "denoise_transformer", "step": 1, "seconds": 30.0, "blocks_executed": 60},
        )
    )

    assert profile.steps[0].category == "no_cache_full"
    assert profile.categories[0].category == "no_cache_full"


def test_summarize_categories_rejects_empty_steps() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        summarize_categories(())
