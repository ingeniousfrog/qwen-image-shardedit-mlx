from __future__ import annotations

from pathlib import Path

from PIL import Image

from shardedit_mlx.product_cli import (
    build_edit_argv,
    build_timed_command,
    main,
    parse_image_option,
)
from shardedit_mlx.product_presets import (
    fit_clarity_dimensions,
    format_mapping_report,
    resolve_product_plan,
)
from shardedit_mlx.time_metrics import format_metrics_report, parse_time_l


SAMPLE_TIME_L = """
100%|██████████| 8/8 [02:28<00:00, 18.56s/it]
      156.42 real        16.40 user        44.17 sys
          5568905216  maximum resident set size
                   0  average shared memory size
               28984  voluntary context switches
              985416  involuntary context switches
        304365852029  instructions retired
        162326953058  cycles elapsed
         18482262616  peak memory footprint
"""


def test_fit_clarity_preserves_aspect_inside_standard_box() -> None:
    assert fit_clarity_dimensions(1080, 1440, box=768) == (576, 768)
    assert fit_clarity_dimensions(1200, 2596, box=768) == (352, 768)
    assert fit_clarity_dimensions(2000, 2000, box=768) == (768, 768)


def test_fit_clarity_high_box_is_1024() -> None:
    assert fit_clarity_dimensions(1080, 1440, box=1024) == (768, 1024)
    assert fit_clarity_dimensions(2000, 2000, box=1024) == (1024, 1024)


def test_resolve_quality_standard_mapping() -> None:
    plan = resolve_product_plan(
        image_width=1080,
        image_height=1440,
        clarity="standard",
        speed="quality",
        seed=7,
    )

    assert plan.width == 576
    assert plan.height == 768
    assert plan.clarity_box == 768
    assert plan.seed == 7
    assert plan.steps == 8
    assert plan.cache_preset == "none"
    assert plan.runtime_options.residency_mode == "shard"
    assert plan.runtime_options.cache_threshold == 0.0
    assert plan.runtime_options.reference_conditioning_size == "fit-box"
    assert plan.runtime_options.reference_conditioning_max_width == 576
    assert plan.runtime_options.reference_conditioning_max_height == 768
    assert plan.runtime_options.profile is True


def test_lightning_steps_4_sets_steps_to_4() -> None:
    plan = resolve_product_plan(
        image_width=1080,
        image_height=1440,
        lightning_steps=4,
    )

    assert plan.steps == 4
    assert any("4-step" in note for note in plan.notes)


def test_resolve_fast_and_balanced_presets() -> None:
    fast = resolve_product_plan(image_width=1080, image_height=1440, speed="fast")
    balanced = resolve_product_plan(
        image_width=2000,
        image_height=2000,
        speed="balanced",
        clarity="high",
    )

    assert fast.cache_preset == "f1b2"
    assert fast.width == 576
    assert fast.height == 768
    assert fast.runtime_options.cache_back_blocks == 2
    assert fast.runtime_options.cache_threshold_schedule == "fixed"
    assert balanced.width == 1024
    assert balanced.height == 1024
    assert balanced.clarity_box == 1024
    assert balanced.cache_preset == "flow-aware"
    assert balanced.runtime_options.cache_threshold_schedule == "flow-aware"
    assert balanced.runtime_options.reference_conditioning_max_width == 1024
    assert balanced.runtime_options.reference_conditioning_max_height == 1024


def test_format_mapping_report_includes_tiers() -> None:
    report = format_mapping_report(
        resolve_product_plan(image_width=1080, image_height=1440, speed="fast")
    )

    assert "clarity: fast" not in report
    assert "box 768x768" in report
    assert "source 1080x1440 -> output 576x768" in report
    assert "speed:   fast -> cache_preset=f1b2" in report
    assert "cache_back_blocks: 2" in report


def test_parse_time_l_extracts_requested_counters() -> None:
    metrics = parse_time_l(SAMPLE_TIME_L)

    assert metrics.real_seconds == 156.42
    assert metrics.user_seconds == 16.40
    assert metrics.sys_seconds == 44.17
    assert metrics.voluntary_context_switches == 28984
    assert metrics.involuntary_context_switches == 985416
    assert metrics.instructions_retired == 304365852029
    assert metrics.cycles_elapsed == 162326953058
    assert metrics.peak_memory_footprint_bytes == 18482262616
    assert metrics.as_dict()["peak_memory_footprint_gb"] == 17.213


def test_format_metrics_report_lists_key_fields() -> None:
    report = format_metrics_report(parse_time_l(SAMPLE_TIME_L))

    assert "voluntary_context_switches: 28984" in report
    assert "involuntary_context_switches: 985416" in report
    assert "instructions_retired: 304365852029" in report
    assert "cycles_elapsed: 162326953058" in report
    assert "peak_memory_footprint_bytes: 18482262616" in report
    assert "real_seconds: 156.42" in report


def test_parse_image_option_supports_comma_separated_paths() -> None:
    assert parse_image_option("/tmp/a.png") == (Path("/tmp/a.png"),)
    assert parse_image_option("/tmp/a.png,/tmp/b.jpg, /tmp/c.png") == (
        Path("/tmp/a.png"),
        Path("/tmp/b.jpg"),
        Path("/tmp/c.png"),
    )


def test_build_edit_argv_maps_product_plan() -> None:
    plan = resolve_product_plan(
        image_width=1080,
        image_height=1440,
        clarity="high",
        speed="balanced",
        seed=42,
    )
    argv = build_edit_argv(
        plan,
        images=(Path("/tmp/in.png"), Path("/tmp/ref.png")),
        prompt="make it studio lighting",
        model=Path("/tmp/model"),
        lora=Path("/tmp/lora.safetensors"),
        output=Path("/tmp/out.png"),
    )

    assert plan.width == 768
    assert plan.height == 1024
    assert "--width" in argv and argv[argv.index("--width") + 1] == "768"
    assert "--height" in argv and argv[argv.index("--height") + 1] == "1024"
    image_index = argv.index("--image-paths")
    assert argv[image_index + 1] == "/tmp/in.png"
    assert argv[image_index + 2] == "/tmp/ref.png"
    assert "--shardedit-cache-threshold-schedule" in argv
    assert (
        argv[argv.index("--shardedit-cache-threshold-schedule") + 1] == "flow-aware"
    )
    assert argv[argv.index("--shardedit-reference-conditioning-max-width") + 1] == "768"
    assert argv[argv.index("--shardedit-reference-conditioning-max-height") + 1] == "1024"
    assert "--shardedit-profile" in argv
    assert "--low-ram" in argv

    timed = build_timed_command(
        argv,
        metrics_path=Path("/tmp/shardedit-time-metrics.txt"),
        python="/usr/bin/python3",
    )
    assert timed[:5] == [
        "/usr/bin/time",
        "-l",
        "-o",
        "/tmp/shardedit-time-metrics.txt",
        "/usr/bin/python3",
    ]
    assert timed[5:7] == ["-m", "shardedit_mlx.mflux_fast_edit"]


def test_dry_run_json_exit_zero(tmp_path: Path, capsys) -> None:
    import json

    image = tmp_path / "ref.png"
    extra = tmp_path / "extra.png"
    Image.new("RGB", (1080, 1440), color=(12, 34, 56)).save(image)
    Image.new("RGB", (640, 640), color=(90, 90, 90)).save(extra)

    code = main(
        [
            "--image",
            f"{image},{extra}",
            "--prompt",
            "test",
            "--speed",
            "fast",
            "--dry-run",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert payload["dry_run"] is True
    assert payload["mapping"]["cache_preset"] == "f1b2"
    assert payload["mapping"]["output_size"] == "576x768"
    assert payload["mapping"]["clarity_box"] == 768
    assert payload["images"] == [str(image), str(extra)]
    assert payload["primary_image"] == str(image)
    assert payload["lightning_steps"] == 8
    assert "8steps" in payload["lora"]
    assert "--image-paths" in payload["command"]
    image_index = payload["command"].index("--image-paths")
    assert payload["command"][image_index + 1] == str(image)
    assert payload["command"][image_index + 2] == str(extra)
    assert "/usr/bin/time" in payload["command"]
    steps_index = payload["command"].index("--steps")
    assert payload["command"][steps_index + 1] == "8"


def test_dry_run_uses_ref_png_default_when_image_is_missing(tmp_path: Path, capsys, monkeypatch) -> None:
    import json

    monkeypatch.chdir(tmp_path)

    code = main(["--prompt", "test", "--dry-run", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert payload["images"] == ["ref.png"]
    assert payload["mapping"]["source_size"] == "576x768"
    assert payload["mapping"]["output_size"] == "576x768"
    assert "dry-run warning: image path does not exist" in captured.err


def test_dry_run_switches_lora_when_lightning_steps_4(tmp_path: Path, capsys) -> None:
    import json

    image = tmp_path / "ref.png"
    Image.new("RGB", (1080, 1440), color=(12, 34, 56)).save(image)

    code = main(
        [
            "--image",
            str(image),
            "--prompt",
            "test",
            "--lightning-steps",
            "4",
            "--dry-run",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert payload["lightning_steps"] == 4
    assert "4steps" in payload["lora"]
    assert "8steps" not in payload["lora"]
    steps_index = payload["command"].index("--steps")
    assert payload["command"][steps_index + 1] == "4"
    lora_index = payload["command"].index("--lora-paths")
    assert payload["command"][lora_index + 1] == payload["lora"]
