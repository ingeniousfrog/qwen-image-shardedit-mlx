from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

from shardedit_mlx.gpu_utilization_sampler import (
    GpuUtilizationSampler,
    collect_sample,
)


IOREG = """+-o AGXAcceleratorG14G
    {
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=1000,"Tiler Utilization %"=12,"Renderer Utilization %"=34,"Device Utilization %"=56,"In use system memory"=200}
    }
"""

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pageins:                                          10.
Pageouts:                                          2.
Swapins:                                           3.
Swapouts:                                          4.
"""

SWAP = "total = 3.00G  used = 1.00G  free = 2.00G  (encrypted)"


def test_collect_sample_parses_command_outputs() -> None:
    def runner(args: tuple[str, ...]) -> str:
        if args[0] == "ioreg":
            return IOREG
        if args[0] == "vm_stat":
            return VM_STAT
        if args[:2] == ("sysctl", "-n"):
            return SWAP
        raise AssertionError(args)

    sample = collect_sample(
        timestamp_monotonic=1.5,
        wall_time=100.0,
        runner=runner,
    )

    assert sample.device_utilization_percent == 56
    assert sample.renderer_utilization_percent == 34
    assert sample.tiler_utilization_percent == 12
    assert sample.pageins == 10
    assert sample.swapins == 3
    assert sample.swap_used_bytes == 1024**3
    assert sample.errors == ()


def test_sampler_writes_jsonl(tmp_path: Path) -> None:
    calls = {"n": 0}

    def runner(args: tuple[str, ...]) -> str:
        calls["n"] += 1
        if args[0] == "ioreg":
            return IOREG
        if args[0] == "vm_stat":
            return VM_STAT
        if args[:2] == ("sysctl", "-n"):
            return SWAP
        raise AssertionError(args)

    output = tmp_path / "gpu_utilization.jsonl"
    sampler = GpuUtilizationSampler(
        output,
        interval_seconds=0.05,
        runner=runner,
    )
    sampler.start()
    time.sleep(0.18)
    sampler.stop()

    assert sampler.sample_count >= 2
    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == sampler.sample_count
    payload = json.loads(lines[0])
    assert payload["device_utilization_percent"] == 56
    assert payload["pageins"] == 10


def test_sample_cli_wraps_command(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    completed = subprocess.run(
        (
            "python3",
            str(Path(__file__).resolve().parents[1] / "tools" / "sample_gpu_utilization.py"),
            "--output-dir",
            str(output_dir),
            "--interval-seconds",
            "0.1",
            "--",
            "python3",
            "-c",
            "import time; time.sleep(0.25)",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    jsonl = output_dir / "gpu_utilization.jsonl"
    metadata = json.loads((output_dir / "sample_metadata.json").read_text(encoding="utf-8"))
    assert jsonl.exists()
    assert metadata["sample_count"] >= 1
    assert metadata["returncode"] == 0
