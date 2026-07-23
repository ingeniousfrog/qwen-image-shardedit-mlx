"""Background GPU utilization and paging sampler for full-miss diagnosis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import TextIO

from shardedit_mlx.system_memory import (
    GpuPerformanceSnapshot,
    SwapUsage,
    VmStatSnapshot,
    parse_ioreg_gpu_performance_statistics,
    parse_swap_usage,
    parse_vm_stat,
)


CommandRunner = Callable[[tuple[str, ...]], str]


@dataclass(frozen=True)
class GpuUtilizationSample:
    timestamp_monotonic: float
    wall_time: float
    device_utilization_percent: int | None
    renderer_utilization_percent: int | None
    tiler_utilization_percent: int | None
    alloc_system_memory_bytes: int | None
    in_use_system_memory_bytes: int | None
    pageins: int | None
    pageouts: int | None
    swapins: int | None
    swapouts: int | None
    page_size_bytes: int | None
    swap_used_bytes: int | None
    swap_total_bytes: int | None
    errors: tuple[str, ...]

    def to_json_dict(self) -> dict:
        return asdict(self)


def run_command(args: tuple[str, ...]) -> str:
    result = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _safe_run(runner: CommandRunner, args: tuple[str, ...]) -> tuple[str | None, str | None]:
    try:
        return runner(args), None
    except OSError as error:
        return None, f"{' '.join(args)}: {error}"
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        return None, f"{' '.join(args)}: {detail}"


def collect_sample(
    *,
    timestamp_monotonic: float | None = None,
    wall_time: float | None = None,
    runner: CommandRunner = run_command,
) -> GpuUtilizationSample:
    """Collect one GPU + paging sample using ioreg / vm_stat / sysctl."""

    mono = time.monotonic() if timestamp_monotonic is None else timestamp_monotonic
    wall = time.time() if wall_time is None else wall_time
    errors: list[str] = []

    ioreg_output, ioreg_error = _safe_run(
        runner, ("ioreg", "-r", "-d", "1", "-c", "IOAccelerator")
    )
    if ioreg_error is not None:
        errors.append(ioreg_error)
    gpu: GpuPerformanceSnapshot | None = None
    if ioreg_output is not None:
        try:
            gpu = parse_ioreg_gpu_performance_statistics(ioreg_output)
        except ValueError as error:
            errors.append(f"ioreg parse: {error}")

    vm_output, vm_error = _safe_run(runner, ("vm_stat",))
    if vm_error is not None:
        errors.append(vm_error)
    vm_stat: VmStatSnapshot | None = None
    if vm_output is not None:
        try:
            vm_stat = parse_vm_stat(vm_output)
        except ValueError as error:
            errors.append(f"vm_stat parse: {error}")

    swap_output, swap_error = _safe_run(runner, ("sysctl", "-n", "vm.swapusage"))
    if swap_error is not None:
        errors.append(swap_error)
    swap_usage: SwapUsage | None = None
    if swap_output is not None:
        swap_usage = parse_swap_usage(swap_output)
        if swap_usage is None:
            errors.append(f"swapusage parse failed: {swap_output!r}")

    page_counters = dict(vm_stat.counters) if vm_stat is not None else {}
    return GpuUtilizationSample(
        timestamp_monotonic=mono,
        wall_time=wall,
        device_utilization_percent=(
            None if gpu is None else gpu.device_utilization_percent
        ),
        renderer_utilization_percent=(
            None if gpu is None else gpu.renderer_utilization_percent
        ),
        tiler_utilization_percent=(
            None if gpu is None else gpu.tiler_utilization_percent
        ),
        alloc_system_memory_bytes=(
            None if gpu is None else gpu.alloc_system_memory_bytes
        ),
        in_use_system_memory_bytes=(
            None if gpu is None else gpu.in_use_system_memory_bytes
        ),
        pageins=page_counters.get("pageins"),
        pageouts=page_counters.get("pageouts"),
        swapins=page_counters.get("swapins"),
        swapouts=page_counters.get("swapouts"),
        page_size_bytes=None if vm_stat is None else vm_stat.page_size_bytes,
        swap_used_bytes=None if swap_usage is None else swap_usage.used_bytes,
        swap_total_bytes=None if swap_usage is None else swap_usage.total_bytes,
        errors=tuple(errors),
    )


class GpuUtilizationSampler:
    """Sample GPU utilization and paging counters on a background thread."""

    def __init__(
        self,
        output_path: Path,
        *,
        interval_seconds: float = 0.2,
        runner: CommandRunner = run_command,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.runner = runner
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: TextIO | None = None
        self.sample_count = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("w", encoding="utf-8")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-utilization-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, join_timeout_seconds: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_seconds)
            self._thread = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> GpuUtilizationSampler:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _run(self) -> None:
        assert self._handle is not None
        next_at = time.monotonic()
        while not self._stop.is_set():
            sample = collect_sample(runner=self.runner)
            self._handle.write(json.dumps(sample.to_json_dict(), sort_keys=True))
            self._handle.write("\n")
            self._handle.flush()
            self.sample_count += 1
            next_at += self.interval_seconds
            delay = next_at - time.monotonic()
            if delay > 0:
                self._stop.wait(timeout=delay)
            else:
                next_at = time.monotonic()
