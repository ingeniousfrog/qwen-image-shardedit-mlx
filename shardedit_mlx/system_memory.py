"""Parsers for macOS virtual-memory telemetry used by benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import re


VM_DELTA_COUNTERS = ("pageins", "pageouts", "swapins", "swapouts")


@dataclass(frozen=True)
class VmStatSnapshot:
    page_size_bytes: int
    counters: tuple[tuple[str, int], ...]

    def value(self, name: str) -> int:
        for counter_name, value in self.counters:
            if counter_name == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True)
class SwapUsage:
    total_bytes: int
    used_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class GpuPerformanceSnapshot:
    device_utilization_percent: int
    renderer_utilization_percent: int
    tiler_utilization_percent: int
    alloc_system_memory_bytes: int
    in_use_system_memory_bytes: int | None = None


def _normalize_counter_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_vm_stat(output: str) -> VmStatSnapshot:
    """Parse the stable text interface emitted by macOS ``vm_stat``."""

    page_size_match = re.search(r"page size of (\d+) bytes", output)
    if page_size_match is None:
        raise ValueError("vm_stat output does not contain a page size")

    counters = tuple(
        (_normalize_counter_name(match.group(1)), int(match.group(2)))
        for line in output.splitlines()[1:]
        if (match := re.match(r'^"?(.+?)"?:\s+(\d+)\.$', line.strip()))
    )
    if not counters:
        raise ValueError("vm_stat output does not contain counters")
    return VmStatSnapshot(
        page_size_bytes=int(page_size_match.group(1)),
        counters=counters,
    )


def parse_memory_pressure_free_percentage(output: str) -> int | None:
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    return int(match.group(1)) if match is not None else None


def _size_to_bytes(value: str, unit: str) -> int:
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return round(float(value) * multipliers[unit])


def parse_swap_usage(output: str) -> SwapUsage | None:
    values = {
        name: _size_to_bytes(value, unit)
        for name, value, unit in re.findall(
            r"(total|used|free)\s*=\s*([0-9.]+)([KMGT])", output
        )
    }
    if set(values) != {"total", "used", "free"}:
        return None
    return SwapUsage(
        total_bytes=values["total"],
        used_bytes=values["used"],
        free_bytes=values["free"],
    )


_IOREG_PERFORMANCE_STATISTICS = re.compile(
    r'"PerformanceStatistics"\s*=\s*\{([^}]*)\}',
    re.DOTALL,
)
_IOREG_STAT_ENTRY = re.compile(r'"([^"]+)"\s*=\s*(-?\d+)')


def parse_ioreg_gpu_performance_statistics(
    output: str,
) -> GpuPerformanceSnapshot:
    """Parse ``ioreg -r -d 1 -c IOAccelerator`` PerformanceStatistics."""

    match = _IOREG_PERFORMANCE_STATISTICS.search(output)
    if match is None:
        raise ValueError("ioreg output does not contain PerformanceStatistics")

    stats = {
        name: int(value)
        for name, value in _IOREG_STAT_ENTRY.findall(match.group(1))
    }
    required = (
        "Device Utilization %",
        "Renderer Utilization %",
        "Tiler Utilization %",
        "Alloc system memory",
    )
    missing = tuple(name for name in required if name not in stats)
    if missing:
        raise ValueError(
            "PerformanceStatistics is missing required fields: "
            + ", ".join(missing)
        )
    return GpuPerformanceSnapshot(
        device_utilization_percent=stats["Device Utilization %"],
        renderer_utilization_percent=stats["Renderer Utilization %"],
        tiler_utilization_percent=stats["Tiler Utilization %"],
        alloc_system_memory_bytes=stats["Alloc system memory"],
        in_use_system_memory_bytes=stats.get("In use system memory"),
    )


def counter_deltas(
    before: VmStatSnapshot,
    after: VmStatSnapshot,
) -> dict[str, int]:
    """Return page and byte deltas for cumulative paging counters."""

    if before.page_size_bytes != after.page_size_bytes:
        raise ValueError("vm_stat page size changed between snapshots")

    page_deltas = {
        name: after.value(name) - before.value(name) for name in VM_DELTA_COUNTERS
    }
    if any(delta < 0 for delta in page_deltas.values()):
        raise ValueError("vm_stat counters decreased between snapshots")
    return {
        metric: value
        for name, pages in page_deltas.items()
        for metric, value in (
            (f"{name}_pages", pages),
            (f"{name}_bytes", pages * before.page_size_bytes),
        )
    }
