"""Parse macOS `/usr/bin/time -l` resource counters."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


_TIME_LINE = re.compile(
    r"^\s*(?P<real>[\d.]+)\s+real\s+(?P<user>[\d.]+)\s+user\s+(?P<sys>[\d.]+)\s+sys\s*$"
)
_COUNTER_LINE = re.compile(r"^\s*(?P<value>\d+)\s+(?P<label>.+?)\s*$")


@dataclass(frozen=True)
class TimeLMetrics:
    """Selected counters from macOS `/usr/bin/time -l`."""

    real_seconds: float | None = None
    user_seconds: float | None = None
    sys_seconds: float | None = None
    voluntary_context_switches: int | None = None
    involuntary_context_switches: int | None = None
    instructions_retired: int | None = None
    cycles_elapsed: int | None = None
    peak_memory_footprint_bytes: int | None = None
    maximum_resident_set_size_bytes: int | None = None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.peak_memory_footprint_bytes is not None:
            payload["peak_memory_footprint_gb"] = round(
                self.peak_memory_footprint_bytes / (1024**3), 3
            )
        if self.maximum_resident_set_size_bytes is not None:
            payload["maximum_resident_set_size_gb"] = round(
                self.maximum_resident_set_size_bytes / (1024**3), 3
            )
        return payload


_LABEL_TO_FIELD = {
    "voluntary context switches": "voluntary_context_switches",
    "involuntary context switches": "involuntary_context_switches",
    "instructions retired": "instructions_retired",
    "cycles elapsed": "cycles_elapsed",
    "peak memory footprint": "peak_memory_footprint_bytes",
    "maximum resident set size": "maximum_resident_set_size_bytes",
}


def parse_time_l(text: str) -> TimeLMetrics:
    """Extract timing and resource counters from `/usr/bin/time -l` stderr."""

    values: dict[str, float | int] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        time_match = _TIME_LINE.match(line)
        if time_match is not None:
            values["real_seconds"] = float(time_match.group("real"))
            values["user_seconds"] = float(time_match.group("user"))
            values["sys_seconds"] = float(time_match.group("sys"))
            continue
        counter_match = _COUNTER_LINE.match(line)
        if counter_match is None:
            continue
        label = counter_match.group("label").strip().lower()
        field = _LABEL_TO_FIELD.get(label)
        if field is not None:
            values[field] = int(counter_match.group("value"))
    return TimeLMetrics(**values)  # type: ignore[arg-type]


def format_metrics_report(metrics: TimeLMetrics) -> str:
    """Human-readable resource summary for the product CLI."""

    payload = metrics.as_dict()

    def _fmt(key: str, label: str) -> str:
        value = payload.get(key)
        if value is None:
            return f"{label}: n/a"
        if isinstance(value, float):
            return f"{label}: {value:.2f}"
        return f"{label}: {value}"

    lines = [
        "=== resource metrics (/usr/bin/time -l) ===",
        _fmt("real_seconds", "real_seconds"),
        _fmt("user_seconds", "user_seconds"),
        _fmt("sys_seconds", "sys_seconds"),
        _fmt("voluntary_context_switches", "voluntary_context_switches"),
        _fmt("involuntary_context_switches", "involuntary_context_switches"),
        _fmt("instructions_retired", "instructions_retired"),
        _fmt("cycles_elapsed", "cycles_elapsed"),
        _fmt("peak_memory_footprint_bytes", "peak_memory_footprint_bytes"),
    ]
    peak_gb = payload.get("peak_memory_footprint_gb")
    if peak_gb is not None:
        lines.append(f"peak_memory_footprint_gb: {peak_gb}")
    max_rss = payload.get("maximum_resident_set_size_bytes")
    if max_rss is not None:
        lines.append(f"maximum_resident_set_size_bytes: {max_rss}")
        lines.append(
            f"maximum_resident_set_size_gb: {payload['maximum_resident_set_size_gb']}"
        )
    return "\n".join(lines)
